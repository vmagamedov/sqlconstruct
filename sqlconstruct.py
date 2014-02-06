"""
    sqlconstruct
    ~~~~~~~~~~~~

    Functional approach to query database using SQLAlchemy.

    Example::

        >>> product_struct = Construct({
        ...     'name': Product.name,
        ...     'url': url_for_product.defn(Product),
        ...     'image_url': if_(
        ...         Image.id,
        ...         then_=url_for_image.defn(Image, 100, 100),
        ...         else_=None,
        ...     ),
        ... })
        ...
        >>> product = (
        ...     session.query(product_struct)
        ...     .outerjoin(Product.image)
        ...     .first()
        ... )
        ...
        >>> product.name
        u'Foo product'
        >>> product.url
        '/p1-foo-product.html'
        >>> product.image_url
        '//images.example.st/1-100x100-foo.jpg'

    See README.rst for more examples and documentation.

    :copyright: (c) 2013 Vladimir Magamedov.
    :license: BSD, see LICENSE.txt for more details.
"""
import sys
import inspect
from operator import attrgetter
from functools import partial, wraps
from collections import defaultdict

import sqlalchemy
from sqlalchemy.sql import ColumnElement
from sqlalchemy.util import immutabledict, ImmutableContainer, OrderedSet
from sqlalchemy.orm.query import Query as _SAQuery, _QueryEntity
from sqlalchemy.orm.attributes import QueryableAttribute


PY3 = sys.version_info[0] == 3

SQLA_ge_09 = sqlalchemy.__version__ >= '0.9'


if PY3:
    import builtins
    def _exec_in(source, globals_dict):
        getattr(builtins, 'exec')(source, globals_dict)

    _map = map
    _range = range
    _iteritems = dict.items
    _im_func = lambda m: m

else:
    def _exec_in(source, globals_dict):
        exec('exec source in globals_dict')

    from itertools import imap as _map
    _range = xrange
    _iteritems = dict.iteritems
    _im_func = lambda m: m.im_func


if SQLA_ge_09:
    from sqlalchemy.orm.query import Bundle
else:
    class Bundle(object):
        def __init__(self, name, *exprs, **kw):
            pass


__all__ = (
    'ConstructQuery', 'Construct',
    'if_', 'apply_', 'map_', 'get_', 'define',
    'QueryMixin',
)


class _QueryPlan(object):

    def __init__(self, session=None):
        self._session = session
        self._queries = OrderedSet({None})
        self._columns = defaultdict(OrderedSet)
        self._children = defaultdict(OrderedSet)

    def add_query(self, query, parent_query):
        self._queries.add(query)
        self._children[parent_query].add(query)

    def add_expr(self, query, column):
        self._columns[query].add(column)

    def query_id(self, query):
        return list(self._queries).index(query)

    def column_id(self, query, column):
        return list(self._columns[query]).index(column)

    def query_columns(self, query):
        return tuple(self._columns.get(query) or ())

    def query_children(self, query):
        return tuple(self._children.get(query) or ())

    def process_rows(self, rows):
        children = self.query_children(None)
        queries = (None,) + children
        results = (rows,) + tuple(child.process(self, None, rows)
                                  for child in children)
        return {0: [
            tuple(zip(map(self.query_id, queries),
                      result_row))
            for result_row in zip(*results)
        ]}


class _Scope(object):

    def __init__(self, query_plan, query=None, parent=None):
        self.query_plan = query_plan
        self.query = query
        self.parent = parent

        if query and parent:
            for expr in query.__requires__():
                self.query_plan.add_query(query, parent.query)
                self.query_plan.add_expr(query, expr)

    def lookup(self, column):
        scope = self
        while scope.query:
            if column in scope.query:
                return scope
            scope = scope.parent
        return scope

    def nested(self, query):
        ext_expr = query.__reference__()
        scope = self.lookup(ext_expr)
        self.query_plan.add_expr(scope.query, ext_expr)
        return type(self)(self.query_plan, query, scope)

    def add(self, column, query=None):
        if query is None:
            query = self.lookup(column).query
        self.query_plan.add_expr(query, column)

        query_id = self.query_plan.query_id(query)
        column_id = self.query_plan.column_id(query, column)
        def proc(result, _query_id=query_id, _column_id=column_id):
            return result[_query_id][_column_id]
        return proc

    def gen_loop(self):
        def loop(result, _query_id=self.query_plan.query_id(self.query)):
            for item in result[_query_id]:
                # TODO: optimize
                yield dict(result, **dict(item))
        return loop

    def gen_getter(self):
        def getter(result, _query_id=self.query_plan.query_id(self.query)):
            return dict(result, **dict(result[_query_id]))
        return getter


class _QueryBase(object):

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __contains__(self, column):
        raise NotImplementedError

    def __reference__(self):
        return None

    def __requires__(self):
        return tuple()

    def process(self, query_plan, outer_query, outer_rows):
        raise NotImplementedError


class _ObjectSubQuery(_QueryBase):
    pass


class _CollectionSubQuery(_QueryBase):
    pass


class _RelativeObjectSubQuery(_QueryBase):

    def __init__(self, ext_expr, int_expr, query, _hash=None):
        self._ext_expr = ext_expr
        self._int_expr = int_expr
        self._sa_query = query
        self._hash = _hash or hash((type(self), ext_expr, int_expr, query))

    @classmethod
    def from_relation(cls, relation_property):
        ext_expr, int_expr = relation_property.local_remote_pairs[0]
        query = _SAQuery([relation_property.mapper.class_])
        hash_ = hash((cls, relation_property))
        return cls(ext_expr, int_expr, query, hash_)

    def __reference__(self):
        return self._ext_expr

    def __requires__(self):
        return [self._int_expr]

    def __hash__(self):
        return self._hash

    def __contains__(self, column):
        return any(el.c.contains_column(column)
                   for el in self._sa_query.statement.froms)

    def process(self, query_plan, outer_query, outer_rows):
        ext_col_id = query_plan.column_id(outer_query, self._ext_expr)
        ext_exprs = [row[ext_col_id] for row in outer_rows]
        if ext_exprs:
            columns = query_plan.query_columns(self) + (self._int_expr,)
            rows = (
                self._sa_query
                .with_session(query_plan._session.registry())
                .with_entities(*columns)
                .filter(self._int_expr.in_(set(ext_exprs)))
                .all()
            )
        else:
            rows = []

        children = query_plan.query_children(self)

        queries = (self,) + children
        results = (rows,) + tuple(child.process(query_plan, self, rows)
                                  for child in children)

        col_id = query_plan.column_id(self, self._int_expr)
        mapping = {}
        for result_row in zip(*results):
            mapping[result_row[0][col_id]] = \
                tuple(zip(map(query_plan.query_id, queries),
                          result_row))

        nulls = (
            (query_plan.query_id(self),
             tuple(None for _ in query_plan.query_columns(self))),
        )

        return [mapping.get(ext_expr) if ext_expr in mapping else nulls
                for ext_expr in ext_exprs]


class _RelativeCollectionSubQuery(_QueryBase):

    def __init__(self, ext_expr, int_expr, query, _hash=None):
        self._ext_expr = ext_expr
        self._int_expr = int_expr
        self._sa_query = query
        self._hash = _hash or hash((type(self), ext_expr, int_expr, query))

    @classmethod
    def from_relation(cls, relation_property):
        if relation_property.secondary is not None:
            ext_expr, int_expr = relation_property.local_remote_pairs[0]
            query = (_SAQuery([relation_property.mapper.class_])
                     .join(relation_property.secondary,
                           relation_property.secondaryjoin))
        else:
            ext_expr, int_expr = relation_property.local_remote_pairs[0]
            query = _SAQuery([relation_property.mapper.class_])
        hash_ = hash((cls, relation_property))
        return cls(ext_expr, int_expr, query, hash_)

    def __reference__(self):
        return self._ext_expr

    def __requires__(self):
        return [self._int_expr]

    def __hash__(self):
        return self._hash

    def __contains__(self, column):
        return any(el.c.contains_column(column)
                   for el in self._sa_query.statement.froms)

    def process(self, query_plan, outer_query, outer_rows):
        ext_col_id = query_plan.column_id(outer_query, self._ext_expr)
        ext_exprs = [row[ext_col_id] for row in outer_rows]
        if ext_exprs:
            columns = query_plan.query_columns(self) + (self._int_expr,)
            rows = (
                self._sa_query
                .with_session(query_plan._session.registry())
                .with_entities(*columns)
                .filter(self._int_expr.in_(ext_exprs))
                .all()
            )
        else:
            rows = []

        children = query_plan.query_children(self)

        queries = (self,) + children
        results = (rows,) + tuple(child.process(query_plan, self, rows)
                                  for child in children)

        col_id = query_plan.column_id(self, self._int_expr)
        groups = defaultdict(list)
        for result_row in zip(*results):
            groups[result_row[0][col_id]].append(
                tuple(zip(map(query_plan.query_id, queries),
                          result_row))
            )
        return [groups[ext_expr] for ext_expr in ext_exprs]


class Processable(object):

    def __processor__(self, scope):
        raise NotImplementedError


def _get_value_processor(scope, value):
    if isinstance(value, ColumnElement):
        return scope.add(value)
    elif isinstance(value, QueryableAttribute):
        return _get_value_processor(scope, value.__clause_element__())
    elif isinstance(value, Processable):
        return value.__processor__(scope)
    else:
        return lambda result, _value=value: _value


class Object(immutabledict):

    __new__ = dict.__new__
    __init__ = dict.__init__

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError('Constructed object has no attribute {0!r}'
                                 .format(attr))

    __delattr__ = ImmutableContainer._immutable

    def __repr__(self):
        return '{cls}({arg})'.format(cls=type(self).__name__,
                                     arg=dict.__repr__(self))

    def __reduce__(self):
        return type(self), (dict(self),)


def _proxy_query_method(unbound_method):
    func = _im_func(unbound_method)
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self._query, *args, **kwargs)
    return wrapper


def _generative_proxy_query_method(unbound_method):
    func = _im_func(unbound_method)
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        cls = type(self)
        clone = cls.__new__(cls)
        clone.__dict__.update(self.__dict__)
        clone._query = func(clone._query, *args, **kwargs)
        return clone
    return wrapper


class ConstructQuery(object):

    def __init__(self, session, spec):
        self._session = session
        self._spec = spec
        self._keys, self._values = zip(*spec.items()) if spec else [(), ()]
        self._scope = _Scope(_QueryPlan(session))
        self._processors = [_get_value_processor(self._scope, val)
                            for val in self._values]
        self._query = _SAQuery(self._scope.query_plan.query_columns(None))

    __str__     = _proxy_query_method(_SAQuery.__str__)

    join        = _generative_proxy_query_method(_SAQuery.join)
    outerjoin   = _generative_proxy_query_method(_SAQuery.outerjoin)
    filter      = _generative_proxy_query_method(_SAQuery.filter)
    order_by    = _generative_proxy_query_method(_SAQuery.order_by)

    all = _im_func(_SAQuery.all)

    def __iter__(self):
        rows = self._query.with_session(self._session.registry()).all()
        result = self._scope.query_plan.process_rows(rows)
        for r in self._scope.gen_loop()(result):
            yield Object(zip(self._keys, [proc(r) for proc in self._processors]))


class Construct(Bundle):
    single_entity = True

    def __init__(self, spec):
        self._keys, self._values = zip(*spec.items()) if spec else [(), ()]
        self._scope = _Scope(_QueryPlan())
        self._processors = [_get_value_processor(self._scope, val)
                            for val in self._values]
        self._range = tuple(_range(len(self._keys)))
        self._columns = self._scope.query_plan.query_columns(None)
        super(Construct, self).__init__(None, *self._columns)

    def _from_row(self, row):
        return Object({self._keys[i]: self._processors[i]([row])
                       for i in self._range})

    def from_query(self, query):
        query = query.with_entities(*self._columns)
        return list(self._from_row(row) for row in query)

    def create_row_processor(self, query, procs, labels):
        def proc(row, result):
            return self._from_row([row[col] for col in self._columns])
        return proc


class if_(Processable):

    def __init__(self, condition, then_=None, else_=None):
        self._cond = condition
        self._then = then_
        self._else = else_

    def __processor__(self, scope):
        def process(result,
                    cond_proc=_get_value_processor(scope, self._cond),
                    then_proc=_get_value_processor(scope, self._then),
                    else_proc=_get_value_processor(scope, self._else)):
            if cond_proc(result):
                return then_proc(result)
            else:
                return else_proc(result)
        return process


class apply_(Processable):

    def __init__(self, func, args=None, kwargs=None):
        self._func = func
        self._args = args or []
        self._kwargs = kwargs or {}

    def __processor__(self, scope):
        args = []
        eval_dict = {'__func__': self._func}

        for i, arg in enumerate(self._args):
            args.append('__proc{0}__(result)'.format(i))
            eval_dict['__proc{0}__'.format(i)] = _get_value_processor(scope, arg)

        for key, arg in self._kwargs.items():
            args.append('{0}=__{0}_proc__(result)'.format(key))
            eval_dict['__{0}_proc__'.format(key)] = _get_value_processor(scope, arg)

        processor_src = (
            'def __processor__(result):\n'
            '    return __func__({args})\n'
            .format(args=', '.join(args))
        )
        _exec_in(compile(processor_src, __name__, 'single'),
                 eval_dict)
        return eval_dict['__processor__']


class map_(Processable):

    def __init__(self, func, collection):
        if isinstance(collection, _RelativeCollectionSubQuery):
            sub_query = collection
        else:
            sub_query = (_RelativeCollectionSubQuery
                         .from_relation(collection.property))
        self._func = func
        self._sub_query = sub_query

    def __processor__(self, scope):
        nested_scope = scope.nested(self._sub_query)
        func_proc = _get_value_processor(nested_scope, self._func)
        loop = nested_scope.gen_loop()
        def process(result):
            return [func_proc(item) for item in loop(result)]
        return process


class get_(Processable):

    def __init__(self, func, obj):
        if isinstance(obj, _RelativeObjectSubQuery):
            sub_query = obj
        else:
            sub_query = _RelativeObjectSubQuery.from_relation(obj.property)
        self._func = func
        self._sub_query = sub_query

    def __processor__(self, scope):
        nested_scope = scope.nested(self._sub_query)
        func_proc = _get_value_processor(nested_scope, self._func)
        getter = nested_scope.gen_getter()
        def process(result):
            return func_proc(getter(result))
        return process


class _arg_helper(object):

    def __init__(self, name):
        self.__name__ = name

    def __getattr__(self, attr_name):
        return _arg_helper(self.__name__ + '.' + attr_name)


def define(func):
    """Universal function definition

    Example::

        >>> @define
        ... def url_for_product(product):
        ...     def body(product_id, product_name):
        ...         return '/p{id}-{name}.html'.format(
        ...             id=product_id,
        ...             name=slugify(product_name),
        ...         )
        ...     return body, [product.id, product.name]
        ...
        >>> url_for_product(product)
        '/p1-foo-product.html'
        >>> url_for_product.defn(Product)
        <sqlconstruct.apply_ at 0x000000000>
        >>> url_for_product.func(product.id, product.name)
        '/p1-foo-product.html'

    """
    spec = inspect.getargspec(func)
    assert not spec.varargs and not spec.keywords,\
        'Variable args are not supported'

    signature = inspect.formatargspec(
        args=spec.args,
        defaults=['__defaults__[{0}]'.format(i)
                  for i in range(len(spec.defaults or []))],
        formatvalue=lambda value: '=' + value,
    )

    body, arg_helpers = func(*_map(_arg_helper, spec.args))
    body_args = ', '.join(_map(attrgetter('__name__'), arg_helpers))

    definition_src = (
        'def {name}{signature}:\n'
        '    return __apply__(__body__, args=[{body_args}])\n'
        .format(
            name=func.__name__,
            signature=signature,
            body_args=body_args
        )
    )
    definition_eval_dict = {
        '__defaults__': spec.defaults,
        '__apply__': apply_,
        '__body__': body,
    }
    _exec_in(compile(definition_src, func.__module__, 'single'),
             definition_eval_dict)
    definition = definition_eval_dict[func.__name__]

    objective_src = (
        'def {name}{signature}:\n'
        '    return __body__({body_args})\n'
        .format(
            name=func.__name__,
            signature=signature,
            body_args=body_args,
        )
    )
    objective_eval_dict = {
        '__defaults__': spec.defaults,
        '__body__': body,
    }
    _exec_in(compile(objective_src, func.__module__, 'single'),
             objective_eval_dict)
    objective = objective_eval_dict[func.__name__]

    objective.func = body
    objective.defn = definition
    return objective


# SQLAlchemy < 0.9 compatibility

def _entity_wrapper(query, entity):
    if isinstance(entity, Construct):
        return _ConstructEntity(query, entity)
    else:
        return _QueryEntity(query, entity)


class _ConstructEntity(_QueryEntity):
    """Queryable construct entities

    Adapted from: http://www.sqlalchemy.org/trac/ticket/2824
    """
    filter_fn = id

    entities = ()
    entity_zero_or_selectable = None

    # hack for sqlalchemy.orm.query:Query class
    class mapper:
        class dispatch:
            append_result = False

    def __init__(self, query, struct):
        query._entities.append(self)
        self.struct = struct

    def corresponds_to(self, entity):
        return False

    def adapt_to_selectable(self, query, sel):
        query._entities.append(self)

    #def setup_entity(self, *args, **kwargs):
    #    raise NotImplementedError

    def setup_context(self, query, context):
        context.primary_columns.extend(self.struct._columns)

    def row_processor(self, query, context, custom_rows):
        def proc(row, result, _column):
            return row[_column]
        procs = [partial(proc, _column=c) for c in self.struct._columns]
        labels = [None] * len(self.struct._columns)
        return self.struct.create_row_processor(query, procs, labels), None


class QueryMixin(object):

    def _set_entities(self, entities, entity_wrapper=_entity_wrapper):
        super(QueryMixin, self)._set_entities(entities, entity_wrapper)
