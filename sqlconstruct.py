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
from itertools import chain
from functools import partial
from collections import defaultdict

import sqlalchemy
from sqlalchemy.sql import ColumnElement
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.util import immutabledict, ImmutableContainer, OrderedSet
from sqlalchemy.orm.query import Query as _SAQuery, _QueryEntity
from sqlalchemy.orm.attributes import QueryableAttribute, InstrumentedAttribute


_PY3 = sys.version_info[0] == 3

_SQLA_ge_09 = sqlalchemy.__version__ >= '0.9'


if _PY3:
    import builtins
    def _exec_in(source, globals_dict):
        getattr(builtins, 'exec')(source, globals_dict)

    _range = range
    _iteritems = dict.items
    _im_func = lambda m: m

else:
    def _exec_in(source, globals_dict):
        exec('exec source in globals_dict')

    _range = xrange
    _iteritems = dict.iteritems
    _im_func = lambda m: m.im_func


if _SQLA_ge_09:
    from sqlalchemy.orm.query import Bundle as _Bundle
else:
    class _Bundle(object):
        def __init__(self, name, *exprs, **kw):
            pass


__all__ = (
    'ConstructQuery', 'construct_query_maker', 'Construct',
    'ObjectSubQuery', 'CollectionSubQuery',
    'RelativeObjectSubQuery', 'RelativeCollectionSubQuery',
    'if_', 'apply_', 'map_', 'get_', 'define',
    'QueryMixin',
)


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


class _QueryPlan(object):

    def __init__(self):
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

    def process_rows(self, rows, session):
        children = self.query_children(None)

        results = [[((self.query_id(None), row),) for row in rows]]
        results.extend(child.__execute__(self, None, rows, session)
                       for child in children)

        return {0: [tuple(chain(*r)) for r in zip(*results)]}


class _Scope(object):

    def __init__(self, query_plan, query=None, parent=None):
        self.query_plan = query_plan
        self.query = query
        self.parent = parent

        if query and parent:
            self.query_plan.add_query(query, parent.query)
            for expr in query.__requires__():
                self.query_plan.add_expr(query, expr)

    def lookup(self, column):
        scope = self
        while scope.query:
            if scope.query.__contains_column__(column):
                return scope
            scope = scope.parent
        return scope

    def root(self):
        scope = self
        while scope.query:
            scope = scope.parent
        return scope

    def nested(self, query):
        ext_expr = query.__reference__()
        if ext_expr is not None:
            scope = self.lookup(ext_expr)
            self.query_plan.add_expr(scope.query, ext_expr)
        else:
            scope = self.root()
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
                yield dict(chain(_iteritems(result), item))
        return loop


class _SubQueryBase(_SAQuery):
    __hash = None

    def __hash__(self):
        if self.__hash is not None:
            id_, hash_ = self.__hash
            if id(self) == id_:
                return hash_
        return super(_SubQueryBase, self).__hash__()

    def __set_hash__(self, hash):
        self.__hash = (id(self), hash)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __contains_column__(self, column):
        return any(el.c.contains_column(column)
                   for el in self.statement.froms)

    def __reference__(self):
        return None

    def __requires__(self):
        return tuple()

    def __execute__(self, query_plan, outer_query, outer_rows, session):
        raise NotImplementedError


class ObjectSubQuery(_SubQueryBase):

    def __init__(self, expr, scoped_session=None):
        self.__scoped_session = scoped_session
        super(ObjectSubQuery, self).__init__([expr])

    def __execute__(self, query_plan, outer_query, outer_rows, session):
        if self.__scoped_session is not None:
            session = self.__scoped_session.registry()

        self_id = query_plan.query_id(self)
        columns = query_plan.query_columns(self)

        rows = (
            self
            .with_session(session)
            .with_entities(*columns)
            .limit(1)
            .all()
        )

        results = [[((self_id, row),) for row in rows]]
        if rows:
            results.extend(child.__execute__(query_plan, self, rows, session)
                           for child in query_plan.query_children(self))
            merged = [tuple(chain(*r)) for r in zip(*results)][0]
        else:
            merged = ((self_id, tuple(None for _ in columns)),)

        return [merged for _ in outer_rows]


class CollectionSubQuery(_SubQueryBase):

    def __init__(self, expr, scoped_session=None):
        self.__scoped_session = scoped_session
        super(CollectionSubQuery, self).__init__([expr])

    def __execute__(self, query_plan, outer_query, outer_rows, session):
        if self.__scoped_session is not None:
            session = self.__scoped_session.registry()

        self_id = query_plan.query_id(self)
        columns = query_plan.query_columns(self)

        rows = (
            self
            .with_session(session)
            .with_entities(*columns)
            .all()
        )

        results = [[((self_id, row),) for row in rows]]
        if rows:
            results.extend(child.__execute__(query_plan, self, rows, session)
                           for child in query_plan.query_children(self))
            merged = ((self_id, [tuple(chain(*r)) for r in zip(*results)]),)
        else:
            merged = ((self_id, []),)

        return [merged for _ in outer_rows]


class RelativeObjectSubQuery(_SubQueryBase):

    def __init__(self, ext_expr, int_expr, scoped_session=None):
        self.__ext_expr = ext_expr
        self.__int_expr = int_expr
        self.__scoped_session = scoped_session
        super(RelativeObjectSubQuery, self).__init__([int_expr])

    @classmethod
    def from_relation(cls, relation_property):
        ext_expr, int_expr = relation_property.local_remote_pairs[0]
        query = cls(ext_expr, int_expr)
        query.__set_hash__(hash((cls, relation_property)))
        return query

    def __reference__(self):
        return self.__ext_expr

    def __requires__(self):
        return [self.__int_expr]

    def __execute__(self, query_plan, outer_query, outer_rows, session):
        if self.__scoped_session is not None:
            session = self.__scoped_session.registry()

        self_id = query_plan.query_id(self)
        columns = query_plan.query_columns(self)

        ext_col_id = query_plan.column_id(outer_query, self.__ext_expr)
        ext_col_values = [row[ext_col_id] for row in outer_rows]

        if outer_rows:
            rows = (
                self
                .with_session(session)
                .with_entities(*chain(columns, (self.__int_expr,)))
                .filter(self.__int_expr.in_(set(ext_col_values) - {None}))
                .all()
            )
        else:
            rows = []

        results = [[((self_id, row),) for row in rows]]
        if rows:
            results.extend(child.__execute__(query_plan, self, rows, session)
                           for child in query_plan.query_children(self))

        # merging query results and putting them into mapping
        mapping = {}
        for r in zip(*results):
            r = tuple(chain(*r))
            _, row = r[0]
            # last column in the row is `self.__int_expr`
            mapping[row[-1]] = r

        # null values are used to fill results of the subquery, when outer query
        # row doesn't have corresponding row in this subquery (for example,
        # foreign key is null)
        nulls = ((self_id, tuple(None for _ in columns)),)

        return [mapping.get(val, nulls) for val in ext_col_values]


class RelativeCollectionSubQuery(_SubQueryBase):

    def __init__(self, ext_expr, int_expr, scoped_session=None):
        self.__ext_expr = ext_expr
        self.__int_expr = int_expr
        self.__scoped_session = scoped_session
        super(RelativeCollectionSubQuery, self).__init__([int_expr])

    @classmethod
    def from_relation(cls, relation_property):
        ext_expr, int_expr = relation_property.local_remote_pairs[0]
        query = cls(ext_expr, int_expr)
        if relation_property.secondary is not None:
            query = query.join(relation_property.mapper.class_,
                               relation_property.secondaryjoin)
        query.__set_hash__(hash((cls, relation_property)))
        return query

    def __reference__(self):
        return self.__ext_expr

    def __requires__(self):
        return [self.__int_expr]

    def __execute__(self, query_plan, outer_query, outer_rows, session):
        if self.__scoped_session is not None:
            session = self.__scoped_session.registry()

        self_id = query_plan.query_id(self)
        columns = query_plan.query_columns(self)

        ext_col_id = query_plan.column_id(outer_query, self.__ext_expr)
        ext_col_values = [row[ext_col_id] for row in outer_rows]

        if ext_col_values:
            rows = (
                self
                .with_session(session)
                .with_entities(*chain(columns, (self.__int_expr,)))
                .filter(self.__int_expr.in_(set(ext_col_values) - {None}))
                .all()
            )
        else:
            rows = []

        results = [[((self_id, row),) for row in rows]]
        if rows:
            results.extend(child.__execute__(query_plan, self, rows, session)
                           for child in query_plan.query_children(self))

        # merging query results and putting them into mapping
        mapping = defaultdict(list)
        for r in zip(*results):
            r = tuple(chain(*r))
            _, row = r[0]
            # last column in the row is `self.__int_expr`
            mapping[row[-1]].append(r)

        return [((self_id, mapping[val]),) for val in ext_col_values]


class _ConstructQueryBase(object):

    def __init__(self, spec, scoped_session=None):
        self.__keys, values = zip(*spec.items()) if spec else [(), ()]
        self.__scope = _Scope(_QueryPlan())
        self.__procs = [_get_value_processor(self.__scope, val)
                        for val in values]
        self.__scoped_session = scoped_session
        self.__session = None

        columns = self.__scope.query_plan.query_columns(None)
        super(_ConstructQueryBase, self).__init__(columns)

    @property
    def session(self):
        if self.__session is not None:
            return self.__session
        if self.__scoped_session is not None:
            return self.__scoped_session.registry()

    @session.setter
    def session(self, value):
        self.__session = value

    def __iter__(self):
        rows = list(super(_ConstructQueryBase, self).__iter__())
        result = self.__scope.query_plan.process_rows(rows, self.session)
        for r in self.__scope.gen_loop()(result):
            values = [proc(r) for proc in self.__procs]
            yield Object(zip(self.__keys, values))


def construct_query_maker(base_cls):
    return type('ConstructQuery', (_ConstructQueryBase, base_cls), {})


ConstructQuery = construct_query_maker(_SAQuery)


class Construct(_Bundle):
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


class _Processable(object):

    def __processor__(self, scope):
        raise NotImplementedError


def _get_value_processor(scope, value):
    if isinstance(value, ColumnElement):
        return scope.add(value)
    elif isinstance(value, QueryableAttribute):
        return _get_value_processor(scope, value.__clause_element__())
    elif isinstance(value, _Processable):
        return value.__processor__(scope)
    else:
        return lambda result, _value=value: _value


class if_(_Processable):

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


class apply_(_Processable):

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


class map_(_Processable):

    def __init__(self, func, collection):
        if isinstance(collection, (CollectionSubQuery, RelativeCollectionSubQuery)):
            sub_query = collection
        else:
            sub_query = (RelativeCollectionSubQuery
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


class get_(_Processable):

    def __init__(self, func, obj):
        if isinstance(obj, (ObjectSubQuery, RelativeObjectSubQuery)):
            sub_query = obj
        else:
            sub_query = RelativeObjectSubQuery.from_relation(obj.property)
        self._func = func
        self._sub_query = sub_query

    def __processor__(self, scope):
        nested_scope = scope.nested(self._sub_query)
        return _get_value_processor(nested_scope, self._func)


class _ArgNameHelper(object):

    def __init__(self, name):
        self.__argname__ = name

    def __getattr__(self, name):
        return _ArgNameAttrHelper(self.__argname__ + '.' + name)


class _ArgNameAttrHelper(_ArgNameHelper):

    def __getattr__(self, name):
        raise AttributeError('It is not allowed to access second-level '
                             'attributes in the function definition')


class _ArgValueHelper(object):

    def __init__(self, arg):
        self.__value__ = arg

    def __getattr__(self, name):
        if (isinstance(self.__value__, InstrumentedAttribute) and
            isinstance(self.__value__.property, RelationshipProperty)):
            column = getattr(self.__value__.property.mapper.class_, name)
            return _ArgValueAttrHelper(get_(column, self.__value__))
        return _ArgValueAttrHelper(getattr(self.__value__, name))


class _ArgValueAttrHelper(_ArgValueHelper):

    def __getattr__(self, name):
        raise AttributeError('It is not allowed to access second-level '
                             'attributes in the function definition')


def _get_definition(func, args):
    body, helpers = func(*[_ArgValueHelper(arg) for arg in args])
    return apply_(body, args=[h.__value__ for h in helpers])


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

    definition_src = (
        'def {name}{signature}:\n'
        '    return __defn__(__func__, [{args}])\n'
        .format(
            name=func.__name__,
            signature=signature,
            args=', '.join(spec.args),
        )
    )
    definition_eval_dict = {
        '__defaults__': spec.defaults,
        '__defn__': _get_definition,
        '__func__': func,
    }
    _exec_in(compile(definition_src, func.__module__, 'single'),
             definition_eval_dict)
    definition = definition_eval_dict[func.__name__]

    body, arg_name_helpers = func(*[_ArgNameHelper(arg) for arg in spec.args])
    body_args = ', '.join(h.__argname__ for h in arg_name_helpers)

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
