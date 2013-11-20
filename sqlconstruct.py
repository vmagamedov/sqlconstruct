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
from functools import partial
from itertools import chain

import sqlalchemy
from sqlalchemy.sql import ColumnElement
from sqlalchemy.util import immutabledict, ImmutableContainer
from sqlalchemy.orm.query import _QueryEntity
from sqlalchemy.orm.attributes import QueryableAttribute


PY3 = sys.version_info[0] == 3

SQLA_ge_09 = sqlalchemy.__version__ >= '0.9'


if PY3:
    import builtins

    def _exec_in(source, globals_dict):
        getattr(builtins, 'exec')(source, globals_dict)

    _map = map
    _zip = zip

    _iteritems = dict.items

else:
    def _exec_in(source, globals_dict):
        exec('exec source in globals_dict')

    from itertools import imap as _map, izip as _zip

    _iteritems = dict.iteritems


if SQLA_ge_09:
    from sqlalchemy.orm.query import Bundle
else:
    class Bundle(object):
        def __init__(self, name, *exprs, **kw):
            pass


__all__ = ('Construct', 'if_', 'apply_', 'define', 'QueryMixin')


class Processable(object):

    def __columns__(self):
        raise NotImplementedError

    def __processor__(self):
        raise NotImplementedError


def _get_value_processor(value):
    if isinstance(value, ColumnElement):
        return lambda row_map, row, _hash=hash(value): row[row_map[_hash]]
    elif isinstance(value, QueryableAttribute):
        return _get_value_processor(value.__clause_element__())
    elif isinstance(value, Processable):
        return value.__processor__()
    else:
        return lambda row_map, row, _value=value: _value


def _yield_columns(value):
    if isinstance(value, ColumnElement):
        yield value
    elif isinstance(value, QueryableAttribute):
        yield value.__clause_element__()
    elif isinstance(value, Processable):
        for column in value.__columns__():
            yield column


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
        return '%s(%s)' % (type(self).__name__, dict.__repr__(self))

    def __reduce__(self):
        return type(self), (dict(self),)


class Construct(Bundle):
    single_entity = True

    def __init__(self, spec):
        self._keys, self._values = zip(*spec.items()) if spec else [(), ()]
        self._columns = tuple(set(chain(*_map(_yield_columns, self._values))))
        self._processors = tuple(_map(_get_value_processor, self._values))
        self._row_map = {hash(col): i for i, col in enumerate(self._columns)}
        self._range = range(len(self._columns))
        super(Construct, self).__init__(None, *self._columns)

    def _from_row(self, row):
        return Object({self._keys[i]: self._processors[i](self._row_map, row)
                       for i in self._range})

    def from_query(self, query):
        query = query.with_entities(*self._columns)
        return list(self._from_row(row) for row in query)

    def create_row_processor(self, query, procs, labels):
        def proc(row, result):
            return self._from_row([proc(row, None) for proc in procs])
        return proc


class if_(Processable):

    def __init__(self, condition, then_=None, else_=None):
        self._cond = condition
        self._then = then_
        self._else = else_

    def __columns__(self):
        for obj in (self._cond, self._then, self._else):
            for column in _yield_columns(obj):
                yield column

    def __processor__(self):
        def process(row_map, row,
                    cond_proc=_get_value_processor(self._cond),
                    then_proc=_get_value_processor(self._then),
                    else_proc=_get_value_processor(self._else)):
            if cond_proc(row_map, row):
                return then_proc(row_map, row)
            else:
                return else_proc(row_map, row)
        return process


class apply_(Processable):

    def __init__(self, func, args=None, kwargs=None):
        self._func = func
        self._args = args or []
        self._kwargs = kwargs or {}

    def __columns__(self):
        for arg in set(self._args).union(self._kwargs.values()):
            for column in _yield_columns(arg):
                yield column

    def __processor__(self):
        args = []
        eval_dict = {'__func__': self._func}

        for i, arg in enumerate(self._args):
            args.append('__proc{i}__(row_map, row)'.format(i=i))
            eval_dict['__proc{i}__'.format(i=i)] = _get_value_processor(arg)

        for key, arg in self._kwargs.items():
            args.append('{key}=__{key}_proc__(row_map, row)'.format(key=key))
            eval_dict['__{key}_proc__'.format(key=key)] = _get_value_processor(arg)

        processor_src = (
            'def __processor__(row_map, row):\n'
            '    return __func__({args})\n'
            .format(args=', '.join(args))
        )
        _exec_in(compile(processor_src, __name__, 'single'),
                 eval_dict)
        return eval_dict['__processor__']


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
        defaults=['__defaults__[%d]' % i
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
