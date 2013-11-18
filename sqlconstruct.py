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
import abc
import inspect
from operator import attrgetter, itemgetter, methodcaller
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
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __columns__(self):
        pass

    @abc.abstractmethod
    def __process__(self, values_map):
        pass


def _get_value_processor(value):
    if isinstance(value, ColumnElement):
        return itemgetter(hash(value))
    elif isinstance(value, QueryableAttribute):
        return itemgetter(hash(value.__clause_element__()))
    elif isinstance(value, Processable):
        return value.__process__
    else:
        return lambda values_map: value


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
        self._spec_keys, self._spec_values = zip(*(
            _iteritems(spec)
        )) if spec else [(), ()]
        self._columns = tuple(set(chain(*_map(_yield_columns,
                                              self._spec_values))))
        self._column_hashes = tuple(_map(hash, self._columns))
        self._processors = tuple(_map(_get_value_processor, self._spec_values))
        super(Construct, self).__init__(None, *self._columns)

    def _from_row(self, row):
        values_map = dict(_zip(self._column_hashes, row))
        mc = methodcaller('__call__', values_map)
        return Object(_zip(self._spec_keys, _map(mc, self._processors)))

    def from_query(self, query):
        query = query.with_entities(*self._columns)
        return list(_map(self._from_row, query))

    def create_row_processor(self, query, procs, labels):
        def proc(row, result):
            return self._from_row([proc(row, None) for proc in procs])
        return proc


class if_(Processable):

    def __init__(self, condition, then_=None, else_=None):
        self._cond = condition
        self._then = then_
        self._else = else_

        self._cond_proc = _get_value_processor(condition)
        self._then_proc = _get_value_processor(then_)
        self._else_proc = _get_value_processor(else_)

    def __columns__(self):
        for obj in (self._cond, self._then, self._else):
            for column in _yield_columns(obj):
                yield column

    def __process__(self, values_map):
        if self._cond_proc(values_map):
            return self._then_proc(values_map)
        else:
            return self._else_proc(values_map)


class apply_(Processable):

    def __init__(self, func, args=None, kwargs=None):
        self._func = func
        self._args = args or []
        self._kw_keys, self._kw_values = zip(*(
            _iteritems(kwargs)
        )) if kwargs else [(), ()]
        self._args_procs = tuple(_map(_get_value_processor, self._args))
        self._kwargs_procs = tuple(_map(_get_value_processor, self._kw_values))
        self._has_kwargs = bool(kwargs)

    def __columns__(self):
        for arg in set(self._args).union(self._kw_values):
            for column in _yield_columns(arg):
                yield column

    def __process__(self, values_map):
        mc = methodcaller('__call__', values_map)
        args = _map(mc, self._args_procs)
        if self._has_kwargs:
            kwargs_values = _map(mc, self._kwargs_procs)
            kwargs = dict(_zip(self._kw_keys, kwargs_values))
            return self._func(*args, **kwargs)
        else:
            return self._func(*args)


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
