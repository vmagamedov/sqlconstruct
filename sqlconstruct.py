"""Presents models to the views (templates)

    Example::

        product_struct = Construct(dict(
            name=Product.name,
            url=apply_(
                get_product_url_fn,
                args=[Product.id, Product.name, Company.domain],
            ),
            image_url=apply_(
                get_image_url_fn,
                args=[Image.id, Image.file_name, Image.store_type, 100, 100],
            ),
        ))

        products = (
            db.session.query(product_struct)
            .join(Product.company)
            .outerjoin(Product.main_image)
            .limit(10)
        )

"""
import sys
import abc
import inspect
from operator import attrgetter
from functools import partial
from itertools import chain

import sqlalchemy
from sqlalchemy.sql import ColumnElement
from sqlalchemy.util import OrderedDict, immutabledict, ImmutableContainer
from sqlalchemy.orm.query import _QueryEntity
from sqlalchemy.orm.attributes import QueryableAttribute


PY3 = sys.version_info[0] == 3

SQLA_ge_09 = sqlalchemy.__version__ >= '0.9'


if PY3:
    import builtins

    def _exec_in(source, globals_dict):
        getattr(builtins, 'exec')(source, globals_dict)
else:
    def _exec_in(source, globals_dict):
        exec('exec source in globals_dict')


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
    def yield_columns(self):
        pass

    @abc.abstractmethod
    def process(self, values_map):
        pass


def _get_value_from_map(values_map, value):
    if isinstance(value, ColumnElement):
        return values_map[value]
    elif isinstance(value, QueryableAttribute):
        return values_map[value.__clause_element__()]
    elif isinstance(value, Processable):
        return value.process(values_map)
    else:
        return value


def _yield_columns(value):
    if isinstance(value, ColumnElement):
        yield value
    elif isinstance(value, QueryableAttribute):
        yield value.__clause_element__()
    elif isinstance(value, Processable):
        for column in value.yield_columns():
            yield column


class Object(immutabledict):

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
        self._spec = OrderedDict(spec)
        self._columns = tuple(set(chain(*map(_yield_columns, spec.values()))))
        super(Construct, self).__init__(None, *self._columns)

    def from_row(self, row):
        values_map = dict(zip(self._columns, row))
        get_value = partial(_get_value_from_map, values_map)
        return Object(zip(
            self._spec.keys(),
            map(get_value, self._spec.values()),
        ))

    def from_query(self, query):
        query = query.with_entities(*self._columns)
        return map(self.from_row, query)

    def create_row_processor(self, query, procs, labels):
        def proc(row, result):
            return self.from_row([proc(row, None) for proc in procs])
        return proc


class if_(Processable):

    def __init__(self, condition, then_=None, else_=None):
        self.condition = condition
        self.then_ = then_
        self.else_ = else_

    def yield_columns(self):
        for obj in (self.condition, self.then_, self.else_):
            for column in _yield_columns(obj):
                yield column

    def process(self, values_map):
        get_value = partial(_get_value_from_map, values_map)
        condition = get_value(self.condition)
        if condition:
            return get_value(self.then_)
        else:
            return get_value(self.else_)


class apply_(Processable):

    def __init__(self, func, args=None, kwargs=None):
        self.func = func
        self.args = args or []
        self.kwargs = OrderedDict(kwargs or [])

    def yield_columns(self):
        for arg in set(self.args).union(self.kwargs.values()):
            for column in _yield_columns(arg):
                yield column

    def process(self, values_map):
        get_value = partial(_get_value_from_map, values_map)
        args = map(get_value, self.args)
        kwargs = dict(zip(
            self.kwargs.keys(),
            map(get_value, self.kwargs.values()),
        ))
        return self.func(*args, **kwargs)


class _arg_helper(object):

    def __init__(self, name):
        self.__name__ = name

    def __getattr__(self, attr_name):
        return _arg_helper(self.__name__ + '.' + attr_name)


def define(func):
    """Universal function definition

    Example::

        @construct.define
        def url(image, width, height, opt=5):

            def body(id_, name, store_type, width, height, opt):
                print id_, name, store_type, width, height, opt

            return body, [image.id, image.file_name, image.store_type, width,
                          height, opt]

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

    body, arg_helpers = func(*map(_arg_helper, spec.args))
    body_args = ', '.join(map(attrgetter('__name__'), arg_helpers))

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
