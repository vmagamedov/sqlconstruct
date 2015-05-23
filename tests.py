import sys
import pickle
import inspect
import operator
import collections
from pstats import Stats
from cProfile import Profile

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

import sqlalchemy
from sqlalchemy import Table, Column, String, Integer, ForeignKey
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, Query as QueryBase, relationship, aliased
from sqlalchemy.orm import subqueryload, scoped_session, sessionmaker
from sqlalchemy.sql.operators import ColumnOperators
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta


PY3 = sys.version_info[0] == 3
SQLA_ge_08 = sqlalchemy.__version__ >= '0.8'
SQLA_ge_09 = sqlalchemy.__version__ >= '0.9'


if PY3:
    _range = range
else:
    _range = xrange

if SQLA_ge_08:
    from sqlalchemy.util import KeyedTuple
else:
    from sqlalchemy.util import NamedTuple as KeyedTuple


from sqlconstruct import Construct, Object, apply_, if_, define, QueryMixin
from sqlconstruct import ConstructQuery, bind, map_, get_, _Scope, _QueryPlan
from sqlconstruct import ObjectSubQuery, CollectionSubQuery
from sqlconstruct import RelativeObjectSubQuery, RelativeCollectionSubQuery


if SQLA_ge_09:
    class Query(QueryBase):
        pass
else:
    class Query(QueryMixin, QueryBase):
        pass


capitalize = lambda s: s.capitalize()


@define
def defined_func(a, b, extra_id=0, extra_name=''):
    """doc"""
    def body(a_id, a_name, b_id, b_name, extra_id, extra_name):
        return a_id, b_id, extra_id, a_name, b_name, extra_name
    return body, [a.id, a.name, b.id, b.name, extra_id, extra_name]


def columns_set(processable):
    scope = _Scope(_QueryPlan())
    processable.__processor__(scope)
    return set(scope.query_plan.query_columns(None))


def proceed(processable, mapping):
    scope = _Scope(_QueryPlan())
    processor = processable.__processor__(scope)
    columns = scope.query_plan.query_columns(None)
    result = {0: [mapping[col] for col in columns]}
    return processor(result)


class BaseMeta(DeclarativeMeta):

    def __new__(mcs, name, bases, attrs):
        attrs.setdefault('__tablename__', name.lower())
        attrs.setdefault('id', Column(Integer, primary_key=True))
        return DeclarativeMeta.__new__(mcs, name, bases, attrs)


class TestConstruct(unittest.TestCase):

    def setUp(self):
        engine = create_engine('sqlite://')
        base_cls = declarative_base(metaclass=BaseMeta)

        self.a_cls = type('A', (base_cls,), dict(
            name=Column(String),
        ))

        self.b_cls = type('B', (base_cls,), dict(
            name=Column(String),
            a_id=Column(Integer, ForeignKey('a.id')),
            a=relationship('A'),
        ))

        base_cls.metadata.create_all(engine)
        self.session = Session(engine, query_cls=Query)

        self.session.add_all([
            self.b_cls(name='b1', a=self.a_cls(name='a1')),
            self.b_cls(name='b2', a=self.a_cls(name='a2')),
        ])

    def test_object_interface(self):
        obj = Object({'a': 1, 'b': 2})
        self.assertEqual(repr(obj), 'Object({})'.format(repr({'a': 1, 'b': 2})))
        self.assertTrue(isinstance(obj, collections.Mapping), type(obj))
        self.assertEqual(obj.a, 1)
        self.assertEqual(obj['a'], 1)
        self.assertEqual(obj.b, 2)
        self.assertEqual(obj['b'], 2)
        self.assertEqual(dict(obj), {'a': 1, 'b': 2})
        with self.assertRaises(KeyError):
            _ = obj['foo']
        with self.assertRaises(AttributeError):
            _ = obj.foo

    def test_object_pickling(self):
        ref = {'a': 1, 'b': 2}

        o1 = pickle.loads(pickle.dumps(Object(ref), 0))
        self.assertIs(type(o1), Object)
        self.assertEqual(dict(o1), ref)

        o2 = pickle.loads(pickle.dumps(Object(ref), 1))
        self.assertIs(type(o2), Object)
        self.assertEqual(dict(o2), ref)

        o3 = pickle.loads(pickle.dumps(Object(ref), 2))
        self.assertIs(type(o3), Object)
        self.assertEqual(dict(o3), ref)

    def test_object_immutability(self):
        obj = Object({'foo': 'bar'})

        with self.assertRaises(TypeError):
            obj.foo = 'baz'

        with self.assertRaises(TypeError):
            obj['foo'] = 'baz'

        with self.assertRaises(TypeError):
            del obj.foo

        with self.assertRaises(TypeError):
            del obj['foo']

        with self.assertRaises(TypeError):
            obj.clear()

        with self.assertRaises(TypeError):
            obj.pop('foo', None)

        with self.assertRaises(TypeError):
            obj.popitem()

        with self.assertRaises(TypeError):
            obj.setdefault('foo', 'baz')

        with self.assertRaises(TypeError):
            obj.update({'foo': 'baz'})

    def test_scalar_construct(self):
        struct = Construct({'foo': 1, 'bar': '2'})
        s = struct._from_row([])
        self.assertEqual(s.foo, 1)
        self.assertEqual(s.bar, '2')

    def test_basic_construct(self):
        struct = Construct({
            'a_id': self.a_cls.id,
            'a_name': self.a_cls.name,
        })
        self.assertEqual(set(struct._columns), {
            self.a_cls.__table__.c.id,
            self.a_cls.__table__.c.name,
        })
        result = {
            self.a_cls.__table__.c.id: 1,
            self.a_cls.__table__.c.name: 'a1',
        }
        row = [result[col] for col in struct._columns]
        s = struct._from_row(row)
        self.assertEqual(s.a_id, 1)
        self.assertEqual(s.a_name, 'a1')

    def test_nested_construct(self):
        struct = Construct({
            'a_id': apply_(operator.add, [self.a_cls.id, 5]),
            'a_name': apply_(operator.concat, [self.a_cls.name, '-test']),
        })
        self.assertEqual(set(struct._columns), {
            self.a_cls.__table__.c.id,
            self.a_cls.__table__.c.name,
        })
        result = {
            self.a_cls.__table__.c.id: 1,
            self.a_cls.__table__.c.name: 'a1',
        }
        row = [result[col] for col in struct._columns]
        s = struct._from_row(row)
        self.assertEqual(s.a_id, 1 + 5)
        self.assertEqual(s.a_name, 'a1' + '-test')

    def test_apply(self):
        add = lambda a, b, c=30, d=400: a + b + c + d

        min_pos_apply = apply_(add, [1, 2])
        self.assertEqual(columns_set(min_pos_apply), set())
        self.assertEqual(proceed(min_pos_apply, {}), 1 + 2 + 30 + 400)

        min_kw_apply = apply_(add, [], {'a': 1, 'b': 2})
        self.assertEqual(columns_set(min_kw_apply), set())
        self.assertEqual(proceed(min_kw_apply, {}), 1 + 2 + 30 + 400)

        max_pos_apply = apply_(add, [1, 2, 33, 444])
        self.assertEqual(columns_set(max_pos_apply), set())
        self.assertEqual(proceed(max_pos_apply, {}), 1 + 2 + 33 + 444)

        max_kw_apply = apply_(add, [], {'a': 1, 'b': 2, 'c': 33, 'd': 444})
        self.assertEqual(columns_set(max_kw_apply), set())
        self.assertEqual(proceed(max_kw_apply, {}), 1 + 2 + 33 + 444)

        mixed_apply = apply_(add, [1, 2], {'c': 33, 'd': 444})
        self.assertEqual(columns_set(mixed_apply), set())
        self.assertEqual(proceed(mixed_apply, {}), 1 + 2 + 33 + 444)

    def test_apply_with_columns(self):
        f1 = self.a_cls.id
        f2 = self.b_cls.id
        c1 = self.a_cls.__table__.c.id
        c2 = self.b_cls.__table__.c.id
        fn1 = func.count(self.a_cls.id)
        fn2 = func.count(self.b_cls.id)

        add = lambda a, b: a + b

        apl1 = apply_(add, [f1], {'b': f2})
        self.assertEqual(columns_set(apl1), {c1, c2})
        self.assertEqual(proceed(apl1, {c1: 3, c2: 4}), 3 + 4)

        apl2 = apply_(add, [c1], {'b': c2})
        self.assertEqual(columns_set(apl2), {c1, c2})
        self.assertEqual(proceed(apl1, {c1: 4, c2: 5}), 4 + 5)

        apl3 = apply_(add, [fn1], {'b': fn2})
        self.assertEqual(columns_set(apl3), {fn1, fn2})
        self.assertEqual(proceed(apl3, {fn1: 5, fn2: 6}), 5 + 6)

    def test_nested_apply(self):
        c1 = self.a_cls.__table__.c.id
        c2 = self.b_cls.__table__.c.id

        add = lambda a, b: a + b

        apl = apply_(add, [
            apply_(add, [
                apply_(add, [
                    0,
                    1,
                ]),
                apply_(add, [
                    2,
                    apply_(add, [
                        3,
                        c1,  # 4
                    ]),
                ]),
            ]),
            apply_(add, [
                apply_(add, [
                    apply_(add, [
                        c2,  # 5
                        6,
                    ]),
                    7,
                ]),
                apply_(add, [
                    8,
                    9,
                ]),
            ]),
        ])
        self.assertEqual(columns_set(apl), {c1, c2})
        self.assertEqual(proceed(apl, {c1: 4, c2: 5}), sum(range(10)))

    def test_if(self):
        add = lambda a, b: a + b
        c1 = self.a_cls.__table__.c.id
        c2 = self.a_cls.__table__.c.name
        c3 = self.b_cls.__table__.c.id
        c4 = self.b_cls.__table__.c.name

        if1 = if_(True, then_=1, else_=2)
        self.assertEqual(columns_set(if1), set())
        self.assertEqual(proceed(if1, {}), 1)

        if2 = if_(False, then_=1, else_=2)
        self.assertEqual(columns_set(if2), set())
        self.assertEqual(proceed(if2, {}), 2)

        if3 = if_(c1, then_=c2, else_=c3)
        self.assertEqual(columns_set(if3), {c1, c2, c3})
        self.assertEqual(proceed(if3, {c1: 0, c2: 3, c3: 6}), 6)
        self.assertEqual(proceed(if3, {c1: 1, c2: 3, c3: 6}), 3)

        if4 = if_(c1, then_=apply_(add, [c2, c3]), else_=apply_(add, [c3, c4]))
        self.assertEqual(columns_set(if4), {c1, c2, c3, c4})
        self.assertEqual(proceed(if4, {c1: 0, c2: 2, c3: 3, c4: 4}), 3 + 4)
        self.assertEqual(proceed(if4, {c1: 1, c2: 2, c3: 3, c4: 4}), 2 + 3)

    def test_defined_signatures(self):
        obj_spec = inspect.getargspec(defined_func)
        self.assertEqual(obj_spec.args, ['a', 'b', 'extra_id', 'extra_name'])
        self.assertEqual(obj_spec.varargs, None)
        self.assertEqual(obj_spec.keywords, None)
        self.assertEqual(obj_spec.defaults, (0, ''))

        defn_spec = inspect.getargspec(defined_func.defn)
        self.assertEqual(defn_spec.args, ['a', 'b', 'extra_id', 'extra_name'])
        self.assertEqual(defn_spec.varargs, None)
        self.assertEqual(defn_spec.keywords, None)
        self.assertEqual(defn_spec.defaults, (0, ''))

        func_spec = inspect.getargspec(defined_func.func)
        self.assertEqual(func_spec.args, ['a_id', 'a_name', 'b_id', 'b_name',
                                          'extra_id', 'extra_name'])
        self.assertEqual(func_spec.varargs, None)
        self.assertEqual(func_spec.keywords, None)
        self.assertEqual(func_spec.defaults, None)

    def test_defined_calls(self):
        c1 = self.a_cls.__table__.c.id
        c2 = self.a_cls.__table__.c.name
        c3 = self.b_cls.__table__.c.id
        c4 = self.b_cls.__table__.c.name

        self.assertEqual(
            defined_func(self.a_cls(id=1, name='foo'),
                         self.b_cls(id=2, name='bar'),
                         extra_id=3,
                         extra_name='baz'),
            (1, 2, 3, 'foo', 'bar', 'baz'),
        )

        self.assertEqual(
            defined_func(self.a_cls(id=1, name='foo'),
                         None,
                         extra_id=3,
                         extra_name='baz'),
            (1, None, 3, 'foo', None, 'baz'),
        )

        self.assertEqual(
            defined_func(None,
                         self.b_cls(id=2, name='bar'),
                         extra_id=3,
                         extra_name='baz'),
            (None, 2, 3, None, 'bar', 'baz'),
        )

        apl1 = defined_func.defn(self.a_cls, self.b_cls,
                                 extra_id=3, extra_name='baz')
        self.assertTrue(isinstance(apl1, apply_), type(apl1))
        self.assertEqual(columns_set(apl1), {c1, c2, c3, c4})
        self.assertEqual(
            proceed(apl1, {c1: 1, c2: 'foo', c3: 2, c4: 'bar'}),
            (1, 2, 3, 'foo', 'bar', 'baz'),
        )

        apl2 = defined_func.defn(self.a_cls, self.b_cls,
                                 extra_id=c1, extra_name=c2)
        self.assertTrue(isinstance(apl2, apply_), type(apl2))
        self.assertEqual(columns_set(apl2), {c1, c2, c3, c4})
        self.assertEqual(
            proceed(apl2, {c1: 1, c2: 'foo', c3: 2, c4: 'bar'}),
            (1, 2, 1, 'foo', 'bar', 'foo'),
        )

        apl3 = defined_func.defn(self.a_cls, self.b_cls,
                                 extra_id=apply_(operator.add, [c1, c3]),
                                 extra_name=apply_(operator.concat, [c2, c4]))
        self.assertTrue(isinstance(apl3, apply_), type(apl3))
        self.assertEqual(columns_set(apl3), {c1, c2, c3, c4})
        self.assertEqual(
            proceed(apl3, {c1: 1, c2: 'foo', c3: 2, c4: 'bar'}),
            (1, 2, (1 + 2), 'foo', 'bar', ('foo' + 'bar')),
        )

        self.assertEqual(
            defined_func.func(1, 'foo', 2, 'bar', 3, 'baz'),
            (1, 2, 3, 'foo', 'bar', 'baz'),
        )

    def test_defined_meta(self):
        self.assertEqual(defined_func.__doc__, "doc")
        self.assertEqual(defined_func.__module__, "tests")

    def test_pipe_operator(self):
        struct = Construct({
            'a_name_hash': apply_(capitalize, [self.a_cls.name]) | hash,
        })

        s1, s2 = self.session.query(struct).order_by(self.a_cls.name).all()

        self.assertEqual(s1.a_name_hash, hash('A1'))
        self.assertEqual(s2.a_name_hash, hash('A2'))

    def test_query_count(self):
        query = self.session.query(
            Construct({'a_id': self.a_cls.id,
                       'a_name': self.a_cls.name}),
        )
        self.assertEqual(query.count(), 2)

    def test_query_expr(self):
        query = self.session.query(
            Construct({'a_id': self.a_cls.id,
                       'a_name': func.upper(self.a_cls.name)}),
        )

        s1, s2 = query.order_by(self.a_cls.name.asc()).all()

        self.assertEqual(s1.a_name, 'A1')
        self.assertEqual(s2.a_name, 'A2')

    def test_query_single_entity(self):
        query = self.session.query(
            Construct({'a_id': self.a_cls.id,
                       'a_name': self.a_cls.name}),
        )

        s1, s2 = query.all()

        self.assertTrue(isinstance(s1, Object), type(s1))
        self.assertEqual(s1.a_id, 1)
        self.assertEqual(s1.a_name, 'a1')

        self.assertTrue(isinstance(s2, Object), type(s2))
        self.assertEqual(s2.a_id, 2)
        self.assertEqual(s2.a_name, 'a2')

    def test_query_row(self):
        query = self.session.query(
            self.a_cls.id,
            Construct({'a_id': self.a_cls.id,
                       'a_name': self.a_cls.name}),
            self.a_cls.name,
        )

        r1, r2 = query.all()

        self.assertTrue(isinstance(r1, KeyedTuple), type(r1))
        self.assertEqual(r1.id, 1)
        self.assertEqual(r1.name, 'a1')
        self.assertTrue(isinstance(r1[1], Object), type(r1[1]))
        self.assertEqual(r1[1].a_id, 1)
        self.assertEqual(r1[1].a_name, 'a1')

        self.assertTrue(isinstance(r2, KeyedTuple), type(r2))
        self.assertEqual(r2.id, 2)
        self.assertEqual(r2.name, 'a2')
        self.assertTrue(isinstance(r2[1], Object), type(r2[1]))
        self.assertEqual(r2[1].a_id, 2)
        self.assertEqual(r2[1].a_name, 'a2')

    def test_query_aliased_models(self):
        a1_cls = aliased(self.a_cls, name='A1')
        a2_cls = aliased(self.a_cls, name='A2')

        query = (
            self.session.query(
                Construct({'a1_id': a1_cls.id,
                           'a1_name': a1_cls.name,
                           'a2_id': a2_cls.id,
                           'a2_name': a2_cls.name}),
            )
            .select_from(a1_cls)
            .join(a2_cls, a2_cls.id == a1_cls.id + 1)
        )

        statement = str(query)
        self.assertIn('"A1".id AS "A1_id"', statement)
        self.assertIn('"A1".name AS "A1_name"', statement)
        self.assertIn('"A2".id AS "A2_id"', statement)
        self.assertIn('"A2".name AS "A2_name"', statement)

        s, = query.all()

        self.assertTrue(isinstance(s, Object), type(s))
        self.assertEqual(s.a1_id, 1)
        self.assertEqual(s.a1_name, 'a1')
        self.assertEqual(s.a2_id, 2)
        self.assertEqual(s.a2_name, 'a2')

    def test_query_labeled_columns(self):
        a1_cls = aliased(self.a_cls, name='A1')
        a2_cls = aliased(self.a_cls, name='A2')

        query = (
            self.session.query(
                Construct({'a1_id': a1_cls.id.label('__a1_id__'),
                           'a1_name': a1_cls.name.label('__a1_name__'),
                           'a2_id': a2_cls.id.label('__a2_id__'),
                           'a2_name': a2_cls.name.label('__a2_name__')}),
            )
            .select_from(a1_cls)
            .join(a2_cls, a2_cls.id == a1_cls.id + 1)
        )

        statement = str(query)
        self.assertIn('"A1".id AS __a1_id__', statement)
        self.assertIn('"A1".name AS __a1_name__', statement)
        self.assertIn('"A2".id AS __a2_id__', statement)
        self.assertIn('"A2".name AS __a2_name__', statement)

        s, = query.all()

        self.assertTrue(isinstance(s, Object), type(s))
        self.assertEqual(s.a1_id, 1)
        self.assertEqual(s.a1_name, 'a1')
        self.assertEqual(s.a2_id, 2)
        self.assertEqual(s.a2_name, 'a2')

    def test_query_with_explicit_join(self):
        query = (
            self.session.query(
                Construct({'a_id': self.a_cls.id,
                           'a_name': self.a_cls.name,
                           'b_id': self.b_cls.id,
                           'b_name': self.b_cls.name}),
            )
            .join(self.b_cls.a)
        )

        s1, s2 = query.all()

        self.assertTrue(isinstance(s1, Object), type(s1))
        self.assertEqual(s1.a_id, 1)
        self.assertEqual(s1.a_name, 'a1')
        self.assertEqual(s1.b_id, 1)
        self.assertEqual(s1.b_name, 'b1')

        self.assertTrue(isinstance(s2, Object), type(s2))
        self.assertEqual(s2.a_id, 2)
        self.assertEqual(s2.a_name, 'a2')
        self.assertEqual(s2.b_id, 2)
        self.assertEqual(s2.b_name, 'b2')

    @unittest.skipIf(SQLA_ge_09, 'SQLAlchemy < 0.9')
    def test_query_with_implicit_join_lt_09(self):
        from sqlalchemy.exc import InvalidRequestError

        with self.assertRaises(InvalidRequestError) as e1:
            (
                self.session.query(
                    Construct({'a_id': self.a_cls.id,
                               'a_name': self.a_cls.name,
                               'b_id': self.b_cls.id,
                               'b_name': self.b_cls.name}),
                )
                .join(self.a_cls)
            )
        if SQLA_ge_08:
            self.assertIn("Don't know how to join", e1.exception.args[0])
        else:
            self.assertEqual(e1.exception.args[0],
                             "Could not find a FROM clause to join from")

        with self.assertRaises(InvalidRequestError) as e2:
            (
                self.session.query(
                    Construct({'a_id': self.a_cls.id,
                               'a_name': self.a_cls.name,
                               'b_id': self.b_cls.id,
                               'b_name': self.b_cls.name}),
                )
                .join(self.b_cls)
            )
        if SQLA_ge_08:
            self.assertIn("Don't know how to join", e2.exception.args[0])
        else:
            self.assertEqual(e2.exception.args[0],
                             "Could not find a FROM clause to join from")

    @unittest.skip('optional')
    def test_performance(self):

        @define
        def test_func(a, b):
            def body(a_id, a_name, b_id, b_name):
                pass
            return body, [a.id, a.name, b.id, b.name]

        struct = Construct({
            'r1': if_(self.a_cls.id,
                      then_=test_func.defn(self.a_cls, self.b_cls)),
            'r2': if_(self.a_cls.name,
                      then_=test_func.defn(self.a_cls, self.b_cls)),
            'r3': if_(self.b_cls.id,
                      then_=test_func.defn(self.a_cls, self.b_cls)),
            'r4': if_(self.b_cls.name,
                      then_=test_func.defn(self.a_cls, self.b_cls)),
        })

        row = (
            self.session.query(*struct._columns)
            .join(self.b_cls.a)
            .first()
        )

        # warm-up
        for _ in _range(5000):
            struct._from_row(row)

        profile1 = Profile()
        profile1.enable()

        for _ in _range(5000):
            struct._from_row(row)

        profile1.disable()
        out1 = StringIO()
        stats1 = Stats(profile1, stream=out1)
        stats1.strip_dirs()
        stats1.sort_stats('calls').print_stats(10)
        print(out1.getvalue().lstrip())
        out1.close()

        row = (
            self.session.query(
                self.a_cls.id.label('a_id'),
                self.a_cls.name.label('a_name'),
                self.b_cls.id.label('b_id'),
                self.b_cls.name.label('b_name'),
            )
            .join(self.b_cls.a)
            .first()
        )

        def make_object(row):
            Object(dict(
                r1=(
                    test_func.func(row.a_id, row.a_name, row.b_id, row.b_name)
                    if row.a_id else None
                ),
                r2=(
                    test_func.func(row.a_id, row.a_name, row.b_id, row.b_name)
                    if row.a_name else None
                ),
                r3=(
                    test_func.func(row.a_id, row.a_name, row.b_id, row.b_name)
                    if row.b_id else None
                ),
                r4=(
                    test_func.func(row.a_id, row.a_name, row.b_id, row.b_name)
                    if row.b_name else None
                ),
            ))

        # warm-up
        for _ in _range(5000):
            make_object(row)

        profile2 = Profile()
        profile2.enable()

        for _ in _range(5000):
            make_object(row)

        profile2.disable()
        out2 = StringIO()
        stats2 = Stats(profile2, stream=out2)
        stats2.strip_dirs()
        stats2.sort_stats('calls').print_stats(10)
        print(out2.getvalue().lstrip())
        out2.close()

        self.assertEqual(stats1.total_calls, stats2.total_calls)


class TestSubQueries(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine('sqlite://')
        self.base_cls = declarative_base(metaclass=BaseMeta)

    def init(self):
        self.base_cls.metadata.create_all(self.engine)
        session = scoped_session(sessionmaker())
        session.configure(bind=self.engine)
        return session

    def test_many_to_one(self):

        class A(self.base_cls):
            name = Column(String)
            b_id = Column(Integer, ForeignKey('b.id'))
            b = relationship('B')

        class B(self.base_cls):
            name = Column(String)

        session = self.init()
        b1, b2, b3 = B(name='b1'), B(name='b2'), B(name='b3')
        session.add_all([
            A(name='a1', b=b1), A(name='a2', b=b1), A(name='a3'),
            A(name='a4', b=b2), A(name='a5'), A(name='a6', b=b2),
            A(name='a7'), A(name='a8', b=b3), A(name='a9', b=b3),
        ])
        session.commit()

        query = (
            ConstructQuery({
                'a_name': A.name,
                'b_name': get_(if_(B.id, apply_(capitalize, [B.name]), '~'),
                               A.b),
            })
            .with_session(session.registry())
        )
        self.assertEqual(
            tuple(dict(obj) for obj in query.all()),
            ({'a_name': 'a1', 'b_name': 'B1'},
             {'a_name': 'a2', 'b_name': 'B1'},
             {'a_name': 'a3', 'b_name': '~'},
             {'a_name': 'a4', 'b_name': 'B2'},
             {'a_name': 'a5', 'b_name': '~'},
             {'a_name': 'a6', 'b_name': 'B2'},
             {'a_name': 'a7', 'b_name': '~'},
             {'a_name': 'a8', 'b_name': 'B3'},
             {'a_name': 'a9', 'b_name': 'B3'}),
        )

    def test_one_to_one(self):

        class A(self.base_cls):
            name = Column(String)
            b = relationship('B', uselist=False)

        class B(self.base_cls):
            name = Column(String)
            a_id = Column(Integer, ForeignKey('a.id'))

        session = self.init()
        session.add_all([
            A(name='a1', b=B(name='b1')),
            A(name='a2'),
            B(name='b2'),
            A(name='a3', b=B(name='b3')),
        ])
        session.commit()

        query = (
            ConstructQuery({
                'a_name': A.name,
                'b_name': get_(if_(B.id, apply_(capitalize, [B.name]), '~'),
                               A.b),
            })
            .with_session(session.registry())
        )
        self.assertEqual(
            tuple(dict(obj) for obj in query.all()),
            ({'a_name': 'a1', 'b_name': 'B1'},
             {'a_name': 'a2', 'b_name': '~'},
             {'a_name': 'a3', 'b_name': 'B3'}),
        )

    def test_one_to_many(self):

        class A(self.base_cls):
            name = Column(String)
            b_list = relationship('B')

        class B(self.base_cls):
            name = Column(String)
            a_id = Column(Integer, ForeignKey('a.id'))

        session = self.init()
        session.add_all([
            A(name='a1', b_list=[B(name='b1'), B(name='b2'), B(name='b3')]),
            A(name='a2', b_list=[B(name='b4'), B(name='b5'), B(name='b6')]),
            A(name='a3', b_list=[B(name='b7'), B(name='b8'), B(name='b9')]),
        ])
        session.commit()

        query = (
            ConstructQuery({
                'a_name': A.name,
                'b_names': map_(apply_(capitalize, [B.name]), A.b_list),
            })
            .with_session(session.registry())
        )
        self.assertEqual(
            tuple(dict(obj) for obj in query.all()),
            ({'a_name': 'a1', 'b_names': ['B1', 'B2', 'B3']},
             {'a_name': 'a2', 'b_names': ['B4', 'B5', 'B6']},
             {'a_name': 'a3', 'b_names': ['B7', 'B8', 'B9']}),
        )

    def test_many_to_many(self):
        ab_table = Table(
            'a_b',
            self.base_cls.metadata,
            Column('a_id', Integer, ForeignKey('a.id')),
            Column('b_id', Integer, ForeignKey('b.id'))
        )

        class A(self.base_cls):
            name = Column(String)
            b_list = relationship('B', secondary=ab_table)

        class B(self.base_cls):
            name = Column(String)
            a_list = relationship('A', secondary=ab_table)

        session = self.init()
        a1, a2, a3, a4 = A(name='a1'), A(name='a2'), A(name='a3'), A(name='a4')
        b1, b2, b3, b4 = B(name='b1'), B(name='b2'), B(name='b3'), B(name='b4')
        a1.b_list = [b2, b3, b4]
        a2.b_list = [b1, b3, b4]
        a3.b_list = [b1, b2, b4]
        a4.b_list = [b1, b2, b3]
        session.add_all([a1, a2, a3, a4])
        session.commit()

        q1 = (
            ConstructQuery({
                'a_name': A.name,
                'b_names': map_(apply_(capitalize, [B.name]), A.b_list),
            })
            .with_session(session.registry())
            .order_by(A.name)
        )
        self.assertEqual(
            tuple((obj.a_name, set(obj.b_names)) for obj in q1.all()),
            (
                ('a1', {'B2', 'B3', 'B4'}),
                ('a2', {'B1', 'B3', 'B4'}),
                ('a3', {'B1', 'B2', 'B4'}),
                ('a4', {'B1', 'B2', 'B3'}),
            )
        )

        q2 = (
            ConstructQuery({
                'b_name': B.name,
                'a_names': map_(apply_(capitalize, [A.name]), B.a_list),
            })
            .with_session(session.registry())
            .order_by(B.name)
        )
        self.assertEqual(
            tuple((obj.b_name, set(obj.a_names)) for obj in q2.all()),
            (
                ('b1', {'A2', 'A3', 'A4'}),
                ('b2', {'A1', 'A3', 'A4'}),
                ('b3', {'A1', 'A2', 'A4'}),
                ('b4', {'A1', 'A2', 'A3'}),
            ),
        )

    def test_non_empty_in_op_in_relative_subqueries(self):
        in_op = ColumnOperators.in_

        class EmptyInOpError(Exception):
            pass

        def wrapper(self, values):
            if not values:
                raise EmptyInOpError
            return in_op(self, values)

        patcher = patch.object(ColumnOperators, 'in_', wrapper)

        class A(self.base_cls):
            value = Column(String)

        class B(self.base_cls):
            value = Column(String)

        session = self.init()
        session.add_all([A(), A(), A()])
        session.commit()

        obj_sq = RelativeObjectSubQuery(A.value, B.value)
        obj_query = ConstructQuery({'a_id': A.id,
                                    'b_id': get_(B.id, obj_sq)},
                                   session)
        with patcher:
            obj_query.all()

        list_sq = RelativeCollectionSubQuery(A.value, B.value)
        list_query = ConstructQuery({'a_id': A.id,
                                     'b_id': map_(B.id, list_sq)},
                                    session)
        with patcher:
            list_query.all()

    def test_nested(self):
        """
        A <- B -> C -> D <- E
        """
        class A(self.base_cls):
            name = Column(String)

        class B(self.base_cls):
            name = Column(String)
            a_id = Column('a_id', Integer, ForeignKey('a.id'))
            a = relationship('A', backref='b_list')
            c_id = Column('c_id', Integer, ForeignKey('c.id'))
            c = relationship('C', backref='b_list')

        class C(self.base_cls):
            name = Column(String)
            d_id = Column('d_id', Integer, ForeignKey('d.id'))
            d = relationship('D', backref='c_list')

        class D(self.base_cls):
            name = Column(String)

        class E(self.base_cls):
            name = Column(String)
            d_id = Column('d_id', Integer, ForeignKey('d.id'))
            d = relationship('D', backref='e_list')

        session = self.init()
        a1, a2, a3 = A(name='a1'), A(name='a2'), A(name='a3')
        d1 = D(name='d1',
               c_list=[C(name='c1',
                         b_list=[B(name='b1'),
                                 B(name='b2', a=a2),
                                 B(name='b3', a=a3)]),
                       C(name='c2',
                         b_list=[B(name='b4', a=a1),
                                 B(name='b5'),
                                 B(name='b6', a=a3)]),
                       C(name='c3',
                         b_list=[B(name='b7', a=a1),
                                 B(name='b8', a=a2),
                                 B(name='b9')])],
               e_list=[E(name='e1'), E(name='e2'), E(name='e3')])
        session.add_all([a1, a2, a3, d1])
        session.commit()

        # A <- B -> C
        r1 = tuple(dict(obj) for obj in ConstructQuery({
            'a_name': A.name,
            'b_names': map_(B.name, A.b_list),
            'c_names': map_(get_(C.name, B.c), A.b_list)
        }).with_session(session.registry()).order_by(A.name).all())
        self.assertEqual(r1, (
            {'a_name': 'a1', 'b_names': ['b4', 'b7'], 'c_names': ['c2', 'c3']},
            {'a_name': 'a2', 'b_names': ['b2', 'b8'], 'c_names': ['c1', 'c3']},
            {'a_name': 'a3', 'b_names': ['b3', 'b6'], 'c_names': ['c1', 'c2']},
        ))

        # B -> C -> D
        r2 = tuple(dict(obj) for obj in ConstructQuery({
            'b_name': B.name,
            'c_name': get_(C.name, B.c),
            'd_name': get_(get_(D.name, C.d), B.c),
        }).with_session(session.registry()).order_by(B.name).all())
        self.assertEqual(r2, (
            {'b_name': 'b1', 'c_name': 'c1', 'd_name': 'd1'},
            {'b_name': 'b2', 'c_name': 'c1', 'd_name': 'd1'},
            {'b_name': 'b3', 'c_name': 'c1', 'd_name': 'd1'},
            {'b_name': 'b4', 'c_name': 'c2', 'd_name': 'd1'},
            {'b_name': 'b5', 'c_name': 'c2', 'd_name': 'd1'},
            {'b_name': 'b6', 'c_name': 'c2', 'd_name': 'd1'},
            {'b_name': 'b7', 'c_name': 'c3', 'd_name': 'd1'},
            {'b_name': 'b8', 'c_name': 'c3', 'd_name': 'd1'},
            {'b_name': 'b9', 'c_name': 'c3', 'd_name': 'd1'},
        ))

        # C -> D <- E
        r3 = tuple(dict(obj) for obj in ConstructQuery({
            'c_name': C.name,
            'd_name': get_(D.name, C.d),
            'e_names': get_(map_(E.name, D.e_list), C.d),
        }).with_session(session.registry()).order_by(C.name).all())
        self.assertEqual(r3, (
            {'c_name': 'c1', 'd_name': 'd1', 'e_names': ['e1', 'e2', 'e3']},
            {'c_name': 'c2', 'd_name': 'd1', 'e_names': ['e1', 'e2', 'e3']},
            {'c_name': 'c3', 'd_name': 'd1', 'e_names': ['e1', 'e2', 'e3']},
        ))

        # D <- C <- B
        r4 = dict(ConstructQuery({
            'd_name': D.name,
            'c_names': map_(C.name, D.c_list),
            'b_names': map_(map_(B.name, C.b_list), D.c_list),
        }).with_session(session.registry()).order_by(D.name).one())
        self.assertEqual(r4['d_name'], 'd1')
        self.assertEqual(set(r4['c_names']), {'c1', 'c2', 'c3'})
        self.assertEqual(set(map(frozenset, r4['b_names'])), {
            frozenset({'b1', 'b2', 'b3'}),
            frozenset({'b4', 'b5', 'b6'}),
            frozenset({'b7', 'b8', 'b9'}),
        })

    def test_with_define(self):

        class A(self.base_cls):
            name = Column(String)
            b_id = Column(Integer, ForeignKey('b.id'))
            b = relationship('B')

        class B(self.base_cls):
            name = Column(String)

        @define
        def full_name(a, b):
            def body(a_name, b_name):
                return ' '.join((a_name.capitalize(), b_name.capitalize()))
            return body, [a.name, b.name]

        session = self.init()
        b1, b2, b3 = B(name='b1'), B(name='b2'), B(name='b3')
        session.add_all([
            A(name='a1', b=b1), A(name='a2', b=b1), A(name='a3', b=b1),
            A(name='a4', b=b2), A(name='a5', b=b2), A(name='a6', b=b2),
            A(name='a7', b=b3), A(name='a8', b=b3), A(name='a9', b=b3),
        ])
        session.commit()

        query = (
            ConstructQuery({
                'full_name': full_name.defn(A, A.b),
            })
            .with_session(session.registry())
            .order_by(A.name)
        )

        self.assertEqual(
            tuple(dict(obj) for obj in query.all()),
            ({'full_name': 'A1 B1'},
             {'full_name': 'A2 B1'},
             {'full_name': 'A3 B1'},
             {'full_name': 'A4 B2'},
             {'full_name': 'A5 B2'},
             {'full_name': 'A6 B2'},
             {'full_name': 'A7 B3'},
             {'full_name': 'A8 B3'},
             {'full_name': 'A9 B3'}),
        )

    def test_with_scoped_session(self):

        class A(self.base_cls):
            name = Column(String)

        session = self.init()
        session.add_all([A(name='a1'), A(name='a2'), A(name='a3')])
        session.commit()

        q1 = ConstructQuery({'name': A.name}, session)
        self.assertEqual({obj.name for obj in q1.all()}, {'a1', 'a2', 'a3'})

        q2 = ConstructQuery({'name': A.name})
        with self.assertRaises(AttributeError):
            q2.all()

    def test_with_custom_queries(self):

        class A(self.base_cls):
            name = Column(String)

        class B(self.base_cls):
            name = Column(String)

        session = self.init()
        session.add_all([
            A(name='a1'), A(name='a2'), A(name='a3'),
            B(name='b1'), B(name='b2'), B(name='b3'),
        ])
        session.commit()

        sq1 = CollectionSubQuery(B).order_by(B.name.desc())
        q1 = (
            ConstructQuery({'a_name': A.name, 'b_list': map_(B.name, sq1)},
                           session)
            .order_by(A.name)
        )
        self.assertEqual({(obj.a_name, tuple(obj.b_list)) for obj in q1.all()},
                         {('a1', ('b3', 'b2', 'b1')),
                          ('a2', ('b3', 'b2', 'b1')),
                          ('a3', ('b3', 'b2', 'b1'))})

        sq2 = ObjectSubQuery(B).order_by(B.name.desc())
        q2 = (
            ConstructQuery({'a_name': A.name, 'b_name': get_(B.name, sq2)},
                           session)
            .order_by(A.name)
        )
        self.assertEqual({(obj.a_name, obj.b_name) for obj in q2.all()},
                         {('a1', 'b3'), ('a2', 'b3'), ('a3', 'b3')})

    def test_query_entities_modification(self):

        class A(self.base_cls):
            name = Column(String)

        session = self.init()
        session.add_all([A(name='a1'), A(name='a2'), A(name='a3')])
        session.commit()

        count = ConstructQuery({'name': A.name}, session).count()
        self.assertEqual(count, 3)

        with self.assertRaises(NotImplementedError):
            ConstructQuery({'id': A.id}).add_columns(A.name)

        with self.assertRaises(NotImplementedError):
            ConstructQuery({'id': A.id}).add_entity(A)

    def test_bound_to_query_expressions(self):

        class A(self.base_cls):
            name = Column(String)
            b_list = relationship('B')

        class B(self.base_cls):
            name = Column(String)
            a_id = Column(Integer, ForeignKey('a.id'))

        session = self.init()
        session.add_all([
            A(name='a1', b_list=[B(name='b1')]),
            A(name='a2', b_list=[B(name='b4'), B(name='b5')]),
            A(name='a3', b_list=[B(name='b7'), B(name='b8'), B(name='b9')]),
        ])
        session.commit()

        sq = (
            RelativeObjectSubQuery.from_relation(A.b_list)
            .group_by(B.a_id)
        )
        query = (
            ConstructQuery({
                'a_name': A.name,
                'b_count': get_(bind(func.count(), sq), sq),
            })
            .order_by(A.name.asc())
            .with_session(session.registry())
        )
        self.assertEqual(
            [dict(obj) for obj in query.all()],
            [{'a_name': 'a1', 'b_count': 1},
             {'a_name': 'a2', 'b_count': 2},
             {'a_name': 'a3', 'b_count': 3}],
        )

    def test_bound_to_query_reference(self):

        class A(self.base_cls):
            name = Column(String)
            b_list = relationship('B')

        class B(self.base_cls):
            name = Column(String)
            a_id = Column(Integer, ForeignKey('a.id'))
            ref = Column(Integer)

        class C(self.base_cls):
            name = Column(String)
            ref = Column(Integer)

        session = self.init()
        session.add_all([
            A(name='a1', b_list=[B(name='b1', ref=-1), B(name='b2', ref=2)]),
            A(name='a2', b_list=[B(name='b3', ref=-3), B(name='b4', ref=4)]),
            C(name='c1', ref=1), C(name='c2', ref=-1),
            C(name='c3', ref=2), C(name='c4', ref=-2),
            C(name='c5', ref=3), C(name='c6', ref=-3),
            C(name='c7', ref=4), C(name='c8', ref=-4),
        ])
        session.commit()

        sq1 = (
            RelativeCollectionSubQuery.from_relation(A.b_list)
            .order_by(B.name.asc())
        )
        sq2 = (
            RelativeCollectionSubQuery(bind(func.abs(B.ref), sq1), func.abs(C.ref))
            .order_by(C.name.asc())
        )
        query = (
            ConstructQuery({
                'a_name': A.name,
                'c_names': map_(map_(C.name, sq2), sq1),
            })
            .order_by(A.name.asc())
            .with_session(session.registry())
        )
        self.assertEqual(
            [dict(obj) for obj in query.all()],
            [
                {'a_name': 'a1', 'c_names': [['c1', 'c2'], ['c3', 'c4']]},
                {'a_name': 'a2', 'c_names': [['c5', 'c6'], ['c7', 'c8']]},
            ],
        )

    def test_bound_to_query_entities(self):

        class A(self.base_cls):
            name = Column(String)
            c_id = Column(Integer, ForeignKey('c.id'))
            c = relationship('C')

        class B(self.base_cls):
            name = Column(String)
            a_id = Column(Integer, ForeignKey('a.id'))
            c_id = Column(Integer, ForeignKey('c.id'))
            a = relationship('A', backref='b_list')
            c = relationship('C')

        class C(self.base_cls):
            name = Column(String)

        session = self.init()
        session.add_all([
            A(name='a1', c=C(name='c11'), b_list=[
                B(name='b1', c=C(name='c21')),
                B(name='b2', c=C(name='c22')),
            ]),
            A(name='a2', c=C(name='c12'), b_list=[
                B(name='b3', c=C(name='c23')),
                B(name='b4', c=C(name='c24')),
            ]),
            A(name='a3', c=C(name='c13'), b_list=[
                B(name='b5', c=C(name='c25')),
                B(name='b6', c=C(name='c26')),
            ]),
        ])
        session.commit()

        @define
        def concat_names(a, c1, b, c2):
            def body(a_name, c1_name, b_name, c2_name):
                return ' '.join((a_name, c1_name, b_name, c2_name))
            return body, [a.name, c1.name, b.name, c2.name]

        sq = (
            RelativeCollectionSubQuery.from_relation(A.b_list)
            .outerjoin(B.c)
            .order_by(B.name.asc())
        )
        query = (
            ConstructQuery({
                'names': map_(concat_names.defn(A, bind(C, None), B, C), sq),
            })
            .outerjoin(A.c)
            .order_by(A.name.asc())
            .with_session(session.registry())
        )
        self.assertEqual(
            [dict(obj) for obj in query.all()],
            [{'names': ['a1 c11 b1 c21', 'a1 c11 b2 c22']},
             {'names': ['a2 c12 b3 c23', 'a2 c12 b4 c24']},
             {'names': ['a3 c13 b5 c25', 'a3 c13 b6 c26']}],
        )

    @unittest.skip('optional')
    def test_performance(self):

        class A(self.base_cls):
            name = Column(String)
            b_list = relationship('B')

        class B(self.base_cls):
            name = Column(String)
            a_id = Column(Integer, ForeignKey('a.id'))

        session = self.init()
        session.add_all([A(name='a_%d' % i,
                           b_list=[B(name='b_%d_%d' % (i, j))
                                   for j in range(10)])
                         for i in _range(20)])
        session.commit()
        session.remove()

        @define
        def b_name(a, b):
            def body(a_id, a_name, b_id, b_name):
                return '%d - %s - %d - %s' % (a_id, a_name, b_id, b_name)
            return body, [a.id, a.name, b.id, b.name]

        tuples_query = session.query(A.id, A.name)
        tuples_subquery = session.query(B.id, B.a_id, B.name)

        def perform_tuples_query():
            a_rows = tuples_query.all()
            _tuples_subquery = (
                tuples_subquery
                .filter(B.a_id.in_({a_row.id for a_row in a_rows}))
            )
            b_rows_map = collections.defaultdict(list)
            for row in _tuples_subquery.all():
                b_rows_map[row.a_id].append(row)

            return [{'a_name': a_row.name,
                     'b_names': [b_name.func(a_row.id, a_row.name,
                                             b_row.id, b_row.name)
                                 for b_row in b_rows_map[a_row.id]]}
                    for a_row in a_rows]

        construct_query = (
            ConstructQuery({
                'a_name': A.name,
                'b_names': map_(b_name.defn(A, B), A.b_list),
            }, session)
        )

        def perform_construct_query():
            return construct_query.all()

        def perform_objects_query():
            session.remove()
            query = session.query(A).options(subqueryload(A.b_list))
            return [{'a_name': a.name,
                     'b_names': [b_name(a, b) for b in a.b_list]}
                    for a in query.all()]


        def do(impl, count):
            # warm-up
            _res = [impl() for _ in _range(count)]

            profile = Profile()
            profile.enable()

            res = [impl() for _ in _range(count)]

            profile.disable()
            out = StringIO()
            stats = Stats(profile, stream=out)
            stats.strip_dirs()
            stats.sort_stats('calls').print_stats(10)
            print(out.getvalue().lstrip())
            out.close()
            return _res, res

        res = []
        res.extend(do(perform_tuples_query, 1000))
        res.extend(do(perform_construct_query, 1000))
        res.extend(do(perform_objects_query, 1000))
        1/0


class TestBugs(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine('sqlite://')
        self.base_cls = declarative_base(metaclass=BaseMeta)

    def init(self):
        self.base_cls.metadata.create_all(self.engine)
        session = scoped_session(sessionmaker())
        session.configure(bind=self.engine)
        return session

    def test_nested_get_with_null_value_in_outer_expr(self):
        class A(self.base_cls):
            name = Column(String)
            b_id = Column('b_id', Integer, ForeignKey('b.id'))
            b = relationship('B', backref='a_list')

        class B(self.base_cls):
            name = Column(String)
            c_id = Column('c_id', Integer, ForeignKey('c.id'))
            c = relationship('C', backref='b_list')

        class C(self.base_cls):
            name = Column(String)

        session = self.init()
        c1 = C(name='c1')
        b1, b2 = B(name='b1', c=c1), B(name='b2')
        a1, a2, a3 = A(name='a1', b=b1), A(name='a2', b=b2), A(name='a3')
        session.add_all([a1, a2, a3, b1, b2, c1])
        session.commit()

        res = tuple(dict(obj) for obj in ConstructQuery({
            'a_name': A.name,
            'b_name': get_(B.name, A.b),
            'c_name': get_(get_(C.name, B.c), A.b),
        }).with_session(session.registry()).order_by(A.name).all())
        self.assertEqual(res, (
            {'a_name': 'a1', 'b_name': 'b1', 'c_name': 'c1'},
            {'a_name': 'a2', 'b_name': 'b2', 'c_name': None},
            {'a_name': 'a3', 'b_name': None, 'c_name': None},
        ))
