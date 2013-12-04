import sys
import pickle
import inspect
import operator
import collections

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import sqlalchemy
from sqlalchemy import Column, String, Integer, create_engine, ForeignKey, func
from sqlalchemy.orm import Session, Query as QueryBase, relationship, aliased
from sqlalchemy.ext.declarative import declarative_base


PY3 = sys.version_info[0] == 3
SQLA_ge_08 = sqlalchemy.__version__ >= '0.8'
SQLA_ge_09 = sqlalchemy.__version__ >= '0.9'


if SQLA_ge_08:
    from sqlalchemy.util import KeyedTuple
else:
    from sqlalchemy.util import NamedTuple as KeyedTuple


from sqlconstruct import Construct, Object, apply_, if_, define, QueryMixin


if SQLA_ge_09:
    class Query(QueryBase):
        pass
else:
    class Query(QueryMixin, QueryBase):
        pass


@define
def defined_func(a, b, extra_id=0, extra_name=''):
    def body(a_id, a_name, b_id, b_name, extra_id, extra_name):
        return a_id + b_id + extra_id, a_name + b_name + extra_name
    return body, [a.id, a.name, b.id, b.name, extra_id, extra_name]


def proceed(processable, mapping):
    keys, row = zip(*mapping.items()) if mapping else [(), ()]
    processor = processable.__processor__()
    row_map = {hash(key): i for i, key in enumerate(keys)}
    return processor(row_map, row)


class TestConstruct(unittest.TestCase):

    def setUp(self):
        engine = create_engine('sqlite://')
        base_cls = declarative_base()

        self.a_cls = type('A', (base_cls,), dict(
            __tablename__='a',
            id=Column(Integer, primary_key=True),
            name=Column(String),
        ))

        self.b_cls = type('B', (base_cls,), dict(
            __tablename__='b',
            id=Column(Integer, primary_key=True),
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
        self.assertEquals(set(struct._columns), {
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
        self.assertEquals(set(struct._columns), {
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
        self.assertEqual(set(min_pos_apply.__columns__()), set())
        self.assertEqual(proceed(min_pos_apply, {}), 1 + 2 + 30 + 400)

        min_kw_apply = apply_(add, [], {'a': 1, 'b': 2})
        self.assertEqual(set(min_kw_apply.__columns__()), set())
        self.assertEqual(proceed(min_kw_apply, {}), 1 + 2 + 30 + 400)

        max_pos_apply = apply_(add, [1, 2, 33, 444])
        self.assertEqual(set(max_pos_apply.__columns__()), set())
        self.assertEqual(proceed(max_pos_apply, {}), 1 + 2 + 33 + 444)

        max_kw_apply = apply_(add, [], {'a': 1, 'b': 2, 'c': 33, 'd': 444})
        self.assertEqual(set(max_kw_apply.__columns__()), set())
        self.assertEqual(proceed(max_kw_apply, {}), 1 + 2 + 33 + 444)

        mixed_apply = apply_(add, [1, 2], {'c': 33, 'd': 444})
        self.assertEqual(set(mixed_apply.__columns__()), set())
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
        self.assertEquals(set(apl1.__columns__()), {c1, c2})
        self.assertEqual(proceed(apl1, {c1: 3, c2: 4}), 3 + 4)

        apl2 = apply_(add, [c1], {'b': c2})
        self.assertEquals(set(apl2.__columns__()), {c1, c2})
        self.assertEqual(proceed(apl1, {c1: 4, c2: 5}), 4 + 5)

        apl3 = apply_(add, [fn1], {'b': fn2})
        self.assertEquals(set(apl3.__columns__()), {fn1, fn2})
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
        self.assertEquals(set(apl.__columns__()), {c1, c2})
        self.assertEqual(proceed(apl, {c1: 4, c2: 5}), sum(range(10)))

    def test_if(self):
        add = lambda a, b: a + b
        c1 = self.a_cls.__table__.c.id
        c2 = self.a_cls.__table__.c.name
        c3 = self.b_cls.__table__.c.id
        c4 = self.b_cls.__table__.c.name

        if1 = if_(True, then_=1, else_=2)
        self.assertEquals(set(if1.__columns__()), set())
        self.assertEqual(proceed(if1, {}), 1)

        if2 = if_(False, then_=1, else_=2)
        self.assertEquals(set(if2.__columns__()), set())
        self.assertEqual(proceed(if2, {}), 2)

        if3 = if_(c1, then_=c2, else_=c3)
        self.assertEquals(set(if3.__columns__()), {c1, c2, c3})
        self.assertEqual(proceed(if3, {c1: 0, c2: 3, c3: 6}), 6)
        self.assertEqual(proceed(if3, {c1: 1, c2: 3, c3: 6}), 3)

        if4 = if_(c1, then_=apply_(add, [c2, c3]), else_=apply_(add, [c3, c4]))
        self.assertEquals(set(if4.__columns__()), {c1, c2, c3, c4})
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
            (1 + 2 + 3, 'foo' + 'bar' + 'baz'),
        )

        apl1 = defined_func.defn(self.a_cls, self.b_cls,
                                 extra_id=3, extra_name='baz')
        self.assertTrue(isinstance(apl1, apply_), type(apl1))
        self.assertEquals(set(apl1.__columns__()), {c1, c2, c3, c4})
        self.assertEqual(
            proceed(apl1, {c1: 1, c2: 'foo', c3: 2, c4: 'bar'}),
            (1 + 2 + 3, 'foo' + 'bar' + 'baz'),
        )

        apl2 = defined_func.defn(self.a_cls, self.b_cls,
                                 extra_id=c1, extra_name=c2)
        self.assertTrue(isinstance(apl2, apply_), type(apl2))
        self.assertEquals(set(apl2.__columns__()), {c1, c2, c3, c4})
        self.assertEqual(
            proceed(apl2, {c1: 1, c2: 'foo', c3: 2, c4: 'bar'}),
            (1 + 2 + 1, 'foo' + 'bar' + 'foo'),
        )

        apl3 = defined_func.defn(self.a_cls, self.b_cls,
                                 extra_id=apply_(operator.add, [c1, c3]),
                                 extra_name=apply_(operator.concat, [c2, c4]))
        self.assertTrue(isinstance(apl3, apply_), type(apl3))
        self.assertEquals(set(apl3.__columns__()), {c1, c2, c3, c4})
        self.assertEqual(
            proceed(apl3, {c1: 1, c2: 'foo', c3: 2, c4: 'bar'}),
            (1 + 2 + (1 + 2), 'foo' + 'bar' + ('foo' + 'bar')),
        )

        self.assertEqual(
            defined_func.func(1, 'foo', 2, 'bar', 3, 'baz'),
            (1 + 2 + 3, 'foo' + 'bar' + 'baz'),
        )

    def test_query_count(self):
        query = self.session.query(
            Construct({'a_id': self.a_cls.id,
                       'a_name': self.a_cls.name}),
        )
        self.assertEqual(query.count(), 2)

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

    @unittest.skipIf(SQLA_ge_08, 'SQLAlchemy < 0.8')
    def test_query_with_implicit_join_lt_08(self):
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
        self.assertEqual(e1.exception.args[0],
                         'Could not find a FROM clause to join from')

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
        self.assertEqual(e2.exception.args[0],
                         'Could not find a FROM clause to join from')

    @unittest.skipIf(not SQLA_ge_08 or SQLA_ge_09, '0.8 <= SQLAlchemy < 0.9')
    def test_query_with_implicit_join_ge_08(self):
        from sqlalchemy.exc import NoInspectionAvailable

        with self.assertRaises(NoInspectionAvailable) as e1:
            (
                self.session.query(
                    Construct({'a_id': self.a_cls.id,
                               'a_name': self.a_cls.name,
                               'b_id': self.b_cls.id,
                               'b_name': self.b_cls.name}),
                )
                .join(self.a_cls)
            )
        self.assertIn('No inspection system is available', e1.exception.args[0])

        with self.assertRaises(NoInspectionAvailable) as e2:
            (
                self.session.query(
                    Construct({'a_id': self.a_cls.id,
                               'a_name': self.a_cls.name,
                               'b_id': self.b_cls.id,
                               'b_name': self.b_cls.name}),
                )
                .join(self.b_cls)
            )
        self.assertIn('No inspection system is available', e2.exception.args[0])

    @unittest.skip('optional')
    def test_performance(self):
        from pstats import Stats
        from cProfile import Profile
        try:
            from cStringIO import StringIO
        except ImportError:
            from io import StringIO

        _range = range if PY3 else xrange

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
