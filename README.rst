============
SQLConstruct
============

`SQLConstruct` is a functional approach to query database using `SQLAlchemy`
library. It was written to reach more speed without introducing unmaintainable
and verbose code. On the contrary, code becomes simpler, so there are less
chances of shooting yourself in the foot.

Main problems it aims to solve:

- ORM overhead in read-only ``SELECT`` queries;
- Network traffic when loading unnecessary columns;
- Code complexity;
- N+1 problem.

Final
=====

You describe what you want to get from the database:

.. code-block:: python

    from sqlconstruct import Construct, if_

    product_struct = Construct({
        'name': Product.name,
        'url': url_for_product.defn(Product),
        'image_url': if_(
            Image.id,
            then_=url_for_image.defn(Image, 100, 100),
            else_=None,
        ),
    })

And you get it. `SQLConstruct` knows which columns you need and how transform
them into suitable to use format:

.. code-block:: python

    >>> product = (
    ...     session.query(product_struct)
    ...     .outerjoin(Product.image)
    ...     .first()
    ... )
    ...
    >>> product.name
    'Foo product'
    >>> product.url
    '/p1-foo-product.html'
    >>> product.image_url
    '//images.example.st/123-100x100-foo.jpg'

Full story
==========

Basic preparations:

.. code-block:: python

    from sqlalchemy import create_engine
    from sqlalchemy import Column, Integer, String, Text, ForeignKey
    from sqlalchemy.orm import Session, relationship, eagerload
    from sqlalchemy.ext.declarative import declarative_base

    engine = create_engine('sqlite://')
    Base = declarative_base()

    class Image(Base):
        __tablename__ = 'image'

        id = Column(Integer, primary_key=True)
        name = Column(String)

    class Product(Base):
        __tablename__ = 'product'

        id = Column(Integer, primary_key=True)
        name = Column(String)
        image_id = Column(Integer, ForeignKey(Image.id))
        description = Column(Text)

        image = relationship(Image)

    Base.metadata.create_all(engine)

    session = Session(engine)
    session.add(Product(name='Foo product', image=Image(name='Foo.jpg')))
    session.commit()

    def slugify(name):
        # very dumb implementation, just for an example
        return name.lower().replace(' ', '-')

    def url_for_product(product):
        return '/p{id}-{name}.html'.format(
            id=product.id,
            name=slugify(product.name),
        )

    def url_for_image(image, width, height):
        return '//images.example.st/{id}-{width}x{height}-{name}'.format(
            id=image.id,
            width=width,
            height=height,
            name=slugify(image.name),
        )

Usual way:

.. code-block:: python

    >>> product = (
    ...     session.query(Product)
    ...     .options(eagerload(Product.image))
    ...     .first()
    ... )
    ...
    >>> product.name
    u'Foo product'
    >>> url_for_product(product)
    '/p1-foo-product.html'
    >>> url_for_image(product.image, 100, 100) if product.image else None
    '//images.example.st/1-100x100-foo.jpg'

Disadvantages:

- ``description`` column isn't deferred, it will be loaded every time;
- if you will mark ``description`` column as deferred, this can introduce N+1
  problem somewhere else in your project;
- if you forgot to ``eagerload`` ``Product.image`` you will also get N+1
  problem;
- you have to pass model instances as arguments everywhere in the project and
  this tends to code complexity, because you don't know how they will be used in
  the future;
- model instances creation isn't cheap, CPU time grows with number of columns,
  even if they are all deferred.

Initial solution:

.. code-block:: python

    from sqlconstruct import Construct, apply_, if_

    def url_for_product(product_id, product_name):
        return '/p{id}-{name}.html'.format(
            id=product_id,
            name=slugify(product_name),
        )

    def url_for_image(image_id, image_name, width, height):
        return '//images.example.st/{id}-{width}x{height}-{name}'.format(
            id=image_id,
            width=width,
            height=height,
            name=slugify(image_name),
        )

    product_struct = Construct({
        'name': Product.name,
        'url': apply_(url_for_product, args=[Product.id, Product.name]),
        'image_url': if_(
            Image.id,
            then_=apply_(url_for_image, args=[Image.id, Image.name, 100, 100]),
            else_=None,
        ),
    })

Usage:

.. code-block:: python

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

Advantages:

- you're loading only what you need, no extra network traffic, no need to
  defer/undefer columns;
- ``url_for_product`` and ``url_for_image`` functions can't add complexity,
  because they are forced to define all needed columns as arguments;
- you're working with precomputed values (urls in this example).

Disadvantages:

- code of functions is hard to refactor and reuse, because you should specify or
  pass all the arguments every time;
- you should be careful with joins, because if you wouldn't specify them
  explicitly, `SQLAlchemy` will produce cartesian product of the tables
  (``SELECT ... FROM product, image WHERE ...``), which will return wrong
  results and hurt your performance.

To address first disadvantage, `SQLConstruct` provides ``define`` decorator,
which gives you ability to define hybrid functions to use them in different
ways:

.. code-block:: python

    from sqlconstruct import define

    @define
    def url_for_product(product):
        def body(product_id, product_name):
            return '/p{id}-{name}.html'.format(
                id=product_id,
                name=slugify(product_name),
            )
        return body, [product.id, product.name]

    @define
    def url_for_image(image, width, height):
        def body(image_id, image_name, width, height):
            return '//images.example.st/{id}-{width}x{height}-{name}'.format(
                id=image_id,
                width=width,
                height=height,
                name=slugify(image_name),
            )
        return body, [image.id, image.name, width, height]

Now these functions can be used in these ways:

.. code-block:: python

    >>> product = session.query(Product).first()
    >>> url_for_product(product)  # objective style
    '/p1-foo-product.html'
    >>> url_for_product.defn(Product)  # apply_ declaration
    <sqlconstruct.apply_ at 0x000000000>
    >>> url_for_product.func(product.id, product.name)  # functional style
    '/p1-foo-product.html'

Modified final ``Construct`` definition:

.. code-block:: python

    product_struct = Construct({
        'name': Product.name,
        'url': url_for_product.defn(Product),
        'image_url': if_(
            Image.id,
            then_=url_for_image.defn(Image, 100, 100),
            else_=None,
        ),
    })

Installation
============

To install `SQLConstruct`, simply::

    pip install https://github.com/vmagamedov/sqlconstruct/archive/rev-0.2.zip

`SQLConstruct` is tested and supported on these Python versions: 2.7 and 3.3;
PyPy is also supported. Supported `SQLAlchemy` versions includes 0.7, 0.8
and 0.9.

Examples above are using `SQLAlchemy` >= 0.9, if you are using older versions,
you will have to do next changes in your project configuration:

.. code-block:: python

    from sqlconstruct import QueryMixin
    from sqlalchemy.orm.query import Query as BaseQuery

    class Query(QueryMixin, BaseQuery):
        pass

    session = Session(engine, query_cls=Query)
    
Flask-SQLAlchemy:

.. code-block:: python

    from flask.ext.sqlalchemy import SQLAlchemy

    db = SQLAlchemy(app, session_options={'query_cls': Query})

or

.. code-block:: python

    db = SQLAlchemy(session_options={'query_cls': Query})
    db.init_app(app)

License
=======

`SQLConstruct` is distributed under the BSD license. See LICENSE.txt for more
details.
