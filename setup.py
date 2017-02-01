from os.path import join, dirname
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open(join(dirname(__file__), 'README.rst')) as f:
    README = f.read()


setup(
    name='SQLConstruct',
    version='0.2.3',
    description='Functional approach to query database using SQLAlchemy',
    long_description=README,
    author='Vladimir Magamedov',
    author_email='vladimir@magamedov.com',
    url='https://github.com/vmagamedov/sqlconstruct',
    py_modules=['sqlconstruct'],
    install_requires=['SQLAlchemy>=0.9'],
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Database :: Front-Ends',
    ],
)
