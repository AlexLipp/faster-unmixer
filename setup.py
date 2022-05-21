import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cyth_scripts.pyx"),
    include_dirs=[numpy.get_include()]
)