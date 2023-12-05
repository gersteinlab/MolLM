from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("Transformer_M/data/algos.pyx"),
    include_dirs=[numpy.get_include()]
)