#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize

ext_modules = cythonize(
    ["subtraction_pipeline/ibme_fast_raster.pyx"]
)


setup(
    name="subtraction_pipeline",
    version="0.1",
    packages=["subtraction_pipeline"],
    ext_modules=ext_modules,
)
