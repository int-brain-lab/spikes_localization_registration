#!/usr/bin/env python
from setuptools import setup, Extension


with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines() if not x.startswith("git+")]


setup(
    name="subtraction_pipeline",
    setup_requires=require + [
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
    ],
    version="0.1",
    packages=["subtraction_pipeline"],
    ext_modules=[
        Extension(
            'sub',
            sources=["subtraction_pipeline/ibme_fast_raster.pyx"],
        ),
    ],
)
