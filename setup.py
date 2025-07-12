#!/usr/bin/env python3
"""
Setup script for MSTML Framework Cython extensions.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define Cython extensions
extensions = [
    Extension(
        "source.fast_encode_tree",
        ["source/fast_encode_tree.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="mstml",
    version="1.0.0",
    description="Multi-Scale Topic Manifold Learning Framework",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    zip_safe=False,
)