#!/usr/bin/env python3
"""
Setup script for MSTML Framework.
"""

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os

def parse_requirements(filename):
    """Parse a pip requirements file into a list of install_requires."""
    if not os.path.exists(filename):
        return []
    with open(filename, "r") as f:
        lines = f.readlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

# Define the Cython extension
extensions = [
    Extension(
        name="mstml.fast_encode_tree.fast_encode_tree",  # Full path to Cython .pyx
        sources=["mstml/fast_encode_tree/fast_encode_tree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="mstml",
    version="1.0.0",
    author="MSTML Research Team",
    description="Multi-Scale Topic Manifold Learning Framework",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
    install_requires=parse_requirements("pip_requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
