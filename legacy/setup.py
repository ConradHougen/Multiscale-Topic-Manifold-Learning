"""
Build the fast_encode_tree Cython extension in-place.

Run from the repo root:
    python legacy/setup.py build_ext --inplace

Or from inside legacy/:
    python setup.py build_ext --inplace

The compiled .so/.pyd is placed next to _tree.py.  If it is present,
_tree.py imports it automatically; otherwise it falls back to the
pure-Python implementation.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    ext_modules=cythonize(
        Extension(
            "legacy.fast_encode_tree",
            sources=[os.path.join(current_dir, "fast_encode_tree.pyx")],
            include_dirs=[np.get_include()],
        ),
        compiler_directives={"language_level": "3"},
    ),
    script_args=["build_ext", "--inplace"],
)
