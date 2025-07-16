"""
Fallback implementation for fast_encode_tree when Cython module is not available.

This module provides the same interface as the Cython version but with pure Python
implementation. It will be used automatically if the Cython module fails to import.
"""

# This is the original Python implementation
# Import everything from the original file
from .fast_encode_tree import *

import warnings
warnings.warn(
    "Using pure Python fallback for fast_encode_tree. "
    "For better performance, compile the Cython extension with: "
    "python setup.py build_ext --inplace",
    UserWarning
)