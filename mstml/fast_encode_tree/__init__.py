"""
This module wraps the fast tree encoding logic with a fallback to pure Python
if the Cython extension fails to load.
"""

import warnings

try:
    # Try importing the compiled Cython version
    from .fast_encode_tree import *  # this is the compiled .so/.pyd version
except ImportError:
    # Fall back to the pure Python version
    from .fast_encode_tree_py import *

    warnings.warn(
        "Cython module 'fast_encode_tree' not compiled. "
        "Falling back to slower pure Python version.\n\n"
        "To improve performance, run:\n"
        "    python setup.py build_ext --inplace\n"
        "or:\n"
        "    pip install -e .\n",
        UserWarning
    )