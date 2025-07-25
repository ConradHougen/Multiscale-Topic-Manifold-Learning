# mstml/__init__.py
import importlib
from types import ModuleType

__version__ = "1.0.0"

__all__ = [
    "core",
    "data_loaders",
    "dataframe_schema",
    "text_preprocessing",
    "utils",
    "mstml_utils",
    "gdltm_utils",
    "model_evaluation",
    "author_disambiguation",
    "__version__",
]

# Map attribute -> submodule for lazy loading
_lazy_submodules = {
    "core": "mstml.core",
    "data_loaders": "mstml.data_loaders",
    "dataframe_schema": "mstml.dataframe_schema",
    "text_preprocessing": "mstml.text_preprocessing",
    "utils": "mstml.utils",
    "mstml_utils": "mstml.mstml_utils",
    "gdltm_utils": "mstml.gdltm_utils",
    "model_evaluation": "mstml.model_evaluation",
    "author_disambiguation": "mstml.author_disambiguation",
}

def __getattr__(name: str) -> ModuleType:
    if name in _lazy_submodules:
        module = importlib.import_module(_lazy_submodules[name])
        globals()[name] = module  # cache for future
        return module
    raise AttributeError(f"module 'mstml' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_submodules.keys()))
