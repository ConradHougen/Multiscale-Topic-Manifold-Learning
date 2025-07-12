"""
Multi-Scale Topic Manifold Learning (MSTML)

A scalable method for predicting collaborative behaviors using textual data 
and probabilistic information geometry of author-topical interests.
"""

__version__ = "1.0.0"
__author__ = "MSTML Research Team"

from .core import Mstml, MstmlParams, MstmlEmbedType, MstmlEnsembleInterdisciplinarity, MstmlLongitudinalAnalysis
from .gdltm import Gdltm, GdltmParams
from .hrg import HierarchicalRandomGraph
from .utils import *

__all__ = [
    'Mstml',
    'MstmlParams', 
    'MstmlEmbedType',
    'MstmlEnsembleInterdisciplinarity',
    'MstmlLongitudinalAnalysis',
    'Gdltm',
    'GdltmParams',
    'HierarchicalRandomGraph'
]