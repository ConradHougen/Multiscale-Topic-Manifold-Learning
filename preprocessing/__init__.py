"""
Preprocessing utilities for MSTML.

This package contains modules for data preprocessing, cleaning, and preparation
for the Multi-Scale Topic Manifold Learning framework.
"""

from .text_processing import TextProcessor
from .author_disambiguation import AuthorDisambiguator
from .data_loaders import DataLoader

__all__ = [
    'TextProcessor',
    'AuthorDisambiguator', 
    'DataLoader'
]