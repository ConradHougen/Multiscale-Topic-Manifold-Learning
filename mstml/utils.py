"""
General purpose utility functions for the MSTML package.

This module contains clean, side-effect-free utility functions that are used across
the MSTML codebase including:
1. File I/O and directory management
2. Distance and similarity metrics (consolidated from duplicates)
3. Logging and validation utilities
4. Basic data manipulation functions
5. Hash and ID generation utilities

For specialized utilities, see:
- gdltm_utils.py: Temporal/longitudinal topic modeling utilities
- mstml_utils.py: Network and hierarchical topic modeling utilities
"""

import numpy as np
import pandas as pd
import pickle
import os
import re
import hashlib
import logging
from scipy.special import rel_entr
from pathlib import Path


# NOTE: Enums have been moved to their appropriate specialized modules:
# - AuthEmbedEnum: moved to mstml_utils.py (network/hierarchical utilities)
# - TopicRelevanceEnum: moved to gdltm_utils.py (temporal topic modeling utilities)


def validate_dataset_name(name: str) -> bool:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(
            f"Invalid dataset name '{name}'. Must start with a letter or underscore and contain only lowercase letters, digits, and underscores."
        )
    return True

def log_print(message: str, level: str = "info", logger: logging.Logger = None, also_print: bool = True):
    """
    Logs and optionally prints a message.

    Parameters:
        message (str): The message to log/print.
        level (str): Logging level: 'debug', 'info', 'warning', 'error', or 'critical'.
        logger (logging.Logger): Logger instance. If None, uses root logger.
        also_print (bool): Whether to also print to stdout.
    """
    if logger is None:
        logger = logging.getLogger()

    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)

    if also_print:
        print(message)


def get_data_int_dir(dset, dsub):
    """Get intermediate data directory path."""
    base_dir = os.path.join("data", dset)
    if dsub:
        return os.path.join(base_dir, dsub, "intermediate/")
    return os.path.join(base_dir, "intermediate/")


def get_data_clean_dir(dset, dsub):
    """Get clean data directory path."""
    base_dir = os.path.join("data", dset)
    if dsub:
        return os.path.join(base_dir, dsub, "clean/")
    return os.path.join(base_dir, "clean/")


def get_data_original_dir(dset, dsub):
    """Get original data directory path."""
    base_dir = os.path.join("data", dset)
    if dsub:
        return os.path.join(base_dir, dsub, "original/")
    return os.path.join(base_dir, "original/")


def get_file_path_without_extension(file_path):
    """Get file path without extension."""
    return os.path.splitext(file_path)[0]


def get_file_stem_only(file_path):
    """Get file stem (filename without path and extension)."""
    return Path(file_path).stem


def hellinger_distance(p, q):
    """
    Compute Hellinger distance between two probability distributions.
    
    Args:
        p, q: Probability distributions (numpy arrays)
        
    Returns:
        Hellinger distance (float)
    """
    # Ensure inputs are numpy arrays and normalized
    p = np.array(p)
    q = np.array(q)
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Compute Hellinger distance
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    return np.sqrt(0.5 * np.sum((sqrt_p - sqrt_q) ** 2))


def hellinger_similarity(p, q):
    """
    Compute Hellinger similarity (1 - Hellinger distance).
    
    Args:
        p, q: Probability distributions (numpy arrays)
        
    Returns:
        Hellinger similarity (float)
    """
    return 1.0 - hellinger_distance(p, q)


def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions using rel_entr.

    Args:
        p, q (array-like): Probability distributions

    Returns:
        float: Jensen-Shannon divergence
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if not np.any(p):
        raise ValueError("Input p must have nonzero sum.")
    if not np.any(q):
        raise ValueError("Input q must have nonzero sum.")

    # Normalize
    p /= np.sum(p)
    q /= np.sum(q)

    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a, b: Input vectors (numpy arrays)
        
    Returns:
        Cosine similarity (float)
    """
    a = np.array(a)
    b = np.array(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


# NOTE: Text preprocessing functionality has been moved to text_preprocessing.py
# Use TextPreprocessor class for comprehensive text preprocessing capabilities


# NOTE: Network analysis functionality has been moved to mstml_utils.py
# Use network analysis functions in mstml_utils.py for author networks and graph operations




def save_pickle(obj, filepath):
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# NOTE: Temporal analysis functions have been moved to gdltm_utils.py
# Use gdltm_utils.py for time windowing, frequency filtering, and topic coherence over time






def normalize_vector(vector):
    """Normalize a vector to unit length."""
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def create_hash_id(text, length=8):
    """Create a hash ID from text."""
    return hashlib.md5(text.encode()).hexdigest()[:length]