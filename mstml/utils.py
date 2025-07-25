"""
Utility functions (generic)

This module contains various utility functions that are general-purpose for:
1. Managing file I/O
2. Computing vector distances
3. Logging information
4. Managing networkx graphs
5. Anything that is not specific to the MSTML framework.

For utility functions or classes specific to MSTML or GDLTM, see:
1. mstml_utils.py
2. gdtlm_utils.py
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
import re
import glob
import matplotlib.pyplot as plt
import nltk
import random
import hashlib
import math
import logging
from scipy.spatial import KDTree
from scipy.special import rel_entr
from collections import Counter
from math import factorial
from pathlib import Path
from bisect import bisect_left
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from gensim.utils import simple_preprocess
from enum import Enum, auto

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class AuthEmbedEnum(Enum):
    """Different types of embedding representations for authors."""
    WORD_FREQ = auto()
    AT_DISTN = auto()
    AT_SPARSIFIED_DISTN = auto()
    TERM_RELEVANCE_N_HOT = auto()
    TERM_RELEVANCE_VMASK = auto()


class TopicRelevanceEnum(Enum):
    """Different methods for representing topics filtered by relevance."""
    N_HOT_ENCODING = auto()  # N nonzero entries for N most relevant terms
    VOCAB_MASK = auto()      # Masked to union of N most relevant terms per topic


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


def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Preprocess text for topic modeling.
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        
    Returns:
        List of preprocessed tokens
    """
    # Convert to lowercase and tokenize
    tokens = simple_preprocess(text, deacc=True)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def create_author_network(author_collaborations):
    """
    Create a NetworkX graph from author collaborations.
    
    Args:
        author_collaborations: List of author lists (each list represents co-authors)
        
    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()
    
    for authors in author_collaborations:
        if len(authors) > 1:
            # Add edges between all pairs of co-authors
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
    
    return G


def compute_network_metrics(G):
    """
    Compute basic network metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of network metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Connectivity
    if nx.is_connected(G):
        metrics['is_connected'] = True
        metrics['diameter'] = nx.diameter(G)
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        metrics['is_connected'] = False
        metrics['num_components'] = nx.number_connected_components(G)
    
    # Centrality measures (for smaller networks)
    if G.number_of_nodes() < 1000:
        metrics['avg_degree_centrality'] = np.mean(list(nx.degree_centrality(G).values()))
        metrics['avg_betweenness_centrality'] = np.mean(list(nx.betweenness_centrality(G).values()))
        metrics['avg_closeness_centrality'] = np.mean(list(nx.closeness_centrality(G).values()))
    
    return metrics


def save_pickle(obj, filepath):
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_time_windows(dates, window_size='1Y'):
    """
    Create time windows from a list of dates.
    
    Args:
        dates: List of datetime objects
        window_size: Pandas frequency string (e.g., '1Y', '6M', '1Q')
        
    Returns:
        List of (start_date, end_date) tuples
    """
    if not dates:
        return []
    
    dates = pd.to_datetime(dates)
    min_date = dates.min()
    max_date = dates.max()
    
    # Create date range
    date_range = pd.date_range(start=min_date, end=max_date, freq=window_size)
    
    windows = []
    for i in range(len(date_range) - 1):
        windows.append((date_range[i], date_range[i + 1]))
    
    # Add final window if needed
    if date_range[-1] < max_date:
        windows.append((date_range[-1], max_date))
    
    return windows


def filter_by_frequency(word_list, min_freq=2, max_freq_ratio=0.8):
    """
    Filter words by frequency.
    
    Args:
        word_list: List of words
        min_freq: Minimum frequency threshold
        max_freq_ratio: Maximum frequency ratio (relative to total documents)
        
    Returns:
        Filtered list of words
    """
    word_counts = Counter(word_list)
    total_docs = len(word_list)
    
    filtered_words = []
    for word, count in word_counts.items():
        freq_ratio = count / total_docs
        if count >= min_freq and freq_ratio <= max_freq_ratio:
            filtered_words.extend([word] * count)
    
    return filtered_words


def compute_topic_coherence(topic_words, texts, measure='c_v'):
    """
    Compute topic coherence score.
    
    Args:
        topic_words: List of words in the topic
        texts: List of documents (tokenized)
        measure: Coherence measure ('c_v', 'c_npmi', 'c_uci', 'u_mass')
        
    Returns:
        Coherence score (float)
    """
    try:
        from gensim.models import CoherenceModel
        from gensim.corpora import Dictionary
        
        # Create dictionary and corpus
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Compute coherence
        coherence_model = CoherenceModel(
            topics=[topic_words],
            texts=texts,
            dictionary=dictionary,
            coherence=measure
        )
        
        return coherence_model.get_coherence()
    
    except ImportError:
        print("Gensim not available for coherence computation")
        return 0.0


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