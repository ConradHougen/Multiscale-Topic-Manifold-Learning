"""
_diversity.py — Topic diversity metrics.

Diversity@k: percentage of unique words in the top-k words across all topics.
Mean pairwise Hellinger / cosine distance across topic vectors.

Ported from:
  AToMS-LP/topic_diversity_comparison.ipynb
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import pdist

log = logging.getLogger(__name__)


def diversity_at_k(
    topic_word_lists: "list[list[str]]",
    k: int = 10,
) -> float:
    """Diversity@k: unique words / (n_topics × k).

    A score of 1.0 means all top-k words are unique across topics.

    Args:
        topic_word_lists: List of per-topic word lists.
        k:                Number of top words to consider per topic.

    Returns:
        Diversity score in [0, 1].
    """
    if not topic_word_lists:
        return 0.0
    unique: set[str] = set()
    total = 0
    for words in topic_word_lists:
        top = words[:k]
        unique.update(top)
        total += len(top)
    return len(unique) / total if total > 0 else 0.0


def mean_pairwise_hellinger(topic_vectors: np.ndarray) -> float:
    """Mean pairwise Hellinger distance over (n_topics, vocab_size) matrix.

    Args:
        topic_vectors: (n_topics, vocab_size) float array.

    Returns:
        Scalar mean distance.  NaN if fewer than 2 topics.
    """
    if topic_vectors.shape[0] < 2:
        return float("nan")

    X = topic_vectors.astype(np.float64)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    X = X / row_sums
    sqrt_X = np.sqrt(X)
    dists = pdist(sqrt_X, metric="euclidean") / np.sqrt(2)  # exact Hellinger
    return float(np.mean(dists))


def mean_pairwise_cosine_distance(topic_vectors: np.ndarray) -> float:
    """Mean pairwise cosine distance over (n_topics, vocab_size) matrix.

    Args:
        topic_vectors: (n_topics, vocab_size) float array.

    Returns:
        Scalar mean cosine distance.  NaN if fewer than 2 topics.
    """
    if topic_vectors.shape[0] < 2:
        return float("nan")
    dists = pdist(topic_vectors.astype(np.float64), metric="cosine")
    return float(np.mean(dists))


def compute_diversity(
    topic_word_lists: "list[list[str]]",
    topic_vectors: "np.ndarray | None" = None,
    k: int = 10,
) -> "dict[str, float]":
    """Compute all diversity metrics and return as a dict.

    Args:
        topic_word_lists: Per-topic word lists.
        topic_vectors:    Optional (n_topics, vocab_size) array for vector-based metrics.
        k:                Top words for Diversity@k.

    Returns:
        Dict with keys: 'diversity_at_k', 'mean_hellinger', 'mean_cosine'.
    """
    results: dict[str, float] = {
        "diversity_at_k": diversity_at_k(topic_word_lists, k),
    }
    if topic_vectors is not None:
        results["mean_hellinger"] = mean_pairwise_hellinger(topic_vectors)
        results["mean_cosine"]    = mean_pairwise_cosine_distance(topic_vectors)
    else:
        results["mean_hellinger"] = float("nan")
        results["mean_cosine"]    = float("nan")
    return results


def compute_ensemble_diversity(
    topic_word_lists_per_chunk: "list[list[list[str]]]",
    topic_vectors_per_chunk: "list[np.ndarray | None] | None" = None,
    k: int = 10,
) -> "dict[str, float]":
    """Average diversity metrics across ensemble chunks.

    Args:
        topic_word_lists_per_chunk: Outer = chunks, middle = topics, inner = words.
        topic_vectors_per_chunk:    Matching per-chunk phi matrices (optional).
        k:                          Top words for Diversity@k.

    Returns:
        Dict of averaged metrics.
    """
    metrics_keys = ["diversity_at_k", "mean_hellinger", "mean_cosine"]
    accumulator: dict[str, list[float]] = {m: [] for m in metrics_keys}

    tvecs = topic_vectors_per_chunk or [None] * len(topic_word_lists_per_chunk)

    for words, vecs in zip(topic_word_lists_per_chunk, tvecs):
        r = compute_diversity(words, vecs, k)
        for m in metrics_keys:
            v = r.get(m, float("nan"))
            if not np.isnan(v):
                accumulator[m].append(v)

    return {
        m: float(np.mean(vals)) if vals else float("nan")
        for m, vals in accumulator.items()
    }
