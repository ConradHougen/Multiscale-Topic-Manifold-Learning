"""
_alignment.py — Forward k-NN topic alignment across temporal chunks.

For each chunk-topic, finds its nearest neighbours in the next chunk and
measures how close they are (low distance = high alignment = stable topics
over time).

Uses the fixed FAISS Hellinger conversion from _math.faiss_sq_l2_to_distance.

Ported from:
  AToMS-LP/topic_alignment_comparison.ipynb
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.stats import skew

log = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False


def _pairwise_hellinger(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Exact pairwise Hellinger distances between rows of X and Y."""
    # Normalise
    Xn = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
    Yn = Y / np.maximum(Y.sum(axis=1, keepdims=True), 1e-12)
    # (n, m) matrix via broadcasting
    diff = np.sqrt(Xn)[:, None, :] - np.sqrt(Yn)[None, :, :]  # (n, m, d)
    return np.sqrt(0.5 * np.sum(diff ** 2, axis=-1))


def _pairwise_cosine_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(X, Y)


def compute_alignment_scores(
    topic_vecs_per_chunk: "list[np.ndarray]",
    distance_metric: str = "hellinger",
    knn: int = 1,
) -> dict:
    """Compute forward k-NN alignment scores across consecutive chunks.

    For each chunk c and each topic t in chunk c, find the k nearest topics in
    chunk c+1 and record the distances.  The distribution of these forward-match
    distances characterises temporal alignment.

    Args:
        topic_vecs_per_chunk: List of (n_topics_c, vocab_size) arrays, one per chunk.
        distance_metric:      'hellinger' | 'cosine' | 'euclidean'.
        knn:                  k for forward matching.

    Returns:
        Dict with keys:
          'all_distances' — flat array of all forward-match distances,
          'mean_distance' — mean of the above,
          'skewness'      — skewness of the above (positive = right-tailed),
          'n_pairs'       — total number of matched pairs.
    """
    from .._math import faiss_sq_l2_to_distance, prepare_faiss_vectors

    all_dists: list[float] = []

    for cidx in range(len(topic_vecs_per_chunk) - 1):
        src = topic_vecs_per_chunk[cidx].astype(np.float32)
        tgt = topic_vecs_per_chunk[cidx + 1].astype(np.float32)
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            continue

        k_actual = min(knn, tgt.shape[0])

        if _FAISS_OK and distance_metric in ("hellinger", "cosine", "euclidean"):
            src_q = prepare_faiss_vectors(src, distance_metric)
            tgt_q = prepare_faiss_vectors(tgt, distance_metric)
            d     = src_q.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(tgt_q)
            sq_D, _ = index.search(src_q, k_actual)
            dists = faiss_sq_l2_to_distance(sq_D, distance_metric).flatten()
        else:
            if distance_metric == "hellinger":
                D = _pairwise_hellinger(src, tgt)
            elif distance_metric == "cosine":
                D = _pairwise_cosine_dist(src, tgt)
            else:
                from scipy.spatial.distance import cdist
                D = cdist(src, tgt, metric="euclidean")
            # Take k nearest per row
            dists = np.sort(D, axis=1)[:, :k_actual].flatten()

        all_dists.extend(dists.tolist())

    arr = np.array(all_dists, dtype=np.float64)
    return {
        "all_distances": arr,
        "mean_distance": float(arr.mean()) if len(arr) > 0 else float("nan"),
        "skewness":      float(skew(arr))  if len(arr) > 1 else float("nan"),
        "n_pairs":       len(arr),
    }
