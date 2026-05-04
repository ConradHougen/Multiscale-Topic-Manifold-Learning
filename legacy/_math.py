"""
_math.py — Self-contained mathematical utilities for the legacy pipeline.

All functions are pure (no I/O, no global state).  This module has no
dependency on any other legacy module and can be imported independently.

FAISS distance conversion — critical bug fix
--------------------------------------------
Original AToMS-LP notebooks used ``IndexFlatL2`` on ``sqrt(X)`` to
approximate Hellinger k-NN, then converted the returned squared-L2
distances with:

    distances / np.sqrt(2)      # WRONG — gives sqrt(2)·H², not H

The correct conversion is:

    np.sqrt(distances / 2.0)    # CORRECT — gives true Hellinger H ∈ [0, 1]

Proof:  IndexFlatL2 returns  ||√p − √q||²  =  Σ(√pᵢ − √qᵢ)²
        Hellinger distance H = √( 0.5 · Σ(√pᵢ − √qᵢ)² )
                              = √( faiss_sq_l2 / 2 )
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Hellinger distance
# ---------------------------------------------------------------------------

def hellinger(p: ndarray, q: ndarray) -> float:
    """Exact Hellinger distance between two probability distributions.

    H(p, q) = sqrt(0.5 · Σ(√pᵢ − √qᵢ)²)

    Equivalent to: ||√p − √q|| / √2

    Args:
        p, q: 1-D arrays representing probability distributions.
              They do not need to be pre-normalised; zero-vectors are safe.

    Returns:
        Hellinger distance in [0, 1].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sp = p.sum()
    sq = q.sum()
    if sp > 0:
        p = p / sp
    if sq > 0:
        q = q / sq
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def hellinger_matrix(X: ndarray) -> ndarray:
    """Vectorised pairwise Hellinger distance matrix (O(n²) space).

    Args:
        X: (n, d) matrix of probability distributions (rows need not sum to 1;
           they are L1-normalised internally).

    Returns:
        (n, n) symmetric distance matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    X = X / row_sums
    sqrt_X = np.sqrt(X)
    # cdist with euclidean on sqrt vectors gives ||√p − √q||;
    # divide by √2 to obtain Hellinger distance.
    return cdist(sqrt_X, sqrt_X, metric="euclidean") / np.sqrt(2)


def hellinger_matrix_cross(X: ndarray, Y: ndarray) -> ndarray:
    """Vectorised cross-pairwise Hellinger distances between two sets of distributions.

    Args:
        X: (n, d) matrix of probability distributions (L1-normalised internally).
        Y: (m, d) matrix of probability distributions (L1-normalised internally).

    Returns:
        (n, m) distance matrix where entry [i, j] = H(X[i], Y[j]).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    rx = X.sum(axis=1, keepdims=True)
    ry = Y.sum(axis=1, keepdims=True)
    X = X / np.where(rx > 0, rx, 1.0)
    Y = Y / np.where(ry > 0, ry, 1.0)
    return cdist(np.sqrt(X), np.sqrt(Y), metric="euclidean") / np.sqrt(2)


# ---------------------------------------------------------------------------
# FAISS distance conversion (the fixed formula)
# ---------------------------------------------------------------------------

def faiss_sq_l2_to_hellinger(sq_l2: ndarray) -> ndarray:
    """Convert FAISS squared-L2 distances (from IndexFlatL2 on sqrt vectors)
    to true Hellinger distances.

    Background
    ----------
    When you index sqrt(X) in FAISS IndexFlatL2, the returned distances are:
        ||√p − √q||²  =  Σ(√pᵢ − √qᵢ)²  =  2 · H(p,q)²

    So H(p,q) = sqrt(sq_l2 / 2).

    Args:
        sq_l2: Array of FAISS squared-L2 distances (any shape).

    Returns:
        Array of Hellinger distances, same shape as input, values in [0, 1].
    """
    return np.sqrt(np.asarray(sq_l2, dtype=np.float64) / 2.0)


def faiss_sq_l2_to_hellinger_legacy(sq_l2: ndarray) -> ndarray:
    """AToMS-LP original (buggy) FAISS conversion for exact historical reproducibility.

    The original AToMS-LP notebooks treated IndexFlatL2 output as an un-squared L2
    distance and divided by sqrt(2).  Because IndexFlatL2 actually returns SQUARED L2,
    this yields sqrt(2)·H²(p,q) instead of H(p,q).

    Use ONLY when --reproduce_legacy_bug is set, to bitwise-reproduce thesis/CAMSAP
    2025 results.  For all new work use faiss_sq_l2_to_hellinger instead.

    Args:
        sq_l2: Array of FAISS squared-L2 distances (any shape).

    Returns:
        sqrt(2)·H²  — NOT a true Hellinger distance.
    """
    return np.asarray(sq_l2, dtype=np.float64) / np.sqrt(2)


def faiss_sq_l2_to_distance(
    sq_l2: ndarray,
    distance_metric: str,
    legacy_bug: bool = False,
) -> ndarray:
    """Convert FAISS squared-L2 distances to the requested distance metric.

    For Hellinger: uses the FAISS sqrt-trick conversion.
    For euclidean: takes sqrt of the squared values.
    For cosine:    assumes the index was built on L2-normalised vectors
                   (cosine distance = 1 - cosine_similarity, and for unit
                   vectors ||u−v||² = 2 - 2·cos → cos_dist = sq_l2/2).

    Args:
        sq_l2: FAISS squared-L2 distances.
        distance_metric: 'hellinger', 'euclidean', or 'cosine'.
        legacy_bug: If True and metric is 'hellinger', use the original AToMS-LP
                    formula (sq_l2 / sqrt(2)) instead of the correct sqrt(sq_l2 / 2).
                    Only set this for exact historical reproduction.

    Returns:
        Converted distances.
    """
    m = distance_metric.lower()
    sq_l2 = np.asarray(sq_l2, dtype=np.float64)
    if m == "hellinger":
        return faiss_sq_l2_to_hellinger_legacy(sq_l2) if legacy_bug \
               else faiss_sq_l2_to_hellinger(sq_l2)
    elif m == "euclidean":
        return np.sqrt(sq_l2)
    elif m == "cosine":
        # For L2-normalised vectors: sq_l2 = ||u-v||² = 2(1-cos) → cos_dist = sq_l2/2
        return np.clip(sq_l2 / 2.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported distance_metric: '{distance_metric}'")


def prepare_faiss_vectors(X: ndarray, distance_metric: str) -> ndarray:
    """Transform probability vectors into the space where L2 distance
    approximates the desired metric, for use with FAISS IndexFlatL2.

    Hellinger:  index sqrt(X)  →  L2 gives ||√p−√q||² = 2H²
    Euclidean:  index X        →  L2 gives Euclidean² directly
    Cosine:     index X / ||X||₂  →  L2 gives 2(1−cos)

    Args:
        X: (n, d) array of row vectors.
        distance_metric: 'hellinger', 'euclidean', or 'cosine'.

    Returns:
        Transformed (n, d) float32 array ready for FAISS IndexFlatL2.
    """
    m = distance_metric.lower()
    X = np.asarray(X, dtype=np.float32)
    if m == "hellinger":
        return np.sqrt(np.maximum(X, 0.0))
    elif m == "euclidean":
        return X
    elif m == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return (X / norms).astype(np.float32)
    else:
        raise ValueError(f"Unsupported distance_metric: '{distance_metric}'")


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def entropy(p: ndarray) -> float:
    """Shannon entropy (base 2) of a probability distribution.

    Safe for zeros (0 · log2(0) := 0).

    Args:
        p: 1-D array; need not be normalised.

    Returns:
        Entropy in bits.
    """
    p = np.asarray(p, dtype=np.float64)
    p_sum = p.sum()
    if p_sum > 0:
        p = p / p_sum
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


# ---------------------------------------------------------------------------
# Term relevance (LDAvis-style)
# ---------------------------------------------------------------------------

def term_relevance(phi: ndarray, p_w: ndarray, lam: float,
                   epsilon: float = 1e-12) -> ndarray:
    """LDAvis-style term relevance score for a single topic.

    relevance(w, t) = λ · log p(w|t) + (1−λ) · log[ p(w|t) / p(w) ]

    Args:
        phi: (vocab_size,) topic-word probability vector p(w|t).
        p_w: (vocab_size,) corpus-wide word probability p(w).
        lam: Relevance weighting parameter λ ∈ [0, 1].
             λ=1 → pure p(w|t); λ=0 → pure log-lift.
        epsilon: Small constant to avoid log(0).

    Returns:
        (vocab_size,) relevance scores (higher = more relevant).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    phi = np.asarray(phi, dtype=np.float64)
    p_w = np.asarray(p_w, dtype=np.float64)
    log_phi = np.log(phi + epsilon)
    log_pw = np.log(p_w + epsilon)
    return lam * log_phi + (1.0 - lam) * (log_phi - log_pw)


def corpus_word_probabilities(tokenized_docs: list[list[str]]) -> dict[str, float]:
    """Compute empirical word probabilities p(w) from tokenised documents.

    Args:
        tokenized_docs: List of tokenised documents (list of word lists).

    Returns:
        Dict mapping word → probability.
    """
    from collections import Counter
    counts: Counter = Counter(w for doc in tokenized_docs for w in doc)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {w: c / total for w, c in counts.items()}


# ---------------------------------------------------------------------------
# Diffusion weight utilities
# ---------------------------------------------------------------------------

def diffusion_weights(distances: ndarray) -> ndarray:
    """Convert k-NN distances to diffusion influence weights.

    Formula (original AToMS-LP design):
        w_ij = 1 − d_ij / Σ_k d_ik

    Closer neighbours receive higher weights.  If there is only one
    neighbour the formula degenerates to 0; the caller should guard
    against this (see ``diffuse_distribution``).

    Args:
        distances: (k,) array of distances to k neighbours (excluding self).

    Returns:
        (k,) array of weights, non-negative, not necessarily summing to 1.
    """
    d = np.asarray(distances, dtype=np.float64)
    total = d.sum()
    if total <= 0:
        return np.ones_like(d) / max(len(d), 1)
    return 1.0 - d / total
