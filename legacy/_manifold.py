"""
_manifold.py — Step 4: Pairwise distances, topic dendrogram, and manifold embedding.

Pipeline
--------
  compute_pairwise_distances  → full O(n²) symmetric distance matrix (for embedding)
  build_topic_dendrogram      → scipy linkage matrix Z + height bounds
  cut_dendrogram              → cluster_labels array
  compute_embedding           → low-dimensional embedding array

Naming is metric-agnostic throughout: the result is always called
``distance_matrix`` regardless of the underlying metric.

Dendrogram uses FAISS kNN (k = cfg.dendrogram.knn, default 100) to build an
approximate sparse distance matrix, then passes the true distances to
scipy Ward linkage — the fixed formula (not the original sqrt(2) scaling).

Embedding (PHATE by default) takes the full precomputed distance matrix and
treats it as a precomputed diffusion-operator input (``knn_dist='precomputed'``).

Ported from:
  AToMS-LP/AToMS_HRG_Ensemble_Interdisciplinarity.ipynb (cells 48-55)
  AToMS-LP/AToMS_HRG_Longitudinal_Analysis.ipynb (cells 5-20)
  AToMS-LP/AToMS/atoms_hrg_library.py (rescale_parameter)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy import ndarray
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from ._math import (
    hellinger_matrix,
    faiss_sq_l2_to_distance,
    prepare_faiss_vectors,
)

log = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False


# ---------------------------------------------------------------------------
# Full pairwise distance matrix
# ---------------------------------------------------------------------------

def compute_pairwise_distances(
    topic_vectors: ndarray,
    distance_metric: str = "hellinger",
) -> ndarray:
    """Compute the full O(n²) symmetric pairwise distance matrix.

    This matrix is passed to the embedding algorithm as a precomputed input.

    Args:
        topic_vectors:   (n_topics, vocab_size) float array.
        distance_metric: 'hellinger' | 'cosine' | 'euclidean'.

    Returns:
        (n_topics, n_topics) symmetric distance matrix, dtype float64.
    """
    m = distance_metric.lower()
    log.info(
        "Computing full pairwise %s distances for %d topics …",
        m, topic_vectors.shape[0],
    )
    if m == "hellinger":
        return hellinger_matrix(topic_vectors)
    elif m == "cosine":
        from sklearn.metrics.pairwise import cosine_distances
        return cosine_distances(topic_vectors.astype(np.float64))
    elif m == "euclidean":
        X = topic_vectors.astype(np.float64)
        return squareform(pdist(X, metric="euclidean"))
    else:
        raise ValueError(f"Unsupported distance_metric: '{distance_metric}'")


# ---------------------------------------------------------------------------
# Topic dendrogram
# ---------------------------------------------------------------------------

def _faiss_knn_distance_matrix(
    topic_vectors: ndarray,
    knn: int,
    distance_metric: str,
) -> ndarray:
    """Build sparse symmetric distance matrix via FAISS kNN.

    Entries for non-kNN pairs remain 0 (approximation used by the original
    AToMS-LP pipeline; Ward linkage still finds meaningful clusters because
    knn=100 captures most of the local structure).
    """
    query = prepare_faiss_vectors(topic_vectors, distance_metric)
    d = query.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(query)
    n = query.shape[0]
    k_actual = min(knn, n - 1)
    sq_D, I = index.search(query, k_actual + 1)
    sq_D, I = sq_D[:, 1:], I[:, 1:]  # exclude self

    # Convert to true metric distances
    dist_vals = faiss_sq_l2_to_distance(sq_D, distance_metric)

    # Symmetrise into full n×n matrix
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for ki in range(k_actual):
            j = int(I[i, ki])
            d_ij = float(dist_vals[i, ki])
            mat[i, j] = d_ij
            mat[j, i] = d_ij
    return mat.astype(np.float64)


def _fallback_knn_distance_matrix(
    topic_vectors: ndarray,
    knn: int,
    distance_metric: str,
) -> ndarray:
    """Exact k-NN fallback when FAISS is not available."""
    if distance_metric == "hellinger":
        full_mat = hellinger_matrix(topic_vectors)
    elif distance_metric == "cosine":
        from sklearn.metrics.pairwise import cosine_distances
        full_mat = cosine_distances(topic_vectors.astype(np.float64))
    else:
        full_mat = squareform(pdist(topic_vectors.astype(np.float64), metric="euclidean"))

    n = full_mat.shape[0]
    k_actual = min(knn, n - 1)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = full_mat[i].copy()
        row[i] = np.inf
        knn_idx = np.argpartition(row, k_actual)[:k_actual]
        for j in knn_idx:
            mat[i, j] = full_mat[i, j]
            mat[j, i] = full_mat[i, j]
    return mat


def build_topic_dendrogram(
    topic_vectors: ndarray,
    knn: int = 100,
    linkage_method: str = "ward",
    distance_metric: str = "hellinger",
) -> tuple[ndarray, float, float]:
    """Build a hierarchical clustering dendrogram over topic vectors.

    Uses FAISS approximate kNN (k=``knn``, default 100) to construct a sparse
    symmetric distance matrix, then calls scipy linkage.  True metric distances
    (via the fixed FAISS conversion) are passed directly to ``linkage`` —
    the original AToMS-LP sqrt(2) scaling is NOT applied.

    Args:
        topic_vectors:   (n_topics, vocab_size) float array.
        knn:             k for the FAISS approximate neighbour search.
        linkage_method:  Scipy linkage method ('ward' | 'complete' | 'average' | 'single').
        distance_metric: 'hellinger' | 'cosine' | 'euclidean'.

    Returns:
        (Z, min_height, max_height)
        - Z:          Linkage matrix, shape (n_topics − 1, 4).
        - min_height: Distance of the first (smallest) merge — used for cut rescaling.
        - max_height: Distance of the last (largest) merge.
    """
    log.info(
        "Building %s dendrogram (knn=%d, metric=%s) for %d topics …",
        linkage_method, knn, distance_metric, topic_vectors.shape[0],
    )
    if _FAISS_OK:
        mat = _faiss_knn_distance_matrix(topic_vectors, knn, distance_metric)
    else:
        log.warning("FAISS unavailable; using exact kNN fallback for dendrogram.")
        mat = _fallback_knn_distance_matrix(topic_vectors, knn, distance_metric)

    condensed = squareform(mat)
    Z = linkage(condensed, method=linkage_method)

    min_height = float(Z[0, 2])
    max_height = float(Z[-1, 2])
    log.info(
        "Dendrogram complete: %d merges, height range [%.4f, %.4f].",
        Z.shape[0], min_height, max_height,
    )
    return Z, min_height, max_height


# ---------------------------------------------------------------------------
# Dendrogram cut
# ---------------------------------------------------------------------------

def cut_dendrogram(
    Z: ndarray,
    cut_height: float,
    min_height: float,
    max_height: float,
) -> ndarray:
    """Cut the dendrogram at a normalised height and return cluster labels.

    Args:
        Z:          Linkage matrix from ``build_topic_dendrogram``.
        cut_height: Normalised cut in [0, 1].  0 = minimum merge height,
                    1 = maximum merge height (root).
        min_height: Smallest merge distance (from ``build_topic_dendrogram``).
        max_height: Largest merge distance.

    Returns:
        Integer cluster-label array of shape (n_topics,), 1-indexed.
    """
    if not 0.0 <= cut_height <= 1.0:
        raise ValueError(f"cut_height must be in [0, 1], got {cut_height}")
    cut_dist = min_height + cut_height * (max_height - min_height)
    labels = fcluster(Z, t=cut_dist, criterion="distance")
    n_clusters = len(np.unique(labels))
    log.info(
        "Dendrogram cut at height=%.4f → %d meta-topic clusters.", cut_dist, n_clusters
    )
    return labels


# ---------------------------------------------------------------------------
# Manifold embedding
# ---------------------------------------------------------------------------

def compute_embedding(
    distance_matrix: ndarray,
    embedding_method: str = "phate",
    knn: int = 5,
    **method_kwargs,
) -> ndarray:
    """Embed the topic-distance matrix into low-dimensional space.

    Dispatches to PHATE, UMAP, t-SNE, or PCA based on ``embedding_method``.
    All methods receive the precomputed distance matrix as input
    (``metric='precomputed'`` or ``knn_dist='precomputed'``).

    Args:
        distance_matrix:  (n_topics, n_topics) full symmetric distance matrix.
        embedding_method: 'phate' | 'umap' | 'tsne' | 'pca'.
        knn:              Neighbourhood size (used by PHATE and UMAP).
        **method_kwargs:  Passed to the embedding algorithm
                          (e.g. n_components, gamma for PHATE).

    Returns:
        (n_topics, n_components) embedding array.
    """
    m = embedding_method.lower().replace("-", "").replace("_", "")

    if m == "phate":
        return _embed_phate(distance_matrix, knn=knn, **method_kwargs)
    elif m == "umap":
        return _embed_umap(distance_matrix, knn=knn, **method_kwargs)
    elif m in ("tsne", "tsne"):
        return _embed_tsne(distance_matrix, **method_kwargs)
    elif m == "pca":
        return _embed_pca(distance_matrix, **method_kwargs)
    else:
        raise ValueError(f"Unsupported embedding_method: '{embedding_method}'")


def _embed_phate(mat: ndarray, knn: int = 5, **kwargs) -> ndarray:
    try:
        import phate  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("phate not installed. Run: pip install phate") from exc
    kwargs.setdefault("n_components", 3)
    kwargs.setdefault("gamma", 0.0)
    kwargs.setdefault("t", "auto")
    op = phate.PHATE(knn=knn, knn_dist="precomputed", **kwargs)
    embedding = op.fit_transform(mat)
    log.info("PHATE embedding shape: %s.", embedding.shape)
    return embedding


def _embed_umap(mat: ndarray, knn: int = 15, **kwargs) -> ndarray:
    try:
        import umap  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn") from exc
    kwargs.setdefault("n_components", 3)
    kwargs.pop("n_neighbors", None)  # override with knn arg
    reducer = umap.UMAP(n_neighbors=knn, metric="precomputed", **kwargs)
    embedding = reducer.fit_transform(mat)
    log.info("UMAP embedding shape: %s.", embedding.shape)
    return embedding


def _embed_tsne(mat: ndarray, **kwargs) -> ndarray:
    from sklearn.manifold import TSNE
    import sklearn
    kwargs.setdefault("n_components", 3)
    kwargs.setdefault("perplexity", 30.0)
    kwargs.setdefault("init", "random")
    # sklearn ≥1.5 renamed n_iter → max_iter; translate for forward compatibility
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        if "n_iter" in kwargs:
            kwargs.setdefault("max_iter", kwargs.pop("n_iter"))
    tsne = TSNE(metric="precomputed", **kwargs)
    embedding = tsne.fit_transform(mat)
    log.info("t-SNE embedding shape: %s.", embedding.shape)
    return embedding


def _embed_pca(mat: ndarray, **kwargs) -> ndarray:
    from sklearn.decomposition import PCA
    kwargs.setdefault("n_components", 3)
    # PCA on the distance matrix (kernel trick approximation)
    n = mat.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * H @ (mat ** 2) @ H
    pca = PCA(**kwargs)
    embedding = pca.fit_transform(gram)
    log.info("PCA embedding shape: %s.", embedding.shape)
    return embedding
