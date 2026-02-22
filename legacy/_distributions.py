"""
_distributions.py — Step 3: Author barycenters, diffusion k-NN graph, and diffusion.

Pipeline
--------
  build_coauthor_graph          → NetworkX co-author graph
  compute_author_barycenters    → author_id → barycenter distribution
  build_diffusion_graph         → NetworkX topic k-NN graph (Hellinger-weighted edges)
  diffuse_distributions         → dict of diffused distributions over the k-NN graph

The diffusion step propagates topic mass from topics that an author/document
directly occupies into adjacent topics according to their Hellinger-distance
neighbourhood, smoothing the sparse distributions.

FAISS distance conversion bug fix
----------------------------------
The original AToMS-LP notebooks convert FAISS squared-L2 distances with
``dist / sqrt(2)`` (wrong).  This module uses the fixed formula
``sqrt(dist / 2)`` from ``_math.faiss_sq_l2_to_distance``.

Single-neighbour diffusion guard
---------------------------------
The weight formula ``w_ij = 1 − d_ij / Σ d_ik`` degenerates to 0 when a node
has only one neighbour (d/d = 1, 1 − 1 = 0).  In that case the module uses
a uniform weight of 1.0 so that mass is still propagated.

Ported from:
  AToMS-LP/AToMS_HRG_Ensemble_Interdisciplinarity.ipynb (cells 36–46)
  AToMS-LP/AToMS/atoms_hrg_library.py (diffuse_distribution)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

from ._math import faiss_sq_l2_to_distance, prepare_faiss_vectors

log = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False


# ---------------------------------------------------------------------------
# Co-author graph
# ---------------------------------------------------------------------------

def build_coauthor_graph(df: pd.DataFrame, author_col: str = "author_ids") -> nx.Graph:
    """Build a weighted co-author graph from the dataframe.

    Each paper contributes an edge (or increments the weight of an existing
    edge) for every pair of co-authors.  Edge ``weight`` = number of papers
    co-authored.

    Args:
        df:          DataFrame with an ``author_col`` column (list of int IDs).
        author_col:  Column name holding per-paper author ID lists.

    Returns:
        Undirected NetworkX graph.
    """
    G: nx.Graph = nx.Graph()
    for auth_ids in df[author_col]:
        ids = list(auth_ids) if auth_ids else []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                u, v = ids[i], ids[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
    log.info(
        "Co-author graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ---------------------------------------------------------------------------
# Author barycenters
# ---------------------------------------------------------------------------

def compute_author_barycenters(
    expanded_distns: "dict[int, np.ndarray]",
    df: pd.DataFrame,
    author_col: str = "author_ids",
) -> "tuple[dict[int, np.ndarray], dict[int, list[int]], dict[int, list[float]]]":
    """Compute per-author barycenter distributions (weighted average over their documents).

    Weight for document d attributed to author a is ``1 / |authors(d)|``, i.e.
    each author is credited equally regardless of team size.

    Args:
        expanded_distns: doc_id → global theta (from ``expand_distributions``).
        df:              DataFrame with author_col and index matching expanded_distns.
        author_col:      Column holding per-paper list of author IDs.

    Returns:
        (author_barycenters, authId_to_docs, authId_to_weights)
        - author_barycenters: author_id → normalised barycenter distribution.
        - authId_to_docs:     author_id → list of doc_ids.
        - authId_to_weights:  author_id → list of weights (1/n_authors per doc).
    """
    # Accumulate weighted distributions
    author_sums: dict[int, np.ndarray] = {}
    author_weight_total: dict[int, float] = {}
    authId_to_docs: dict[int, list[int]] = {}
    authId_to_weights: dict[int, list[float]] = {}

    for doc_id, auth_ids in zip(df.index, df[author_col]):
        if doc_id not in expanded_distns:
            continue
        theta = expanded_distns[doc_id]
        n_auth = max(len(auth_ids), 1)
        w = 1.0 / n_auth
        for aid in auth_ids:
            if aid not in author_sums:
                author_sums[aid]         = np.zeros_like(theta, dtype=np.float64)
                author_weight_total[aid] = 0.0
                authId_to_docs[aid]      = []
                authId_to_weights[aid]   = []
            author_sums[aid]         += w * theta
            author_weight_total[aid] += w
            authId_to_docs[aid].append(doc_id)
            authId_to_weights[aid].append(w)

    # Normalise to unit-sum distributions
    author_barycenters: dict[int, np.ndarray] = {}
    for aid, cumsum in author_sums.items():
        total = author_weight_total[aid]
        vec = cumsum / total if total > 0.0 else cumsum
        s = vec.sum()
        author_barycenters[aid] = (vec / s).astype(np.float32) if s > 0 else vec.astype(np.float32)

    log.info("Computed barycenters for %d authors.", len(author_barycenters))
    return author_barycenters, authId_to_docs, authId_to_weights


# ---------------------------------------------------------------------------
# Diffusion k-NN graph
# ---------------------------------------------------------------------------

def _build_faiss_knn(
    vectors: np.ndarray,
    knn: int,
    distance_metric: str,
) -> "tuple[np.ndarray, np.ndarray]":
    """FAISS-based approximate k-NN.  Returns (distances, indices) for each row."""
    query = prepare_faiss_vectors(vectors, distance_metric)  # float32, transformed
    d = query.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(query)
    D, I = index.search(query, knn + 1)  # +1 to include self (removed below)
    return D[:, 1:], I[:, 1:]  # exclude self-match


def _build_scipy_knn(
    vectors: np.ndarray,
    knn: int,
    distance_metric: str,
) -> "tuple[np.ndarray, np.ndarray]":
    """Pure-numpy fallback k-NN (O(n²), exact)."""
    from scipy.spatial.distance import cdist
    from ._math import hellinger_matrix

    if distance_metric == "hellinger":
        dist_mat = hellinger_matrix(vectors)
    elif distance_metric == "cosine":
        from sklearn.metrics.pairwise import cosine_distances
        dist_mat = cosine_distances(vectors)
    else:
        dist_mat = cdist(vectors, vectors, metric="euclidean")

    n = dist_mat.shape[0]
    k_actual = min(knn, n - 1)
    D_rows = []
    I_rows = []
    for i in range(n):
        row = dist_mat[i].copy()
        row[i] = np.inf  # exclude self
        idx = np.argpartition(row, k_actual)[:k_actual]
        idx = idx[np.argsort(row[idx])]
        D_rows.append(row[idx])
        I_rows.append(idx)
    return np.array(D_rows, dtype=np.float32), np.array(I_rows, dtype=np.int64)


def build_diffusion_graph(
    topic_vectors: np.ndarray,
    knn: int = 5,
    distance_metric: str = "hellinger",
) -> nx.Graph:
    """Build a k-NN graph over topic vectors for the diffusion step.

    Edge weights are true distances in the requested metric (Hellinger, cosine,
    or Euclidean).  The fixed FAISS conversion formula is used when FAISS is
    available.

    Args:
        topic_vectors:   (n_topics, vocab_size) array from ``train_ensemble``.
        knn:             Number of nearest neighbours per topic node.
        distance_metric: 'hellinger' | 'cosine' | 'euclidean'.

    Returns:
        Undirected NetworkX graph with ``weight`` = distance on each edge.
    """
    n = topic_vectors.shape[0]

    if _FAISS_OK:
        sq_D, I = _build_faiss_knn(topic_vectors, knn, distance_metric)
        # Convert FAISS squared-L2 to requested metric (fixed formula)
        distances = faiss_sq_l2_to_distance(sq_D, distance_metric)
    else:
        log.warning("FAISS not available; using O(n²) fallback for diffusion graph.")
        distances, I = _build_scipy_knn(topic_vectors, knn, distance_metric)

    G: nx.Graph = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for k_idx in range(distances.shape[1]):
            j   = int(I[i, k_idx])
            d   = float(distances[i, k_idx])
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=d)

    log.info(
        "Diffusion graph: %d nodes, %d edges (knn=%d).", n, G.number_of_edges(), knn
    )
    return G


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def _diffuse_one(
    graph: nx.Graph,
    known_dist: np.ndarray,
    num_iterations: int,
    diffusion_rate: float,
) -> np.ndarray:
    """Diffuse a single distribution over the graph.

    For each topic node with zero mass in ``known_dist``, propagate mass from
    its k-NN neighbours weighted by inverse distance.

    Single-neighbour guard: when a node has exactly one neighbour, the
    standard weight formula gives 0; we use weight 1.0 instead.

    Ported from atoms_hrg_library.py::diffuse_distribution.
    """
    n = len(known_dist)
    distribution = known_dist.copy().astype(np.float32)
    buffer = np.zeros(n, dtype=np.float32)

    for _ in range(num_iterations):
        buffer.fill(0.0)
        for node in range(n):
            if known_dist[node] > 0.0:
                buffer[node] = distribution[node]
            else:
                neighbours = list(graph.neighbors(node))
                if not neighbours:
                    continue
                weights = np.array(
                    [graph[node][nb]["weight"] for nb in neighbours], dtype=np.float32
                )
                w_sum = weights.sum()
                if len(neighbours) == 1 or w_sum <= 0.0:
                    # Single-neighbour guard or degenerate case
                    norm_w = np.ones(len(neighbours), dtype=np.float32)
                else:
                    norm_w = 1.0 - weights / w_sum
                nb_dists = np.array(
                    [distribution[nb] for nb in neighbours], dtype=np.float32
                )
                buffer[node] = diffusion_rate * float(np.dot(norm_w, nb_dists))
        distribution[:] = buffer

    total = distribution.sum()
    if total > 0.0:
        distribution /= total
    return distribution


def diffuse_distributions(
    graph: nx.Graph,
    distributions: "dict[int, np.ndarray]",
    num_iterations: int = 1,
    diffusion_rate: float = 0.7,
) -> "dict[int, np.ndarray]":
    """Apply diffusion to every distribution in the dict.

    Args:
        graph:          Topic k-NN graph (from ``build_diffusion_graph``).
        distributions:  Mapping of id → (n_topics,) distribution to diffuse.
        num_iterations: Message-passing passes over the graph.
        diffusion_rate: Neighbour influence weight ∈ [0, 1].

    Returns:
        New dict with the same keys and diffused, normalised distributions.
    """
    result: dict[int, np.ndarray] = {}
    for key, dist in distributions.items():
        result[key] = _diffuse_one(graph, dist, num_iterations, diffusion_rate)
    log.info("Diffused %d distributions.", len(result))
    return result
