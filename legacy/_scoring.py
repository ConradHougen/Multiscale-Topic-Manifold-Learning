"""
_scoring.py — Step 5: Interdisciplinarity scoring via HRG dendrogram MLE.

Pipeline
--------
  encode_dendrogram_tree           → encoded TreeNode root + author_index_map
  compute_author_meta_distributions → author_id → distribution over meta-topics
  score_interdisciplinarity_docs   → doc_id → entropy-based score
  score_interdisciplinarity_links  → list[(frozenset(u,v), score)] sorted ascending
  rank_authors_by_interdisciplinarity → OrderedDict author_id → entropy score

The HRG scoring framework (Clauset et al., 2008) computes the maximum
likelihood probability that any co-author link existed, given the hierarchical
structure of the topic dendrogram.  Low-likelihood links are the most
surprising/interdisciplinary.

Ported from:
  AToMS-LP/AToMS_HRG_Ensemble_Interdisciplinarity.ipynb (cells 56-80)
  AToMS-LP/AToMS/atoms_hrg_library.py (truncate_dendrogram, get_new_leaf_nodes,
    calculate_author_distributions, setup_author_probs_matrix,
    setup_link_prob_matrix, compute_link_likelihood_scores, score_interdisciplinarity)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import networkx as nx
import numpy as np
from numpy import ndarray

from ._tree import TreeNode, fast_encode_tree_structure

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree encoding
# ---------------------------------------------------------------------------

def encode_dendrogram_tree(
    Z: ndarray,
    author_ct_distns: "dict[int, ndarray]",
    coauthor_graph: nx.Graph,
) -> "tuple[TreeNode, dict[int, int]]":
    """Encode a scipy linkage matrix into an HRG tree.

    Each internal node stores the MLE link probability across its left/right
    subtree split (calculated from the co-author graph).

    Args:
        Z:                Linkage matrix (n_topics − 1, 4).
        author_ct_distns: author_id → diffused distribution over all chunk-topics.
        coauthor_graph:   NetworkX co-author graph.

    Returns:
        (root, author_index_map)
        - root:             Encoded tree root (TreeNode).
        - author_index_map: author_id → row index inside node probability arrays.
    """
    log.info(
        "Encoding dendrogram tree (%d topics, %d authors) …",
        Z.shape[0] + 1, len(author_ct_distns),
    )
    return fast_encode_tree_structure(Z, author_ct_distns, coauthor_graph)


# ---------------------------------------------------------------------------
# Tree truncation helpers (ported from atoms_hrg_library.py)
# ---------------------------------------------------------------------------

def _truncate_dendrogram(node: Optional[TreeNode], cut_distance: float) -> Optional[TreeNode]:
    """Recursively truncate the encoded tree at ``cut_distance``.

    Nodes whose merge distance is below ``cut_distance`` are collapsed into
    leaves; their accumulated author-topic probabilities are summed.
    """
    if node is None:
        return None

    def _accumulate(current: TreeNode):
        probs = np.zeros_like(current.author_topic_space_probs, dtype=np.float64)
        ids: set[int] = set()
        stack = [current]
        while stack:
            n = stack.pop()
            if n:
                probs += n.author_topic_space_probs
                if n.left is None and n.right is None:
                    ids.add(n.id)
                else:
                    ids.update(n.original_leaf_ids)
                if n.left:
                    stack.append(n.left)
                if n.right:
                    stack.append(n.right)
        return probs, ids

    if node.distance >= cut_distance:
        new_node = TreeNode(
            id=node.id,
            type=node.type,
            distance=node.distance,
            author_topic_space_probs=node.author_topic_space_probs,
            left=None,
            right=None,
            left_right_link_prob=node.left_right_link_prob,
            original_leaf_ids=node.original_leaf_ids,
        )
        new_node.left  = _truncate_dendrogram(node.left,  cut_distance)
        new_node.right = _truncate_dendrogram(node.right, cut_distance)
        return new_node
    else:
        accumulated_probs, accumulated_ids = _accumulate(node)
        return TreeNode(
            id=node.id,
            type=node.type,
            distance=node.distance,
            author_topic_space_probs=accumulated_probs,
            left=None,
            right=None,
            left_right_link_prob=node.left_right_link_prob,
            original_leaf_ids=accumulated_ids,
        )


def _get_new_leaf_nodes(
    node: TreeNode,
) -> "list[tuple[TreeNode, set[int]]]":
    """Return the leaf nodes of the truncated tree and the original ids each covers."""
    if node.left is None and node.right is None:
        return [(node, node.original_leaf_ids)]
    leaves: list[tuple[TreeNode, set[int]]] = []
    if node.left:
        leaves.extend(_get_new_leaf_nodes(node.left))
    if node.right:
        leaves.extend(_get_new_leaf_nodes(node.right))
    return leaves


def _find_first_common_parent(
    root: Optional[TreeNode],
    node1: TreeNode,
    node2: TreeNode,
) -> Optional[TreeNode]:
    if root is None:
        return None
    if root is node1 or root is node2:
        return root
    left  = _find_first_common_parent(root.left,  node1, node2)
    right = _find_first_common_parent(root.right, node1, node2)
    if left is not None and right is not None:
        return root
    return left if left is not None else right


# ---------------------------------------------------------------------------
# Author meta-topic distributions
# ---------------------------------------------------------------------------

def compute_author_meta_distributions(
    root: TreeNode,
    cut_dist: float,
    author_ct_distns: "dict[int, ndarray]",
) -> "dict[int, ndarray]":
    """Compute each author's distribution over meta-topic clusters.

    The dendrogram is truncated at ``cut_dist``, yielding a set of new leaf
    nodes (each representing one meta-topic).  Each author's probability of
    belonging to each meta-topic is computed by summing their chunk-topic
    probabilities that fall under that leaf.

    Args:
        root:             Encoded tree root from ``encode_dendrogram_tree``.
        cut_dist:         Actual (un-normalised) linkage distance for the cut.
        author_ct_distns: author_id → diffused distribution over all chunk-topics.

    Returns:
        author_id → (n_meta_topics,) distribution (sums to 1, dtype float32).
    """
    truncated   = _truncate_dendrogram(root, cut_dist)
    new_leaves  = _get_new_leaf_nodes(truncated)
    n_leaves    = len(new_leaves)
    log.info("Truncated tree at dist=%.4f → %d meta-topic leaves.", cut_dist, n_leaves)

    author_distns: dict[int, ndarray] = {}
    for author, distn in author_ct_distns.items():
        new_dist = np.zeros(n_leaves, dtype=np.float32)
        for i, (leaf_node, original_ids) in enumerate(new_leaves):
            for orig_id in original_ids:
                if orig_id < len(distn):
                    new_dist[i] += float(distn[orig_id])
        total = new_dist.sum()
        author_distns[author] = new_dist / total if total > 0.0 else new_dist

    return author_distns


# ---------------------------------------------------------------------------
# Link probability matrices
# ---------------------------------------------------------------------------

def _setup_author_probs_matrix(
    new_leaves: "list[tuple[TreeNode, set[int]]]",
    author_index_map: "dict[int, int]",
) -> ndarray:
    """Build (n_authors, n_meta_topics) author probability matrix."""
    n_leaves  = len(new_leaves)
    n_authors = len(author_index_map)
    mat = np.zeros((n_authors, n_leaves), dtype=np.float64)
    for i, (node, _) in enumerate(new_leaves):
        mat[:, i] = node.author_topic_space_probs
    return mat.astype(np.float32)


def _setup_link_prob_matrix(
    truncated_root: TreeNode,
    new_leaves: "list[tuple[TreeNode, set[int]]]",
) -> ndarray:
    """Build (n_meta_topics, n_meta_topics) link probability matrix."""
    n_leaves = len(new_leaves)
    lpm = np.zeros((n_leaves, n_leaves), dtype=np.float32)
    cache: dict[frozenset, Optional[TreeNode]] = {}

    for i in range(n_leaves):
        for j in range(i, n_leaves):
            n1, _ = new_leaves[i]
            n2, _ = new_leaves[j]
            key = frozenset((n1.id, n2.id))
            if key not in cache:
                cache[key] = _find_first_common_parent(truncated_root, n1, n2)
            parent = cache[key]
            prob = float(parent.left_right_link_prob) if parent else 0.0
            lpm[i, j] = prob
            lpm[j, i] = prob

    return lpm


# ---------------------------------------------------------------------------
# Interdisciplinarity scoring
# ---------------------------------------------------------------------------

def score_interdisciplinarity_docs(
    df: "pd.DataFrame",
    author_meta_distns: "dict[int, ndarray]",
    author_col: str = "author_ids",
) -> "OrderedDict[int, float]":
    """Score each document's author team by Shannon entropy of their combined distribution.

    High entropy → the team collectively spans many meta-topics → interdisciplinary.

    Args:
        df:                DataFrame with ``author_col`` column.
        author_meta_distns: author_id → meta-topic distribution (from
                            ``compute_author_meta_distributions``).
        author_col:        Column holding per-paper author ID lists.

    Returns:
        OrderedDict sorted by score descending (most interdisciplinary first).
    """
    scores: dict[int, float] = {}
    zero_dist = next(iter(author_meta_distns.values())) * 0.0  # reference shape

    for doc_id, auth_ids in zip(df.index, df[author_col]):
        valid = [
            author_meta_distns[a].astype(np.float32)
            for a in (auth_ids or [])
            if a in author_meta_distns
        ]
        if not valid:
            continue
        combined = np.zeros_like(zero_dist, dtype=np.float32)
        for v in valid:
            combined += v
        combined /= len(valid)
        entropy = float(
            -np.sum(combined * np.log2(combined + np.float32(1e-10)))
        )
        scores[doc_id] = entropy

    return OrderedDict(
        sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    )


def score_interdisciplinarity_links(
    coauthor_graph: nx.Graph,
    author_index_map: "dict[int, int]",
    author_probs_matrix: ndarray,
    link_prob_matrix: ndarray,
) -> "list[tuple[frozenset, float]]":
    """Compute HRG link likelihood scores for every co-author edge.

    A low score means the link is surprising given the hierarchical topic
    structure — these are the most interdisciplinary co-author pairs.

    Args:
        coauthor_graph:     NetworkX co-author graph.
        author_index_map:   author_id → row index in ``author_probs_matrix``.
        author_probs_matrix: (n_authors, n_meta_topics) float32 matrix.
        link_prob_matrix:   (n_meta_topics, n_meta_topics) float32 MLE link probs.

    Returns:
        List of (frozenset{u, v}, likelihood_score) sorted ascending by score
        (lowest-likelihood = most interdisciplinary first).
    """
    edges = list(coauthor_graph.edges())
    scores = np.zeros(len(edges), dtype=np.float32)

    for idx, (u, v) in enumerate(edges):
        if u not in author_index_map or v not in author_index_map:
            continue
        u_probs = author_probs_matrix[author_index_map[u], :]
        v_probs = author_probs_matrix[author_index_map[v], :]
        outer   = np.outer(u_probs, v_probs)
        scores[idx] = float(np.sum(outer * link_prob_matrix))

    ranked = sorted(enumerate(scores), key=lambda x: x[1])
    return [(frozenset(edges[idx]), float(score)) for idx, score in ranked]


def rank_authors_by_interdisciplinarity(
    author_meta_distns: "dict[int, ndarray]",
) -> "OrderedDict[int, float]":
    """Rank authors by Shannon entropy of their meta-topic distribution.

    High entropy → the author's work spans many meta-topics.

    Args:
        author_meta_distns: author_id → meta-topic distribution.

    Returns:
        OrderedDict sorted by score descending (most interdisciplinary first).
    """
    scores: dict[int, float] = {}
    for auth_id, dist in author_meta_distns.items():
        d = dist.astype(np.float32)
        entropy = float(-np.sum(d * np.log2(d + np.float32(1e-10))))
        scores[auth_id] = entropy

    return OrderedDict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Full scoring pipeline (convenience wrapper)
# ---------------------------------------------------------------------------

def run_scoring(
    Z: ndarray,
    min_height: float,
    max_height: float,
    cut_height: float,
    author_ct_distns: "dict[int, ndarray]",
    coauthor_graph: nx.Graph,
    df: "pd.DataFrame",
    author_col: str = "author_ids",
) -> dict:
    """Run the full scoring pipeline in one call.

    Returns a dict with keys:
      root, author_index_map, author_meta_distns,
      doc_scores, link_scores, author_ranking.
    """
    import pandas as pd

    root, author_index_map = encode_dendrogram_tree(Z, author_ct_distns, coauthor_graph)

    cut_dist = min_height + cut_height * (max_height - min_height)
    author_meta_distns = compute_author_meta_distributions(root, cut_dist, author_ct_distns)

    # Rebuild truncated tree for matrix setup
    truncated  = _truncate_dendrogram(root, cut_dist)
    new_leaves = _get_new_leaf_nodes(truncated)
    author_probs_mat = _setup_author_probs_matrix(new_leaves, author_index_map)
    link_prob_mat    = _setup_link_prob_matrix(truncated, new_leaves)

    doc_scores    = score_interdisciplinarity_docs(df, author_meta_distns, author_col)
    link_scores   = score_interdisciplinarity_links(
        coauthor_graph, author_index_map, author_probs_mat, link_prob_mat
    )
    author_ranking = rank_authors_by_interdisciplinarity(author_meta_distns)

    return {
        "root":               root,
        "author_index_map":   author_index_map,
        "author_meta_distns": author_meta_distns,
        "doc_scores":         doc_scores,
        "link_scores":        link_scores,
        "author_ranking":     author_ranking,
    }
