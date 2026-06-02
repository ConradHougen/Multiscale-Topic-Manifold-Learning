"""
_tree.py — Hierarchical-tree data structure for the legacy pipeline.

Provides:
  TreeNode                     — binary-tree node carrying author-topic distributions
  fast_encode_tree_structure   — builds the full encoded tree from a scipy linkage matrix

Cython fast path
----------------
If fast_encode_tree.pyx has been compiled (``python legacy/setup.py build_ext
--inplace`` from the repo root), this module re-exports TreeNode and
fast_encode_tree_structure directly from the compiled extension — identical to
the AToMS-LP Cython implementation and substantially faster for large corpora.

If the compiled extension is absent (e.g. fresh clone without building), this
module falls back to a numpy-vectorised pure-Python implementation that
produces identical results.

To compile:
    cd Multiscale-Topic-Manifold-Learning
    python legacy/setup.py build_ext --inplace
"""

from __future__ import annotations

import gc
import logging
import numpy as np
from numpy import ndarray

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python fallback (always defined first)
# ---------------------------------------------------------------------------

class TreeNode:
    """Binary tree node for the encoded dendrogram.

    Each leaf corresponds to one chunk-topic; each internal node is created
    by the linkage algorithm merging two subtrees.

    Attributes
    ----------
    id : int
        Node identifier.  Leaf ids are 0 … (n_topics − 1);
        internal node ids start at n_topics.
    type : int
        0 = leaf, 1 = internal.
    distance : float
        Linkage merge distance at this node (0.0 for leaves).
    author_topic_space_probs : ndarray, shape (n_authors,)
        Sum of per-author topic probability for every original leaf topic
        beneath this node.  Used to compute subtree author distributions.
    left, right : TreeNode or None
        Child nodes.
    left_right_link_prob : float
        MLE probability that a random co-author link crosses the left/right
        split at this node (computed by calculate_left_right_link_prob).
    original_leaf_ids : set[int]
        Set of original leaf node ids encompassed by this subtree.
    """

    __slots__ = (
        "id", "type", "distance", "author_topic_space_probs",
        "left", "right", "left_right_link_prob", "original_leaf_ids",
    )

    def __init__(
        self,
        id: int,
        type: int,
        distance: float,
        author_topic_space_probs: ndarray,
        left: "TreeNode | None" = None,
        right: "TreeNode | None" = None,
        left_right_link_prob: float = 0.0,
        original_leaf_ids: "set[int] | None" = None,
    ) -> None:
        self.id = id
        self.type = type
        self.distance = float(distance)
        self.author_topic_space_probs = np.array(
            author_topic_space_probs, dtype=np.float64
        ).copy()
        self.left = left
        self.right = right
        self.left_right_link_prob = float(left_right_link_prob)
        self.original_leaf_ids = original_leaf_ids if original_leaf_ids is not None else set()

    def __reduce__(self):
        args = (
            self.id, self.type, self.distance,
            np.asarray(self.author_topic_space_probs),
            self.left, self.right,
            self.left_right_link_prob, self.original_leaf_ids,
        )
        return self.__class__, args

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def get_leaf_count(self) -> int:
        if self.is_leaf():
            return 1
        return (
            (self.left.get_leaf_count() if self.left else 0) +
            (self.right.get_leaf_count() if self.right else 0)
        )

    def get_all_leaf_ids(self) -> "set[int]":
        if self.is_leaf():
            return {self.id}
        ids: set[int] = set()
        if self.left:
            ids.update(self.left.get_all_leaf_ids())
        if self.right:
            ids.update(self.right.get_all_leaf_ids())
        return ids


def calculate_left_right_link_prob(
    left_probs: ndarray,
    right_probs: ndarray,
    G,
    author_index_map: "dict[int, int]",
) -> float:
    """MLE probability that a co-author edge crosses the left/right split."""
    n = len(left_probs)
    u_idxs, v_idxs = [], []
    for u, v in G.edges():
        if u not in author_index_map or v not in author_index_map:
            continue
        ui, vi = author_index_map[u], author_index_map[v]
        if ui < n and vi < n:
            u_idxs.append(ui)
            v_idxs.append(vi)

    if u_idxs:
        ui = np.asarray(u_idxs, dtype=np.intp)
        vi = np.asarray(v_idxs, dtype=np.intp)
        lu = left_probs[ui];  ru = right_probs[ui]
        lv = left_probs[vi];  rv = right_probs[vi]
        numerator = float(np.sum((lu * (1.0 - ru)) * (rv * (1.0 - lv))
                                 + (lv * (1.0 - rv)) * (ru * (1.0 - lu))))
    else:
        numerator = 0.0

    expected_left  = float(np.dot(left_probs,  1.0 - right_probs))
    expected_right = float(np.dot(right_probs, 1.0 - left_probs))
    denominator = expected_left * expected_right
    return numerator / denominator if denominator > 0.0 else 0.0


def fast_encode_tree_structure(
    Z: ndarray,
    author_chunk_topic_distns: "dict[int, ndarray]",
    G,
) -> "tuple[TreeNode, dict[int, int]]":
    """Build an encoded binary tree from a scipy linkage matrix."""
    n_topics  = Z.shape[0] + 1
    n_authors = len(author_chunk_topic_distns)

    author_index_map: dict[int, int] = {
        a: i for i, a in enumerate(author_chunk_topic_distns.keys())
    }

    author_topic_probs = np.zeros((n_authors, n_topics), dtype=np.float64)
    for author, distn in author_chunk_topic_distns.items():
        author_topic_probs[author_index_map[author], :] = distn

    node_map: dict[int, TreeNode] = {
        i: TreeNode(id=i, type=0, distance=0.0,
                    author_topic_space_probs=author_topic_probs[:, i],
                    original_leaf_ids={i})
        for i in range(n_topics)
    }

    # Free the base matrix now that leaf nodes have their own copies (~22 GB).
    del author_topic_probs
    gc.collect()

    # Precompute edge index arrays once — avoids repeated dict lookups per merge.
    _eu, _ev = [], []
    for u, v in G.edges():
        if u in author_index_map and v in author_index_map:
            ui, vi = author_index_map[u], author_index_map[v]
            if ui < n_authors and vi < n_authors:
                _eu.append(ui); _ev.append(vi)
    _edge_u = np.asarray(_eu, dtype=np.intp)
    _edge_v = np.asarray(_ev, dtype=np.intp)

    for row in range(Z.shape[0]):
        left_node  = node_map[int(Z[row, 0])]
        right_node = node_map[int(Z[row, 1])]
        dist       = float(Z[row, 2])

        lp = left_node.author_topic_space_probs
        rp = right_node.author_topic_space_probs

        if _edge_u.size:
            lu = lp[_edge_u]; ru = rp[_edge_u]
            lv = lp[_edge_v]; rv = rp[_edge_v]
            numerator = float(np.sum((lu * (1.0 - ru)) * (rv * (1.0 - lv))
                                     + (lv * (1.0 - rv)) * (ru * (1.0 - lu))))
        else:
            numerator = 0.0
        el = float(np.dot(lp, 1.0 - rp))
        er = float(np.dot(rp, 1.0 - lp))
        link_prob = numerator / (el * er) if el * er > 0.0 else 0.0

        new_id = n_topics + row
        node_map[new_id] = TreeNode(
            id=new_id, type=1, distance=dist,
            author_topic_space_probs=lp + rp,
            left=left_node, right=right_node,
            left_right_link_prob=link_prob,
            original_leaf_ids=left_node.original_leaf_ids | right_node.original_leaf_ids,
        )

    root = node_map[n_topics + Z.shape[0] - 1]
    return root, author_index_map


# ---------------------------------------------------------------------------
# Override with Cython extension if compiled
# ---------------------------------------------------------------------------
try:
    from .fast_encode_tree import (  # type: ignore[import]
        TreeNode as TreeNode,
        fast_encode_tree_structure as fast_encode_tree_structure,
    )
except ImportError:
    log.warning(
        "Cython extension 'fast_encode_tree' not found — falling back to pure-Python "
        "tree encoding. This will be significantly slower for large corpora. "
        "Build the extension with: python legacy/setup.py build_ext --inplace"
    )
