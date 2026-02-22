"""
_tree.py — Hierarchical-tree data structure for the legacy pipeline.

Provides:
  TreeNode                     — binary-tree node carrying author-topic distributions
  fast_encode_tree_structure   — builds the full encoded tree from a scipy linkage matrix

Ported from:
  AToMS-LP/AToMS/fast_encode_tree (Cython implementation)
  mstml/fast_encode_tree/fast_encode_tree_py.py (pure-Python fallback)

The tree is used by _scoring.py to compute HRG link-likelihood scores and
author meta-topic distributions after dendrogram truncation.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# TreeNode
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

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __reduce__(self):
        args = (
            self.id, self.type, self.distance,
            np.asarray(self.author_topic_space_probs),
            self.left, self.right,
            self.left_right_link_prob, self.original_leaf_ids,
        )
        return self.__class__, args

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Link-probability helper
# ---------------------------------------------------------------------------

def calculate_left_right_link_prob(
    left_probs: ndarray,
    right_probs: ndarray,
    G,
    author_index_map: "dict[int, int]",
) -> float:
    """MLE probability that a co-author edge crosses the left/right split.

    For each edge (u, v) in G, compute the probability that u is in the left
    subtree AND v is in the right subtree (or vice versa), then normalise by
    the expected number of such cross-links.

    Args:
        left_probs:       (n_authors,) probability of each author being in left subtree.
        right_probs:      (n_authors,) probability of each author being in right subtree.
        G:                NetworkX graph of the co-author network.
        author_index_map: Mapping author_id → row index in left_probs / right_probs.

    Returns:
        Link probability in [0, 1].  Returns 0.0 when denominator is zero.
    """
    numerator = 0.0

    for u, v in G.edges():
        if u not in author_index_map or v not in author_index_map:
            continue
        u_idx = author_index_map[u]
        v_idx = author_index_map[v]
        if u_idx >= len(left_probs) or v_idx >= len(left_probs):
            continue

        left_u  = left_probs[u_idx]  * (1.0 - right_probs[u_idx])
        right_v = right_probs[v_idx] * (1.0 - left_probs[v_idx])
        left_v  = left_probs[v_idx]  * (1.0 - right_probs[v_idx])
        right_u = right_probs[u_idx] * (1.0 - left_probs[u_idx])

        numerator += left_u * right_v + left_v * right_u

    expected_left  = sum(
        left_probs[i] * (1.0 - right_probs[i]) for i in range(len(left_probs))
    )
    expected_right = sum(
        right_probs[i] * (1.0 - left_probs[i]) for i in range(len(right_probs))
    )
    denominator = expected_left * expected_right
    return numerator / denominator if denominator > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Tree encoder
# ---------------------------------------------------------------------------

def fast_encode_tree_structure(
    Z: ndarray,
    author_chunk_topic_distns: "dict[int, ndarray]",
    G,
) -> "tuple[TreeNode, dict[int, int]]":
    """Build an encoded binary tree from a scipy linkage matrix.

    Each leaf corresponds to one chunk-topic (column of the topic-vector
    matrix); each internal node is formed by merging the two nodes indicated
    by the linkage row.

    At every internal node we compute:
      * the sum of left + right author-topic probability vectors
      * the HRG link probability between the left and right subtrees

    Args:
        Z:                         Linkage matrix, shape (n_topics − 1, 4).
        author_chunk_topic_distns: Mapping author_id → (n_topics,) float array.
        G:                         NetworkX co-author graph.

    Returns:
        (root_node, author_index_map) where author_index_map maps
        author_id → row index inside each node's author_topic_space_probs.
    """
    n_topics  = Z.shape[0] + 1
    n_authors = len(author_chunk_topic_distns)

    # Contiguous index map for authors
    author_index_map: dict[int, int] = {
        a: i for i, a in enumerate(author_chunk_topic_distns.keys())
    }

    # Build (n_authors × n_topics) probability matrix
    author_topic_probs = np.zeros((n_authors, n_topics), dtype=np.float64)
    for author, distn in author_chunk_topic_distns.items():
        idx = author_index_map[author]
        author_topic_probs[idx, :] = distn

    # Create leaf nodes
    node_map: dict[int, TreeNode] = {}
    for i in range(n_topics):
        node_map[i] = TreeNode(
            id=i,
            type=0,
            distance=0.0,
            author_topic_space_probs=author_topic_probs[:, i],
            original_leaf_ids={i},
        )

    # Build internal nodes from linkage rows
    for row in range(Z.shape[0]):
        left_idx  = int(Z[row, 0])
        right_idx = int(Z[row, 1])
        dist      = float(Z[row, 2])

        left_node  = node_map[left_idx]
        right_node = node_map[right_idx]

        combined_probs = (
            left_node.author_topic_space_probs +
            right_node.author_topic_space_probs
        )
        link_prob = calculate_left_right_link_prob(
            left_node.author_topic_space_probs,
            right_node.author_topic_space_probs,
            G,
            author_index_map,
        )

        new_id = n_topics + row
        node_map[new_id] = TreeNode(
            id=new_id,
            type=1,
            distance=dist,
            author_topic_space_probs=combined_probs,
            left=left_node,
            right=right_node,
            left_right_link_prob=link_prob,
            original_leaf_ids=left_node.original_leaf_ids | right_node.original_leaf_ids,
        )

    root = node_map[n_topics + Z.shape[0] - 1]
    return root, author_index_map
