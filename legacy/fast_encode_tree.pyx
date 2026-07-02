# Command line to compile: python legacy/setup.py build_ext --inplace
# Delete fast_encode_tree.c, fast_encode_tree.cp*.pyd, and build/ first to clean.

import numpy as np
cimport numpy as np

cdef class TreeNode:
    cdef public int id
    cdef public int type
    cdef public double distance
    cdef public object left
    cdef public object right
    cdef public double[:] author_topic_space_probs
    cdef public double left_right_link_prob
    cdef public set original_leaf_ids

    def __init__(self, int id, int type, double distance, double[:] author_topic_space_probs,
                 object left=None, object right=None,
                 double left_right_link_prob=0.0, original_leaf_ids=None):
        self.id = id
        self.type = type
        self.distance = distance
        self.author_topic_space_probs = np.copy(author_topic_space_probs)
        self.left = left
        self.right = right
        self.left_right_link_prob = left_right_link_prob
        self.original_leaf_ids = original_leaf_ids if original_leaf_ids is not None else set()

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.id, self.type, self.distance,
                np.asarray(self.author_topic_space_probs),
                self.left, self.right,
                self.left_right_link_prob, self.original_leaf_ids,
            ),
        )


cdef double calculate_left_right_link_prob(double[:] left_probs, double[:] right_probs,
                                           object G, dict author_index_map):
    cdef double numerator = 0.0
    cdef double expected_authors_left = 0.0
    cdef double expected_authors_right = 0.0
    cdef double denominator
    cdef int u_idx, v_idx, i

    for u, v in G.edges():
        u_idx = author_index_map[u]
        v_idx = author_index_map[v]
        numerator += (
            (left_probs[u_idx] * (1 - right_probs[u_idx])) * (right_probs[v_idx] * (1 - left_probs[v_idx]))
            + (left_probs[v_idx] * (1 - right_probs[v_idx])) * (right_probs[u_idx] * (1 - left_probs[u_idx]))
        )

    for i in range(left_probs.shape[0]):
        expected_authors_left  += left_probs[i]  * (1.0 - right_probs[i])
        expected_authors_right += right_probs[i] * (1.0 - left_probs[i])

    denominator = expected_authors_left * expected_authors_right
    if denominator == 0:
        return 0.0
    return numerator / denominator


def fast_encode_tree_structure(np.ndarray[double, ndim=2] Z, dict author_chunk_topic_distns, object G):
    cdef int num_chunk_topics = Z.shape[0] + 1
    cdef int num_authors = len(author_chunk_topic_distns)
    cdef dict author_index_map = {author: i for i, author in enumerate(author_chunk_topic_distns.keys())}
    cdef np.ndarray[double, ndim=2] author_topic_probs = np.zeros(
        (num_authors, num_chunk_topics), dtype=np.float64
    )
    cdef double[:] new_author_topic_space_probs
    cdef int i

    for author, distn in author_chunk_topic_distns.items():
        author_topic_probs[author_index_map[author], :] = distn

    cdef dict node_map = {
        i: TreeNode(i, 0, 0.0, author_topic_probs[:, i])
        for i in range(num_chunk_topics)
    }

    # Free base matrix — leaf nodes have their own copies (~22 GB saved).
    del author_topic_probs

    for i in range(Z.shape[0]):
        left_node  = node_map[int(Z[i, 0])]
        right_node = node_map[int(Z[i, 1])]
        dist       = Z[i, 2]

        left_array  = np.asarray(left_node.author_topic_space_probs)
        right_array = np.asarray(right_node.author_topic_space_probs)
        new_author_topic_space_probs_mv = np.ascontiguousarray(left_array + right_array, dtype=np.float64)
        new_author_topic_space_probs = new_author_topic_space_probs_mv

        new_node = TreeNode(num_chunk_topics + i, 1, dist, new_author_topic_space_probs,
                            left_node, right_node)
        new_node.left_right_link_prob = calculate_left_right_link_prob(
            left_node.author_topic_space_probs, right_node.author_topic_space_probs,
            G, author_index_map,
        )
        node_map[num_chunk_topics + i] = new_node

    return node_map[max(node_map.keys())], author_index_map
