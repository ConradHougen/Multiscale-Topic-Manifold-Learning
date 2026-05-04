# Command line to compile: python setup.py build_ext --inplace
# First, delete the fast_encode_tree.c and fast_encode_tree.cp... files and the "build" folder at this level to clean

import numpy as np
cimport numpy as np
from libc.math cimport pow

cdef class TreeNode:
    cdef public int id
    cdef public int type
    cdef public double distance
    cdef public object left
    cdef public object right
    cdef public double[:] author_topic_space_probs
    cdef public double left_right_link_prob
    cdef public set original_leaf_ids

    def __init__(self, int id, int type, double distance, double[:] author_topic_space_probs, object left=None, object right=None,
                 double left_right_link_prob=0.0, original_leaf_ids=None):
        self.id = id
        self.type = type
        self.distance = distance
        # Ensure the data is copied to prevent overwriting
        self.author_topic_space_probs = np.copy(author_topic_space_probs)
        self.left = left
        self.right = right
        self.left_right_link_prob = left_right_link_prob
        self.original_leaf_ids = original_leaf_ids if original_leaf_ids is not None else set()

        # Custom __reduce__ method for pickling
        def __reduce__(self):
            """ Return a tuple containing class info and instance data for pickling. """
            # Convert memoryview back to a numpy array for pickling
            author_topic_space_probs_as_numpy = np.asarray(self.author_topic_space_probs)

            # Tuple to return: (class, arguments to recreate object)
            args = (self.id, self.type, self.distance, author_topic_space_probs_as_numpy, self.left, self.right,
                    self.left_right_link_prob, self.original_leaf_ids)
            return self.__class__, args

        # Custom __setstate__ method for unpickling
        def __setstate__(self, state):
            """ Restore the object state from the unpickled data. """
            # Unpack the state tuple and restore the attributes
            self.id, self.type, self.distance, author_topic_space_probs_as_numpy, self.left, self.right, self.left_right_link_prob, self.original_leaf_ids = state

            # Convert numpy array back to memoryview
            self.author_topic_space_probs = np.asarray(author_topic_space_probs_as_numpy, dtype=np.double)

cdef double calculate_left_right_link_prob(double[:] left_probs, double[:] right_probs, object G, dict author_index_map):
    cdef double numerator, expected_authors_left, expected_authors_right, denominator

    numerator = 0.0

    # Use NetworkX to find all the edge pairs and compute the sum of products for the numerator
    for u, v in G.edges():
        u_idx = author_index_map[u]
        v_idx = author_index_map[v]

        # Compute the numerator as the sum over all edges of the probability of u and v being in opposite sub-trees
        numerator += ((left_probs[u_idx]*(1-right_probs[u_idx])) * (right_probs[v_idx]*(1-left_probs[v_idx]))) + \
                     ((left_probs[v_idx]*(1-right_probs[v_idx])) * (right_probs[u_idx]*(1-left_probs[u_idx])))

    expected_authors_left = 0.0
    expected_authors_right = 0.0

    for i in range(left_probs.shape[0]):
        expected_authors_left += left_probs[i] * (1.0 - right_probs[i])
        expected_authors_right += right_probs[i] * (1.0 - left_probs[i])

    denominator = expected_authors_left * expected_authors_right

    if denominator == 0:
        return 0.0

    return numerator / denominator


def fast_encode_tree_structure(np.ndarray[double, ndim=2] Z, dict author_chunk_topic_distns, object G):
    cdef int num_chunk_topics = Z.shape[0] + 1
    cdef int num_authors = len(author_chunk_topic_distns)

    # Create an index map for author IDs to contiguous indices
    cdef dict author_index_map = {author: i for i, author in enumerate(author_chunk_topic_distns.keys())}

    # Initialize author-topic space probabilities as numpy arrays
    cdef np.ndarray[double, ndim=2] author_topic_probs = np.zeros((num_authors, num_chunk_topics), dtype=np.float64)
    for author, distn in author_chunk_topic_distns.items():
        author_index = author_index_map[author]
        author_topic_probs[author_index, :] = distn

    cdef dict node_map = {i: TreeNode(i, 0, 0.0, author_topic_probs[:, i]) for i in range(num_chunk_topics)}

    cdef double[:] new_author_topic_space_probs

    for i in range(Z.shape[0]):
        left = int(Z[i, 0])
        right = int(Z[i, 1])
        dist = Z[i, 2]

        left_node = node_map[left]
        right_node = node_map[right]

        # Convert memoryviews to numpy arrays for the addition operation
        left_array = np.asarray(left_node.author_topic_space_probs)
        right_array = np.asarray(right_node.author_topic_space_probs)
        new_author_topic_space_probs_np = left_array + right_array

        # Create a contiguous numpy array and get its memoryview
        new_author_topic_space_probs_mv = np.ascontiguousarray(new_author_topic_space_probs_np, dtype=np.float64)
        new_author_topic_space_probs = new_author_topic_space_probs_mv

        new_node = TreeNode(num_chunk_topics + i, 1, dist, new_author_topic_space_probs, left_node, right_node)
        new_node.left_right_link_prob = calculate_left_right_link_prob(left_node.author_topic_space_probs, right_node.author_topic_space_probs, G, author_index_map)

        node_map[num_chunk_topics + i] = new_node

    return node_map[max(node_map.keys())], author_index_map
