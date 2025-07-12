# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Fast tree encoding module for MSTML (Cython implementation).

This module provides optimized tree data structures and encoding functions for 
hierarchical topic modeling and network analysis. This Cython implementation
provides significant performance improvements over the pure Python version.
"""

import numpy as np
import pickle
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, log
from libc.stdlib cimport malloc, free

# NumPy array type definitions
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int_t INT_t

# Initialize NumPy
cnp.import_array()


cdef class TreeNode:
    """
    Cython tree node class for hierarchical topic modeling.
    
    This class represents nodes in a dendrogram structure used for
    hierarchical clustering of topics and authors.
    """
    
    cdef public int id
    cdef public int type
    cdef public double distance
    cdef public cnp.ndarray author_topic_space_probs
    cdef public TreeNode left
    cdef public TreeNode right
    cdef public double left_right_link_prob
    cdef public set original_leaf_ids
    
    def __init__(self, int id, int type, double distance, 
                 cnp.ndarray[DTYPE_t, ndim=1] author_topic_space_probs,
                 TreeNode left=None, TreeNode right=None, 
                 double left_right_link_prob=0.0, 
                 set original_leaf_ids=None):
        """
        Initialize a tree node.
        
        Args:
            id: Unique identifier for the node
            type: Node type (0 for leaf, 1 for internal)
            distance: Distance value for hierarchical clustering
            author_topic_space_probs: Probability distribution over topics/authors
            left: Left child node
            right: Right child node
            left_right_link_prob: Probability of links between left and right subtrees
            original_leaf_ids: Set of original leaf node IDs encompassed by this node
        """
        self.id = id
        self.type = type
        self.distance = distance
        # Ensure the data is copied to prevent overwriting
        self.author_topic_space_probs = np.copy(author_topic_space_probs).astype(np.float64)
        self.left = left
        self.right = right
        self.left_right_link_prob = left_right_link_prob
        self.original_leaf_ids = original_leaf_ids if original_leaf_ids is not None else set()
    
    def __reduce__(self):
        """Return a tuple containing class info and instance data for pickling."""
        # Convert to regular numpy array for pickling
        author_topic_space_probs_as_numpy = np.asarray(self.author_topic_space_probs)
        
        # Tuple to return: (class, arguments to recreate object)
        args = (self.id, self.type, self.distance, author_topic_space_probs_as_numpy, 
                self.left, self.right, self.left_right_link_prob, self.original_leaf_ids)
        return self.__class__, args
    
    def __setstate__(self, state):
        """Restore the object state from the unpickled data."""
        # Unpack the state tuple and restore the attributes
        (self.id, self.type, self.distance, author_topic_space_probs_as_numpy, 
         self.left, self.right, self.left_right_link_prob, self.original_leaf_ids) = state
        
        # Convert numpy array back to the expected format
        self.author_topic_space_probs = np.asarray(author_topic_space_probs_as_numpy, dtype=np.float64)
    
    cpdef bint is_leaf(self):
        """Check if this node is a leaf node."""
        return self.left is None and self.right is None
    
    cpdef int get_leaf_count(self):
        """Get the number of leaf nodes in this subtree."""
        cdef int count = 0
        
        if self.is_leaf():
            return 1
        
        if self.left is not None:
            count += self.left.get_leaf_count()
        if self.right is not None:
            count += self.right.get_leaf_count()
        return count
    
    cpdef set get_all_leaf_ids(self):
        """Get all leaf node IDs in this subtree."""
        cdef set leaf_ids
        
        if self.is_leaf():
            return {self.id}
        
        leaf_ids = set()
        if self.left is not None:
            leaf_ids.update(self.left.get_all_leaf_ids())
        if self.right is not None:
            leaf_ids.update(self.right.get_all_leaf_ids())
        return leaf_ids


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double calculate_left_right_link_prob_c(cnp.ndarray[DTYPE_t, ndim=1] left_probs,
                                           cnp.ndarray[DTYPE_t, ndim=1] right_probs,
                                           object G,
                                           dict author_index_map):
    """
    Calculate the probability of links between left and right subtrees (Cython optimized).
    
    Args:
        left_probs: Probability distribution for left subtree
        right_probs: Probability distribution for right subtree
        G: NetworkX graph representing the network
        author_index_map: Dictionary mapping author IDs to indices
        
    Returns:
        Link probability between subtrees
    """
    cdef double numerator = 0.0
    cdef double denominator = 0.0
    cdef double expected_authors_left = 0.0
    cdef double expected_authors_right = 0.0
    cdef int u_idx, v_idx, i
    cdef double left_u, right_v, left_v, right_u
    cdef int n_authors = left_probs.shape[0]
    
    # Use NetworkX to find all the edge pairs and compute the sum of products for the numerator
    for u, v in G.edges():
        if u not in author_index_map or v not in author_index_map:
            continue
            
        u_idx = author_index_map[u]
        v_idx = author_index_map[v]
        
        # Ensure indices are within bounds
        if u_idx >= n_authors or v_idx >= n_authors:
            continue
        
        # Compute the numerator as the sum over all edges of the probability 
        # of u and v being in opposite sub-trees
        left_u = left_probs[u_idx] * (1.0 - right_probs[u_idx])
        right_v = right_probs[v_idx] * (1.0 - left_probs[v_idx])
        left_v = left_probs[v_idx] * (1.0 - right_probs[v_idx])
        right_u = right_probs[u_idx] * (1.0 - left_probs[u_idx])
        
        numerator += left_u * right_v + left_v * right_u
    
    # Calculate expected number of authors in each subtree
    for i in range(n_authors):
        expected_authors_left += left_probs[i] * (1.0 - right_probs[i])
        expected_authors_right += right_probs[i] * (1.0 - left_probs[i])
    
    denominator = expected_authors_left * expected_authors_right
    
    if denominator == 0.0:
        return 0.0
    
    return numerator / denominator


def calculate_left_right_link_prob(cnp.ndarray[DTYPE_t, ndim=1] left_probs,
                                  cnp.ndarray[DTYPE_t, ndim=1] right_probs,
                                  object G,
                                  dict author_index_map):
    """
    Python wrapper for calculate_left_right_link_prob_c.
    
    Args:
        left_probs: Probability distribution for left subtree
        right_probs: Probability distribution for right subtree
        G: NetworkX graph representing the network
        author_index_map: Dictionary mapping author IDs to indices
        
    Returns:
        Link probability between subtrees
    """
    return calculate_left_right_link_prob_c(left_probs, right_probs, G, author_index_map)


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_encode_tree_structure(cnp.ndarray[DTYPE_t, ndim=2] Z,
                              dict author_chunk_topic_distns,
                              object G):
    """
    Fast encoding of tree structure from hierarchical clustering (Cython optimized).
    
    Args:
        Z: Linkage matrix from hierarchical clustering
        author_chunk_topic_distns: Dictionary mapping authors to topic distributions
        G: NetworkX graph representing the network
        
    Returns:
        Tuple of (root_node, author_index_map)
    """
    cdef int num_chunk_topics = Z.shape[0] + 1
    cdef int num_authors = len(author_chunk_topic_distns)
    cdef int i, left_idx, right_idx, new_node_id
    cdef double dist, left_right_link_prob
    cdef cnp.ndarray[DTYPE_t, ndim=2] author_topic_probs
    cdef cnp.ndarray[DTYPE_t, ndim=1] new_author_topic_space_probs
    cdef TreeNode left_node, right_node, new_node, node
    cdef dict node_map = {}
    cdef set combined_leaf_ids
    
    # Create an index map for author IDs to contiguous indices
    author_index_map = {author: i for i, author in enumerate(author_chunk_topic_distns.keys())}
    
    # Initialize author-topic space probabilities as numpy arrays
    author_topic_probs = np.zeros((num_authors, num_chunk_topics), dtype=np.float64)
    for author, distn in author_chunk_topic_distns.items():
        author_index = author_index_map[author]
        author_topic_probs[author_index, :] = distn
    
    # Create initial leaf nodes
    for i in range(num_chunk_topics):
        node = TreeNode(
            id=i, 
            type=0,  # leaf node
            distance=0.0, 
            author_topic_space_probs=author_topic_probs[:, i],
            original_leaf_ids={i}
        )
        node_map[i] = node
    
    # Build the tree from the linkage matrix
    for i in range(Z.shape[0]):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        dist = Z[i, 2]
        
        # Get left and right nodes
        left_node = node_map[left_idx]
        right_node = node_map[right_idx]
        
        # Calculate combined author-topic space probabilities
        new_author_topic_space_probs = left_node.author_topic_space_probs + right_node.author_topic_space_probs
        
        # Calculate link probability between left and right subtrees
        left_right_link_prob = calculate_left_right_link_prob_c(
            left_node.author_topic_space_probs,
            right_node.author_topic_space_probs,
            G,
            author_index_map
        )
        
        # Combine original leaf IDs
        combined_leaf_ids = left_node.original_leaf_ids.union(right_node.original_leaf_ids)
        
        # Create new internal node
        new_node_id = num_chunk_topics + i
        new_node = TreeNode(
            id=new_node_id,
            type=1,  # internal node
            distance=dist,
            author_topic_space_probs=new_author_topic_space_probs,
            left=left_node,
            right=right_node,
            left_right_link_prob=left_right_link_prob,
            original_leaf_ids=combined_leaf_ids
        )
        
        # Add to node map
        node_map[new_node_id] = new_node
    
    # The root is the last node created
    root_node = node_map[num_chunk_topics + Z.shape[0] - 1]
    
    return root_node, author_index_map


def save_tree_structure(TreeNode root_node, dict author_index_map, str filepath):
    """
    Save tree structure and author index map to file.
    
    Args:
        root_node: Root node of the tree
        author_index_map: Dictionary mapping authors to indices
        filepath: Path to save the data
    """
    data = {
        'root_node': root_node,
        'author_index_map': author_index_map
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_tree_structure(str filepath):
    """
    Load tree structure and author index map from file.
    
    Args:
        filepath: Path to load the data from
        
    Returns:
        Tuple of (root_node, author_index_map)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['root_node'], data['author_index_map']


def traverse_tree_preorder(TreeNode node, object visit_func=None):
    """
    Traverse tree in pre-order (root, left, right).
    
    Args:
        node: Starting node
        visit_func: Function to call on each node (optional)
        
    Returns:
        List of nodes in pre-order
    """
    if node is None:
        return []
    
    result = [node]
    if visit_func is not None:
        visit_func(node)
    
    if node.left is not None:
        result.extend(traverse_tree_preorder(node.left, visit_func))
    if node.right is not None:
        result.extend(traverse_tree_preorder(node.right, visit_func))
    
    return result


def traverse_tree_postorder(TreeNode node, object visit_func=None):
    """
    Traverse tree in post-order (left, right, root).
    
    Args:
        node: Starting node
        visit_func: Function to call on each node (optional)
        
    Returns:
        List of nodes in post-order
    """
    if node is None:
        return []
    
    result = []
    
    if node.left is not None:
        result.extend(traverse_tree_postorder(node.left, visit_func))
    if node.right is not None:
        result.extend(traverse_tree_postorder(node.right, visit_func))
    
    result.append(node)
    if visit_func is not None:
        visit_func(node)
    
    return result


cpdef int get_tree_depth(TreeNode node):
    """
    Get the maximum depth of the tree (Cython optimized).
    
    Args:
        node: Root node of the tree
        
    Returns:
        Maximum depth of the tree
    """
    cdef int left_depth, right_depth
    
    if node is None:
        return 0
    
    if node.is_leaf():
        return 1
    
    left_depth = get_tree_depth(node.left) if node.left is not None else 0
    right_depth = get_tree_depth(node.right) if node.right is not None else 0
    
    return 1 + max(left_depth, right_depth)


def print_tree_structure(TreeNode node, int indent=0, object max_depth=None):
    """
    Print tree structure for debugging.
    
    Args:
        node: Node to start printing from
        indent: Current indentation level
        max_depth: Maximum depth to print (None for unlimited)
    """
    if node is None or (max_depth is not None and indent > max_depth):
        return
    
    prefix = "  " * indent
    node_type = "LEAF" if node.is_leaf() else "INTERNAL"
    print(f"{prefix}Node {node.id} ({node_type}): dist={node.distance:.4f}, "
          f"link_prob={node.left_right_link_prob:.4f}")
    
    if not node.is_leaf():
        if node.left is not None:
            print_tree_structure(node.left, indent + 1, max_depth)
        if node.right is not None:
            print_tree_structure(node.right, indent + 1, max_depth)