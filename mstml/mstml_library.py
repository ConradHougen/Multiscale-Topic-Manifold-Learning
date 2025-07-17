import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import gensim.corpora as corpora
import pickle
import os
import re
import glob
import nltk
import random
import mplcursors
import hypernetx as hnx
import community as community_louvain  # For Louvain method

from itertools import islice
from matplotlib import cm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter, defaultdict, OrderedDict
from math import factorial
from pathlib import Path
from bisect import bisect_left
from datetime import datetime
from wordcloud import WordCloud
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from gensim.utils import simple_preprocess
from enum import Enum, auto
from gensim.models import AuthorTopicModel
from scipy.stats import kendalltau
from itertools import combinations
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Try to import the compiled Cython module, fall back to Python implementation
from fast_encode_tree import fast_encode_tree_structure, TreeNode


# Function that extracts 'text_processed' column per doc chunk
def preprocess_documents(chunks):
    preprocessed_chunks = []
    for chunk_df in chunks:
        preprocessed_docs = {}
        for doc_id in chunk_df.index:
            preprocessed_docs[doc_id] = chunk_df.loc[doc_id, 'text_processed']
        preprocessed_chunks.append(preprocessed_docs)
    return preprocessed_chunks


# Expands distributions over ensemble of topics
def expand_doc_topic_distns(doc_topic_distns, inds_by_chunk, num_chunks, ntopics_by_chunk):
    """
    Expand the document-topic distributions to the full set of chunk topics based on dynamic topics per chunk.

    Parameters:
    doc_topic_distns (dict): Dictionary keyed by doc_id mapping to numpy arrays of topic distributions.
    inds_by_chunk (dict): Dictionary mapping chunk index to a list of doc_ids in that chunk.
    num_chunks (int): Total number of chunks.
    ntopics_by_chunk (dict): Dictionary mapping chunk index to the number of topics in that chunk.

    Returns:
    dict: Expanded document-topic distributions keyed by doc_id.
    """
    expanded_distns = {}

    # Calculate the total number of topics across all chunks
    total_topics = sum(ntopics_by_chunk[chunk] for chunk in range(num_chunks))

    # Calculate starting index for each chunk based on cumulative topics
    chunk_start_index = {}
    current_index = 0
    for chunk in range(num_chunks):
        chunk_start_index[chunk] = current_index
        current_index += ntopics_by_chunk[chunk]

    # Create a reverse mapping from doc_id to chunk index
    doc_id_to_chunk = {doc_id: chunk_index for chunk_index, doc_ids in inds_by_chunk.items() for doc_id in doc_ids}

    for doc_id, dist in doc_topic_distns.items():
        if doc_id not in doc_id_to_chunk:
            continue  # Skip if doc_id is not found in the chunk mapping

        chunk = doc_id_to_chunk[doc_id]
        start_index = chunk_start_index[chunk]
        num_topics = ntopics_by_chunk[chunk]

        # Initialize an array for the full set of chunk topics in float32 for memory efficiency
        expanded_dist = np.zeros(total_topics, dtype=np.float32)

        # Place the original distribution in the expanded array
        expanded_dist[start_index:start_index + num_topics] = dist

        # Store the expanded distribution
        expanded_distns[doc_id] = expanded_dist

    return expanded_distns


def mstml_term_relevance_stable(word_freq_X_mat, topic_Xrow_idx, lambda_param, epsilon=1e-12):
    if lambda_param < 0 or lambda_param > 1:
        raise ValueError(f"lambda_param {lambda_param} is not a legal input")

    # Calculate topic-specific word probabilities p(w|z)
    topic_word_freqs = word_freq_X_mat[topic_Xrow_idx]
    p_w_given_z = topic_word_freqs / topic_word_freqs.sum()

    # Calculate overall word probabilities p(w)
    overall_word_freqs = word_freq_X_mat.sum(axis=0)
    p_w = overall_word_freqs / overall_word_freqs.sum()

    # Smooth probabilities
    p_w_given_z = p_w_given_z + epsilon
    p_w = p_w + epsilon

    # Compute term relevance
    log_p_w_given_z = np.log(p_w_given_z)
    log_p_w = np.log(p_w)
    adjusted_relevance = lambda_param * log_p_w_given_z + (1 - lambda_param) * (log_p_w_given_z - log_p_w)

    return adjusted_relevance


# Function to perform diffusion for the author topic distributions over the chunk topic kNN graph
def diffuse_distribution(graph, known_dist, num_iterations=1, diffusion_rate=0.7):
    num_topics = len(known_dist)
    distribution = np.copy(known_dist).astype(np.float32)  # Convert to float32
    buffer = np.zeros(num_topics, dtype=np.float32)  # Preallocate buffer for in-place updates

    for _ in range(num_iterations):
        buffer.fill(0)  # Reset buffer instead of reallocating

        for node in range(num_topics):
            if known_dist[node] > 0:
                buffer[node] += distribution[node]
            else:
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    weights = np.array([graph[node][neighbor]['weight'] for neighbor in neighbors], dtype=np.float32)
                    norm_weights = 1 - (weights / weights.sum())  # Convert distances to similarity weights in-place
                    neighbors_dist = np.array([distribution[neighbor] for neighbor in neighbors], dtype=np.float32)
                    buffer[node] += diffusion_rate * np.dot(norm_weights, neighbors_dist)  # Vectorized dot product

        distribution[:] = buffer  # Update distribution in-place

    # Normalize the distribution to sum to 1
    distribution /= distribution.sum(dtype=np.float32)
    return distribution


# Function to precompute weights for each node in the graph
def precompute_weights(graph, num_topics):
    weights_dict = {}
    for node in range(num_topics):
        neighbors = list(graph.neighbors(node))
        if neighbors:
            weights = np.array([graph[node][neighbor]['weight'] for neighbor in neighbors])
            weights = 1 - (weights / np.sum(weights))  # Convert distances to similarity weights
            weights_dict[node] = (neighbors, weights)
        else:
            weights_dict[node] = ([], np.array([]))
    return weights_dict


# Function to compute new distribution for a single node (can be parallelized)
def compute_new_distribution(node, known_dist, distribution, weights_dict, diffusion_rate):
    if known_dist[node] > 0:
        return distribution[node]  # No diffusion for nodes with known distribution
    else:
        neighbors, weights = weights_dict[node]
        if neighbors:
            neighbors_dist = np.array([distribution[neighbor] for neighbor in neighbors])
            return diffusion_rate * np.sum(weights * neighbors_dist)
        return 0.0  # No diffusion if no neighbors


# Optimized and parallelized function to perform diffusion
def diffuse_distribution_parallel(graph, known_dist, num_iterations=1, diffusion_rate=0.7, num_workers=4):
    num_topics = len(known_dist)
    distribution = np.copy(known_dist)

    # Precompute weights for neighbors
    weights_dict = precompute_weights(graph, num_topics)

    for _ in range(num_iterations):

        # Parallelize the diffusion process across nodes
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                compute_new_distribution,
                range(num_topics),
                [known_dist] * num_topics,
                [distribution] * num_topics,
                [weights_dict] * num_topics,
                [diffusion_rate] * num_topics
            ))

        new_distribution = np.array(results)

        # Add the contributions from the known distribution
        new_distribution += np.where(known_dist > 0, distribution, 0)

        # Update distribution for the next iteration
        distribution = new_distribution

    # Normalize the distribution to sum to 1
    distribution /= np.sum(distribution)
    return distribution


# Find the maximum and non-trivial minimum cut distances in the topic dendrogram
def find_max_min_cut_distance(root):
    """
    Parameters:
    - root: The root node of the dendrogram (TreeNode).

    Returns:
    - (max_distance, min_distance): A tuple containing the maximum and minimum cut distances.
    """

    def traverse_tree(node):
        """ Helper function to recursively traverse the tree and collect distances. """
        if node is None:
            return []

        # If it's a leaf node, it doesn't have a meaningful distance
        if node.left is None and node.right is None:
            return []

        # Recursively traverse the left and right children to collect distances
        left_distances = traverse_tree(node.left)
        right_distances = traverse_tree(node.right)

        # Collect the current node's distance (internal node)
        return [node.distance] + left_distances + right_distances

    # Get all internal node distances
    distances = traverse_tree(root)

    if not distances:
        return None, None  # No internal nodes

    # Maximum and minimum distances
    max_distance = max(distances)
    min_distance = min(distances)

    return max_distance, min_distance


# Rescale a parameter from the interval [0, 1] to [min_cut_distance, max_cut_distance].
def rescale_parameter(x, min_cut_distance, max_cut_distance):
    """
    Parameters:
    - x: The parameter in the range [0, 1].
    - min_cut_distance: The minimum cut distance from the dendrogram.
    - max_cut_distance: The maximum cut distance from the dendrogram.

    Returns:
    - rescaled_value: The rescaled value in the range [min_cut_distance, max_cut_distance].
    """
    if not (0 <= x <= 1):
        raise ValueError("Parameter x must be in the range [0, 1]")

    # Linear transformation to rescale x to the new range
    rescaled_value = min_cut_distance + x * (max_cut_distance - min_cut_distance)

    return rescaled_value


# Function to cut the topic dendrogram at a specific hellinger cut distance
def truncate_dendrogram(node, cut_distance):
    if node is None:
        return None

    def accumulate_probs_and_ids(current_node):
        # Initialize probs as float64 for consistent precision
        probs = np.zeros_like(current_node.author_topic_space_probs, dtype=np.float64)
        original_ids = set()
        stack = [current_node]

        while stack:
            n = stack.pop()
            if n:
                # Accumulate without any type conversion, maintaining float64 precision
                probs += n.author_topic_space_probs
                if n.left is None and n.right is None:
                    original_ids.add(n.id)
                else:
                    original_ids.update(n.original_leaf_ids)
                if n.left:
                    stack.append(n.left)
                if n.right:
                    stack.append(n.right)

        return probs, original_ids

    if node.distance >= cut_distance:
        truncated_node = TreeNode(
            id=node.id,
            type=node.type,
            left=None,
            right=None,
            distance=node.distance,
            author_topic_space_probs=node.author_topic_space_probs,  # Keep original precision
            left_right_link_prob=node.left_right_link_prob,
            original_leaf_ids=node.original_leaf_ids
        )
        left_truncated = truncate_dendrogram(node.left, cut_distance)
        right_truncated = truncate_dendrogram(node.right, cut_distance)
        truncated_node.left = left_truncated
        truncated_node.right = right_truncated
        return truncated_node
    else:
        # Accumulate probabilities without any dtype conversion
        accumulated_probs, accumulated_ids = accumulate_probs_and_ids(node)
        return TreeNode(
            id=node.id,
            type=node.type,
            left=None,
            right=None,
            distance=node.distance,
            author_topic_space_probs=accumulated_probs,  # Use accumulated float64 probs
            left_right_link_prob=node.left_right_link_prob,
            original_leaf_ids=accumulated_ids
        )


def get_leaf_nodes(node):
    """ Helper function to get all leaf nodes in a subtree. """
    leaves = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current:
            if current.left is None and current.right is None:
                leaves.append(current)
            else:
                if current.left:
                    stack.append(current.left)
                if current.right:
                    stack.append(current.right)
    return leaves


def find_first_common_parent(root, node1, node2):
    if root is None:
        return None

    if root == node1 or root == node2:
        return root

    left = find_first_common_parent(root.left, node1, node2)
    right = find_first_common_parent(root.right, node1, node2)

    if left is not None and right is not None:
        return root

    return left if left is not None else right


# This function gathers the author distributions over topics/meta-topics at the current truncation level of the topic
# dendrogram. Each row in the author_probs_matrix is an author-topic distribution
def setup_author_probs_matrix(new_leaf_nodes, author_index_map):
    """ Set up the author probabilities matrix. """
    # Number of leaf nodes and authors
    num_leaf_nodes = len(new_leaf_nodes)
    num_authors = len(author_index_map)

    # Initialize the author probabilities matrix with float64 for accurate population
    author_probs_matrix = np.zeros((num_authors, num_leaf_nodes), dtype=np.float64)

    # Populate the author probabilities matrix
    for i, (node, original_ids) in enumerate(new_leaf_nodes):
        author_probs_matrix[:, i] = node.author_topic_space_probs

    # Convert to float32 for memory efficiency in the return value
    return author_probs_matrix.astype(np.float32)


# Creates the MLE values for the probability of two topics at the truncated level of the dendrogram, being linked. This
# MLE estimate is based on counting the number of authors that are linked in the co-author network, according to their
# likelihoods of being in disjoint topics, at the current truncation level of the dendrogram.
def setup_link_prob_matrix(truncated_root, new_leaf_nodes):
    """ Set up the link probabilities matrix. """
    num_leaf_nodes = len(new_leaf_nodes)
    link_prob_matrix = np.zeros((num_leaf_nodes, num_leaf_nodes), dtype=np.float32)  # Set to float32 for efficiency

    common_parents = {}

    for i in range(num_leaf_nodes):
        for j in range(i, num_leaf_nodes):
            node1, _ = new_leaf_nodes[i]
            node2, _ = new_leaf_nodes[j]
            key = frozenset((node1.id, node2.id))
            if key not in common_parents:
                common_parents[key] = find_first_common_parent(truncated_root, node1, node2)
            common_parent = common_parents[key]
            link_prob_matrix[i, j] = np.float32(common_parent.left_right_link_prob)  # Ensure float32 consistency
            link_prob_matrix[j, i] = link_prob_matrix[i, j]  # Symmetric matrix

    return link_prob_matrix


def compute_link_likelihood_scores(G, author_index_map, author_probs_matrix, link_prob_matrix):
    """ Compute the link likelihood scores. """
    # Initialize an array to store link likelihood scores with float32 for efficiency
    link_likelihood_scores = np.zeros(len(G.edges()), dtype=np.float32)

    # Create a list of edges
    edges = list(G.edges())

    # Iterate through edges to calculate likelihood scores
    for idx, (u, v) in enumerate(edges):
        u_idx = author_index_map[u]
        v_idx = author_index_map[v]

        # Extract probability vectors for authors u and v
        u_probs = author_probs_matrix[u_idx, :].astype(np.float32)  # Ensure float32 for consistency
        v_probs = author_probs_matrix[v_idx, :].astype(np.float32)  # Ensure float32 for consistency

        # Calculate the outer product of u_probs and v_probs
        outer_product = np.outer(u_probs, v_probs)

        # Calculate the link likelihood score using vectorized operations
        link_likelihood_scores[idx] = np.sum(outer_product * link_prob_matrix).astype(np.float32)

    # Sort the likelihood scores in ascending order
    sorted_link_likelihood_scores = sorted(enumerate(link_likelihood_scores), key=lambda x: x[1])

    # Create a sorted list of link likelihood scores
    sorted_link_likelihood_scores_list = [(frozenset(edges[idx]), score) for idx, score in
                                          sorted_link_likelihood_scores]

    return sorted_link_likelihood_scores_list


def get_new_leaf_nodes(node):
    """
    Retrieve new leaf nodes and the set of original leaf nodes they encompass.
    """
    if node.left is None and node.right is None:
        return [(node, node.original_leaf_ids)]

    leaves = []
    if node.left:
        leaves.extend(get_new_leaf_nodes(node.left))
    if node.right:
        leaves.extend(get_new_leaf_nodes(node.right))

    return leaves


def calculate_author_distributions(new_leaf_nodes, author_chunk_topic_distns):
    """
    Calculate the distribution of each author over the new leaf nodes.
    """
    author_distributions = {}
    for author, distn in author_chunk_topic_distns.items():
        new_distn = np.zeros(len(new_leaf_nodes), dtype=np.float32)  # Use float32 for efficiency
        for i, (node, original_set) in enumerate(new_leaf_nodes):
            for original_node_id in original_set:
                new_distn[i] += np.float32(distn[original_node_id])  # Ensure consistency in float32
        author_distributions[author] = new_distn / new_distn.sum(dtype=np.float32)  # Normalize in float32
    return author_distributions


def score_interdisciplinarity(doc_authors, author_distributions):
    """
    Score the interdisciplinarity of author teams associated with each document.
    """
    doc_scores = {}
    for doc_id, authors in doc_authors.items():
        combined_dist = np.zeros_like(next(iter(author_distributions.values())), dtype=np.float32)  # Set to float32
        for author in authors:
            combined_dist += author_distributions[author].astype(np.float32)  # Ensure float32 consistency
        combined_dist /= len(authors)
        entropy = -np.sum(combined_dist * np.log2(combined_dist + np.float32(1e-10)),
                          dtype=np.float32)  # Consistent float32
        doc_scores[doc_id] = entropy
    return doc_scores


def calculate_major_n_topic_score(doc2auth_dict, author_distns, n_major_topics=1):
    auth_team_scores = OrderedDict()

    for document_id, author_ids in doc2auth_dict.items():
        # Skip documents with only one author
        if len(author_ids) == 1:
            continue

        major_n_tpc_weights = np.zeros_like(next(iter(author_distns.values())), dtype=np.float32)  # Set to float32

        for a_id in author_ids:
            distribution = author_distns[a_id].astype(np.float32)  # Ensure float32 consistency
            major_topic_inds = np.argsort(distribution)[-n_major_topics:][::-1]
            major_topic_distn_vals = distribution[major_topic_inds]
            major_n_tpc_weights[major_topic_inds] += major_topic_distn_vals

        # Normalize major_n_tpc_weights to be a distribution that sums to 1
        major_n_tpc_weights /= major_n_tpc_weights.sum(dtype=np.float32)

        # Use the entropy of this vector to score the document
        score = -np.sum(major_n_tpc_weights * np.log2(major_n_tpc_weights + np.float32(1e-10)), dtype=np.float32)
        auth_team_scores[document_id] = score

    return auth_team_scores


# Function to get the nth key-value pair
def get_nth_item_from_ordered_dict(ordered_dict, n):
    # Convert n to a positive index if it is negative
    if n < 0:
        n += len(ordered_dict)

    # Raise an error if the index is out of range
    if n < 0 or n >= len(ordered_dict):
        raise IndexError("Index out of range.")

    # Retrieve the nth item
    return next(islice(ordered_dict.items(), n, n + 1))


# Returns all documents that were authored or co-authored by authors in auth_team
def get_doc_set_from_author_team(authId_to_docs, auth_team):
    return set.union(set(), *[set(authId_to_docs[auth_i]) for auth_i in auth_team])


def map_chunk_and_topic_to_chunk_topic_index(cidx, tidx, num_chunks, topics_per_chunk):
    if cidx < 0 or cidx >= num_chunks:
        raise ValueError("Chunk number out of range")

    if isinstance(tidx, int):
        if tidx < 0 or tidx >= topics_per_chunk:
            raise ValueError("Topic number out of range")
        return cidx * topics_per_chunk + tidx

    elif isinstance(tidx, list):
        indices = []
        for t in tidx:
            if t < 0 or t >= topics_per_chunk:
                raise ValueError("One or more topic numbers out of range")
            indices.append(cidx * topics_per_chunk + t)
        return indices

    else:
        raise TypeError("Topic must be an int or a list of int")


# Assuming 'topic_vectors' is a dictionary where the key is the author (node) and
# the value is the topic vector (a list of probabilities over k topics)...
# This function splits the authors into topic communities.
def assign_topic_communities(topic_vectors):
    """
    Assign each author (node) to the topic with the maximum probability.
    :param topic_vectors: A dictionary of node: topic_vector.
    :return: A dictionary of node: assigned_topic.
    """
    topic_communities = {}
    for node, topic_vector in topic_vectors.items():
        assigned_topic = np.argmax(topic_vector)  # Assign to the topic with maximum probability
        topic_communities[node] = assigned_topic
    return topic_communities


# This function takes a networkx graph and target number of communities, which is
# the number of topics from our topic model, and partitions the network
def assign_louvain_communities(nxg, num_communities):
    """
    Partition the graph using the Louvain method and return community assignments.
    :param nxg: The co-author network graph.
    :param num_communities: The number of communities (topics) to match.
    :return: A dictionary of node: assigned_community.
    """
    # Perform Louvain method
    louvain_partition = community_louvain.best_partition(nxg, resolution=num_communities)
    return louvain_partition


# This function uses the Adjusted Rand Index to compare community partitions
def compare_communities_adjusted_rand_index(topic_communities, network_communities):
    """
    Compare the topic-based communities with the graph-based communities using ARI.
    :param topic_communities: A dictionary of node: assigned_topic.
    :param network_communities: A dictionary of node: assigned_community from network method.
    :return: Adjusted Rand Index score.
    """
    nodes = list(topic_communities.keys())  # Nodes should be the same in both
    topic_labels = [topic_communities[node] for node in nodes]
    network_labels = [network_communities[node] for node in nodes]

    # Calculate Adjusted Rand Index
    ari_score = adjusted_rand_score(topic_labels, network_labels)
    return ari_score


# This function uses the Normalized Mutual Information to compare community partitions
def compare_communities_norm_mutual_information(topic_communities, network_communities):
    """
    Compare the topic-based communities with the graph-based communities using NMI.
    :param topic_communities: A dictionary of node: assigned_topic.
    :param network_communities: A dictionary of node: assigned_community from network method.
    :return: Normalized Mutual Information score.
    """
    nodes = list(topic_communities.keys())  # Nodes should be the same in both
    topic_labels = [topic_communities[node] for node in nodes]
    network_labels = [network_communities[node] for node in nodes]

    # Calculate Normalized Mutual Information
    nmi_score = normalized_mutual_info_score(topic_labels, network_labels)
    return nmi_score


def calculate_rank_correlations(ranked_lists):
    """
    Calculate the rank correlations (Kendall's Tau) between successive rank-ordered lists.
    :param ranked_lists: List of rank-ordered objects (lists or OrderedDicts).
    :return: List of Kendall's Tau correlation coefficients between consecutive iterations.
    """
    correlations = []

    for i in range(len(ranked_lists) - 1):
        # Extract the ranks of items in the two consecutive lists
        list1 = ranked_lists[i]
        list2 = ranked_lists[i + 1]

        # Convert the list/OrderedDict keys to rank positions
        if isinstance(list1, OrderedDict):
            # If it's an OrderedDict, the ranks are implicit (insertion order)
            keys1 = list(list1.keys())
            keys2 = list(list2.keys())
        else:
            # If it's a list of pairs, use the first element (the frozenset)
            keys1 = [pair[0] for pair in list1]
            keys2 = [pair[0] for pair in list2]

        # Create a common ranking list by matching keys between list1 and list2
        common_keys = set(keys1).intersection(set(keys2))
        ranks1 = [keys1.index(k) for k in common_keys]
        ranks2 = [keys2.index(k) for k in common_keys]

        # Calculate Kendall's Tau rank correlation
        tau, _ = kendalltau(ranks1, ranks2)
        correlations.append(tau)

    return correlations


def get_top_percent(ranked_list, percent=5):
    """
    Extract the top percent of ranked items from the list or OrderedDict.
    :param ranked_list: Rank-ordered list or OrderedDict.
    :param percent: Percentage of top-ranked items to return.
    :return: Subset of the top-ranked items.
    """
    num_items = len(ranked_list)
    top_n = max(1, int(num_items * (percent / 100.0)))  # Ensure at least 1 element
    if isinstance(ranked_list, OrderedDict):
        return list(ranked_list.items())[:top_n]
    else:
        return ranked_list[:top_n]


def save_dendrogram_and_index_map(dendrogram, author_index_map, exp_dir):
    """ Save the dendrogram and author_index_map to separate files in the specified directory. """
    # Create full paths for the files
    dendrogram_path = os.path.join(exp_dir, "encoded_root.pkl")
    author_index_map_path = os.path.join(exp_dir, "author_index_map.pkl")

    # Save the dendrogram to the file
    with open(dendrogram_path, 'wb') as dendrogram_file:
        pickle.dump(dendrogram, dendrogram_file)

    # Save the author_index_map to the file
    with open(author_index_map_path, 'wb') as index_map_file:
        pickle.dump(author_index_map, index_map_file)

    print(f"Saved dendrogram to {dendrogram_path}")
    print(f"Saved author_index_map to {author_index_map_path}")


def load_dendrogram_and_index_map(exp_dir):
    """ Load the dendrogram and author_index_map from separate files in the specified directory. """
    # Create full paths for the files
    dendrogram_path = os.path.join(exp_dir, "encoded_root.pkl")
    author_index_map_path = os.path.join(exp_dir, "author_index_map.pkl")

    # Load the dendrogram from the file
    with open(dendrogram_path, 'rb') as dendrogram_file:
        dendrogram = pickle.load(dendrogram_file)

    # Load the author_index_map from the file
    with open(author_index_map_path, 'rb') as index_map_file:
        author_index_map = pickle.load(index_map_file)

    print(f"Loaded dendrogram from {dendrogram_path}")
    print(f"Loaded author_index_map from {author_index_map_path}")

    return dendrogram, author_index_map


# Helper function to count the number of unique topics or categories in the dataframe
def count_unique_topics_or_categories(df2count, colname, cat_filter=None):
    if colname in df2count.columns and colname == 'topic':
        unique_topics = df2count['topic'].nunique()  # Count unique integer topics
        return unique_topics
    elif colname in df2count.columns and colname == 'categories':
        # Case where the 'categories' column is present (containing space-separated strings)
        all_categories = df2count[
            'categories'].str.split().explode()  # Split each row into separate categories and explode into rows

        if cat_filter is not None:
            # Filter out categories not in the filter list
            all_categories = all_categories[all_categories.isin(cat_filter)]

        unique_categories = all_categories.nunique()  # Count unique categories
        return unique_categories
    else:
        print("Neither 'topic' nor 'categories' column found.")
        return None


def find_centroids_and_create_wordclouds(cut_height, dendrogram_linkage_mat_Z, WordsInTopicsAll):
    """
    Function to find the centroid topic vectors from WordsInTopicsAll based on dendrogram clusters at a given cut height.
    Generates word clouds for each cluster's centroid topic vector.

    Parameters:
    - cut_height: The cut height of the dendrogram to determine the clusters.
    - dendrogram_linkage_mat_Z: Linkage matrix from hierarchical clustering.
    - WordsInTopicsAll: List of lists of dataframes, where each dataframe holds topic-word distributions.

    Returns:
    - wordclouds_dict: Dictionary containing word clouds, where the key is the cluster number and the value is the word cloud object.
    """
    # Get the cluster labels based on the specified cut height
    cluster_labels = fcluster(dendrogram_linkage_mat_Z, t=cut_height, criterion='distance')

    # Create an empty dictionary to store the word clouds for each cluster
    wordclouds_dict = {}

    # Create a list to store all the topic vectors from WordsInTopicsAll
    all_topic_vectors = []

    # Flatten the WordsInTopicsAll list of lists into a single list of dataframes
    for topics_list in WordsInTopicsAll:
        all_topic_vectors.extend(topics_list)  # Now all_topic_vectors is a flat list of dataframes

    # Ensure that the number of clusters matches the number of topic vectors
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        # Find the topic vectors that belong to the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Collect all dataframes (topic vectors) corresponding to the current cluster
        cluster_topic_vectors = [all_topic_vectors[i] for i in cluster_indices]

        # Compute the centroid of the topic vectors by averaging the word frequencies
        if cluster_topic_vectors:
            # Merge dataframes on 'word' and average their 'freq' columns
            centroid_vector = pd.concat(cluster_topic_vectors).groupby('word').mean().reset_index()

            # Generate a word cloud for the centroid vector
            word_freq = pd.Series(centroid_vector['freq'].values, index=centroid_vector['word']).to_dict()
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

            # Store the word cloud in the dictionary
            wordclouds_dict[cluster_id] = wordcloud

    return wordclouds_dict


def display_wordcloud(wordclouds_dict, dict_key):
    """
    Function to generate and return the figure containing the word cloud for a specific key in the dictionary.

    Parameters:
    - wordclouds_dict: Dictionary containing word clouds.
    - dict_key: The key in the dictionary whose word cloud needs to be displayed.

    Returns:
    - fig: The matplotlib figure containing the word cloud.
    """
    if dict_key in wordclouds_dict:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordclouds_dict[dict_key], interpolation='bilinear')
        ax.axis('off')
        return fig
    else:
        print(f"Key {dict_key} not found in wordclouds_dict.")
        return None


def plot_phate_embedding_with_filtered_chunks(exp_dir, cut_height, cluster_labels, phate_data,
                                              start_chunk_index, end_chunk_index, time_chunk_names, ntopics_by_chunk):
    """
    Function to plot PHATE embedding, with points filtered by the specified chunk index range,
    and colored by topic cluster labels. The title includes the min and max date range.

    Parameters:
    - cut_height: Dendrogram cut height.
    - cluster_labels: Cluster labels from the hierarchical clustering.
    - phate_data: PHATE transformed data.
    - start_chunk_index: Start index of the chunk range.
    - end_chunk_index: End index of the chunk range.
    - time_chunk_names: List of time chunk names (e.g., 'Month Year (topic_count)').
    - ntopics_by_chunk: Dictionary mapping chunk indices to the number of topics in that chunk.

    Returns:
    - A scatter plot of the PHATE embedding with points filtered by chunk index range.
    """

    # Handle out-of-bounds indices
    start_chunk_index = max(0, start_chunk_index)
    end_chunk_index = min(len(time_chunk_names) - 1, end_chunk_index)

    # Get the filtered range of chunk indices
    chunk_indices = np.arange(start_chunk_index, end_chunk_index + 1)

    # Generate the full chunk labels dynamically based on ntopics_by_chunk
    full_chunk_labels = []
    for chunk in range(len(cluster_labels)):
        num_topics = ntopics_by_chunk.get(chunk, 0)
        full_chunk_labels.extend([chunk] * num_topics)
    full_chunk_labels = np.array(full_chunk_labels)

    # Filter phate_data and cluster_labels based on chunk indices
    filtered_indices = np.isin(full_chunk_labels, chunk_indices)
    filtered_phate_data = phate_data[filtered_indices]
    filtered_cluster_labels = cluster_labels[filtered_indices]

    # Normalize the cluster labels for color mapping
    cmap = cm.get_cmap('rainbow')
    norm_cluster_labels = (filtered_cluster_labels - min(cluster_labels)) / (max(cluster_labels) - min(cluster_labels))
    colors = cmap(norm_cluster_labels)  # Assign colors based on normalized cluster labels

    # Create the scatter plot with full X and Y limits from the entire dataset
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot using the precomputed colors
    scatter = ax.scatter(filtered_phate_data[:, 0], filtered_phate_data[:, 1],
                         c=colors, s=50, edgecolor='k')  # Using colors variable

    # Add interactive hover tool using mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"Index: {sel.index}"))

    # Compute appropriate x and y limits with a margin
    x_min, x_max = np.min(phate_data[:, 0]), np.max(phate_data[:, 0])
    y_min, y_max = np.min(phate_data[:, 1]), np.max(phate_data[:, 1])
    margin_x = (x_max - x_min) * 0.05  # Add 5% margin on each side for x
    margin_y = (y_max - y_min) * 0.05  # Add 5% margin on each side for y

    # Set axis limits with margins
    ax.set_xlim([x_min - margin_x, x_max + margin_x])
    ax.set_ylim([y_min - margin_y, y_max + margin_y])

    # Colorbar based on the full cluster labels range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(cluster_labels), vmax=max(cluster_labels)))
    sm.set_array([])  # Only needed for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)

    # Set colorbar ticks
    num_topics = len(np.unique(cluster_labels))
    desired_num_ticks = 20
    tick_interval = max(1, int(num_topics / desired_num_ticks))
    cbar.set_ticks(np.arange(min(cluster_labels), max(cluster_labels) + 1, tick_interval))
    cbar.set_ticklabels(np.arange(min(cluster_labels), max(cluster_labels) + 1, tick_interval))
    cbar.set_label('Topic Labels', rotation=270, labelpad=15)

    # Extract the date range for the selected chunks
    dates = [chunk.split(' (')[0] for chunk in time_chunk_names[start_chunk_index:end_chunk_index + 1]]
    time_chunk_start_times = pd.to_datetime(dates, format='%b %Y')
    min_date = time_chunk_start_times.min().strftime('%b %Y')
    max_date = time_chunk_start_times.max().strftime('%b %Y')

    # Add axis labels and title with the min-max date range
    ax.set_title(f'FAISS Hellinger-PHATE Embedding: {min_date} - {max_date}')
    ax.set_xlabel('PHATE 1')
    ax.set_ylabel('PHATE 2')

    min_month, min_year = min_date.split()
    max_month, max_year = max_date.split()

    # Save the plot with the cut_height, start-end date range, and chunk indices
    filename = f'phate_filtered_clusters_cut{cut_height:.2f}_{start_chunk_index}_{min_year}{min_month}_to_{max_year}{max_month}.png'
    plt.savefig(exp_dir + filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_wordcloud_for_topic(word_freq_array, vocab, topic_index, lambda_param=0.6, num_top_terms=100,
                             use_contrastive_relevance=False, topic_labels_for_contrastive_relevance=None, epsilon=1e-12):
    """
    Generate and return a word cloud for a given topic index using either the default or contrastive relevance metric.

    Parameters:
    word_freq_array: np.array
        A 2D array where each row represents word frequencies for a specific topic (e.g., WordsFreqMerged).
    vocab: list
        The list of words corresponding to the columns in word_freq_array.
    topic_index: int
        The index of the topic for which to generate the word cloud.
    lambda_param: float
        The lambda parameter (0 to 1) controlling the relevance metric.
    num_top_terms: int
        The number of top relevant terms to include in the word cloud.
    use_contrastive_relevance: bool
        If True, computes the contrastive relevance scores; otherwise, computes the default relevance scores.
    topic_labels_for_contrastive_relevance: np.array, optional
        An array of cluster labels for each topic (required if use_contrastive_relevance is True).
    epsilon: float
        A small constant to avoid division by zero and ensure numerical stability.

    Returns:
    WordCloud
        A WordCloud object for the specified topic, ready for saving or further use.
    """
    # Calculate topic-specific word probabilities p(w|z)
    topic_word_freqs = word_freq_array[topic_index]
    p_w_given_z = topic_word_freqs / topic_word_freqs.sum()

    # Calculate overall word probabilities p(w)
    overall_word_freqs = word_freq_array.sum(axis=0)
    p_w = overall_word_freqs / overall_word_freqs.sum()

    if use_contrastive_relevance and topic_labels_for_contrastive_relevance is not None:
        # Identify out-of-cluster topic indices
        cluster_label = topic_labels_for_contrastive_relevance[topic_index]
        out_cluster_indices = np.where(topic_labels_for_contrastive_relevance != cluster_label)[0]

        # Calculate out-of-cluster word probabilities
        out_cluster_word_freqs = word_freq_array[out_cluster_indices].sum(axis=0)
        p_w_given_z_out_cluster = out_cluster_word_freqs / (out_cluster_word_freqs.sum() + epsilon)

        # Smooth probabilities to avoid log(0)
        p_w_given_z = p_w_given_z + epsilon
        p_w = p_w + epsilon
        p_w_given_z_out_cluster = p_w_given_z_out_cluster + epsilon

        # Compute contrastive relevance scores using log odds
        log_p_w_given_z = np.log(p_w_given_z)
        log_p_w_given_z_out_cluster = np.log(p_w_given_z_out_cluster)
        log_p_w = np.log(p_w)

        relevance_scores = lambda_param * log_p_w_given_z + \
                           (1 - lambda_param) * (log_p_w_given_z - log_p_w - (log_p_w_given_z_out_cluster - log_p_w))

    else:
        # Default relevance score
        relevance_scores = mstml_term_relevance_stable(word_freq_array, topic_index, lambda_param=lambda_param)

    # Get the top relevant terms based on the relevance scores
    top_term_indices = np.argsort(relevance_scores)[-num_top_terms:][::-1]
    top_term_freqs = {vocab[i]: p_w_given_z[i] for i in top_term_indices}

    # Generate the word cloud
    wc = WordCloud(width=800, height=400, max_words=num_top_terms).generate_from_frequencies(top_term_freqs)

    # Plot the word cloud
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')  # Turn off the axis
    plt.show()

    return fig


def refine_smooth_path(phate_data, selected_points, tolerance=0.5, max_iters=100):
    """
    Refine the selected path by replacing points that introduce significant jumps with better alternatives.

    Parameters:
    - phate_data: 2D array with PHATE embedding coordinates of shape (n_samples, 3)
    - selected_points: list of int, indices for the initially selected path
    - tolerance: float, tolerance threshold for considering a point part of a smooth path
    - max_iters: int, maximum number of iterations to attempt before stopping

    Returns:
    - refined_path: list of int, indices for a smoother path without abrupt jumps
    """
    refined_path = selected_points.copy()
    path_stable = False  # Flag to check if path is stable
    iter_count = 0  # Keep track of the number of iterations

    while not path_stable and iter_count < max_iters:
        path_stable = True  # Assume the path is stable at the start of each iteration
        iter_count += 1  # Increment iteration counter

        # Iterate over internal points (skip the first and last)
        for i in range(1, len(refined_path) - 1):
            prev_idx, curr_idx, next_idx = refined_path[i - 1], refined_path[i], refined_path[i + 1]
            prev_point, curr_point, next_point = phate_data[prev_idx], phate_data[curr_idx], phate_data[next_idx]

            # Calculate the midpoint and distance metrics
            midpoint = (prev_point + next_point) / 2
            dist_to_midpoint = np.linalg.norm(curr_point - midpoint)
            dist_prev_next = np.linalg.norm(prev_point - next_point)

            # If the current point deviates too far, find a better candidate
            if dist_to_midpoint > tolerance * dist_prev_next:
                path_stable = False  # Mark the path as unstable because we need to replace this point

                # Find a better candidate close to the midpoint
                candidate_indices = [
                    idx for idx in range(len(phate_data))
                    if np.linalg.norm(phate_data[idx] - midpoint) <= tolerance * dist_prev_next
                ]

                # Replace the point with the candidate closest to the midpoint
                if candidate_indices:
                    best_candidate = min(candidate_indices, key=lambda idx: np.linalg.norm(phate_data[idx] - midpoint))
                    refined_path[i] = best_candidate

    # Log a warning if the maximum number of iterations is reached
    if iter_count >= max_iters:
        print(f"Warning: Path refinement reached the maximum number of iterations ({max_iters}) without stabilizing.")

    return refined_path


def compute_interdisciplinarity_score_fast(doc_authors, author_distributions, author_to_docs,
                                           n_hot=1, min_hot_threshold=0.2):
    """
    Compute interdisciplinarity scores for all documents efficiently.

    Parameters:
    doc_authors (dict): Mapping of document IDs to lists of authors.
    author_distributions (dict): Mapping of authors to their topic distribution vectors.
    author_to_docs (dict): Mapping of authors to lists of their associated documents.
    n_hot (int): Number of "hot" positions to retain in topic distributions (default: 1).
    min_hot_threshold (float): Minimum threshold for a topic to be considered "hot" (default: 0.2).

    Returns:
    OrderedDict: Sorted interdisciplinarity scores for all documents (descending order).
    """
    # Precompute weights (sqrt of number of documents for each author)
    author_weights = {author: np.sqrt(len(docs)) for author, docs in author_to_docs.items()}

    # Prepare document scores
    doc_scores = {}

    # Process each document
    for doc_id, authors in doc_authors.items():
        num_topics = len(next(iter(author_distributions.values())))  # Number of topics
        weighted_sum = np.zeros(num_topics, dtype=np.float32)  # Use float32 for efficiency

        for author in authors:
            # Get author distribution and weight
            topic_dist = author_distributions[author]
            weight = author_weights.get(author, 0)

            # Compute n-hot vector
            sorted_indices = np.argsort(topic_dist)[::-1]  # Sort indices by topic probability (descending)
            n_hot_indices = sorted_indices[:n_hot]  # Top n_hot indices
            n_hot_vector = np.zeros(num_topics, dtype=np.float32)
            for idx in n_hot_indices:
                if topic_dist[idx] >= min_hot_threshold:
                    n_hot_vector[idx] = 1
                else:
                    break

            # Add weighted n-hot vector
            weighted_sum += weight * n_hot_vector

        # Compute total probability mass and normalize
        total_mass = np.sum(weighted_sum, dtype=np.float32)
        if total_mass > 0:
            normalized_vector = weighted_sum / total_mass
        else:
            normalized_vector = np.zeros_like(weighted_sum)

        # Compute entropy of the normalized vector
        entropy = -np.sum(normalized_vector * np.log2(normalized_vector + np.float32(1e-10)), dtype=np.float32)

        # Compute final score as mass-weighted entropy
        doc_scores[doc_id] = total_mass * entropy

    # Sort scores in descending order
    sorted_doc_interdisciplinarity_scores = OrderedDict(
        sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_doc_interdisciplinarity_scores


def normalize_vector(vector):
    """
    Normalize a vector to make it a valid probability distribution.

    Parameters:
    vector (np.array): Input vector.

    Returns:
    np.array: Normalized vector.
    """
    total = np.sum(vector)
    if total > 0:
        return vector / total
    return vector


def compute_pairwise_interdisciplinarity(doc_authors, author_distributions, author_to_docs,
                                         topics, n_hot=1, min_hot_threshold=0.2):
    """
    Compute interdisciplinarity scores for each document between all pairs of topics in the given list.

    Parameters:
    doc_authors (dict): Mapping of document IDs to lists of authors.
    author_distributions (dict): Mapping of authors to their topic distribution vectors.
    author_to_docs (dict): Mapping of authors to lists of their associated documents.
    topics (list): List of topic indices to consider.
    n_hot (int): Number of "hot" positions to retain in topic distributions (default: 1).
    min_hot_threshold (float): Minimum threshold for a topic to be considered "hot" (default: 0.2).

    Returns:
    dict: A dictionary of interdisciplinarity scores, supporting symmetric topic pairs:
          {topic_pair: OrderedDict(doc_id: score, ...), ...}
    """
    # Generate all unique pairs of topics
    topic_pairs = list(combinations(topics, 2))

    # Initialize results dictionary
    pairwise_scores = {}

    for topic_pair in topic_pairs:
        i, j = topic_pair
        mask = np.zeros(len(next(iter(author_distributions.values()))), dtype=np.float32)
        mask[i] = 1
        mask[j] = 1

        # Apply mask to author distributions
        masked_distributions = {
            author: normalize_vector(dist * mask) for author, dist in author_distributions.items()
        }

        # Compute interdisciplinarity scores using the masked distributions
        scores = compute_interdisciplinarity_score_fast(
            doc_authors, masked_distributions, author_to_docs, n_hot=n_hot, min_hot_threshold=min_hot_threshold
        )

        # Store scores under both (i, j) and (j, i) for symmetry
        pairwise_scores[(i, j)] = scores
        pairwise_scores[(j, i)] = scores

    return pairwise_scores


def get_chunk_to_meta_mapping(Z, cut_height):
    """
    Generate a mapping from chunk topics to meta topics at a given cut height.

    Parameters:
    Z (ndarray): Linkage matrix from scipy's hierarchical clustering.
    cut_height (float): Normalized cut height (0 to 1).

    Returns:
    dict: A dictionary mapping chunk topic indices to meta topic indices.
    int: Number of meta topics.
    """
    # Determine min and max cut heights from the dendrogram
    min_cut_height = np.min(Z[:, 2])  # Minimum distance (height)
    max_cut_height = np.max(Z[:, 2])  # Maximum distance (height)

    # Rescale normalized cut_height to the actual cut distance
    cut_dist = rescale_parameter(cut_height, min_cut_height, max_cut_height)

    # Determine the number of chunk topics (leaf nodes)
    num_chunk_topics = Z.shape[0] + 1

    # Get cluster labels for chunk topics at the specified cut distance
    cluster_labels = fcluster(Z, t=cut_dist, criterion='distance')

    # Create a mapping from chunk topics to meta topics
    chunk_to_meta = {chunk_topic: cluster_labels[chunk_topic] - 1 for chunk_topic in range(num_chunk_topics)}

    return chunk_to_meta, max(cluster_labels)


def get_meta_topic_distributions(ct_distns, chunk_to_meta, num_meta_topics):
    """
    Map chunk topic distributions to meta topic distributions.

    Parameters:
    ct_distns (dict): A dictionary where keys are entity IDs (documents, authors, etc.)
                      and values are their chunk topic distributions.
    chunk_to_meta (dict): Mapping from chunk topic indices to meta topic indices.
    num_meta_topics (int): Number of meta topics.

    Returns:
    dict: A dictionary where keys are entity IDs and values are their meta topic distributions.
    """
    # Convert `ct_distns` dictionary into a 2D array for vectorized processing
    entity_ids = list(ct_distns.keys())
    num_entities = len(entity_ids)
    num_chunk_topics = len(next(iter(ct_distns.values())))  # Length of chunk topic distributions

    # Create an array of shape (num_entities, num_chunk_topics)
    ct_distns_array = np.zeros((num_entities, num_chunk_topics), dtype=np.float32)
    for i, entity_id in enumerate(entity_ids):
        ct_distns_array[i] = ct_distns[entity_id]

    # Create a matrix mapping chunk topics to meta topics
    chunk_to_meta_matrix = np.zeros((num_chunk_topics, num_meta_topics), dtype=np.float32)
    for chunk_idx, meta_idx in chunk_to_meta.items():
        chunk_to_meta_matrix[chunk_idx, meta_idx] = 1

    # Multiply chunk topic distributions by the mapping matrix
    meta_topic_distns_array = np.dot(ct_distns_array, chunk_to_meta_matrix)

    # Normalize each entity's meta topic distribution
    row_sums = meta_topic_distns_array.sum(axis=1, keepdims=True)
    meta_topic_distns_array = np.divide(meta_topic_distns_array, row_sums, where=row_sums > 0)

    # Convert back to dictionary with original entity IDs
    meta_topic_distns = {
        entity_ids[i]: meta_topic_distns_array[i] for i in range(num_entities)
    }

    return meta_topic_distns

def select_smooth_path(phate_data, primary_dim=0, secondary_dim=1, tertiary_dim=2,
                       num_points=10, secondary_method='max', secondary_percentile_range=None,
                       tertiary_method='max', tolerance_ratio=0.05, path_step_tol=0.5):
    """
    Select points for traversal along a specified primary dimension with smooth path selection.

    Parameters:
    - phate_data: 2D array with PHATE embedding coordinates of shape (n_samples, 3)
    - primary_dim: int, dimension to traverse along (0, 1, or 2 for PHATE 1, 2, or 3)
    - secondary_dim: int, dimension to filter by (0, 1, or 2 for PHATE 1, 2, or 3)
    - tertiary_dim: int, dimension to optimize within filtered set (0, 1, or 2 for PHATE 1, 2, or 3)
    - num_points: int, number of points to sample along the primary dimension
    - secondary_method: str, method for filtering by secondary dimension ('max', 'min', or 'percentile')
    - secondary_percentile_range: tuple, optional, range (lower, upper) for percentile selection on the secondary dimension
    - tertiary_method: str, method for optimizing the tertiary dimension ('max' or 'min')
    - tolerance_ratio: float, fraction of the range for selection tolerance across all dimensions
    - path_step_tol: float, fraction of distance from prev to next point in path that a middle point can jump

    Returns:
    - optimized_path: list of int, indices of selected points along a smooth path
    """
    # Determine the range for the primary dimension
    primary_min, primary_max = phate_data[:, primary_dim].min(), phate_data[:, primary_dim].max()
    primary_range = primary_max - primary_min
    primary_values = np.linspace(primary_min, primary_max, num_points)

    # Set tolerance based on tolerance_ratio and primary dimension range
    tolerance = tolerance_ratio * primary_range

    # Precompute percentile bounds for the secondary dimension (if needed)
    if secondary_method == 'percentile' and secondary_percentile_range is not None:
        lower_bound = np.percentile(phate_data[:, secondary_dim], secondary_percentile_range[0])
        upper_bound = np.percentile(phate_data[:, secondary_dim], secondary_percentile_range[1])

    # Initial selection of indices based on primary, secondary, and tertiary dimension criteria
    initial_indices = []
    for i, primary_val in enumerate(primary_values):
        # Filter points within the tolerance interval around primary_val in the primary dimension
        interval_mask = np.abs(phate_data[:, primary_dim] - primary_val) <= tolerance
        candidates = phate_data[interval_mask]
        indices = np.where(interval_mask)[0]

        if len(candidates) == 0:
            continue  # Skip if no points are within the tolerance range

        # Apply the secondary dimension criterion
        if secondary_method == 'percentile' and secondary_percentile_range is not None:
            secondary_mask = (phate_data[:, secondary_dim] >= lower_bound) & (phate_data[:, secondary_dim] <= upper_bound)
            secondary_mask = secondary_mask[interval_mask]  # Apply the mask to the filtered candidates
        elif secondary_method == 'max':
            max_val = np.max(phate_data[:, secondary_dim])
            min_val = np.min(phate_data[:, secondary_dim])
            secondary_range = max_val - min_val
            secondary_tolerance = tolerance_ratio * secondary_range
            secondary_mask = (candidates[:, secondary_dim] >= max_val - secondary_tolerance)
        elif secondary_method == 'min':
            max_val = np.max(phate_data[:, secondary_dim])
            min_val = np.min(phate_data[:, secondary_dim])
            secondary_range = max_val - min_val
            secondary_tolerance = tolerance_ratio * secondary_range
            secondary_mask = (candidates[:, secondary_dim] <= min_val + secondary_tolerance)
        else:
            secondary_mask = np.ones(len(candidates), dtype=bool)  # Keep all points if no method specified

        # Filter by tertiary dimension with tolerance within the secondary-filtered set
        combined_indices = indices[secondary_mask]
        filtered_candidates = candidates[secondary_mask]

        if len(filtered_candidates) == 0:
            continue  # Skip if no points remain after filtering

        if i == 0 or i == num_points - 1:
            # Strict max/min for start and end points in the tertiary dimension
            if tertiary_method == 'max':
                tertiary_optimized_idx = combined_indices[np.argmax(filtered_candidates[:, tertiary_dim])]
            elif tertiary_method == 'min':
                tertiary_optimized_idx = combined_indices[np.argmin(filtered_candidates[:, tertiary_dim])]
            else:
                tertiary_optimized_idx = combined_indices[0]  # Default to first if method not recognized
        else:
            # Apply tertiary tolerance for intermediate points to allow smoother transitions
            max_tertiary = np.max(filtered_candidates[:, tertiary_dim])
            min_tertiary = np.min(filtered_candidates[:, tertiary_dim])
            tertiary_range = max_tertiary - min_tertiary
            tertiary_tolerance = tolerance_ratio * tertiary_range

            if tertiary_method == 'max':
                tertiary_mask = (filtered_candidates[:, tertiary_dim] >= max_tertiary - tertiary_tolerance)
            elif tertiary_method == 'min':
                tertiary_mask = (filtered_candidates[:, tertiary_dim] <= min_tertiary + tertiary_tolerance)
            else:
                tertiary_mask = np.ones(len(filtered_candidates), dtype=bool)

            # Select the first point within tolerance bounds for smooth path optimization
            tertiary_optimized_idx = combined_indices[tertiary_mask][0] if np.any(tertiary_mask) else combined_indices[0]

        initial_indices.append(tertiary_optimized_idx)

    # Refine the path using the iterative smoothing function
    refined_path = refine_smooth_path(phate_data, initial_indices, tolerance=path_step_tol)

    return refined_path