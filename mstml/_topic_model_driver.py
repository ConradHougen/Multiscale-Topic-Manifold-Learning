"""
Topic modeling functions and utilities for MSTML.

This module contains functions for topic model processing, author-topic distributions,
interdisciplinarity scoring, and hierarchical topic analysis including HRG functionality.
"""

import numpy as np
import pandas as pd
import pickle
import os
from itertools import islice
from collections import OrderedDict, defaultdict
from enum import Enum, auto
from gensim.models import AuthorTopicModel
from scipy.stats import kendalltau
from .fast_encode_tree import TreeNode
from .dataframe_schema import MainDataSchema


# ============================================================================
# Enums and Classes
# ============================================================================

class AuthEmbedEnum(Enum):
    """Different types of embedding representations for authors"""
    WORD_FREQ = auto()
    AT_DISTN = auto()
    AT_SPARSIFIED_DISTN = auto()
    TERM_RELEVANCE_N_HOT = auto()
    TERM_RELEVANCE_VMASK = auto()


class TermRelevanceTopicType(Enum):
    """Different methods for representing topics filtered by relevance
    N_HOT_ENCODING: each topic is represented by a vector with N nonzero entries, for the N most relevant terms
    VOCAB_MASK: each topic is masked to only the union over T topics of the N most relevant terms for each topic."""
    N_HOT_ENCODING = auto()
    VOCAB_MASK = auto()


class TermRelevanceTopicFilter:
    """This class implements a handler for generating topics, filtered by relevance"""
    def __init__(self,
                 atmodel: AuthorTopicModel,
                 corpus,
                 term_rel_weight=0.6,
                 num_rel_terms=100,
                 trtt=TermRelevanceTopicType.N_HOT_ENCODING):

        self.__atmodel = atmodel
        self.__corpus = corpus
        self.__term_rel_weight = term_rel_weight
        self.__num_rel_terms = num_rel_terms
        self.__trtt = trtt

        word_counts = {}
        for text in corpus:
            for word_id, count in text:
                if word_id in word_counts:
                    word_counts[word_id] += count
                else:
                    word_counts[word_id] = count

        tot_counts = sum(word_counts.values())

        phi_t_w = atmodel.get_topics()
        nterms = atmodel.num_terms
        p_w = np.array([word_counts.get(word_id, 0) / tot_counts for word_id in range(nterms)])

        relevance = term_rel_weight * np.log(phi_t_w + 1e-12) + (1 - term_rel_weight) * np.log((phi_t_w + 1e-12) / p_w)
        self.__filtered_topics = np.zeros_like(phi_t_w)

        if trtt == TermRelevanceTopicType.N_HOT_ENCODING:
            for i in range(relevance.shape[0]):
                top_rel_inds = np.argsort(relevance[i])[-num_rel_terms:]
                self.__filtered_topics[i, top_rel_inds] = phi_t_w[i, top_rel_inds]
        elif trtt == TermRelevanceTopicType.VOCAB_MASK:
            all_relevant_inds = set()
            for i in range(relevance.shape[0]):
                all_relevant_inds.update(np.argsort(relevance[i])[-num_rel_terms:])
            all_relevant_inds = np.array(list(all_relevant_inds))
            for i in range(relevance.shape[0]):
                self.__filtered_topics[i, all_relevant_inds] = phi_t_w[i, all_relevant_inds]
        else:
            raise NotImplementedError

    def get_topics(self):
        return self.__filtered_topics

    def get_trtt(self):
        return self.__trtt


# ============================================================================
# Document and Corpus Processing
# ============================================================================

def preprocess_documents(chunks):
    """Function that extracts 'preprocessed_text' column per doc chunk"""
    
    preprocessed_chunks = []
    for chunk_df in chunks:
        preprocessed_docs = {}
        for doc_id in chunk_df.index:
            preprocessed_docs[doc_id] = chunk_df.loc[doc_id, MainDataSchema.PREPROCESSED_TEXT.colname]
        preprocessed_chunks.append(preprocessed_docs)
    return preprocessed_chunks


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


def count_unique_topics_or_categories(df2count, colname, cat_filter=None):
    """Helper function to count the number of unique topics or categories in the dataframe"""
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


# ============================================================================
# Author Embedding Functions
# ============================================================================

def author_word_freq_embedding(atmodel, auth_id):
    """Function to compute author word-freq embedding, given an author name and fitted author topic model."""
    ntopics = atmodel.num_topics
    nwords = atmodel.num_terms
    topic_wf_distns = atmodel.get_topics()

    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.00)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')

    auth_wf_embedding = np.zeros((nwords,)).astype('float32')
    for tpc in range(ntopics):
        auth_wf_embedding = np.add(auth_wf_embedding, atopics[tpc] * topic_wf_distns[tpc])

    return auth_wf_embedding


def author_topic_distn(atmodel, auth_id):
    """Function to retrieve author distribution over topics, given an author name and fitted author topic model"""
    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.0)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')
    return atopics


def author_topic_sparsified_distn_embedding(atmodel, auth_id, atmodel_alpha, mult=1.01, normalize=False):
    """Function to create sparsified embeddings for an author, based on their author-topic distribution.
    The idea is that the topics which have small contribution to the author-topic distribution are likely
    to be noise. We want to sparsify the vector representations to boost signal.
    atmodel_alpha should be the initial alpha parameter that was passed to AuthorTopicModel.
    mult determines the threshold for sparsification, e.g. zero out any value less than mult * atmodel_alpha
    normalize determines whether to apply normalization after sparsification."""
    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.0)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')
    atopics[atopics < (mult * atmodel_alpha)] = 1e-9
    if normalize:
        if atopics.sum() < 0.01:
            print("Warning: topic distn for auth_id {} may be numerically unstable")
        atopics = atopics / atopics.sum()
    return atopics


def author_topics_by_term_relevance_embedding(atmodel, term_top_filter, auth_id):
    """Function to compute author embeddings based on filtering each topic to the relevant terms
    term_rel_weight is an optional parameter in [0, 1] which is based on the lambda in "Sievert 2014: LDAvis..." """
    ntopics = atmodel.num_topics
    nterms = atmodel.num_terms

    if term_top_filter.get_trtt() != TermRelevanceTopicType.N_HOT_ENCODING:
        raise ValueError("term_top_filter was initialized with wrong trtt parameter")

    filt_topics = term_top_filter.get_topics().astype('float32')
    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.00)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')

    auth_term_rel_embedding = np.zeros((nterms,)).astype('float32')
    for tpc in range(ntopics):
        auth_term_rel_embedding = np.add(auth_term_rel_embedding, atopics[tpc] * filt_topics[tpc])

    # Normalize to probability distribution again
    return auth_term_rel_embedding / auth_term_rel_embedding.sum()


def author_vocab_by_term_relevance_embedding(atmodel, term_top_filter, auth_id):
    """Function to compute author embeddings based on filtering the vocabulary to terms relevant to each topic. Unlike
    the method for filtering topics by term relevance, here we focus on vocab filtering rather than simply summarizing
    each topic's representation by the most relevant terms. Once the vocab is filtered, we use the remaining terms as
    the domain over which all topics are defined.
    term_rel_weight is an optional parameter in [0, 1] which is based on the lambda in "Sievert 2014: LDAvis..." """
    ntopics = atmodel.num_topics
    nterms = atmodel.num_terms

    if term_top_filter.get_trtt() != TermRelevanceTopicType.VOCAB_MASK:
        raise ValueError("term_top_filter was initialized with wrong trtt parameter")

    filt_topics = term_top_filter.get_topics().astype('float32')
    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.00)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')

    auth_term_rel_embedding = np.zeros((nterms,)).astype('float32')
    for tpc in range(ntopics):
        auth_term_rel_embedding = np.add(auth_term_rel_embedding, atopics[tpc] * filt_topics[tpc])

    # Normalize to probability distribution again
    return auth_term_rel_embedding / auth_term_rel_embedding.sum()


# ============================================================================
# HRG and Dendrogram Functions
# ============================================================================

def find_max_min_cut_distance(root):
    """Find the maximum and non-trivial minimum cut distances in the topic dendrogram
    Parameters:
    - root: The root node of the dendrogram (TreeNode).

    Returns:
    - (max_distance, min_distance): A tuple containing the maximum and minimum cut distances."""

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


def truncate_dendrogram(node, cut_distance):
    """Function to cut the topic dendrogram at a specific hellinger cut distance"""
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


# ============================================================================
# Interdisciplinarity and Analysis Functions
# ============================================================================

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


# ============================================================================
# Utility Functions
# ============================================================================

def get_nth_item_from_ordered_dict(ordered_dict, n):
    """Function to get the nth key-value pair"""
    # Convert n to a positive index if it is negative
    if n < 0:
        n += len(ordered_dict)

    # Raise an error if the index is out of range
    if n < 0 or n >= len(ordered_dict):
        raise IndexError("Index out of range.")

    # Retrieve the nth item
    return next(islice(ordered_dict.items(), n, n + 1))


def get_doc_set_from_author_team(authId_to_docs, auth_team):
    """Returns all documents that were authored or co-authored by authors in auth_team"""
    return set.union(set(), *[set(authId_to_docs[auth_i]) for auth_i in auth_team])


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


def compute_author_barycenters(expanded_doc_topic_distns, documents_df, author_column='authors'):
    """
    Compute author barycenters over chunk topics with inverse author count weighting.
    
    Authors are mapped to the barycenter of the documents they authored, where the "mass" 
    for the barycenter calculation depends on the reciprocal of the number of authors per doc.
    
    Args:
        expanded_doc_topic_distns: Dict of {doc_id: topic_distribution_array} over all chunk topics
        documents_df: DataFrame with document metadata including author information
        author_column: Column name containing author information
        
    Returns:
        tuple: (author_topic_barycenters, authId_to_docs, authId2docweights)
            - author_topic_barycenters: Dict of {author_id: barycenter_vector}  
            - authId_to_docs: Dict mapping each author to list of doc_ids
            - authId2docweights: Dict with structured array of doc weights per author
    """
    # Create mapping of authors to documents
    authId_to_docs = defaultdict(list)
    
    for doc_id, row in documents_df.iterrows():
        # Handle different author column formats
        authors = row.get(author_column, [])
        if isinstance(authors, str):
            authors = [authors]  
        elif authors is None:
            continue
            
        for author_id in authors:
            if author_id:  # Skip empty author IDs
                authId_to_docs[author_id].append(doc_id)
    
    # Precompute author counts for faster weighting
    author_counts = documents_df[author_column].apply(
        lambda x: len(x) if isinstance(x, list) else (1 if x else 0)
    ).to_numpy()
    
    # Compute barycenter distribution per author over topics
    author_topic_barycenters = {}
    authId2docweights = {}
    
    # Create the docs_weights data type with float32 for weights
    docs_weights = [('doc_ids', int), ('weights', 'f4')]
    
    for author_id, doc_ids in authId_to_docs.items():
        # Use precomputed author counts to avoid repeated df.loc calls
        weights = np.array([1.0 / author_counts[doc_id] for doc_id in doc_ids], dtype=np.float32)
        authId2docweights[author_id] = np.array(
            [(doc_id, w) for doc_id, w in zip(doc_ids, weights)], dtype=docs_weights
        )
        
        # Collect document topic vectors in advance 
        doc_topic_vectors = []
        valid_weights = []
        
        for i, doc_id in enumerate(doc_ids):
            if doc_id in expanded_doc_topic_distns:
                doc_topic_vectors.append(expanded_doc_topic_distns[doc_id])
                valid_weights.append(weights[i])
        
        if doc_topic_vectors:
            doc_topic_vectors = np.array(doc_topic_vectors, dtype=np.float32)
            valid_weights = np.array(valid_weights, dtype=np.float32)
            
            # Calculate the weighted average for the author
            auth_topic_vector = np.average(doc_topic_vectors, axis=0, weights=valid_weights)
            author_topic_barycenters[author_id] = auth_topic_vector.astype(np.float32)
    
    return author_topic_barycenters, dict(authId_to_docs), authId2docweights