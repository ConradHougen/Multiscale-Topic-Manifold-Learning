"""
Mathematical functions and utilities for MSTML.

This module contains distance metrics, clustering functions, statistical measures,
and other mathematical utilities used throughout the MSTML framework.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import fcluster
from scipy.special import rel_entr
from sklearn.cluster import AgglomerativeClustering
from math import log
from collections import Counter
from scipy.sparse import csr_matrix


# ============================================================================
# Distance Metrics and Similarity Functions
# ============================================================================

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


def hellinger(p, q):
    """
    Compute Hellinger distance between two probability distributions.
    
    Args:
        p, q: Probability distributions (numpy arrays)
        
    Returns:
        Hellinger distance (float)
    """
    # Ensure inputs are numpy arrays and normalized
    p = np.array(p)
    q = np.array(q)
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Compute Hellinger distance
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    return np.sqrt(0.5 * np.sum((sqrt_p - sqrt_q) ** 2))


def hellinger_sim(p, q, alpha=10):
    """Function to compute hellinger similarity between multinomial distributions, using form sim(x) = b/(1+c*x)"""
    d = hellinger(p, q)
    sim = 1 / (1 + (alpha * d))
    return sim


def hellinger_similarity(p, q):
    """
    Compute Hellinger similarity (1 - Hellinger distance).
    
    Args:
        p, q: Probability distributions (numpy arrays)
        
    Returns:
        Hellinger similarity (float)
    """
    return 1.0 - hellinger(p, q)


def euclidean(p, q):
    """Function to calculate the Euclidean distance between two vectors."""
    return np.linalg.norm(p - q)


def kl_divergence(p, q):
    """Function to calculate the Kullback-Leibler Divergence between two distributions."""
    # Ensure the distributions sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Only consider the elements where P > 0 and Q > 0 for calculation
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def js_divergence(p, q):
    """Function to calculate the Jensen-Shannon Divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions using rel_entr.

    Args:
        p, q (array-like): Probability distributions

    Returns:
        float: Jensen-Shannon divergence
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if not np.any(p):
        raise ValueError("Input p must have nonzero sum.")
    if not np.any(q):
        raise ValueError("Input q must have nonzero sum.")

    # Normalize
    p /= np.sum(p)
    q /= np.sum(q)

    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))


def min_weighted_tv(p, q):
    """Function to compute element-wise minimum-weighted total variation distance"""
    w_abs_diff = np.abs(p - q) * np.minimum(p, q)
    return np.sum(w_abs_diff)


def mean_weighted_tv(p, q):
    """Function to compute element-wise mean-valued total variation distance"""
    means = (p + q) / 2
    w_abs_diff = np.abs(p - q) * means
    return np.sum(w_abs_diff)


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a, b: Input vectors (numpy arrays)
        
    Returns:
        Cosine similarity (float)
    """
    a = np.array(a)
    b = np.array(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def normalize_vector(vector):
    """Normalize a vector to unit length."""
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


# ============================================================================
# Statistical Measures
# ============================================================================

def entropy(p):
    """Function to compute entropy of a given probability distribution"""
    p = np.array(p)  # Ensure input is a numpy array
    # Filter out zero probabilities to avoid log(0)
    p_nz = p[p > 0]
    return -np.sum(p_nz * np.log2(p_nz))


def gini_coefficient(p):
    """Function to calculate Gini coefficient"""
    p = p[p != 0]  # Remove zero entries
    n = len(p)
    if n == 0:
        return 0
    sorted_p = np.sort(p)
    tmp_idx = np.arange(1, n + 1)
    return (np.sum((2 * tmp_idx - n - 1) * sorted_p)) / (n * np.sum(sorted_p))


def max_weighted_mean(vecs: list[np.ndarray]):
    """Function to compute the max-weighted mean of a list of vectors of the same length"""
    mwmean = np.zeros_like(vecs[0])
    for vec in vecs:
        mwmean += vec * vec.max()
    mwmean /= mwmean.sum()
    return mwmean


def entropy_of_max_weighted_mean(vecs: list[np.ndarray]):
    """Function to compute entropy of max-weighted mean of multiple vectors, passed as
    a list of numpy arrays of the same length"""
    return entropy(max_weighted_mean(vecs))


# ============================================================================
# Diffusion and Graph Processing
# ============================================================================

def build_diffusion_matrix(indices, distances, num_topics, diffusion_knnk):
    """Build sparse diffusion matrix from KNN search results"""
    
    row_indices = []
    col_indices = []  
    weights = []
    
    for i in range(num_topics):
        # Get neighbors (skip first index which is self)
        max_neighbors = min(diffusion_knnk, indices.shape[1] - 1)
        neighbors = indices[i, 1:max_neighbors+1]
        dists = distances[i, 1:max_neighbors+1]
        
        if len(dists) > 0 and dists.sum() > 0:
            # Convert distances to similarity weights
            norm_weights = 1 - (dists / dists.sum())
            
            for neighbor, weight in zip(neighbors, norm_weights):
                row_indices.append(i)
                col_indices.append(neighbor)
                weights.append(weight)
    
    return csr_matrix((weights, (row_indices, col_indices)), shape=(num_topics, num_topics))


def diffuse_distribution_matrix(diffusion_matrix, known_dist, num_iterations=1, diffusion_rate=0.7):
    """Matrix-based diffusion using sparse matrix operations"""
    distribution = np.copy(known_dist).astype(np.float32)
    mask = (known_dist == 0)
    
    for _ in range(num_iterations):
        if mask.any():
            # Apply diffusion only to nodes without known distributions
            diffused_values = diffusion_rate * (diffusion_matrix @ distribution)
            distribution[mask] = diffused_values[mask]
        
        # Preserve known distributions
        distribution[~mask] = known_dist[~mask]
    
    # Normalize the distribution to sum to 1
    total = distribution.sum()
    if total > 0:
        distribution /= total
    
    return distribution


def diffuse_distribution(graph, known_dist, num_iterations=1, diffusion_rate=0.7):
    """Function to perform diffusion for the author topic distributions over the chunk topic kNN graph"""
    num_topics = len(known_dist)
    distribution = np.copy(known_dist).astype(np.float32)
    buffer = np.zeros(num_topics, dtype=np.float32)

    for _ in range(num_iterations):
        buffer.fill(0)

        for node in range(num_topics):
            if known_dist[node] > 0:
                buffer[node] += distribution[node]
            else:
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    weights = np.array([graph[node][neighbor]['weight'] for neighbor in neighbors], dtype=np.float32)
                    norm_weights = 1 - (weights / weights.sum())
                    neighbors_dist = np.array([distribution[neighbor] for neighbor in neighbors], dtype=np.float32)
                    buffer[node] += diffusion_rate * np.dot(norm_weights, neighbors_dist)

        distribution[:] = buffer

    # Normalize the distribution to sum to 1
    distribution /= distribution.sum(dtype=np.float32)
    return distribution


def precompute_weights(graph, num_topics):
    """Function to precompute weights for each node in the graph"""
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


def compute_new_distribution(node, known_dist, distribution, weights_dict, diffusion_rate):
    """Function to compute new distribution for a single node (can be parallelized)"""
    if known_dist[node] > 0:
        return distribution[node]  # No diffusion for nodes with known distribution
    else:
        neighbors, weights = weights_dict[node]
        if neighbors:
            neighbors_dist = np.array([distribution[neighbor] for neighbor in neighbors])
            return diffusion_rate * np.sum(weights * neighbors_dist)
        return 0.0  # No diffusion if no neighbors


def diffuse_distribution_parallel(graph, known_dist, num_iterations=1, diffusion_rate=0.7, num_workers=4):
    """Optimized and parallelized function to perform diffusion"""
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


# ============================================================================
# Clustering Functions
# ============================================================================

def hier_cluster(distance, n_clusters):
    """Hierarchical (agglomerative) clustering by number of clusters and pairwise distance matrix"""
    cl = AgglomerativeClustering(n_clusters=n_clusters,
                                 affinity='precomputed',
                                 linkage='average')
    cl.fit(distance)
    return (cl.labels_)


def assign_clusters(distance, n_clusters):
    labs = hier_cluster(distance=distance, n_clusters=n_clusters)
    uniq = [list(labs).index(x) for x in set(labs)]
    uniq.sort()

    topic_inds = []
    for elem in uniq:
        topic_inds.append([i for i in range(len(labs)) if labs[i] == labs[elem]])

    cluster = ['Other' for _ in range(len(labs))]
    topic_cnt = 1
    for Topic in topic_inds:
        for i in range(len(labs)):
            if i in Topic:
                cluster[i] = 'Topic' + str(topic_cnt)
        topic_cnt += 1

    return cluster


# ============================================================================
# Utility Functions
# ============================================================================

def rescale_parameter(x, min_cut_distance, max_cut_distance):
    """Rescale a parameter from the interval [0, 1] to [min_cut_distance, max_cut_distance]."""
    if not (0 <= x <= 1):
        raise ValueError("Parameter x must be in the range [0, 1]")

    # Linear transformation to rescale x to the new range
    rescaled_value = min_cut_distance + x * (max_cut_distance - min_cut_distance)

    return rescaled_value


def intersection(lst1, lst2):
    """Function to perform set intersection on two lists"""
    return list(set(lst1) & set(lst2))


def get_sim_from_piter(piter):
    """Helper function for using networkx index generators"""
    # TODO: Fix ZeroDivisionError
    try:
        piter = list(piter)
        sim = piter[0][2]
    except ZeroDivisionError:
        sim = 0.00
    return sim


def pairwise_alignment_score(chunk1, chunk2, metric='cosine'):
    """
    Compute the average best-match similarity between topics in two chunks.
    Each topic is a vector (e.g. probability distribution or embedding).
    """
    sim_func = {
        'cosine': lambda u, v: 1 - cosine(u, v),
        'hellinger': lambda u, v: 1 - hellinger(u, v)
    }[metric]

    scores = []
    for vec1 in chunk1:
        best_sim = max(sim_func(vec1, vec2) for vec2 in chunk2)
        scores.append(best_sim)
    return np.mean(scores)


def compute_alignment_scores(topic_vectors_per_chunk, metric='cosine'):
    """
    Compute alignment scores across all consecutive time chunks.
    Returns a list of alignment scores for each chunk pair.
    """
    all_scores = []
    for t in range(len(topic_vectors_per_chunk) - 1):
        chunk1 = topic_vectors_per_chunk[t]
        chunk2 = topic_vectors_per_chunk[t + 1]
        score = pairwise_alignment_score(chunk1, chunk2, metric=metric)
        all_scores.append(score)
    return all_scores


def rerank_topic_terms_by_relevance(all_topics, dictionary, tokenized_texts, lambda_=0.6, top_n=10):
    """
    Rerank terms for each topic in all_topics based on term relevance.
    """
    # Compute p(w): empirical probability of word in corpus
    word_counts = Counter(word for doc in tokenized_texts for word in doc)
    total_tokens = sum(word_counts.values())
    pw = {w: count / total_tokens for w, count in word_counts.items()}

    reranked_topics = []

    for topic in all_topics:
        relevance_scores = []
        for word, prob in topic:
            if word not in dictionary.token2id:
                continue
            p_w = pw.get(word, 1e-12)
            p_w_given_t = prob
            # Relevance = lambda * log(p(w|t)) + (1 - lambda) * log(p(w|t)/p(w))
            relevance = lambda_ * log(p_w_given_t + 1e-12) + (1 - lambda_) * (log(p_w_given_t + 1e-12) - log(p_w))
            relevance_scores.append((word, relevance))
        
        reranked = sorted(relevance_scores, key=lambda x: x[1], reverse=True)
        reranked_topics.append([word for word, _ in reranked[:top_n]])

    return reranked_topics
