"""
Embedding and manifold learning functions for MSTML.

This module contains functions for dimensional reduction, PHATE embeddings,
path selection, and visualization utilities for topic manifolds.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import mplcursors
import pandas as pd
from matplotlib import cm
from wordcloud import WordCloud
from scipy.cluster.hierarchy import fcluster
from ._math_driver import mstml_term_relevance_stable
from ._file_driver import get_exp_dir, get_data_int_dir, read_pickle


# ============================================================================
# PHATE and Path Selection Functions
# ============================================================================

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


# ============================================================================
# Visualization Functions
# ============================================================================

def generate_colors(num_colors, cset=cc.glasbey):
    """
    Generate a list of perceptually uniform, categorical colors using cset colormap.

    Parameters:
    num_colors: int
        The number of colors to generate.

    Returns:
    list of str
        A list of hexadecimal color codes.
    """
    # Use the Glasbey colormap from colorcet
    glasbey_colors = cset
    if num_colors > len(glasbey_colors):
        raise ValueError(f"Cannot generate more than {len(glasbey_colors)} colors.")
    return glasbey_colors[:num_colors]


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
    cmap = plt.get_cmap('rainbow')
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
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(cluster_labels), vmax=max(cluster_labels)))
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


# ============================================================================
# Word Cloud Functions
# ============================================================================

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


def create_and_save_wordcloud(dset, dsub, max_words=200):
    """Function to create wordcloud of entire data, given dataset and subset strings"""
    
    data_fig_dir = get_exp_dir(dset, dsub)
    int_dir = get_data_int_dir(dset, dsub)

    # Load data words
    try:
        data_words = read_pickle(os.path.join(int_dir, "data_words.pkl"))
    except:
        print(f"Fatal error when loading data_words from {int_dir}")
        print(f"create_and_save_wordcloud exiting without creating wordcloud")
        return

    flat_data = list(np.concatenate(data_words).flat)
    long_string = ','.join(flat_data)

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=max_words, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    plt.figure(facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")

    if not os.path.exists(data_fig_dir):
        os.makedirs(data_fig_dir)

    plt.savefig(data_fig_dir + dset + '_' + dsub + '_complete_wc.pdf', dpi=1200, bbox_inches="tight")
    plt.show()


# ============================================================================
# Utility Functions for Embeddings
# ============================================================================

def convert_slc_tpc_idx_to_yr_and_tpc_idx(slc_tpc_idx, ntop):
    """Function to convert a slice topic index into separate year and topic indices (based on number of topics)"""
    yr_idx = slc_tpc_idx // ntop
    tpc_idx = slc_tpc_idx - (yr_idx * ntop)
    return yr_idx, tpc_idx


def generate_slc_tpc_wordclouds(nslicetopics, ntopics, exp_dir):
    """Function to generate wordclouds from a set of slice topics, and save them in a returned dictionary for re-use."""
    
    wc_lib = {}

    # Load the slice topic dictionary from directory (indexed by year, topic)
    WordsInTopicsAll = read_pickle(os.path.join(exp_dir, 'WordsInTopicsAll.pkl'))

    for slc_tpc in range(nslicetopics):
        # convert to year and topic indices to index into WordsInTopicsAll to retrieve word-freq distns
        yr_idx, tpc_idx = convert_slc_tpc_idx_to_yr_and_tpc_idx(slc_tpc, ntopics)

        distn = {}
        for word, freq in WordsInTopicsAll[yr_idx][tpc_idx].values:
            distn[word] = freq

        wordcloud = WordCloud(max_words=40, width=1000, height=600)
        wc_lib[slc_tpc] = wordcloud.generate_from_frequencies(frequencies=distn)

        wordcloud.to_file(os.path.join(exp_dir, f"slc_tpc_wordcloud{slc_tpc}.png"))

    return wc_lib


def generate_tpc_wordclouds(X, int_dir):
    """Function to generate a dictionary of wordclouds from a matrix of topic word freq vectors."""
    
    wc_lib = {}

    # Get list of words in vocab from dictionary
    with open(int_dir + 'id2word.pkl', 'rb') as f:
        id2word = pickle.load(f)
    wordlist = list(id2word.values())

    # Generate wordcloud for each vector in X
    for vec_idx in range(X.shape[0]):

        distn = {}
        widx = 0
        for word in wordlist:
            distn[word] = X[vec_idx, widx]
            widx += 1

        wordcloud = WordCloud(max_words=40, width=1000, height=600)
        wc_lib[vec_idx] = wordcloud.generate_from_frequencies(frequencies=distn)

    return wc_lib