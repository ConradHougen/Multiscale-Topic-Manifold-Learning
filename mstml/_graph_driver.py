"""
Graph and network analysis functions for MSTML.

This module contains functions for co-author network construction, community detection,
link prediction, and network analysis utilities.
"""

import os
import glob
import pickle
import random
import networkx as nx
import hypernetx as hnx
import pandas as pd
import numpy as np
import community as community_louvain  # For Louvain method
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pathlib import Path
from ._file_driver import get_data_int_dir


# ============================================================================
# Network Construction and I/O
# ============================================================================

def get_coauthors(G, authorid):
    """Function to retrieve all coauthors of a given author (by author ID) and coauthorship network"""
    return [n for n in G.neighbors((str(authorid)))]


def compose_coauthorship_network(dset, dsub, yr_start, yr_end, overwrite=False, d_lims=None, label=''):
    """Function to compose networkx multigraph object by multiple networkx
    multigraphs from yr_start to yr_end."""
    nx_out_fname = get_fname_for_graph_composition(dset, dsub, yr_start, yr_end, label)

    Z = nx.MultiGraph()

    if not overwrite:
        if os.path.exists(nx_out_fname):
            print("Warning, using existing version of file: {}".format(nx_out_fname))
            Z = nx.read_graphml(nx_out_fname, node_type=int, force_multigraph=True)
            return Z

    for year in range(yr_start, yr_end + 1):
        fname = get_fname_for_single_yr_graph(dset, dsub, year, label)
        X = nx.read_graphml(fname, node_type=int, force_multigraph=True)
        Z = nx.compose(Z, X)

    # If degree limits were passed, use to filter the nodes. d_lims should be passed as a 2-tuple of (min, max)
    if d_lims:
        if not is_n_tuple(d_lims, 2):
            raise Exception("d_lims should be a 2-tuple of (min, max) degrees permitted")
        else:
            nodes_to_remove = [node for node, degree in dict(Z.degree()).items() if degree < d_lims[0]]
            nodes_to_remove.extend([node for node, degree in dict(Z.degree()).items() if degree > d_lims[1]])
            print("Removing {} nodes for degree limits ({}, {})...".format(len(nodes_to_remove), d_lims[0], d_lims[1]))
            Z.remove_nodes_from(nodes_to_remove)

    nx.write_graphml(Z, nx_out_fname)

    print("Returning composed network from {} to {}".format(yr_start, yr_end))
    print("Warning: The returned network is a MultiGraph. Cast it as a Graph if needed.")
    return Z


def nx_graph_to_pairs_file(nxg_in, exp_dir, pairs_fname):
    """Function to take networkx co-author network graph and convert to a .pairs file in the
    appropriate experiment directory based on the dataset"""
    # TODO: Check if HRG model handles multigraphs. For now, reduce to simple graph
    nxg = nx.Graph(nxg_in)
    # Now, write the edge list as a .pairs file for HRG processing
    # Ensure the directory exists, or create it if it doesn't
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    # Construct the full file path
    out_path = os.path.join(exp_dir, pairs_fname)
    print("Writing networkx graph to file: {}".format(out_path))
    nx.write_edgelist(nxg, out_path, delimiter="\t", data=False)


def gen_nx_multigraph_from_dataframe(src_df, dset, dsub, overwrite=False, label=''):
    """Function to create networkx multigraph graphml file using dataframe with an
    appropriately formatted column of author IDs"""
    if src_df.empty:
        raise ValueError("src_df is empty")

    for idx, row in src_df.iterrows():
        raise NotImplementedError


def gen_nx_multigraphs_per_year(dataset, data_subset, overwrite=False, label=''):
    """Function to create networkx multigraph graphml file for each year of documents in
    a given pandas dataframe (main_df.pkl from data/dataset/intermediate/data_subset/)"""
    # Load the source dataframe
    df_fname = get_data_int_dir(dataset, data_subset) + '/main_df.pkl'
    try:
        with open(df_fname, 'rb') as f:
            src_df = pickle.load(f)
    except:
        print("Error: couldn't open source dataframe at {}".format(df_fname))
        print("Exiting...")
        return

    # Ensure that the date column is in datetime format
    src_df['date'] = pd.to_datetime(src_df['date'])

    dt1 = src_df.loc[0, 'date']
    dt2 = src_df['date'].iloc[-1]
    print("Generating NetworkX multigraphs for data in years [{},  {}]".format(dt1.year, dt2.year))

    out_dir = '../data/' + dataset + '/networks/' + data_subset + '/'
    os.makedirs(out_dir, exist_ok=True)

    # Loop through years of dataframe and create networks
    for year in range(dt1.year, dt2.year + 1):
        include = src_df[src_df['date'].dt.year == year]

        if not include.empty:
            nx_out_fname = get_fname_for_single_yr_graph(dataset, data_subset, year, label)
            # If the file already exists, may skip if overwrite == False
            if not overwrite:
                if os.path.exists(nx_out_fname):
                    print("Skip since {} exists".format(nx_out_fname))
                    continue

            G = nx.MultiGraph()

            for index, row in include.iterrows():
                coauthors = str(row['authorID']).split(',')
                coauthors = coauthors[:-1]

                G.add_nodes_from(coauthors)

                # Get all possible pairs from article co-authors and add as links in network
                auth_pairs = [(a, b) for idx, a in enumerate(coauthors) for b in coauthors[idx + 1:]]

                # TODO: Fix EID label handling. Don't use for now
                # if label == 'EID':
                # Add edge IDs as a concatenation of node IDs
                #    edges_to_add = [tuple(sorted(ap)) + (int(str(tuple(sorted(ap))[0])+str(tuple(sorted(ap))[1])),) for ap in auth_pairs]
                #    G.add_edges_from(edges_to_add)
                # else:
                #    G.add_edges_from(auth_pairs)

                G.add_edges_from(auth_pairs)

            # TODO: Figure out why selfloops are occurring (author disambiguation step?)
            G.remove_edges_from(nx.selfloop_edges(G))

            # print("Saving graphml file: {}".format(nx_out_fname))
            nx.write_graphml(G, nx_out_fname)


# ============================================================================
# Hypergraph Functions
# ============================================================================

def flatten_df_column_with_str_list(df, list_column, new_column, retained_cols):
    """Helper function to flatten dataframe according to a column of lists for hypergraph constructor"""
    df[list_column] = df[list_column].map(lambda x: str.split(x, ',')[:-1])
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    cols_to_retain = ([idx for idx, col in enumerate(df.columns) if col in retained_cols])
    expanded_df = df.iloc[destination_rows, cols_to_retain].copy()
    expanded_df[new_column] = ([item for items in df[list_column] for item in items])
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df


def gen_hypergraph(dset, dsub, yr_start=None, yr_end=None):
    """Function to generate and return hypergraph using authors column in doc dataframe"""
    # Load the source dataframe
    df_fname = '../data/' + dset + '/intermediate/' + dsub + '/main_df.pkl'
    try:
        with open(df_fname, 'rb') as f:
            src_df = pickle.load(f)
    except:
        print("Error: couldn't open source dataframe at {}".format(df_fname))
        print("Exiting...")
        return

    # Ensure that the date column is in datetime format
    src_df['date'] = pd.to_datetime(src_df['date'])

    # Filter dataframe on years from yr_start to yr_end
    if yr_start is not None:
        if yr_end is not None:
            src_df = src_df.loc[((src_df['date'] >= yr_start) & (src_df['date'] <= yr_end))]
        else:
            src_df = src_df.loc[src_df['date'] >= yr_start]
    elif yr_end is not None:
        src_df = src_df.loc[src_df['date'] <= yr_end]

    try:
        dt1 = src_df['date'].iloc[0]
        dt2 = src_df['date'].iloc[-1]
        print("Generating HyperNetX hypergraph for data in years [{},  {}]".format(dt1.year, dt2.year))
    except:
        print("Error: Couldn't generate hypergraph from potentially empty dataframe")
        return

    out_dir = '../data/' + dset + '/networks/' + dsub + '/'
    os.makedirs(out_dir, exist_ok=True)

    # Create dataframe format for hypergraph constructor
    print("Generating hypergraph...")
    h_constructor_df = flatten_df_column_with_str_list(src_df, 'authorID', 'single_authIDs', ['id'])

    # Save and return hypergraph
    H = hnx.Hypergraph(h_constructor_df, edge_col='id', node_col='single_authIDs')

    if yr_start is not None and yr_end is not None:
        hfn = out_dir + 'h_incidence_' + dset + '_' + dsub + '_' + str(yr_start) + '_' + str(yr_end) + '.pkl'
    else:
        hfn = out_dir + 'h_incidence_' + dset + '_' + dsub + '.pkl'

    print('Saving incidence dictionary to file: ' + hfn)
    with open(hfn, 'wb') as f:
        pickle.dump(H.incidence_dict, f)

    return H


# ============================================================================
# Community Detection and Analysis
# ============================================================================

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


def fixed_source_label_propagation(G, fixed_nodes):
    """Function to perform label propagation where the set of fixed_nodes is immutable.
    Nodes in G should have a 'label' attribute"""
    # Initialize all node labels
    labels = {node: data['label'] for node, data in G.nodes(data=True)}

    # Set to keep track of nodes which should not change
    fixed_set = set(fixed_nodes)

    # Container to hold the nodes to be updated
    to_be_updated = set(G.nodes()) - fixed_set

    # Continue until all nodes keep their labels unchanged after an iteration
    while to_be_updated:
        # List to collect changes for this iteration
        next_labels = {}

        for node in to_be_updated:
            if node in fixed_set:
                continue

            # Count all labels of the neighborhood, including its own current label
            neighbor_labels = [labels[neighbor] for neighbor in G[node] if labels[neighbor] is not None]
            if neighbor_labels:
                # Assign the label that is most frequent in neighbors
                new_label = Counter(neighbor_labels).most_common(1)[0][0]
                if labels[node] != new_label:
                    next_labels[node] = new_label

        # Update labels and determine which nodes to check in the next round
        to_be_updated = set()
        for node, new_label in next_labels.items():
            labels[node] = new_label
            to_be_updated.update([node] + list(G[node]))

        # Remove fixed nodes from the update set
        to_be_updated -= fixed_set

    # Apply final labels back to the graph
    for node, label in labels.items():
        G.nodes[node]['label'] = label


# ============================================================================
# Network Analysis and Metrics
# ============================================================================

def find_kth_largest_connected_component(nxg, k=0):
    """Takes a networkx graph and returns the kth largest connected component"""
    if k == 0:
        # Return the largest connected component
        return max(nx.connected_components(nxg), key=len)
    else:
        # Find all connected components, sorted by size
        components = sorted(list(nx.connected_components(nxg)), key=len, reverse=True)
        # Return the kth largest component
        return components[k]


def topic_space_dist_vs_path_len_for_non_overlapping_max_paths(nxg, auth_vecs, dmetric, nsamples=10000):
    """Samples node pairs up to nsamples from a networkx graph, where the node pairs come from the greedily sampled
    maximum length path in the network, with each path removed after sampling to avoid edge-sample repeats."""
    topic_distances = []
    path_len = []

    sampled_pairs = []
    remaining_graph = nxg.copy()

    while len(sampled_pairs) < nsamples and remaining_graph.number_of_edges() > 0:
        # Select a random source node from the remaining graph
        source = random.choice(list(remaining_graph.nodes))

        # Compute the node with the furthest distance from the source
        lengths = nx.single_source_shortest_path_length(remaining_graph, source)
        max_length = max(lengths.values())
        target = random.choice([node for node, length in lengths.items() if length == max_length])

        # Get the shortest path between the source and the target
        path = nx.shortest_path(remaining_graph, source, target)

        # Generate all node pairs from the source node along this path
        path_pairs = [(source, node) for node in path if node != source]

        # Add as many pairs as possible without exceeding num_samples
        for dos in range(len(path_pairs)):
            pair = path_pairs[dos]
            if len(sampled_pairs) >= nsamples:
                break
            sampled_pairs.append(pair)
            # Add 1 since we exclude the (source, source) pair from path_pairs
            path_len.append(dos + 1)
            topic_distances.append(dmetric(auth_vecs[pair[0]], auth_vecs[pair[1]]))

        # Remove the edges used in this path from the graph to prevent overlap
        edges_in_path = list(zip(path, path[1:]))
        remaining_graph.remove_edges_from(edges_in_path)

    return np.array(topic_distances), np.array(path_len), sampled_pairs


def topic_space_dist_vs_path_length(nxg, auth_vecs, auth_ids, log_file, dmetric, nsamples=10000):
    """Function to get degrees of separation (path length) in co-author network vs. topic distance"""
    from ._file_driver import atoms_print  # Import here to avoid circular imports
    
    topic_distances = np.zeros(nsamples)
    path_len = np.zeros(nsamples)

    # Until we have enough total node pairs, randomly select a source node,
    # then, choose a path to a target node at max distance, and sample all pairs along the same path
    nsp = 0
    sampled_pairs = []
    max_attempts = nsamples * 10
    attempts = 0
    while nsp < nsamples:
        if attempts >= max_attempts:
            atoms_print("Early exit: Attempts at limit:{}".format(attempts), log_file)
            break

        # Start by randomly sampling a source node
        src_node = random.choice(auth_ids)
        # Get the path lengths starting from the source
        path_lengths = nx.single_source_shortest_path_length(nxg, src_node)
        max_length = max(path_lengths.values())

        # Randomly sample one maximum length path

        # For each possible degree of separation, sample a connected node
        for dos in range(1, max_length + 1):
            tgt_nodes = [node for node, length in path_lengths.items() if length == dos]

            # If there are no nodes at dos degrees of separation, skip
            if not tgt_nodes:
                continue

            # Select one such target node at random
            tgt_node = random.choice(tgt_nodes)

            # Add the pair to the set of sampled pairs
            sampled_pairs.append((src_node, tgt_node))
            # Store the degrees of separation and computed topic space distance
            topic_distances[nsp] = dmetric(auth_vecs[src_node], auth_vecs[tgt_node])
            path_len[nsp] = dos
            nsp += 1

            # If we have sampled enough pairs, break out of loop
            if nsp >= nsamples:
                break

        attempts += 1

    return topic_distances, path_len, sampled_pairs


# ============================================================================
# Link Prediction and Evaluation
# ============================================================================

def lp_expA_score(pred_net, train_net, test_net, name, out_dir):
    """Function to compute and store F1 score stats from link prediction results"""
    score = {}

    fp_net = nx.difference(pred_net, test_net)
    fn_net = nx.difference(test_net, pred_net)
    notrain_net = nx.difference(pred_net, train_net)
    tp_net = nx.difference(notrain_net, fp_net)

    nx.write_graphml(fp_net, out_dir + name + '_fp_net.graphml')
    nx.write_graphml(fn_net, out_dir + name + '_fn_net.graphml')
    nx.write_graphml(notrain_net, out_dir + name + '_notrain_net.graphml')
    nx.write_graphml(tp_net, out_dir + name + '_tp_net.graphml')

    tp = tp_net.size()
    fp = fp_net.size()
    fn = fn_net.size()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    score['tp'] = tp
    score['fp'] = fp
    score['fn'] = fn
    score['precision'] = precision
    score['recall'] = recall
    score['f1'] = f1

    return score


# ============================================================================
# Graph File Format Conversion
# ============================================================================

def graphml_viz_convert_file(dset, dsub, filename, overwrite=False, specific_directory=None):
    """Function to convert NX graphml file into a visualizable graphml file with unique edge ids"""
    from ._file_driver import get_net_src_dir  # Import here to avoid circular imports
    
    if specific_directory != None:
        net_dir = specific_directory
    else:
        net_dir = get_net_src_dir(dset, dsub)

    fpath = net_dir + filename
    fname = os.path.basename(fpath)
    print(fname)
    # Create an output file of the same name, prefixed by 'viz_'
    foutname = os.path.join(net_dir, 'viz_'+fname)

    if not overwrite:
        if os.path.exists(foutname):
            print("Early exit since {} exists".format(fname))
            return

    fin = open(fpath, 'r')
    fout = open(foutname, 'w')

    # Read line by line of input file, and process lines with edge data
    while True:
        line = fin.readline()

        # Stop at end of input file
        if not line:
            break

        import re
        source = re.search(r'(?<=source=").{7}', line)
        target = re.search(r'(?<=target=").{7}', line)
        if source:
            src = source.group(0)
            tar = target.group(0)
            srctar = [src, tar]
            srctar.sort(key=int)
            line = r'    <edge id="' + srctar[0] + srctar[1] + '" source="' + src + '" target="' + tar + '" />\n'
            fout.writelines([line])
        else:
            fout.writelines([line])

    fin.close()
    fout.close()


def graphml_viz_convert(dset, dsub, overwrite=False, specific_directory=None):
    """Function to convert NX graphml files into visualizable graphml files with unique edge ids"""
    from ._file_driver import get_net_src_dir  # Import here to avoid circular imports
    
    if specific_directory != None:
        net_dir = specific_directory
    else:
        net_dir = get_net_src_dir(dset, dsub)

    pattern = str(net_dir) + '*.graphml'
    exclude = str(net_dir) + 'viz_*.graphml'

    files = glob.glob(pattern)
    fexc = set(glob.glob(exclude))

    files = [x for x in files if x not in fexc]

    # For each .graphml file not in the exclusion list, change edge ids
    for f in files:
        fname = os.path.basename(f)
        print(fname)
        # Create an output file of the same name, prefixed by 'viz_'
        foutname = str(net_dir) + 'viz_' + fname

        if os.path.exists(foutname):
            if not overwrite:
                print("Skip, since {} exists".format(fname))
                continue

        fin = open(f, 'r')
        fout = open(foutname, 'w')

        # Read line by line of input file, and process lines with edge data
        while True:
            line = fin.readline()

            # Stop at end of input file
            if not line:
                break

            import re
            source = re.search(r'(?<=source=").{7}', line)
            target = re.search(r'(?<=target=").{7}', line)
            if source:
                src = source.group(0)
                tar = target.group(0)
                srctar = [src, tar]
                srctar.sort(key=int)
                line = r'    <edge id="' + srctar[0] + srctar[1] + '" source="' + src + '" target="' + tar + '" />\n'
                fout.writelines([line])
            else:
                fout.writelines([line])

        fin.close()
        fout.close()


def convert_viz_graphml_to_pairs_file(dset, dsub, filename, specific_directory=None):
    """Function to convert viz_ .graphml file into a .pairs file (edge list) for HRG model"""
    from ._file_driver import get_net_src_dir  # Import here to avoid circular imports
    
    if specific_directory != None:
        in_dir = specific_directory
    else:
        in_dir = get_net_src_dir(dset, dsub)

    fpath = in_dir + filename
    fin = open(fpath, 'r')
    fname = fpath.split('/')[-1]
    # Input check for viz_ prefix (otherwise likely incorrect input file format)
    if fname.find("viz_") != 0:
        raise Exception("Abort function due to likely improper input file format")

    print("Converting to .pairs file: " + fname)
    # Create an output file of the same name with .pairs extension
    fdir = "/".join(fpath.split("/")[0:-1]) + "/"
    ext_idx = fname.rfind(".")
    foutname = fname[0:ext_idx] + ".pairs"
    fout = open(fdir + foutname, 'w')

    # Read line by line of input file, and process lines with unique edge ids
    # Note that this converts multiple edges between node pairs into single edges.
    uniq_edge_id_set = set()
    while True:
        line = fin.readline()

        # Stop at end of input file
        if not line:
            break

        import re
        source = re.search(r'(?<=source=").{7}', line)
        target = re.search(r'(?<=target=").{7}', line)
        if source:
            src = source.group(0)
            tar = target.group(0)
            srctar = [src, tar]
            srctar.sort(key=int)
            if srctar[0] + srctar[1] in uniq_edge_id_set:
                continue
            else:
                uniq_edge_id_set.add(srctar[0] + srctar[1])
                line = "{}\t{}\n".format(srctar[0], srctar[1])
                fout.write(line)

    fin.close()
    fout.close()


def atoms_read_graphml_to_netx(dset, dsub, filename, specific_directory=None):
    """Wrapper for reading graphml file to networkx object"""
    from ._file_driver import get_net_src_dir  # Import here to avoid circular imports
    
    if specific_directory != None:
        in_dir = specific_directory
    else:
        in_dir = get_net_src_dir(dset, dsub)

    fpath = in_dir + filename
    nxg = nx.read_graphml(fpath, node_type=int)
    return nxg


# ============================================================================
# Helper Functions
# ============================================================================

def get_fname_for_single_yr_graph(dset, dsub, yr, label=''):
    """Function to generate file name for single year graph"""
    net_dir = '../data/' + dset + '/networks/' + dsub + '/'
    fname = net_dir + dset + '_' + dsub + '_' + str(yr) + '_' + label + '.graphml'
    return fname


def get_fname_for_graph_composition(dset, dsub, yr_start, yr_end, label=''):
    """Function to generate file name for graph composition"""
    net_dir = '../data/' + dset + '/networks/' + dsub + '/'
    fname = net_dir + 'composed_' + dset + '_' + dsub + '_' \
            + str(yr_start) + '-' + str(yr_end) + '_' + label + '.graphml'
    return fname


def is_n_tuple(test_var, n: int):
    """Helper function to confirm if a variable is a tuple of the expected length"""
    return isinstance(test_var, tuple) and len(test_var) == n