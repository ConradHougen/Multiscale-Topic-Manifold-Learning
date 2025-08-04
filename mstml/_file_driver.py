"""
File I/O and utility functions for MSTML.

This module contains functions for data loading, directory management,
file operations, and other utility functions.
"""

import os
import pickle
import hashlib
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from pathlib import Path
from bisect import bisect_left
from math import factorial
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from PyPDF2 import PdfReader, PdfWriter


# ============================================================================
# Directory and Path Management
# ============================================================================

def get_net_src_dir(dset, dsub, create=False):
    """Function to return network source directory"""
    net_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir,
                                           'data', dset, 'networks', dsub))
    net_dir += os.sep
    if create:
        if not os.path.exists(net_dir):
            os.makedirs(net_dir)
    return net_dir


def get_data_int_dir(dset, dsub, create=False):
    """Function to return intermediate data source directory"""
    int_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir,
                                           'data', dset, 'intermediate', dsub))
    int_dir += os.sep
    if create:
        if not os.path.exists(int_dir):
            os.makedirs(int_dir)
    return int_dir


def get_data_original_dir(dset="news20"):
    base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data", dset))
    original_dir = os.path.join(base_dir, "original")
    return original_dir


def get_data_clean_dir(dset, dsub):
    """Get clean data directory path."""
    base_dir = os.path.join("data", dset)
    if dsub:
        return os.path.join(base_dir, dsub, "clean/")
    return os.path.join(base_dir, "clean/")


def get_data_intermediate_dir(dset="news20", dsub=None):
    base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data", dset))
    intermediate_dir = os.path.join(base_dir, "intermediate")
    if dsub:
        intermediate_dir = os.path.join(intermediate_dir, dsub)
    return intermediate_dir


def get_data_networks_dir(dset="news20", dsub=None):
    base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data", dset))
    networks_dir = os.path.join(base_dir, "networks")
    if dsub:
        networks_dir = os.path.join(networks_dir, dsub)
    return networks_dir


def get_exp_dir(dset, dsub, leaf='', create=False):
    """Function to return the experiment directory associated with a dataset and time of run
    of the experiment. leaf is the name of the bottom-level directory"""
    exp_dir_name = os.path.join(os.path.abspath(os.getcwd()),
                                os.pardir,
                                'experiments',
                                dset,
                                dsub,
                                leaf)
    if leaf != '':
        exp_dir_name += os.sep

    if create:
        if not os.path.exists(exp_dir_name):
            os.makedirs(exp_dir_name)
    return exp_dir_name


def initialize_dataset_directories(dir_type="data", dset="news20", dsub=None):
    if dir_type != "data" and dir_type != "experiments":
        raise ValueError(f"dir_type {dir_type} is unsupported!")

    base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, dir_type, dset))

    try:
        # Create the main dataset directory
        os.makedirs(base_dir, exist_ok=False)

        if dir_type == "data":
            # Create sub-directories
            intermediate_dir = os.path.join(base_dir, "intermediate")
            original_dir = os.path.join(base_dir, "original")
            networks_dir = os.path.join(base_dir, "networks")

            os.makedirs(intermediate_dir, exist_ok=False)
            os.makedirs(original_dir, exist_ok=False)
            os.makedirs(networks_dir, exist_ok=False)

            # If dsub is provided, create additional sub-folders within intermediate and networks
            if dsub:
                intermediate_sub_dir = os.path.join(intermediate_dir, dsub)
                networks_sub_dir = os.path.join(networks_dir, dsub)
                os.makedirs(intermediate_sub_dir, exist_ok=False)
                os.makedirs(networks_sub_dir, exist_ok=False)
            else:
                intermediate_sub_dir = intermediate_dir
                networks_sub_dir = networks_dir
        elif dir_type == "experiments":
            # Create the dsub directory if specified
            if dsub:
                experiments_sub_dir = os.path.join(base_dir, dsub)
                os.makedirs(experiments_sub_dir, exist_ok=False)
            else:
                experiments_sub_dir = base_dir

    except FileExistsError:
        # Handle the case where the directory already exists
        if dir_type == "data":
            intermediate_dir = os.path.join(base_dir, "intermediate")
            original_dir = os.path.join(base_dir, "original")
            networks_dir = os.path.join(base_dir, "networks")

            if dsub:
                intermediate_sub_dir = os.path.join(intermediate_dir, dsub)
                networks_sub_dir = os.path.join(networks_dir, dsub)
            else:
                intermediate_sub_dir = intermediate_dir
                networks_sub_dir = networks_dir

            return base_dir, intermediate_sub_dir, original_dir, networks_sub_dir

        elif dir_type == "experiments":
            if dsub:
                experiments_sub_dir = os.path.join(base_dir, dsub)
            else:
                experiments_sub_dir = base_dir

            return base_dir, experiments_sub_dir, None, None

    except OSError as e:
        print(f"Error: {e}")
        return None, None, None, None

    if dir_type == "data":
        return base_dir, intermediate_sub_dir, original_dir, networks_sub_dir
    elif dir_type == "experiments":
        return base_dir, experiments_sub_dir, None, None


def remove_empty_folders(root_dir):
    """Function to remove empty folders in a given directory"""
    folders = list(os.walk(root_dir))
    for path, _, _ in folders[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
            print("Removing empty directory: {}".format(path))


def create_exp_dir(params, inds, dir_prefix='', suffix_prefix=''):
    """Function to create directory unique name, given suffix and dataset info"""
    exp_root_dir = os.path.join(os.getcwd(), os.pardir, 'experiments') + os.sep

    if dir_prefix == '':
        dir_prefix = gen_time_str()

    dir_suffix = gen_dir_suffix(params, inds)
    dir_suffix = suffix_prefix + dir_suffix
    dataset = params['dset'][inds['dset']]
    data_subset = params['dsub'][inds['dsub']]

    dir_name = dir_prefix + dir_suffix

    out_root_dir = os.path.join(exp_root_dir, dataset, data_subset) + os.sep
    out_dir = os.path.join(out_root_dir, dir_name) + os.sep

    # Cleanup empty prior directories in out_root_dir that were unused
    remove_empty_folders(out_root_dir)

    os.makedirs(out_dir, exist_ok=True)
    print("Using directory {}".format(out_dir))

    return out_dir


# ============================================================================
# File I/O Operations
# ============================================================================

def write_pickle(file_path, data, overwrite=True):
    """Write data to a pickle file"""
    if not overwrite and os.path.exists(file_path):
        print(f"File '{file_path}' already exists. Skipping write as overwrite=False.")
        return

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data successfully written to '{file_path}'.")


def read_pickle(file_path):
    """Read data from a pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# Time and Naming Utilities
# ============================================================================

def generate_unique_4digit_code():
    """Generate unique 4-digit code based on current time"""
    # Get the current date and time as a string
    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Hash the string
    hash_object = hashlib.sha256(current_time.encode())
    hash_hex = hash_object.hexdigest()

    # Take the last 4 digits of the integer representation of the hash
    unique_code = int(hash_hex, 16) % 10000

    return unique_code


def get_date_hour_minute():
    """Function to generate a 6 digit _ 4 digit timestr of [month|day|year]_[hour|minute]"""
    today = datetime.now()
    timestr = today.strftime("%m%d%Y_%H%M")
    return timestr


def get_hour_minute():
    """Function to generate 4 digit timestr of hour|minute i.e. hrmi"""
    today = datetime.now()
    timestr = today.strftime('%H%M')
    return timestr


def get_hour_minute_second():
    """Function to generate 6 digit timestr of hour|minute|second"""
    today = datetime.now()
    timestr = today.strftime("%H%M%S")
    return timestr


def gen_time_str():
    """Function to generate unique time string based on current datetime up to minute"""
    today = datetime.now()
    time_str = today.strftime('%m%d%y') + '_' + today.strftime('%H%M')
    return time_str


def gen_date_str():
    """Function to generate time string that is only unique up to current date."""
    today = datetime.now()
    date_str = today.strftime('%m%d%y')
    return date_str


def get_lda_model_file_name(dset, dsub, chunk):
    """Helper function for naming (or retrieving names of) saved lda models"""
    return str(chunk) + '_' + dset + '_' + dsub + '_lda.model'


# ============================================================================
# Path and File Utilities
# ============================================================================

def get_file_path_without_extension(file_path):
    """Function to get the directory and filename, without the file extension"""
    path = Path(file_path)
    return path.parent / path.stem


def get_file_stem_only(file_path):
    """Function to get the filename only, without the file extension"""
    path = Path(file_path)
    return path.stem


def catalan_number(n):
    """Function to compute the Catalan numbers (number of possible dendrograms of n nodes)"""
    return factorial(2 * n) // (factorial(n) * factorial(n + 1))


# ============================================================================
# Parameter Management
# ============================================================================

def init_inds(params):
    """Function to initialize inds"""
    inds = dict.fromkeys(params.keys(), 0)
    return inds


def incr_inds(params, inds):
    """Function to increment inds for selecting next set of parameters"""
    out = inds.copy()

    pkeys = list(params.keys())
    ikeys = list(inds.keys())

    if pkeys != ikeys:
        print("Error: inds and params must have same key sets")
        return False, out, []

    # Iterate from the last key up, incrementing inds when possible.
    # If not possible, return (False, inds)
    pkeys.reverse()

    pkey_change_list = []
    for pkey in pkeys:
        l = len(params[pkey]) - 1
        i = out[pkey]
        if i < l:
            out[pkey] += 1
            # Reset all lower order inds
            for r in pkey_change_list:
                out[r] = 0
            # Also, add the current pkey that we just incremented, to the change list
            pkey_change_list.append(pkey)
            return True, out, pkey_change_list
        elif i == l and i > 0:
            pkey_change_list.append(pkey)

    return False, out, []


def gen_dir_suffix(params, inds):
    """Function to generate a dir_suffix for usage in create_exp_dir"""
    dir_suffix = ''

    pkeys = params.keys()
    ikeys = inds.keys()

    if pkeys != ikeys:
        print("Error: inds and params must share the same key set")
        return dir_suffix

    for pkey in pkeys:
        # Don't need dataset or subset in dir name due to upper directory tree
        if pkey == 'dset' or pkey == 'dsub':
            continue
        # Special handling for years
        elif pkey == 'st':
            dir_suffix += '_' + str(params[pkey][inds[pkey]])
            continue
        elif pkey == 'ends':
            yr1 = params[pkey][inds[pkey]][0]
            yr2 = params[pkey][inds[pkey]][1]
            dir_suffix += '_' + str(yr1) + '_' + str(yr2)
            continue
        else:
            to_add = str(params[pkey][inds[pkey]])
            # remove any decimal points or other punctuation
            to_add = ''.join(filter(str.isalnum, to_add))
            dir_suffix += '_' + pkey + to_add

    return dir_suffix


def get_key(d, v):
    """Function which takes a dictionary and a value, then returns the key"""
    for key, value in d.items():
        if v == value:
            return key


# ============================================================================
# Data Processing Utilities
# ============================================================================

def filter_dataframe(df, yr1=None, yr2=None, authids=None, reset_index=True):
    """Function to filter dataframe according to constraints"""
    # Convert date column to datetime type
    df['date'] = pd.to_datetime(df['date'])
    # Perform set logic to determine filters, then apply them to dataframe
    condition = pd.Series([True] * len(df))
    condition.index = df.index
    if yr1 is not None:
        condition &= (df['date'].dt.year >= int(yr1))
    if yr2 is not None:
        condition &= (df['date'].dt.year <= int(yr2))
    if authids is not None:
        authids_cond = pd.Series([False] * len(df))
        authids_cond.index = df.index
        for authid in authids:
            authids_cond |= (df['authorID'].str.contains(str(authid)))
        condition &= authids_cond
    filtered_df = df[condition]
    if reset_index:
        filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def load_filtered_dataframe(dset, dsub, yr1=None, yr2=None, authids=None, reset_index=True):
    """Function to load partial dataframe according to constraints"""
    # First, get the directory where the data are located
    data_src_dir = get_data_int_dir(dset, dsub)
    # Load main dataframe from the data directory
    with open(data_src_dir + 'main_df.pkl', 'rb') as f:
        main_df = pickle.load(f)
    return filter_dataframe(main_df, yr1, yr2, authids, reset_index)


def get_authorId2doc(df):
    """Function to generate authorId2doc, given a dataframe with an authorID column"""
    authorId2doc = {}

    df = df.reset_index(drop=True)

    # Using the first (zeroth index) element to determine which processing to use
    # (If the type changes within the column, this is not expected)
    if type(df['authorID'][0]) == list:
        for idx, row in df.iterrows():
            authorIds = row['authorID']
            for authId in authorIds:
                authId = int(authId)
                if authId not in authorId2doc:
                    authorId2doc[authId] = [idx]
                else:
                    authorId2doc[authId].append(idx)
    # Legacy support for if the type is string instead of list
    else:
        print(type(df['authorID'][0]))
        for idx, row in df.iterrows():
            authorIds = row['authorID'].split(',')[:-1]
            for authId in authorIds:
                authId = int(authId)
                if authId not in authorId2doc:
                    authorId2doc[authId] = [idx]
                else:
                    authorId2doc[authId].append(idx)
    return authorId2doc


def get_author2doc(df, int_data_dir):
    """Function to generate author2doc, given a dataframe and authorId_to_author"""
    print("Warning: Author names are potentially ambiguous. Should use get_authorId2doc() instead.")

    with open(int_data_dir + 'authorId_to_author.pkl', 'rb') as f:
        authorId_to_author = pickle.load(f)

    author2doc = {}

    for idx, row in df.iterrows():
        authorIds = row['authorID'].split(',')[:-1]
        for authId in authorIds:
            authId = int(authId)
            auth = authorId_to_author[authId][0]
            if re.search("^[a-zA-Z]", auth) is not None:
                if auth not in author2doc:
                    author2doc[auth] = [idx]
                else:
                    author2doc[auth].append(idx)
            else:
                print("Error: auth={}".format(auth))

    return author2doc


def merge_lists_to_set(df_in, rows, column):
    """Function to merge a pandas column containing lists into a single set, for the specified rows."""
    # Extract the lists from the specified rows and column
    lists_to_merge = df_in.loc[list(rows), column].tolist()  # Convert rows to list directly here

    # Merge the lists into a single set of integers
    merged_set = set()
    for lst in lists_to_merge:
        merged_set.update(lst)

    return merged_set


# ============================================================================
# Text Processing Utilities
# ============================================================================

def super_simple_preprocess(sentence):
    """Wrapper to allow mapping of simple_preprocess"""
    return simple_preprocess(str(sentence), deacc=True)


def sent_to_words(sentences):
    """Takes in sentences and yields the simple_preprocess applied to each sentence"""
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (simple_preprocess(str(sentence), deacc=True))


def filter_list_bisect(l_to_filter, positive_sorted_filter, filtered_l):
    """Method to filter list using bisect method"""
    for i, x in enumerate(l_to_filter):
        index = bisect_left(positive_sorted_filter, x)
        if index < len(positive_sorted_filter):
            if positive_sorted_filter[index] == x:
                filtered_l.append(x)


def remove_stopwords(texts):
    """Removes stop_words from a set of texts"""
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'sub', 'sup', 'used', 'using'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def filter_stopwords_by_approval(doc, approved_words):
    """Removes stop_words from a single document (for a multiprocessing approach). extension param
    allows one to arbitrarily extend the list of stop words."""
    new_doc = []
    filter_list_bisect(doc, approved_words, new_doc)

    return new_doc


def remove_singly_occurring_words(texts, vocab):
    """Removes singly-occurring words from a set of texts and dictionary"""
    once_ids = [tokenid for tokenid, docfreq in vocab.cfs.items() if docfreq == 1]
    new_vocab = vocab
    new_vocab.filter_tokens(once_ids)
    new_vocab.compactify()

    tot_words = sum([len(doc) for doc in texts])
    print("Total words prior to filtering: {}".format(tot_words))

    token2id = new_vocab.token2id

    new_texts = [[word for word in doc if word in token2id] for doc in texts]
    tot_words = sum([len(doc) for doc in new_texts])
    print("Total words after filtering: {}".format(tot_words))

    return new_texts, new_vocab


def lemmatize_and_update(texts):
    """Lemmatizes words from a set of texts. Returns an updated vocab dictionary also."""
    import gensim.corpora as corpora
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the texts and return an updated vocabulary with the lemmatized texts
    new_texts = [[lemmatizer.lemmatize(word) for word in doc] for doc in texts]
    new_vocab = corpora.Dictionary(new_texts)

    return new_texts, new_vocab


def lemmatize_mp(text):
    """Multiprocessing version of function to lemmatize words from a single document.
    This doesn't update the vocab dictionary or return it."""
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the texts and return the lemmatized texts
    new_text = [lemmatizer.lemmatize(word) for word in text]

    return new_text


def count_doc_freq_of_words(texts, vocab):
    """Counts the frequency of occurrences of each word among a corpus of texts, where
    a word is only counted once per document. Returns a dictionary of word id ->
    number of documents containing the word at least once."""
    id2docfreq = {}
    token2id = vocab.token2id

    for text in texts:
        # Increment the words that appear at least once
        to_increment = list(set(text))
        # If the entry exists in id2docfreq, increment, otherwise, add 1
        for word in to_increment:
            if token2id[word] in id2docfreq:
                id2docfreq[token2id[word]] += 1
            else:
                id2docfreq[token2id[word]] = 1

    return id2docfreq


def filter_terms(doc, vocabulary):
    """Reduces document to terms in vocab only"""
    # Only keep terms in the reduced vocabulary
    return [term for term in doc if term in vocabulary]


# ============================================================================
# PDF Operations
# ============================================================================

def crop_pdf(input_pdf, output_pdf, x1, y1, x2, y2, pdf_dpi=72):
    """
    Crop a pdf image according to (x1, y1) in bottom left and (x2, y2) in top right
    Scaling to figure size is automatic (using 72 dpi pdf standard by default)
    
    Parameters:
        input_pdf (str): Path to the input PDF file.
        output_pdf (str): Path to the output cropped PDF file.
        x1, y1, x2, y2 (float): Bounding box in inches (lower-left to upper-right).
        pdf_dpi (int): Conversion factor for inches to points (default: 72).
    """
    with open(input_pdf, "rb") as input_file:
        reader = PdfReader(input_file)
        writer = PdfWriter()

        for page in reader.pages:
            # Get original media box for validation
            original_box = page.mediabox

            # Convert crop dimensions to points
            lower_left = (x1 * pdf_dpi, y1 * pdf_dpi)
            upper_right = (x2 * pdf_dpi, y2 * pdf_dpi)

            # Validate dimensions against the media box
            original_lower_left = (original_box.left, original_box.bottom)
            original_upper_right = (original_box.right, original_box.top)

            # Ensure crop box fits within the media box
            lower_left = (
                max(lower_left[0], original_lower_left[0]),
                max(lower_left[1], original_lower_left[1]),
            )
            upper_right = (
                min(upper_right[0], original_upper_right[0]),
                min(upper_right[1], original_upper_right[1]),
            )

            # Check for valid crop box dimensions
            if lower_left[0] >= upper_right[0] or lower_left[1] >= upper_right[1]:
                raise ValueError(
                    f"Invalid crop box after adjustment: {lower_left} to {upper_right}"
                )

            # Apply crop box and media box changes
            page.mediabox.lower_left = lower_left
            page.mediabox.upper_right = upper_right
            page.cropbox.lower_left = lower_left
            page.cropbox.upper_right = upper_right

            writer.add_page(page)

        with open(output_pdf, "wb") as output_file:
            writer.write(output_file)


# ============================================================================
# Data Filtering and Analysis Utilities
# ============================================================================

def filter_doc_df_by_specified_authors(doc_df, authId_list):
    """Function to extract subset of doc dataframe including authors in list"""
    if not authId_list:
        print("Error: authId_list was empty")
        return doc_df

    # Get mapping from authors (by ID) to documents
    authorId2doc = get_authorId2doc(doc_df)

    # Create list of doc indices
    init_doc_inds = [authorId2doc[authId] for authId in authId_list]
    doc_inds = [doc_idx for sublist in init_doc_inds for doc_idx in sublist]

    return doc_df.loc[doc_df.index(doc_inds)]


def filter_all_by_connected_authors(int_data_dir, doc_df, source_G, authId_list):
    """Function to extract subset of docs and network by any connection to authors"""
    import networkx as nx
    
    if not authId_list:
        print("Error: authId_list was empty")
        return doc_df, source_G

    # Get all nodes connected to the nodes in authId_list
    sub_G_nodes = set()
    for authId in authId_list:
        sub_G_nodes.add(nx.node_connected_component(source_G, authId))

    # Create the subgraph of all connected nodes to the original authId_list
    sub_G = source_G.subgraph(sub_G_nodes)

    # Create the sub dataframe of all authors connected to authors in authId_list
    sub_df = filter_doc_df_by_specified_authors(doc_df, list(sub_G_nodes))

    return sub_df, sub_G


def filter_all_by_num_publications(int_data_dir, doc_df, source_G, authId_list, nmin, nmax):
    """Function to extract subset of docs and network by number of author publications"""
    if not authId_list:
        print("Using all authors for npub filter")
        authId_list = list(source_G.nodes())

    # Get mapping from authors (by ID) to documents
    authorId2doc = get_authorId2doc(doc_df, int_data_dir)

    # Iterate through list of authors and check number of publications
    new_authId_list = []
    for authId in authId_list:
        npub = len(authorId2doc[authId])
        if nmin <= npub and npub <= nmax:
            new_authId_list.append(authId)

    # Get the sub dataframe of all authors in authId_list with the correct number of publications
    sub_df = filter_doc_df_by_specified_authors(int_data_dir, doc_df, new_authId_list)

    # Get the subgraph of all authors in authId_list with the correct number of publications
    sub_G = source_G.subgraph(new_authId_list)

    return sub_df, sub_G


def filter_all_by_num_direct_coauthors(int_data_dir, doc_df, source_G, authId_list, dmin, dmax):
    """Function to extract subset of docs and network by number of direct coauthors (node degree)"""
    if not authId_list:
        print("Using all authors for direct coauthor filter")
        authId_list = list(source_G.nodes())

    # Iterate though list of authors and check number of direct coauthors (node degree)
    new_authId_list = []
    for authId in authId_list:
        deg = source_G.degree[authId]
        if dmin <= deg and deg <= dmax:
            new_authId_list.append(authId)

    # Get the sub dataframe of all authors with the correct node degrees
    sub_df = filter_doc_df_by_specified_authors(int_data_dir, doc_df, new_authId_list)

    # Get the subgraph of all authors with the correct node degrees
    sub_G = source_G.subgraph(new_authId_list)

    return sub_df, sub_G


def filter_all_by_connected_component_size(int_data_dir, doc_df, source_G, authId_list, smin, smax):
    """Function to extract subset of docs and network by size of connected component"""
    import networkx as nx
    
    if not authId_list:
        print("Using all authors for conn comp size filter")
        authId_list = list(source_G.nodes())

    # Iterate through list of authors and get connected component sizes
    new_authId_list = []
    for authId in authId_list:
        sz = len(nx.node_connected_component(source_G, authId))
        if smin <= sz and sz <= smax:
            new_authId_list.append(authId)

    # Get the sub dataframe of all authors in the specified sizes of connected components
    sub_df = filter_doc_df_by_specified_authors(int_data_dir, doc_df, new_authId_list)

    # Get the subgraph of the specified authors in the specified sizes of connected components
    sub_G = source_G.subgraph(new_authId_list)

    return sub_df, sub_G


def filter_all_by_connected_component_density(int_data_dir, doc_df, source_G, authId_list, dmin=0.0, dmax=1.0):
    """Function to extract subset of docs and network by connected component edge density"""
    # TODO: Function to extract subset of docs and network by connected component edge density
    pass


# ============================================================================
# Statistics and Analysis Functions (Placeholder)
# ============================================================================

def stats_local_node_connectivity(source_G, edge_list):
    """Function to return mean/stdev pairwise node connectivity (minimum separating cutset) among edges in list"""
    # TODO: Function to return mean/stdev pairwise node connectivity (minimum separating cutset) among edges in list
    pass


def stats_clustering_coefficients(source_G, node_list):
    """Function to return mean/stdev clustering coefficient for nodes in list"""
    # TODO: Function to return mean/stdev clustering coefficient for nodes in list
    pass


def stats_degree_assortativity(source_G, node_list):
    """Function to return mean/stdev degree assortativity for nodes in list"""
    # TODO: Function to return mean/stdev degree assortativity for nodes in list
    pass


def stats_neighborhood_degree(source_G, node_list):
    """Function to return mean/stdev neighborhood degree for nodes in list"""
    # TODO: Function to return mean/stdev neighborhood degree for nodes in list
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def validate_dataset_name(name: str) -> bool:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(
            f"Invalid dataset name '{name}'. Must start with a letter or underscore and contain only lowercase letters, digits, and underscores."
        )
    return True


def log_print(message: str, level: str = "info", logger: logging.Logger = None, also_print: bool = True):
    """
    Logs and optionally prints a message.

    Parameters:
        message (str): The message to log/print.
        level (str): Logging level: 'debug', 'info', 'warning', 'error', or 'critical'.
        logger (logging.Logger): Logger instance. If None, uses root logger.
        also_print (bool): Whether to also print to stdout.
    """
    if logger is None:
        logger = logging.getLogger()

    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)

    if also_print:
        print(message)


def create_hash_id(text, length=8):
    """Create a hash ID from text."""
    return hashlib.md5(text.encode()).hexdigest()[:length]