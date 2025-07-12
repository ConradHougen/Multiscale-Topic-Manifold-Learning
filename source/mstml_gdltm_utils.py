import numpy as np
import pandas as pd
import networkx as nx

# Optional imports - framework works without these
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import hypernetx as hnx
    HAS_HYPERNETX = True
except ImportError:
    HAS_HYPERNETX = False
import gensim.corpora as corpora
import pickle
import os
import re
import glob
import nltk
import random
import hashlib
import colorcet as cc
import py4cytoscape as p4c
import math

from scipy.spatial import KDTree
from collections import Counter
from math import factorial
from pathlib import Path
from bisect import bisect_left
from datetime import datetime
from wordcloud import WordCloud
from sklearn.cluster import AgglomerativeClustering
from gensim.utils import simple_preprocess
from enum import Enum, auto
from gensim.models import AuthorTopicModel
from PyPDF2 import PdfReader, PdfWriter

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Different types of embedding representations for authors
class AuthEmbedEnum(Enum):
    WORD_FREQ = auto()
    AT_DISTN = auto()
    AT_SPARSIFIED_DISTN = auto()
    TERM_RELEVANCE_N_HOT = auto()
    TERM_RELEVANCE_VMASK = auto()


# Different methods for representing topics filtered by relevance
# N_HOT_ENCODING: each topic is represented by a vector with N nonzero entries, for the N most relevant terms
# VOCAB_MASK: each topic is masked to only the union over T topics of the N most relevant terms for each topic.
class TermRelevanceTopicType(Enum):
    N_HOT_ENCODING = auto()
    VOCAB_MASK = auto()


# This class implements a handler for generating topics, filtered by relevance
class TermRelevanceTopicFilter:
    def __init__(self,
                 atmodel: AuthorTopicModel,
                 corpus: corpora.MmCorpus,
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


def generate_unique_4digit_code():
    # Get the current date and time as a string
    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Hash the stri0ng
    hash_object = hashlib.sha256(current_time.encode())
    hash_hex = hash_object.hexdigest()

    # Take the last 4 digits of the integer representation of the hash
    unique_code = int(hash_hex, 16) % 10000

    return unique_code


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


# Function to compute the Catalan numbers (number of possible dendrograms of n nodes)
def catalan_number(n):
    return factorial(2 * n) // (factorial(n) * factorial(n + 1))


# Function to get the directory and filename, without the file extension
def get_file_path_without_extension(file_path):
    path = Path(file_path)
    return path.parent / path.stem


# Function to get the filename only, without the file extension
def get_file_stem_only(file_path):
    path = Path(file_path)
    return path.stem


# Function to both print to console and write out to file
def mstml_print(str_to_print, fpath, force_overwrite=False):
    if not os.path.exists(fpath) or force_overwrite:
        with open(fpath, 'w') as f:
            print(str_to_print, file=f)
    else:
        with open(fpath, 'a') as f:
            print(str_to_print, file=f)
    print(str_to_print)


# Function to compute author word-freq embedding, given an author name and fitted author topic model.
def author_word_freq_embedding(atmodel, auth_id):
    ntopics = atmodel.num_topics
    nwords = atmodel.num_terms
    topic_wf_distns = atmodel.get_topics()

    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.00)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')

    auth_wf_embedding = np.zeros((nwords,)).astype('float32')
    for tpc in range(ntopics):
        auth_wf_embedding = np.add(auth_wf_embedding, atopics[tpc] * topic_wf_distns[tpc])

    return auth_wf_embedding


# Function to retrieve author distribution over topics, given an author name and fitted author topic model
def author_topic_distn(atmodel, auth_id):
    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.0)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')
    return atopics


# Function to create sparsified embeddings for an author, based on their author-topic distribution.
# The idea is that the topics which have small contribution to the author-topic distribution are likely
#   to be noise. We want to sparsify the vector representations to boost signal.
# atmodel_alpha should be the initial alpha parameter that was passed to AuthorTopicModel.
# mult determines the threshold for sparsification, e.g. zero out any value less than mult * atmodel_alpha
# normalize determines whether to apply normalization after sparsification.
def author_topic_sparsified_distn_embedding(atmodel, auth_id, atmodel_alpha, mult=1.01, normalize=False):
    atopics = atmodel.get_author_topics(auth_id, minimum_probability=0.0)
    atopics = np.array([k for (j, k) in atopics]).astype('float32')
    atopics[atopics < (mult * atmodel_alpha)] = 1e-9
    if normalize:
        if atopics.sum() < 0.01:
            print("Warning: topic distn for auth_id {} may be numerically unstable")
        atopics = atopics / atopics.sum()
    return atopics


# Function to compute author embeddings based on filtering each topic to the relevant terms
# term_rel_weight is an optional parameter in [0, 1] which is based on the lambda in "Sievert 2014: LDAvis..."
def author_topics_by_term_relevance_embedding(atmodel, term_top_filter, auth_id):
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


# Function to compute author embeddings based on filtering the vocabulary to terms relevant to each topic. Unlike
#   the method for filtering topics by term relevance, here we focus on vocab filtering rather than simply summarizing
#   each topic's representation by the most relevant terms. Once the vocab is filtered, we use the remaining terms as
#   the domain over which all topics are defined.
# term_rel_weight is an optional parameter in [0, 1] which is based on the lambda in "Sievert 2014: LDAvis..."
def author_vocab_by_term_relevance_embedding(atmodel, term_top_filter, auth_id):
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


# Function to merge a pandas column containing lists into a single set, for the specified rows.
def merge_lists_to_set(df_in, rows, column):
    # Extract the lists from the specified rows and column
    lists_to_merge = df_in.loc[list(rows), column].tolist()  # Convert rows to list directly here

    # Merge the lists into a single set of integers
    merged_set = set()
    for lst in lists_to_merge:
        merged_set.update(lst)

    return merged_set


# Function to return network source directory
def get_net_src_dir(dset, dsub, create=False):
    net_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir,
                                           'data', dset, 'networks', dsub))
    net_dir += os.sep
    if create:
        if not os.path.exists(net_dir):
            os.makedirs(net_dir)
    return net_dir


# Function to return intermediate data source directory
def get_data_int_dir(dset, dsub, create=False):
    int_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir,
                                           'data', dset, 'intermediate', dsub))
    int_dir += os.sep
    if create:
        if not os.path.exists(int_dir):
            os.makedirs(int_dir)
    return int_dir


# Function to filter dataframe according to constraints
def filter_dataframe(df, yr1=None, yr2=None, authids=None, reset_index=True):
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


# Function to load partial dataframe according to constraints
def load_filtered_dataframe(dset, dsub, yr1=None, yr2=None, authids=None, reset_index=True):
    # First, get the directory where the data are located
    data_src_dir = get_data_int_dir(dset, dsub)
    # Load main dataframe from the data directory
    with open(data_src_dir + 'main_df.pkl', 'rb') as f:
        main_df = pickle.load(f)
    return filter_dataframe(main_df, yr1, yr2, authids, reset_index)


# Function to initialize inds
def init_inds(params):
    inds = dict.fromkeys(params.keys(), 0)
    return inds


# Function to increment inds for selecting next set of parameters
def incr_inds(params, inds):
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


# Function to perform set intersection on two lists
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


# Function to generate a dir_suffix for usage in create_exp_dir
def gen_dir_suffix(params, inds):
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


# Function to remove empty folders in a given directory
def remove_empty_folders(root_dir):
    folders = list(os.walk(root_dir))
    for path, _, _ in folders[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
            print("Removing empty directory: {}".format(path))


# Function which takes a dictionary and a value, then returns the key
def get_key(d, v):
    for key, value in d.items():
        if v == value:
            return key


# Function to generate a 6 digit _ 4 digit timestr of [month|day|year]_[hour|minute]
def get_date_hour_minute():
    today = datetime.now()
    timestr = today.strftime("%m%d%Y_%H%M")
    return timestr


# Function to generate 4 digit timestr of hour|minute i.e. hrmi
def get_hour_minute():
    today = datetime.now()
    timestr = today.strftime('%H%M')
    return timestr


# Function to generate 6 digit timestr of hour|minute|second
def get_hour_minute_second():
    today = datetime.now()
    timestr = today.strftime("%H%M%S")
    return timestr


# Function to generate unique time string based on current datetime up to minute
def gen_time_str():
    today = datetime.now()
    time_str = today.strftime('%m%d%y') + '_' + today.strftime('%H%M')
    return time_str


# Function to generate time string that is only unique up to current date.
def gen_date_str():
    today = datetime.now()
    date_str = today.strftime('%m%d%y')
    return date_str


# Helper function for naming (or retrieving names of) saved lda models
def get_lda_model_file_name(dset, dsub, chunk):
    return str(chunk) + '_' + dset + '_' + dsub + '_lda.model'


# Function to return the experiment directory associated with a dataset and time of run
# of the experiment. leaf is the name of the bottom-level directory
def get_exp_dir(dset, dsub, leaf='', create=False):
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


# Function to create directory unique name, given suffix and dataset info
def create_exp_dir(params, inds, dir_prefix='', suffix_prefix=''):
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


# Function to calculate the Kullback-Leibler Divergence between two distributions.
def kl_divergence(p, q):
    # Ensure the distributions sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Only consider the elements where P > 0 and Q > 0 for calculation
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


# Function to calculate the Jensen-Shannon Divergence between two distributions.
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# Function to calculate the Euclidean distance between two vectors.
def euclidean(p, q):
    return np.linalg.norm(p - q)


# Function to compute hellinger distance between multinomial distributions
def hellinger(p, q):
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)


# Function to compute hellinger similarity between multinomial distributions, using form sim(x) = b/(1+c*x)
def hellinger_sim(p, q, alpha=10):
    d = hellinger(p, q)
    sim = 1 / (1 + (alpha * d))
    return sim


# Function to compute element-wise minimum-weighted total variation distance
def min_weighted_tv(p, q):
    w_abs_diff = np.abs(p - q) * np.minimum(p, q)
    return np.sum(w_abs_diff)


# Function to compute element-wise mean-valued total variation distance
def mean_weighted_tv(p, q):
    means = (p + q) / 2
    w_abs_diff = np.abs(p - q) * means
    return np.sum(w_abs_diff)


# Function to compute entropy of a given probability distribution
def entropy(p):
    p = np.array(p)  # Ensure input is a numpy array
    # Filter out zero probabilities to avoid log(0)
    p_nz = p[p > 0]
    return -np.sum(p_nz * np.log2(p_nz))


# Function to calculate Gini coefficient
def gini_coefficient(p):
    p = p[p != 0]  # Remove zero entries
    n = len(p)
    if n == 0:
        return 0
    sorted_p = np.sort(p)
    tmp_idx = np.arange(1, n + 1)
    return (np.sum((2 * tmp_idx - n - 1) * sorted_p)) / (n * np.sum(sorted_p))


# Function to compute the max-weighted mean of a list of vectors of the same length
def max_weighted_mean(vecs: list[np.ndarray]):
    mwmean = np.zeros_like(vecs[0])
    for vec in vecs:
        mwmean += vec * vec.max()
    mwmean /= mwmean.sum()
    return mwmean


# Function to compute entropy of max-weighted mean of multiple vectors, passed as
#   a list of numpy arrays of the same length
def entropy_of_max_weighted_mean(vecs: list[np.ndarray]):
    return entropy(max_weighted_mean(vecs))


# Helper function for using networkx index generators
def get_sim_from_piter(piter):
    # TODO: Fix ZeroDivisionError
    try:
        piter = list(piter)
        sim = piter[0][2]
    except ZeroDivisionError:
        sim = 0.00
    return sim


# Function to retrieve all coauthors of a given author (by author ID) and coauthorship network
def get_coauthors(G, authorid):
    return [n for n in G.neighbors((str(authorid)))]


# Wrapper to allow mapping of simple_preprocess
def super_simple_preprocess(sentence):
    return simple_preprocess(str(sentence), deacc=True)


# Takes in sentences and yields the simple_preprocess applied to each sentence
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (simple_preprocess(str(sentence), deacc=True))


# Method to filter list using bisect method
def filter_list_bisect(l_to_filter, positive_sorted_filter, filtered_l):
    for i, x in enumerate(l_to_filter):
        index = bisect_left(positive_sorted_filter, x)
        if index < len(positive_sorted_filter):
            if positive_sorted_filter[index] == x:
                filtered_l.append(x)


# Removes stop_words from a set of texts
def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'sub', 'sup', 'used', 'using'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


# Removes stop_words from a single document (for a multiprocessing approach). extension param
#   allows one to arbitrarily extend the list of stop words.
def filter_stopwords_by_approval(doc, approved_words):
    new_doc = []
    filter_list_bisect(doc, approved_words, new_doc)

    return new_doc


# Removes singly-occurring words from a set of texts and dictionary
def remove_singly_occurring_words(texts, vocab):
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


# Lemmatizes words from a set of texts. Returns an updated vocab dictionary also.
def lemmatize_and_update(texts):
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the texts and return an updated vocabulary with the lemmatized texts
    new_texts = [[lemmatizer.lemmatize(word) for word in doc] for doc in texts]
    new_vocab = corpora.Dictionary(new_texts)

    return new_texts, new_vocab


# Multiprocessing version of function to lemmatize words from a single document.
# This doesn't update the vocab dictionary or return it.
def lemmatize_mp(text):
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the texts and return the lemmatized texts
    new_text = [lemmatizer.lemmatize(word) for word in text]

    return new_text


# Counts the frequency of occurrences of each word among a corpus of texts, where
#   a word is only counted once per document. Returns a dictionary of word id ->
#   number of documents containing the word at least once.
def count_doc_freq_of_words(texts, vocab):
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


# Hierarchical (agglomerative) clustering by number of clusters and pairwise distance matrix
def hier_cluster(distance, n_clusters):
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


# Function to generate file name for single year graph
def get_fname_for_single_yr_graph(dset, dsub, yr, label=''):
    net_dir = '../data/' + dset + '/networks/' + dsub + '/'
    fname = net_dir + dset + '_' + dsub + '_' + str(yr) + '_' + label + '.graphml'
    return fname


# Function to generate file name for graph composition
def get_fname_for_graph_composition(dset, dsub, yr_start, yr_end, label=''):
    net_dir = '../data/' + dset + '/networks/' + dsub + '/'
    fname = net_dir + 'composed_' + dset + '_' + dsub + '_' \
            + str(yr_start) + '-' + str(yr_end) + '_' + label + '.graphml'
    return fname


# Helper function to confirm if a variable is a tuple of the expected length
def is_n_tuple(test_var, n: int):
    return isinstance(test_var, tuple) and len(test_var) == n


# Write data to a pickle file
def write_pickle(file_path, data, overwrite=True):
    if not overwrite and os.path.exists(file_path):
        print(f"File '{file_path}' already exists. Skipping write as overwrite=False.")
        return

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data successfully written to '{file_path}'.")


# Read data from a pickle file
def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Function to perform label propagation where the set of fixed_nodes is immutable.
#   Nodes in G should have a 'label' attribute
def fixed_source_label_propagation(G, fixed_nodes):
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


def get_data_original_dir(dset="news20"):
    base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data", dset))
    original_dir = os.path.join(base_dir, "original")
    return original_dir


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


# Reduces document to terms in vocab only
def filter_terms(doc, vocabulary):
    # Only keep terms in the reduced vocabulary
    return [term for term in doc if term in vocabulary]


# Crop a pdf image according to (x1, y1) in bottom left and (x2, y2) in top right
# Scaling to figure size is automatic (using 72 dpi pdf standard by default)
def crop_pdf(input_pdf, output_pdf, x1, y1, x2, y2, pdf_dpi=72):
    """
    Crop a PDF file to the specified dimensions.

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


# Function to compose networkx multigraph object by multiple networkx
#   multigraphs from yr_start to yr_end.
def compose_coauthorship_network(dset, dsub, yr_start, yr_end, overwrite=False, d_lims=None, label=''):
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


# Function to take networkx co-author network graph and convert to a .pairs file in the
# appropriate experiment directory based on the dataset
def nx_graph_to_pairs_file(nxg_in, exp_dir, pairs_fname):
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


# Function to create networkx multigraph graphml file using dataframe with an
#   appropriately formatted column of author IDs
def gen_nx_multigraph_from_dataframe(src_df, dset, dsub, overwrite=False, label=''):
    if src_df.empty:
        raise ValueError("src_df is empty")

    for idx, row in src_df.iterrows():
        raise NotImplementedError


# Function to create networkx multigraph graphml file for each year of documents in
#   a given pandas dataframe (main_df.pkl from data/dataset/intermediate/data_subset/)
def gen_nx_multigraphs_per_year(dataset, data_subset, overwrite=False, label=''):
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


# Helper function to flatten dataframe according to a column of lists for hypergraph constructor
def flatten_df_column_with_str_list(df, list_column, new_column, retained_cols):
    df[list_column] = df[list_column].map(lambda x: str.split(x, ',')[:-1])
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    cols_to_retain = ([idx for idx, col in enumerate(df.columns) if col in retained_cols])
    expanded_df = df.iloc[destination_rows, cols_to_retain].copy()
    expanded_df[new_column] = ([item for items in df[list_column] for item in items])
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df


# Function to generate and return hypergraph using authors column in doc dataframe
def gen_hypergraph(dset, dsub, yr_start=None, yr_end=None):
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


# Function to compute and store F1 score stats from link prediction results
def lp_expA_score(pred_net, train_net, test_net, name, out_dir):
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


# Function to generate authorId2doc, given a dataframe with an authorID column
def get_authorId2doc(df):
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


# TODO: Deprecated. Should use get_authorId2doc instead
# Function to generate author2doc, given a dataframe and authorId_to_author
def get_author2doc(df, int_data_dir):
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


# Function to convert NX graphml file into a visualizable graphml file with unique edge ids
def graphml_viz_convert_file(dset, dsub, filename, overwrite=False, specific_directory=None):
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


# Function to convert NX graphml files into visualizable graphml files with unique edge ids
def graphml_viz_convert(dset, dsub, overwrite=False, specific_directory=None):
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


# Function to convert viz_ .graphml file into a .pairs file (edge list) for HRG model
def convert_viz_graphml_to_pairs_file(dset, dsub, filename, specific_directory=None):
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


# Wrapper for reading graphml file to networkx object
def mstml_read_graphml_to_netx(dset, dsub, filename, specific_directory=None):
    if specific_directory != None:
        in_dir = specific_directory
    else:
        in_dir = get_net_src_dir(dset, dsub)

    fpath = in_dir + filename
    nxg = nx.read_graphml(fpath, node_type=int)
    return nxg


# Function to convert a slice topic index into separate year and topic indices (based on number of topics)
def convert_slc_tpc_idx_to_yr_and_tpc_idx(slc_tpc_idx, ntop):
    yr_idx = slc_tpc_idx // ntop
    tpc_idx = slc_tpc_idx - (yr_idx * ntop)
    return yr_idx, tpc_idx


# Function to generate wordclouds from a set of slice topics, and save them in a returned dictionary for re-use.
def generate_slc_tpc_wordclouds(nslicetopics, ntopics, exp_dir):
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


# Function to generate a dictionary of wordclouds from a matrix of topic word freq vectors.
def generate_tpc_wordclouds(X, int_dir):
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


# Takes a networkx graph and returns the kth largest connected component
def find_kth_largest_connected_component(nxg, k=0):
    if k == 0:
        # Return the largest connected component
        return max(nx.connected_components(nxg), key=len)
    else:
        # Find all connected components, sorted by size
        components = sorted(list(nx.connected_components(nxg)), key=len, reverse=True)
        # Return the kth largest component
        return components[k]


# Samples node pairs up to nsamples from a networkx graph, where the node pairs come from the greedily sampled
#   maximum length path in the network, with each path removed after sampling to avoid edge-sample repeats.
def topic_space_dist_vs_path_len_for_non_overlapping_max_paths(nxg, auth_vecs, dmetric=hellinger, nsamples=10000):
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


# Function to get degrees of separation (path length) in co-author network vs. topic distance
def topic_space_dist_vs_path_length(nxg, auth_vecs, auth_ids, log_file, dmetric=hellinger, nsamples=10000):
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
            mstml_print("Early exit: Attempts at limit:{}".format(attempts), log_file)
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


# Function to extract subset of doc dataframe including authors in list
def filter_doc_df_by_specified_authors(doc_df, authId_list):
    if not authId_list:
        print("Error: authId_list was empty")
        return doc_df

    # Get mapping from authors (by ID) to documents
    authorId2doc = get_authorId2doc(doc_df)

    # Create list of doc indices
    init_doc_inds = [authorId2doc[authId] for authId in authId_list]
    doc_inds = [doc_idx for sublist in init_doc_inds for doc_idx in sublist]

    return doc_df.loc[doc_df.index(doc_inds)]


# Function to extract subset of docs and network by any connection to authors
def filter_all_by_connected_authors(int_data_dir, doc_df, source_G, authId_list):
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


# Function to extract subset of docs and network by number of author publications
def filter_all_by_num_publications(int_data_dir, doc_df, source_G, authId_list, nmin, nmax):
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


# Function to extract subset of docs and network by number of direct coauthors (node degree)
def filter_all_by_num_direct_coauthors(int_data_dir, doc_df, source_G, authId_list, dmin, dmax):
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


# Function to extract subset of docs and network by size of connected component
def filter_all_by_connected_component_size(int_data_dir, doc_df, source_G, authId_list, smin, smax):
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


# TODO: Function to extract subset of docs and network by connected component edge density
def filter_all_by_connected_component_density(int_data_dir, doc_df, source_G, authId_list, dmin=0.0, dmax=1.0):
    pass


# TODO: Function to return mean/stdev pairwise node connectivity (minimum separating cutset) among edges in list
def stats_local_node_connectivity(source_G, edge_list):
    pass


# TODO: Function to return mean/stdev clustering coefficient for nodes in list
def stats_clustering_coefficients(source_G, node_list):
    pass


# TODO: Function to return mean/stdev degree assortativity for nodes in list
def stats_degree_assortativity(source_G, node_list):
    pass


# TODO: Function to return mean/stdev neighborhood degree for nodes in list
def stats_neighborhood_degree(source_G, node_list):
    pass


# Function to create wordcloud of entire data, given dataset and subset strings
def create_and_save_wordcloud(dset, dsub, max_words=200):
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