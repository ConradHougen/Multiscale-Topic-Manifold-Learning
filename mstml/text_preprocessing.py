"""
Text preprocessing utilities for MSTML.

This module provides comprehensive text preprocessing functionality
including tokenization, cleaning, and preparation for topic modeling.
"""

import re
import nltk
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Optional
from multiprocessing import Pool
from tqdm import tqdm


def super_simple_preprocess(text):
    return simple_preprocess(str(text), deacc=True)


def lemmatize_doc(doc):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in doc]


def filter_doc_with_approved_tokens(doc, approved_tokens_set):
    return [token for token in doc if token in approved_tokens_set]


class TextPreprocessor:
    def __init__(self, stopword_lang='english', custom_stopwords=None):
        self.stopwords = set(stopwords.words(stopword_lang))
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        self.lemmatizer = WordNetLemmatizer()
        self.id2word = None
        self.corpus = None
        self.tokenized_docs = None
        self.filtered_docs = None
        self.lda_model = None
        self.reduced_vocabulary = None
        self.stats_log = {}

    ### === STEP 1: Author and Category Filtering === ###
    def standardize_authors(self, df):
        df['authors'] = df['authors'].map(
            lambda authlist: [str.upper(str.strip(f"{a[0]}, {a[1]}")) for a in authlist]
        )
        return df

    def filter_by_categories(self, df, allowed_categories: List[str]):
        return df[df['categories'].apply(lambda x: any(cat in x for cat in allowed_categories))]

    ### === STEP 2: Basic Preprocessing === ###
    def preprocess_raw_text(self, df, text_column='text'):
        df['text_processed'] = df[text_column].map(super_simple_preprocess)
        return df

    def lemmatize_all(self, docs):
        with Pool() as pool:
            return pool.map(lemmatize_doc, docs)

    def build_dictionary(self, tokenized_docs):
        self.id2word = corpora.Dictionary(tokenized_docs)
        self.tokenized_docs = tokenized_docs
        self.stats_log['vocab_initial'] = len(self.id2word)

    ### === STEP 3: Frequency and Stopword Filtering === ###
    def compute_doc_frequencies(self, docs):
        doc_freq = defaultdict(int)
        for doc in docs:
            for token in set(doc):
                doc_freq[token] += 1
        return doc_freq

    def apply_frequency_stopword_filter(self, docs, low_thresh=1, high_frac=0.995, extra_stopwords=None):
        ndocs = len(docs)
        doc_freqs = self.compute_doc_frequencies(docs)
        high_thresh = int(ndocs * high_frac)

        all_tokens = set(self.id2word.token2id.keys())
        cut_tokens = set(
            token for token, freq in doc_freqs.items()
            if freq <= low_thresh or freq > high_thresh
        )

        if extra_stopwords:
            cut_tokens.update(extra_stopwords)
        cut_tokens.update(self.stopwords)

        approved_tokens = {t for t in all_tokens if t not in cut_tokens and len(t) > 2}

        # Apply in parallel
        args = [(doc, approved_tokens) for doc in docs]
        with Pool() as pool:
            self.filtered_docs = pool.starmap(filter_doc_with_approved_tokens, args)

        self.id2word = corpora.Dictionary(self.filtered_docs)
        self.stats_log['vocab_filtered'] = len(self.id2word)

    ### === STEP 4: Global LDA and Term Relevancy Filtering === ###
    def train_lda_model(self, num_topics=50, passes=1):
        self.corpus = [self.id2word.doc2bow(doc) for doc in self.filtered_docs]
        self.lda_model = LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.id2word, passes=passes)

    def compute_relevancy_scores(self, lambda_param=0.6, top_n=2000):
        topic_term_matrix = self.lda_model.get_topics()
        vocab = self.id2word

        # Build P(w)
        all_terms = [t for doc in self.filtered_docs for t in doc]
        term_counts = Counter(all_terms)
        total_terms = sum(term_counts.values())
        p_w = {term: count / total_terms for term, count in term_counts.items()}

        # Build relevancy scores
        epsilon = 1e-12
        relevancy_scores = {}
        for topic_id, topic_dist in enumerate(topic_term_matrix):
            for term_id in np.argsort(topic_dist)[::-1]:
                term = vocab[term_id]
                p_w_given_t = max(topic_dist[term_id], epsilon)
                p_w_term = max(p_w.get(term, epsilon), epsilon)

                log_p_w_given_t = np.log(p_w_given_t)
                log_p_w = np.log(p_w_term)
                log_odds = log_p_w_given_t - log_p_w
                relevancy = lambda_param * log_p_w_given_t + (1 - lambda_param) * log_odds
                relevancy_scores[(term, topic_id)] = relevancy

        # Top terms per topic
        top_terms = set()
        for topic_id in range(self.lda_model.num_topics):
            topic_terms = {term: relevancy_scores[(term, topic_id)]
                           for (term, tid) in relevancy_scores if tid == topic_id}
            sorted_terms = sorted(topic_terms.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_terms.update([term for term, _ in sorted_terms])

        self.reduced_vocabulary = top_terms

    def apply_relevancy_filter(self):
        self.filtered_docs = [
            [term for term in doc if term in self.reduced_vocabulary]
            for doc in self.filtered_docs
        ]
        self.filtered_docs = [doc for doc in self.filtered_docs if len(doc) > 0]
        self.id2word = corpora.Dictionary(self.filtered_docs)
        self.corpus = [self.id2word.doc2bow(doc) for doc in self.filtered_docs]
        self.stats_log['vocab_final'] = len(self.id2word)

    ### === STEP 5: Accessors and Utilities === ###
    def drop_empty_rows(self, df, text_col='text_processed'):
        return df[df[text_col].map(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)

    def get_dictionary(self):
        return self.id2word

    def get_corpus(self):
        return self.corpus

    def get_vocab_size(self):
        return len(self.id2word) if self.id2word else 0

    def get_stats_log(self):
        return self.stats_log

    def get_processed_docs(self):
        return self.filtered_docs

    def get_lda_model(self):
        return self.lda_model