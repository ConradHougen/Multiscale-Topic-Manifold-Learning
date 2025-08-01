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
from ._file_driver import log_print


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
    
    ### === PIPELINE WRAPPER METHOD === ###
    def update_dataframe(self, df, text_column, 
                        standardize_authors=False, 
                        allowed_categories=None,
                        low_thresh=1, 
                        high_frac=0.995, 
                        extra_stopwords=None,
                        train_lda=True,
                        num_topics=50,
                        lda_passes=1,
                        lambda_param=0.6,
                        top_n_terms=2000):
        """
        Main wrapper method for complete text preprocessing pipeline.
        Called by DataLoader to preprocess text data following the established flow.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of column containing raw text
            standardize_authors: Whether to standardize author names
            allowed_categories: List of allowed categories for filtering
            low_thresh: Minimum document frequency threshold
            high_frac: Maximum document frequency fraction
            extra_stopwords: Additional stopwords to remove
            train_lda: Whether to train LDA model for relevancy filtering
            num_topics: Number of topics for LDA model
            lda_passes: Number of passes for LDA training
            lambda_param: Lambda parameter for relevancy scoring
            top_n_terms: Number of top relevant terms to keep
        
        Returns:
            Series of preprocessed tokenized documents (lists of tokens)
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
        
        log_print("Starting text preprocessing pipeline", level="info")
        
        ### === STEP 1: Author and Category Filtering === ###
        if standardize_authors and 'authors' in df.columns:
            df = self.standardize_authors(df)
            log_print("Applied author name standardization", level="info")
            
        if allowed_categories and 'categories' in df.columns:
            df = self.filter_by_categories(df, allowed_categories)
            log_print(f"Filtered to {len(allowed_categories)} allowed categories", level="info")
        
        ### === STEP 2: Basic Preprocessing === ###
        # Preprocess raw text (tokenization) - modifies df in place
        log_print("Tokenizing documents", level="info")
        df = self.preprocess_raw_text(df, text_column)
        
        # Extract tokenized documents for further processing
        tokenized_docs = df['text_processed'].tolist()
        
        # Apply lemmatization in parallel
        log_print("Applying lemmatization using multiprocessing", level="info")
        tokenized_docs = self.lemmatize_all(tokenized_docs)
        
        # Build initial dictionary
        log_print("Building initial vocabulary dictionary", level="info")
        self.build_dictionary(tokenized_docs)
        
        ### === STEP 3: Frequency and Stopword Filtering === ###
        log_print(f"Applying frequency filtering (low_thresh={low_thresh}, high_frac={high_frac})", level="info")
        self.apply_frequency_stopword_filter(
            tokenized_docs, 
            low_thresh=low_thresh, 
            high_frac=high_frac, 
            extra_stopwords=extra_stopwords
        )
        
        ### === STEP 4: Global LDA and Term Relevancy Filtering === ###
        if train_lda:
            # Train LDA model with progress logging
            log_print(f"Training LDA model with {num_topics} topics and {lda_passes} passes - this may take a moment...", level="info")
            self.train_lda_model(num_topics=num_topics, passes=lda_passes)
            log_print("LDA model training completed", level="info")
            
            # Compute relevancy scores and get reduced vocabulary
            log_print(f"Computing term relevancy scores (λ={lambda_param}, top_n={top_n_terms})", level="info")
            self.compute_relevancy_scores(lambda_param=lambda_param, top_n=top_n_terms)
            
            # Apply relevancy filter
            log_print("Applying relevancy-based vocabulary reduction", level="info")
            self.apply_relevancy_filter()
        
        ### === STEP 5: Final Cleanup === ###
        log_print("Performing final cleanup and dictionary rebuild", level="info")
        # Create temporary dataframe with processed text for dropping empty rows
        temp_df = pd.DataFrame({'text_processed': self.filtered_docs})
        temp_df = self.drop_empty_rows(temp_df, 'text_processed')
        
        # Update filtered docs to only include non-empty documents
        self.filtered_docs = temp_df['text_processed'].tolist()
        
        # Rebuild dictionary one final time
        self.id2word = corpora.Dictionary(self.filtered_docs)
        self.corpus = [self.id2word.doc2bow(doc) for doc in self.filtered_docs]
        self.stats_log['vocab_final'] = len(self.id2word)
        
        # Return processed documents, ensuring we match the original dataframe indices
        # Handle case where filtering may have removed some documents
        if len(self.filtered_docs) == len(df):
            # No documents were removed during processing
            log_print(f"Text preprocessing completed successfully for {len(self.filtered_docs)} documents", level="info")
            return pd.Series(self.filtered_docs, index=df.index)
        else:
            # Some documents were removed - this is more complex to handle
            # For now, return what we have and log a warning
            log_print(f"Document count changed during preprocessing: {len(df)} -> {len(self.filtered_docs)}", level="warning")
            # Pad or truncate to match original length
            if len(self.filtered_docs) < len(df):
                padded_docs = self.filtered_docs + [[] for _ in range(len(df) - len(self.filtered_docs))]
                return pd.Series(padded_docs, index=df.index)
            else:
                return pd.Series(self.filtered_docs[:len(df)], index=df.index)