"""
Text processing utilities for MSTML.

This module provides comprehensive text preprocessing functionality
including tokenization, cleaning, and preparation for topic modeling.
"""

import re
import nltk
import pandas as pd
import numpy as np
from collections import Counter
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Optional, Union

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class TextProcessor:
    """
    Comprehensive text processing class for academic documents.
    """
    
    def __init__(self, 
                 language='english',
                 min_word_length=2,
                 max_word_length=50,
                 remove_numbers=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 custom_stopwords=None):
        """
        Initialize text processor.
        
        Args:
            language: Language for stopwords
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            remove_numbers: Whether to remove numeric tokens
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            custom_stopwords: Additional stopwords to remove
        """
        self.language = language
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize stopwords
        if remove_stopwords:
            self.stopwords = set(stopwords.words(language))
            if custom_stopwords:
                self.stopwords.update(custom_stopwords)
        else:
            self.stopwords = set()
            
        # Initialize lemmatizer
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing unwanted characters and formatting.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Use gensim's simple_preprocess for robust tokenization
        tokens = simple_preprocess(text, deacc=True, min_len=self.min_word_length, max_len=self.max_word_length)
        
        # Remove numbers if specified
        if self.remove_numbers:
            tokens = [token for token in tokens if not token.isdigit()]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def process_text(self, text: str) -> List[str]:
        """
        Complete text processing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            List of processed tokens
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        return tokens
    
    def process_documents(self, documents: List[str], show_progress=True) -> List[List[str]]:
        """
        Process multiple documents.
        
        Args:
            documents: List of text documents
            show_progress: Whether to show progress bar
            
        Returns:
            List of tokenized documents
        """
        processed_docs = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(documents, desc="Processing documents")
            except ImportError:
                iterator = documents
                print(f"Processing {len(documents)} documents...")
        else:
            iterator = documents
        
        for doc in iterator:
            processed_docs.append(self.process_text(doc))
        
        return processed_docs
    
    def filter_by_frequency(self, 
                          tokenized_docs: List[List[str]], 
                          min_freq: int = 2, 
                          max_freq_ratio: float = 0.8) -> List[List[str]]:
        """
        Filter tokens by document frequency.
        
        Args:
            tokenized_docs: List of tokenized documents
            min_freq: Minimum frequency threshold
            max_freq_ratio: Maximum frequency ratio (0-1)
            
        Returns:
            Filtered tokenized documents
        """
        # Count document frequencies
        doc_freq = Counter()
        for doc in tokenized_docs:
            unique_tokens = set(doc)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        total_docs = len(tokenized_docs)
        max_freq = int(total_docs * max_freq_ratio)
        
        # Filter tokens
        valid_tokens = set()
        for token, freq in doc_freq.items():
            if min_freq <= freq <= max_freq:
                valid_tokens.add(token)
        
        # Apply filter to documents
        filtered_docs = []
        for doc in tokenized_docs:
            filtered_doc = [token for token in doc if token in valid_tokens]
            filtered_docs.append(filtered_doc)
        
        return filtered_docs
    
    def create_vocabulary(self, tokenized_docs: List[List[str]]) -> Dict[str, int]:
        """
        Create vocabulary mapping from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            
        Returns:
            Dictionary mapping tokens to IDs
        """
        vocab = set()
        for doc in tokenized_docs:
            vocab.update(doc)
        
        # Sort for consistent ordering
        sorted_vocab = sorted(vocab)
        vocab_dict = {token: idx for idx, token in enumerate(sorted_vocab)}
        
        return vocab_dict
    
    def get_statistics(self, tokenized_docs: List[List[str]]) -> Dict:
        """
        Get statistics about processed documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            
        Returns:
            Dictionary of statistics
        """
        if not tokenized_docs:
            return {}
        
        doc_lengths = [len(doc) for doc in tokenized_docs]
        all_tokens = [token for doc in tokenized_docs for token in doc]
        token_counts = Counter(all_tokens)
        
        stats = {
            'num_documents': len(tokenized_docs),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(token_counts),
            'avg_doc_length': np.mean(doc_lengths),
            'median_doc_length': np.median(doc_lengths),
            'min_doc_length': min(doc_lengths),
            'max_doc_length': max(doc_lengths),
            'vocabulary_size': len(set(all_tokens)),
            'most_common_tokens': token_counts.most_common(10)
        }
        
        return stats
    
    def extract_ngrams(self, 
                      tokenized_docs: List[List[str]], 
                      n: int = 2, 
                      min_freq: int = 2) -> List[str]:
        """
        Extract n-grams from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            n: N-gram size
            min_freq: Minimum frequency threshold
            
        Returns:
            List of frequent n-grams
        """
        ngram_counts = Counter()
        
        for doc in tokenized_docs:
            if len(doc) >= n:
                for i in range(len(doc) - n + 1):
                    ngram = '_'.join(doc[i:i+n])
                    ngram_counts[ngram] += 1
        
        # Filter by frequency
        frequent_ngrams = [ngram for ngram, count in ngram_counts.items() if count >= min_freq]
        
        return frequent_ngrams
    
    def add_custom_stopwords(self, stopwords_list: List[str]):
        """Add custom stopwords to the existing set."""
        self.stopwords.update(stopwords_list)
    
    def remove_short_documents(self, 
                             tokenized_docs: List[List[str]], 
                             min_length: int = 5) -> List[List[str]]:
        """
        Remove documents that are too short.
        
        Args:
            tokenized_docs: List of tokenized documents
            min_length: Minimum document length
            
        Returns:
            Filtered list of documents
        """
        return [doc for doc in tokenized_docs if len(doc) >= min_length]


def create_academic_text_processor():
    """
    Create a text processor optimized for academic documents.
    
    Returns:
        TextProcessor configured for academic texts
    """
    # Academic-specific stopwords
    academic_stopwords = [
        'paper', 'study', 'research', 'analysis', 'method', 'approach',
        'result', 'conclusion', 'discussion', 'introduction', 'abstract',
        'figure', 'table', 'section', 'chapter', 'article', 'journal',
        'conference', 'proceedings', 'university', 'department', 'et', 'al'
    ]
    
    return TextProcessor(
        min_word_length=3,
        max_word_length=30,
        remove_numbers=True,
        remove_stopwords=True,
        lemmatize=True,
        custom_stopwords=academic_stopwords
    )