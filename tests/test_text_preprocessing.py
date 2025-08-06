"""
Test cases for text_preprocessing.py module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mstml.text_preprocessing import (
    TextPreprocessor, super_simple_preprocess, lemmatize_doc,
    filter_doc_with_approved_tokens
)


class TestUtilityFunctions:
    """Test standalone utility functions"""
    
    def test_super_simple_preprocess(self):
        """Test super_simple_preprocess function"""
        # Normal text
        result = super_simple_preprocess("Hello World! This is a test.")
        expected = ['hello', 'world', 'this', 'is', 'test']
        assert result == expected
        
        # Text with numbers and special characters
        result = super_simple_preprocess("Test123 with @#$% symbols & numbers 456")
        expected = ['test', 'with', 'symbols', 'numbers']  # Numbers and symbols removed
        assert result == expected
        
        # Empty string
        result = super_simple_preprocess("")
        assert result == []
        
        # None input (converted to string)
        result = super_simple_preprocess(None)
        assert result == ['none']
    
    def test_lemmatize_doc(self):
        """Test lemmatize_doc function"""
        # Test with various word forms
        doc = ['running', 'cats', 'better', 'studies']
        result = lemmatize_doc(doc)
        expected = ['running', 'cat', 'better', 'study']  # Expected lemmatized forms
        assert result == expected
        
        # Empty document
        result = lemmatize_doc([])
        assert result == []
    
    def test_filter_doc_with_approved_tokens(self):
        """Test filter_doc_with_approved_tokens function"""
        doc = ['apple', 'banana', 'cherry', 'date']
        approved_tokens = {'apple', 'cherry', 'elderberry'}
        
        result = filter_doc_with_approved_tokens(doc, approved_tokens)
        expected = ['apple', 'cherry']
        assert result == expected
        
        # Empty approved tokens
        result = filter_doc_with_approved_tokens(doc, set())
        assert result == []
        
        # Empty document
        result = filter_doc_with_approved_tokens([], approved_tokens)
        assert result == []


class TestTextPreprocessorInit:
    """Test TextPreprocessor initialization"""
    
    def test_default_initialization(self):
        """Test TextPreprocessor with default parameters"""
        processor = TextPreprocessor()
        
        assert processor.stopwords is not None
        assert len(processor.stopwords) > 0  # Should have English stopwords
        assert processor.lemmatizer is not None
        assert processor.id2word is None
        assert processor.corpus is None
        assert processor.tokenized_docs is None
        assert processor.filtered_docs is None
        assert processor.lda_model is None
        assert processor.reduced_vocabulary is None
        assert processor.stats_log == {}
    
    def test_custom_stopwords_initialization(self):
        """Test TextPreprocessor with custom stopwords"""
        custom_stopwords = {'custom', 'words', 'to', 'remove'}
        processor = TextPreprocessor(custom_stopwords=custom_stopwords)
        
        # Should include both English stopwords and custom ones
        assert 'custom' in processor.stopwords
        assert 'words' in processor.stopwords
        assert 'the' in processor.stopwords  # Default English stopword
    
    def test_different_language_stopwords(self):
        """Test TextPreprocessor with different language stopwords"""
        processor = TextPreprocessor(stopword_lang='spanish')
        
        # Should have Spanish stopwords (if available)
        assert processor.stopwords is not None


class TestTextPreprocessorBasicMethods:
    """Test basic preprocessing methods"""
    
    @pytest.fixture
    def processor(self):
        """Create TextPreprocessor instance for testing"""
        return TextPreprocessor()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'text': [
                'This is the first document about machine learning.',
                'Second document discusses deep learning algorithms.',
                'Third text covers natural language processing.'
            ],
            'authors': [
                [['John', 'Doe'], ['Jane', 'Smith']],
                [['Alice', 'Johnson']],
                [['Bob', 'Wilson'], ['Carol', 'Brown']]
            ],
            'categories': [
                ['cs.ML', 'cs.AI'],
                ['cs.LG'],
                ['cs.CL', 'cs.AI']
            ]
        })
    
    def test_standardize_authors(self, processor, sample_df):
        """Test standardize_authors method"""
        result_df = processor.standardize_authors(sample_df)
        
        # Check that authors are properly formatted
        expected_first = ['DOE, JOHN', 'SMITH, JANE']
        assert result_df.iloc[0]['authors'] == expected_first
        
        expected_second = ['JOHNSON, ALICE']
        assert result_df.iloc[1]['authors'] == expected_second
    
    def test_filter_by_categories(self, processor, sample_df):
        """Test filter_by_categories method"""
        # Filter for AI-related categories
        filtered_df = processor.filter_by_categories(sample_df, ['cs.AI'])
        
        # Should return rows 0 and 2 (which contain cs.AI)
        assert len(filtered_df) == 2
        assert 'cs.AI' in filtered_df.iloc[0]['categories']
        assert 'cs.AI' in filtered_df.iloc[1]['categories']
        
        # Filter for non-existent category
        filtered_df = processor.filter_by_categories(sample_df, ['cs.XX'])
        assert len(filtered_df) == 0
    
    def test_preprocess_raw_text(self, processor, sample_df):
        """Test preprocess_raw_text method"""
        from mstml.dataframe_schema import MainDataSchema
        
        result_df = processor.preprocess_raw_text(sample_df, text_column='text', target_column=MainDataSchema.PREPROCESSED_TEXT.colname)
        
        # Check that preprocessed_text column was added
        assert MainDataSchema.PREPROCESSED_TEXT.colname in result_df.columns
        
        # Check that text was properly preprocessed
        first_processed = result_df.iloc[0][MainDataSchema.PREPROCESSED_TEXT.colname]
        assert isinstance(first_processed, list)
        assert 'this' in first_processed
        assert 'machine' in first_processed
        assert 'learning' in first_processed
    
    @patch('mstml.text_preprocessing.Pool')
    def test_lemmatize_all(self, mock_pool, processor):
        """Test lemmatize_all method with mocked multiprocessing"""
        # Mock the pool and map method
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [['lemmatized', 'doc1'], ['lemmatized', 'doc2']]
        
        docs = [['running', 'cats'], ['better', 'studies']]
        result = processor.lemmatize_all(docs)
        
        # Check that pool.map was called correctly
        mock_pool_instance.map.assert_called_once()
        assert result == [['lemmatized', 'doc1'], ['lemmatized', 'doc2']]
    
    def test_build_dictionary(self, processor):
        """Test build_dictionary method"""
        tokenized_docs = [
            ['word1', 'word2', 'word3'],
            ['word2', 'word3', 'word4'],
            ['word1', 'word4', 'word5']
        ]
        
        processor.build_dictionary(tokenized_docs)
        
        # Check that dictionary was created
        assert processor.id2word is not None
        assert processor.tokenized_docs == tokenized_docs
        assert 'vocab_initial' in processor.stats_log
        assert processor.stats_log['vocab_initial'] > 0


class TestTextPreprocessorFrequencyFiltering:
    """Test frequency and stopword filtering methods"""
    
    @pytest.fixture
    def processor(self):
        """Create TextPreprocessor instance for testing"""
        return TextPreprocessor()
    
    def test_compute_doc_frequencies(self, processor):
        """Test compute_doc_frequencies method"""
        docs = [
            ['word1', 'word2', 'word1'],  # word1 appears twice in doc
            ['word2', 'word3'],
            ['word1', 'word3', 'word4']
        ]
        
        doc_freqs = processor.compute_doc_frequencies(docs)
        
        # Check document frequencies (not term frequencies)
        assert doc_freqs['word1'] == 2  # Appears in 2 documents
        assert doc_freqs['word2'] == 2  # Appears in 2 documents  
        assert doc_freqs['word3'] == 2  # Appears in 2 documents
        assert doc_freqs['word4'] == 1  # Appears in 1 document
    
    @patch('mstml.text_preprocessing.Pool')
    def test_apply_frequency_stopword_filter(self, mock_pool, processor):
        """Test apply_frequency_stopword_filter method"""
        # Setup mock dictionary
        mock_dict = Mock()
        mock_dict.token2id = {'word1': 0, 'word2': 1, 'word3': 2, 'the': 3, 'and': 4}
        processor.id2word = mock_dict
        
        # Mock the pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [['word1'], ['word2'], ['word1', 'word3']]
        
        docs = [
            ['word1', 'the', 'and'],
            ['word2', 'the'],
            ['word1', 'word3', 'the']
        ]
        
        processor.apply_frequency_stopword_filter(docs, low_thresh=1, high_frac=0.8)
        
        # Check that filtered_docs was set
        assert processor.filtered_docs == [['word1'], ['word2'], ['word1', 'word3']]
        assert 'vocab_filtered' in processor.stats_log


class TestTextPreprocessorLDAMethods:
    """Test LDA-related methods"""
    
    @pytest.fixture
    def processor_with_docs(self):
        """Create TextPreprocessor with sample filtered docs"""
        processor = TextPreprocessor()
        processor.filtered_docs = [
            ['machine', 'learning', 'algorithm'],
            ['deep', 'learning', 'neural'],
            ['natural', 'language', 'processing'],
            ['machine', 'learning', 'data']
        ]
        # Mock dictionary
        from gensim import corpora
        processor.id2word = corpora.Dictionary(processor.filtered_docs)
        return processor
    
    def test_train_lda_model(self, processor_with_docs):
        """Test train_lda_model method"""
        processor_with_docs.train_lda_model(num_topics=2, passes=1)
        
        # Check that LDA model was created
        assert processor_with_docs.lda_model is not None
        assert processor_with_docs.corpus is not None
        assert len(processor_with_docs.corpus) == len(processor_with_docs.filtered_docs)
    
    def test_compute_relevancy_scores(self, processor_with_docs):
        """Test compute_relevancy_scores method"""
        # First train LDA model
        processor_with_docs.train_lda_model(num_topics=2, passes=1)
        
        # Then compute relevancy scores
        processor_with_docs.compute_relevancy_scores(lambda_param=0.6, top_n=3)
        
        # Check that reduced vocabulary was created
        assert processor_with_docs.reduced_vocabulary is not None
        assert isinstance(processor_with_docs.reduced_vocabulary, set)
        assert len(processor_with_docs.reduced_vocabulary) > 0
    
    def test_apply_relevancy_filter(self, processor_with_docs):
        """Test apply_relevancy_filter method"""
        # Setup reduced vocabulary
        processor_with_docs.reduced_vocabulary = {'machine', 'learning', 'data'}
        
        processor_with_docs.apply_relevancy_filter()
        
        # Check that documents were filtered
        assert processor_with_docs.filtered_docs is not None
        assert 'vocab_final' in processor_with_docs.stats_log
        
        # Check that only approved terms remain
        for doc in processor_with_docs.filtered_docs:
            for term in doc:
                assert term in processor_with_docs.reduced_vocabulary


class TestTextPreprocessorUtilityMethods:
    """Test utility and accessor methods"""
    
    @pytest.fixture
    def processor(self):
        """Create TextPreprocessor instance for testing"""
        return TextPreprocessor()
    
    def test_drop_empty_rows(self, processor):
        """Test drop_empty_rows method"""
        from mstml.dataframe_schema import MainDataSchema
        
        df = pd.DataFrame({
            MainDataSchema.PREPROCESSED_TEXT.colname: [
                ['word1', 'word2'],
                [],  # Empty list
                ['word3'],
                None,  # None value
                ['word4', 'word5']
            ],
            'other_col': [1, 2, 3, 4, 5]
        })
        
        result_df = processor.drop_empty_rows(df, MainDataSchema.PREPROCESSED_TEXT.colname)
        
        # Should only keep rows with non-empty lists
        assert len(result_df) == 3
        assert result_df.iloc[0]['other_col'] == 1
        assert result_df.iloc[1]['other_col'] == 3
        assert result_df.iloc[2]['other_col'] == 5
    
    def test_get_dictionary(self, processor):
        """Test get_dictionary method"""
        # Initially None
        assert processor.get_dictionary() is None
        
        # After setting
        mock_dict = Mock()
        processor.id2word = mock_dict
        assert processor.get_dictionary() == mock_dict
    
    def test_get_corpus(self, processor):
        """Test get_corpus method"""
        # Initially None
        assert processor.get_corpus() is None
        
        # After setting
        mock_corpus = [[(0, 1), (1, 2)], [(2, 1)]]
        processor.corpus = mock_corpus
        assert processor.get_corpus() == mock_corpus
    
    def test_get_vocab_size(self, processor):
        """Test get_vocab_size method"""
        # Initially 0 (no dictionary)
        assert processor.get_vocab_size() == 0
        
        # After setting dictionary
        mock_dict = Mock()
        mock_dict.__len__ = Mock(return_value=100)
        processor.id2word = mock_dict
        assert processor.get_vocab_size() == 100
    
    def test_get_stats_log(self, processor):
        """Test get_stats_log method"""
        # Initially empty
        assert processor.get_stats_log() == {}
        
        # After adding stats
        processor.stats_log['test_stat'] = 42
        assert processor.get_stats_log()['test_stat'] == 42
    
    def test_get_processed_docs(self, processor):
        """Test get_processed_docs method"""
        # Initially None
        assert processor.get_processed_docs() is None
        
        # After setting
        docs = [['word1', 'word2'], ['word3', 'word4']]
        processor.filtered_docs = docs
        assert processor.get_processed_docs() == docs
    
    def test_get_lda_model(self, processor):
        """Test get_lda_model method"""
        # Initially None
        assert processor.get_lda_model() is None
        
        # After setting
        mock_model = Mock()
        processor.lda_model = mock_model
        assert processor.get_lda_model() == mock_model


class TestTextPreprocessorIntegration:
    """Integration tests for TextPreprocessor"""
    
    @pytest.fixture
    def sample_texts(self):
        """Create sample text data for integration testing"""
        return [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Data science combines statistics with computer science."
        ]
    
    def test_basic_preprocessing_pipeline(self, sample_texts):
        """Test basic preprocessing pipeline without LDA"""
        processor = TextPreprocessor()
        
        # Process texts
        tokenized_docs = [super_simple_preprocess(text) for text in sample_texts]
        processor.build_dictionary(tokenized_docs)
        
        # Apply basic filtering
        processor.apply_frequency_stopword_filter(
            tokenized_docs, 
            low_thresh=0,  # Keep all words
            high_frac=1.0   # Don't filter high-frequency words
        )
        
        # Check results
        assert processor.filtered_docs is not None
        assert len(processor.filtered_docs) == len(sample_texts)
        assert processor.stats_log['vocab_initial'] > 0
        assert processor.stats_log['vocab_filtered'] > 0
    
    def test_update_dataframe_method_not_implemented(self):
        """Test that update_dataframe method is not implemented in base class"""
        processor = TextPreprocessor()
        df = pd.DataFrame({'text': ['sample text']})
        
        # This method is not implemented in the provided code
        # This test documents the expected behavior
        with pytest.raises(AttributeError):
            processor.update_dataframe(df, 'text')


@pytest.fixture
def mock_nltk_data():
    """Mock NLTK data to avoid download requirements in tests"""
    with patch('nltk.corpus.stopwords.words') as mock_stopwords:
        mock_stopwords.return_value = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        ]
        yield


class TestTextPreprocessorWithMockedNLTK:
    """Test TextPreprocessor with mocked NLTK components"""
    
    def test_initialization_with_mocked_stopwords(self, mock_nltk_data):
        """Test initialization with mocked NLTK stopwords"""
        processor = TextPreprocessor()
        
        # Should have mocked stopwords
        assert 'the' in processor.stopwords
        assert 'and' in processor.stopwords
        assert len(processor.stopwords) > 0