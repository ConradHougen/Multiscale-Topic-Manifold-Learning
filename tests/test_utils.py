"""
Test cases for utils.py module
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mstml.utils import (
    AuthEmbedEnum, TopicRelevanceEnum, validate_dataset_name, log_print,
    get_data_int_dir, get_data_clean_dir, get_data_original_dir,
    get_file_path_without_extension, get_file_stem_only,
    hellinger_distance, hellinger_similarity, jensen_shannon_divergence,
    cosine_similarity, preprocess_text, create_author_network
)


class TestEnums:
    """Test cases for enum classes"""
    
    def test_auth_embed_enum(self):
        """Test AuthEmbedEnum contains expected values"""
        # Check that all expected enum values exist
        expected_values = {
            'WORD_FREQ', 'AT_DISTN', 'AT_SPARSIFIED_DISTN',
            'TERM_RELEVANCE_N_HOT', 'TERM_RELEVANCE_VMASK'
        }
        actual_values = {member.name for member in AuthEmbedEnum}
        assert expected_values == actual_values
        
        # Check that values are unique
        values = [member.value for member in AuthEmbedEnum]
        assert len(values) == len(set(values))
    
    def test_topic_relevance_enum(self):
        """Test TopicRelevanceEnum contains expected values"""
        expected_values = {'N_HOT_ENCODING', 'VOCAB_MASK'}
        actual_values = {member.name for member in TopicRelevanceEnum}
        assert expected_values == actual_values
        
        # Check that values are unique
        values = [member.value for member in TopicRelevanceEnum]
        assert len(values) == len(set(values))


class TestDatasetValidation:
    """Test cases for validate_dataset_name function"""
    
    def test_valid_dataset_names(self):
        """Test that valid dataset names pass validation"""
        valid_names = [
            "dataset1",
            "my_dataset",
            "data_set_123",
            "_private_dataset",
            "a",
            "dataset_v2"
        ]
        
        for name in valid_names:
            assert validate_dataset_name(name) is True
    
    def test_invalid_dataset_names(self):
        """Test that invalid dataset names raise ValueError"""
        invalid_names = [
            "Dataset1",      # Capital letters
            "123dataset",    # Starts with number
            "data-set",      # Contains hyphen
            "data set",      # Contains space
            "data.set",      # Contains period
            "",              # Empty string
            "data@set",      # Special character
            "dataSet"        # Mixed case
        ]
        
        for name in invalid_names:
            with pytest.raises(ValueError):
                validate_dataset_name(name)
    
    def test_dataset_name_error_message(self):
        """Test that error messages are informative"""
        with pytest.raises(ValueError) as exc_info:
            validate_dataset_name("Invalid-Name")
        
        error_msg = str(exc_info.value)
        assert "Invalid dataset name" in error_msg
        assert "Invalid-Name" in error_msg
        assert "lowercase letters" in error_msg


class TestLogPrint:
    """Test cases for log_print function"""
    
    def test_log_print_default_behavior(self, capfd):
        """Test log_print with default parameters"""
        message = "Test message"
        log_print(message)
        
        # Check that message was printed to stdout
        captured = capfd.readouterr()
        assert message in captured.out
    
    def test_log_print_with_custom_logger(self):
        """Test log_print with custom logger"""
        mock_logger = Mock()
        message = "Test message"
        
        log_print(message, logger=mock_logger)
        
        mock_logger.info.assert_called_once_with(message)
    
    def test_log_print_different_levels(self):
        """Test log_print with different logging levels"""
        mock_logger = Mock()
        message = "Test message"
        
        # Test different log levels
        levels = ['debug', 'info', 'warning', 'error', 'critical']
        for level in levels:
            mock_logger.reset_mock()
            log_print(message, level=level, logger=mock_logger)
            
            # Check that the appropriate log method was called
            expected_method = getattr(mock_logger, level)
            expected_method.assert_called_once_with(message)
    
    def test_log_print_no_print(self, capfd):
        """Test log_print with also_print=False"""
        message = "Test message"
        mock_logger = Mock()
        
        log_print(message, logger=mock_logger, also_print=False)
        
        # Check that logger was called but nothing printed
        mock_logger.info.assert_called_once_with(message)
        captured = capfd.readouterr()
        assert message not in captured.out
    
    def test_log_print_invalid_level(self):
        """Test log_print with invalid logging level"""
        mock_logger = Mock()
        mock_logger.invalid_level = Mock()
        message = "Test message"
        
        # Should fall back to info level
        log_print(message, level="invalid_level", logger=mock_logger)
        mock_logger.info.assert_called_once_with(message)


class TestDirectoryUtilities:
    """Test cases for directory utility functions"""
    
    def test_get_data_int_dir(self):
        """Test get_data_int_dir function"""
        # Test without subdirectory
        result = get_data_int_dir("testset", None)
        expected = os.path.join("data", "testset", "intermediate/")
        assert result == expected
        
        # Test with subdirectory
        result = get_data_int_dir("testset", "subset")
        expected = os.path.join("data", "testset", "subset", "intermediate/")
        assert result == expected
    
    def test_get_data_clean_dir(self):
        """Test get_data_clean_dir function"""
        # Test without subdirectory
        result = get_data_clean_dir("testset", None)
        expected = os.path.join("data", "testset", "clean/")
        assert result == expected
        
        # Test with subdirectory
        result = get_data_clean_dir("testset", "subset")
        expected = os.path.join("data", "testset", "subset", "clean/")
        assert result == expected
    
    def test_get_data_original_dir(self):
        """Test get_data_original_dir function"""
        # Test without subdirectory
        result = get_data_original_dir("testset", None)
        expected = os.path.join("data", "testset", "original/")
        assert result == expected
        
        # Test with subdirectory
        result = get_data_original_dir("testset", "subset")
        expected = os.path.join("data", "testset", "subset", "original/")
        assert result == expected


class TestFileUtilities:
    """Test cases for file utility functions"""
    
    def test_get_file_path_without_extension(self):
        """Test get_file_path_without_extension function"""
        # Test with various file paths
        test_cases = [
            ("file.txt", "file"),
            ("/path/to/file.txt", "/path/to/file"),
            ("complex.file.name.txt", "complex.file.name"),
            ("no_extension", "no_extension"),
            (".hidden", ""),
            ("path/to/.hidden", "path/to/")
        ]
        
        for input_path, expected in test_cases:
            result = get_file_path_without_extension(input_path)
            assert result == expected
    
    def test_get_file_stem_only(self):
        """Test get_file_stem_only function"""
        # Test with various file paths
        test_cases = [
            ("file.txt", "file"),
            ("/path/to/file.txt", "file"),
            ("complex.file.name.txt", "complex.file.name"),
            ("no_extension", "no_extension"),
            (".hidden", ""),
            ("path/to/.hidden", "")
        ]
        
        for input_path, expected in test_cases:
            result = get_file_stem_only(input_path)
            assert result == expected


class TestDistanceMetrics:
    """Test cases for distance and similarity metrics"""
    
    @pytest.fixture
    def sample_distributions(self):
        """Create sample probability distributions for testing"""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        return p, q
    
    def test_hellinger_distance(self, sample_distributions):
        """Test hellinger_distance function"""
        p, q = sample_distributions
        
        # Test basic computation
        distance = hellinger_distance(p, q)
        assert isinstance(distance, float)
        assert 0 <= distance <= 1  # Hellinger distance is bounded [0,1]
        
        # Test identical distributions
        assert hellinger_distance(p, p) == pytest.approx(0, abs=1e-10)
        
        # Test normalization
        unnormalized_p = np.array([1, 0.6, 0.4])  # Sums to 2
        unnormalized_q = np.array([0.8, 0.8, 0.4])  # Sums to 2
        distance = hellinger_distance(unnormalized_p, unnormalized_q)
        assert isinstance(distance, float)
        assert 0 <= distance <= 1
    
    def test_hellinger_similarity(self, sample_distributions):
        """Test hellinger_similarity function"""
        p, q = sample_distributions
        
        similarity = hellinger_similarity(p, q)
        distance = hellinger_distance(p, q)
        
        # Similarity should be 1 - distance
        assert similarity == pytest.approx(1.0 - distance)
        
        # Test identical distributions
        assert hellinger_similarity(p, p) == pytest.approx(1.0, abs=1e-10)
    
    def test_jensen_shannon_divergence(self, sample_distributions):
        """Test jensen_shannon_divergence function"""
        p, q = sample_distributions
        
        # Test basic computation
        divergence = jensen_shannon_divergence(p, q)
        assert isinstance(divergence, float)
        assert divergence >= 0  # JS divergence is non-negative
        
        # Test identical distributions
        assert jensen_shannon_divergence(p, p) == pytest.approx(0, abs=1e-10)
        
        # Test symmetry: JS(p,q) = JS(q,p)
        js_pq = jensen_shannon_divergence(p, q)
        js_qp = jensen_shannon_divergence(q, p)
        assert js_pq == pytest.approx(js_qp, abs=1e-10)
    
    def test_cosine_similarity(self):
        """Test cosine_similarity function"""
        # Test with sample vectors
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        
        similarity = cosine_similarity(a, b)
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1  # Cosine similarity is bounded [-1,1]
        
        # Test identical vectors
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-10)
        
        # Test orthogonal vectors
        orthogonal_a = np.array([1, 0])
        orthogonal_b = np.array([0, 1])
        assert cosine_similarity(orthogonal_a, orthogonal_b) == pytest.approx(0, abs=1e-10)
        
        # Test zero vectors
        zero_vector = np.array([0, 0, 0])
        assert cosine_similarity(zero_vector, a) == 0.0
        assert cosine_similarity(a, zero_vector) == 0.0


class TestTextPreprocessing:
    """Test cases for text preprocessing functions"""
    
    @patch('mstml.utils.stopwords.words')
    def test_preprocess_text_basic(self, mock_stopwords):
        """Test basic text preprocessing"""
        mock_stopwords.return_value = ['the', 'a', 'an', 'is']
        
        text = "This is a sample text for testing."
        result = preprocess_text(text)
        
        assert isinstance(result, list)
        assert all(isinstance(token, str) for token in result)
        # Should not contain stopwords
        assert 'is' not in result
        assert 'a' not in result
    
    @patch('mstml.utils.stopwords.words')
    def test_preprocess_text_no_stopwords(self, mock_stopwords):
        """Test preprocessing without stopword removal"""
        mock_stopwords.return_value = ['the', 'a', 'an', 'is']
        
        text = "This is a test."
        result = preprocess_text(text, remove_stopwords=False)
        
        # Should contain stopwords when remove_stopwords=False
        assert 'is' in result
        assert 'a' in result
    
    @patch('mstml.utils.WordNetLemmatizer')
    def test_preprocess_text_no_lemmatization(self, mock_lemmatizer):
        """Test preprocessing without lemmatization"""
        text = "running cats"
        result = preprocess_text(text, lemmatize=False)
        
        # Lemmatizer should not be used
        mock_lemmatizer.assert_not_called()
        assert 'running' in result
        assert 'cats' in result
    
    def test_preprocess_text_empty_string(self):
        """Test preprocessing of empty string"""
        result = preprocess_text("")
        assert result == []


class TestNetworkUtilities:
    """Test cases for network utility functions"""
    
    def test_create_author_network_basic(self):
        """Test basic author network creation"""
        collaborations = [
            ["Alice", "Bob"],
            ["Bob", "Charlie"],
            ["Alice", "Charlie", "David"]
        ]
        
        G = create_author_network(collaborations)
        
        # Check that result is a NetworkX graph
        assert isinstance(G, nx.Graph)
        
        # Check that all authors are nodes
        expected_authors = {"Alice", "Bob", "Charlie", "David"}
        assert set(G.nodes()) == expected_authors
        
        # Check that collaborations create edges
        assert G.has_edge("Alice", "Bob")
        assert G.has_edge("Bob", "Charlie")
        assert G.has_edge("Alice", "Charlie")
        assert G.has_edge("Alice", "David")
        assert G.has_edge("Charlie", "David")
    
    def test_create_author_network_empty(self):
        """Test network creation with empty input"""
        G = create_author_network([])
        assert isinstance(G, nx.Graph)
        assert len(G.nodes()) == 0
        assert len(G.edges()) == 0
    
    def test_create_author_network_single_authors(self):
        """Test network creation with single authors"""
        collaborations = [["Alice"], ["Bob"]]
        G = create_author_network(collaborations)
        
        # Should have nodes but no edges
        assert "Alice" in G.nodes()
        assert "Bob" in G.nodes()
        assert len(G.edges()) == 0


class TestDistanceMetricsEdgeCases:
    """Test edge cases for distance metrics"""
    
    def test_hellinger_distance_zero_vectors(self):
        """Test Hellinger distance with zero vectors"""
        zero_vec = np.array([0, 0, 0])
        normal_vec = np.array([0.5, 0.3, 0.2])
        
        # Should handle zero vectors gracefully
        distance = hellinger_distance(zero_vec, normal_vec)
        assert isinstance(distance, float)
        assert distance >= 0

    def test_jensen_shannon_divergence_zero_vectors(self):
        """Test JS divergence with zero vectors should raise"""
        zero_vec = np.array([0, 0, 0])
        normal_vec = np.array([0.5, 0.3, 0.2])

        with pytest.raises(ValueError):
            jensen_shannon_divergence(zero_vec, normal_vec)

    
    def test_distance_metrics_with_single_element(self):
        """Test distance metrics with single-element arrays"""
        p = np.array([1.0])
        q = np.array([1.0])
        
        assert hellinger_distance(p, q) == pytest.approx(0, abs=1e-10)
        assert jensen_shannon_divergence(p, q) == pytest.approx(0, abs=1e-10)
        assert cosine_similarity(p, q) == pytest.approx(1.0, abs=1e-10)


class TestUtilityFunctionIntegration:
    """Integration tests for utility functions"""
    
    def test_directory_path_consistency(self):
        """Test that directory utility functions are consistent"""
        dataset = "test_dataset"
        subdataset = "subset"
        
        # All functions should use consistent path structure
        int_dir = get_data_int_dir(dataset, subdataset)
        clean_dir = get_data_clean_dir(dataset, subdataset)
        orig_dir = get_data_original_dir(dataset, subdataset)
        
        # Check that all paths start with the same base
        base_path = os.path.join("data", dataset, subdataset)
        assert int_dir.startswith(base_path)
        assert clean_dir.startswith(base_path)
        assert orig_dir.startswith(base_path)
        
        # Check that they end with the correct subdirectories
        assert int_dir.endswith("intermediate/")
        assert clean_dir.endswith("clean/")
        assert orig_dir.endswith("original/")
    
    def test_file_utilities_consistency(self):
        """Test that file utility functions work together"""
        test_path = "/path/to/complex.file.name.txt"
        
        # Test that functions produce consistent results
        without_ext = get_file_path_without_extension(test_path)
        stem_only = get_file_stem_only(test_path)
        
        assert without_ext == "/path/to/complex.file.name"
        assert stem_only == "complex.file.name"
        
        # Stem should be the basename of the path without extension
        expected_stem = os.path.basename(without_ext)
        assert stem_only == expected_stem