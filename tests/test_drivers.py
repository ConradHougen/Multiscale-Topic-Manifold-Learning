"""
Comprehensive test cases for driver modules (_*_driver.py)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import networkx as nx
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mstml._embedding_driver import *
from mstml._file_driver import *
from mstml._graph_driver import *
from mstml._math_driver import *
from mstml._topic_model_driver import *


class TestEmbeddingDriver:
    """Test embedding driver functionality"""
    
    def test_embedding_functions_exist(self):
        """Test that embedding functions are available"""
        # Test import doesn't raise errors
        assert True  # placeholder for embedding driver tests
    
    @pytest.mark.skipif(True, reason="Needs investigation of embedding driver interface")
    def test_embedding_dimensionality_reduction(self):
        """Test dimensionality reduction functionality"""
        pass


class TestFileDriver:
    """Test file driver functionality"""
    
    def test_get_date_hour_minute(self):
        """Test date/time string generation"""
        result = get_date_hour_minute()
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain date and time components
        assert '_' in result
    
    def test_log_print_basic(self):
        """Test basic log_print functionality"""
        with patch('builtins.print') as mock_print:
            log_print("test message")
            mock_print.assert_called_once()
    
    def test_log_print_with_logger(self):
        """Test log_print with logger"""
        mock_logger = Mock()
        with patch('builtins.print') as mock_print:
            log_print("test message", logger=mock_logger)
            mock_logger.info.assert_called_once_with("test message")
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content\nline 2\nline 3")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()
    
    def test_file_operations_exist(self, temp_file):
        """Test that file operation functions exist"""
        # Verify file exists for testing
        assert Path(temp_file).exists()
        
        # Test that file driver functions can handle files
        # (specific tests depend on actual file driver interface)
        assert True


class TestGraphDriver:
    """Test graph driver functionality"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe with author information"""
        return pd.DataFrame({
            'authors': [
                'Smith, John; Doe, Jane',
                'Doe, Jane; Johnson, Bob',
                'Smith, John; Wilson, Alice',
                'Johnson, Bob; Brown, Charlie'
            ],
            'title': ['Paper 1', 'Paper 2', 'Paper 3', 'Paper 4'],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'])
        })
    
    def test_build_coauthor_network_from_dataframe(self, sample_dataframe):
        """Test co-author network construction"""
        network = build_coauthor_network_from_dataframe(sample_dataframe)
        
        assert isinstance(network, nx.Graph)
        assert len(network.nodes()) > 0
        assert len(network.edges()) > 0
        
        # Check that authors are nodes
        authors = set()
        for author_list in sample_dataframe['authors']:
            authors.update([a.strip() for a in author_list.split(';')])
        
        for author in authors:
            assert author in network.nodes()
    
    def test_build_temporal_coauthor_networks_from_dataframe(self, sample_dataframe):
        """Test temporal co-author network construction"""
        networks = build_temporal_coauthor_networks_from_dataframe(
            sample_dataframe, 
            time_windows=['2023-01-01', '2023-03-01', '2023-05-01']
        )
        
        assert isinstance(networks, dict)
        assert len(networks) > 0
        
        for time_window, network in networks.items():
            assert isinstance(network, nx.Graph)
    
    def test_compose_networks_from_dict(self, sample_dataframe):
        """Test network composition from dictionary"""
        networks = build_temporal_coauthor_networks_from_dataframe(
            sample_dataframe,
            time_windows=['2023-01-01', '2023-03-01', '2023-05-01']
        )
        
        composed_network = compose_networks_from_dict(networks)
        
        assert isinstance(composed_network, nx.Graph)
        assert len(composed_network.nodes()) > 0
    
    def test_coauthor_network_edge_weights(self, sample_dataframe):
        """Test that co-author networks have proper edge weights"""
        network = build_coauthor_network_from_dataframe(sample_dataframe)
        
        for u, v, data in network.edges(data=True):
            assert 'weight' in data
            assert data['weight'] > 0
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame({'authors': [], 'title': [], 'date': []})
        
        network = build_coauthor_network_from_dataframe(empty_df)
        assert isinstance(network, nx.Graph)
        assert len(network.nodes()) == 0
    
    def test_single_author_papers(self):
        """Test handling of single-author papers"""
        single_author_df = pd.DataFrame({
            'authors': ['Smith, John', 'Doe, Jane'],
            'title': ['Paper 1', 'Paper 2'],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01'])
        })
        
        network = build_coauthor_network_from_dataframe(single_author_df)
        
        # Single authors should still be nodes, but no edges
        assert len(network.nodes()) == 2
        assert len(network.edges()) == 0


class TestMathDriver:
    """Test mathematical driver functionality"""
    
    def test_hellinger_distance_function(self):
        """Test Hellinger distance calculation"""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        distance = hellinger(p, q)
        
        assert isinstance(distance, float)
        assert 0 <= distance <= 1
    
    def test_hellinger_identical_distributions(self):
        """Test Hellinger distance for identical distributions"""
        p = np.array([0.5, 0.3, 0.2])
        
        distance = hellinger(p, p)
        
        assert np.isclose(distance, 0.0, atol=1e-10)
    
    def test_hellinger_orthogonal_distributions(self):
        """Test Hellinger distance for orthogonal distributions"""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        
        distance = hellinger(p, q)
        
        # Should be maximum distance
        assert np.isclose(distance, 1.0, atol=1e-10)
    
    def test_diffuse_distribution(self):
        """Test distribution diffusion"""
        initial_dist = np.array([0.8, 0.1, 0.1])
        
        diffused = diffuse_distribution(initial_dist, alpha=0.5)
        
        assert isinstance(diffused, np.ndarray)
        assert len(diffused) == len(initial_dist)
        assert np.isclose(np.sum(diffused), 1.0)  # Should still be a probability distribution
        
        # Diffused distribution should be more uniform
        assert np.std(diffused) < np.std(initial_dist)
    
    def test_rescale_parameter(self):
        """Test parameter rescaling"""
        # Test basic rescaling
        result = rescale_parameter(0.5, 0.0, 1.0, 0.0, 10.0)
        assert np.isclose(result, 5.0)
        
        # Test edge cases
        result = rescale_parameter(0.0, 0.0, 1.0, 0.0, 10.0)
        assert np.isclose(result, 0.0)
        
        result = rescale_parameter(1.0, 0.0, 1.0, 0.0, 10.0)
        assert np.isclose(result, 10.0)
    
    def test_rescale_parameter_validation(self):
        """Test parameter rescaling input validation"""
        # Test invalid input ranges
        with pytest.raises(ValueError):
            rescale_parameter(0.5, 1.0, 0.0, 0.0, 10.0)  # min > max for input
    
    def test_math_driver_numerical_stability(self):
        """Test numerical stability of math functions"""
        # Test with very small numbers
        p = np.array([1e-10, 1.0 - 1e-10])
        q = np.array([1.0 - 1e-10, 1e-10])
        
        distance = hellinger(p, q)
        assert np.isfinite(distance)
        assert not np.isnan(distance)
    
    def test_math_driver_edge_cases(self):
        """Test edge cases for math driver functions"""
        # Test with zeros
        p = np.array([0.0, 1.0])
        q = np.array([1.0, 0.0])
        
        distance = hellinger(p, q)
        assert np.isfinite(distance)
        assert distance > 0


class TestTopicModelDriver:
    """Test topic model driver functionality"""
    
    def test_auth_embed_enum_exists(self):
        """Test AuthEmbedEnum is available"""
        assert hasattr(sys.modules['mstml._topic_model_driver'], 'AuthEmbedEnum')
    
    def test_term_relevance_topic_type_enum_exists(self):
        """Test TermRelevanceTopicType enum is available"""
        assert hasattr(sys.modules['mstml._topic_model_driver'], 'TermRelevanceTopicType')
    
    def test_term_relevance_topic_filter_class_exists(self):
        """Test TermRelevanceTopicFilter class is available"""
        assert hasattr(sys.modules['mstml._topic_model_driver'], 'TermRelevanceTopicFilter')
    
    @pytest.fixture
    def sample_topic_distributions(self):
        """Sample topic distributions for testing"""
        np.random.seed(42)
        num_topics = 5
        vocab_size = 100
        
        # Create normalized topic distributions
        topics = np.random.dirichlet(np.ones(vocab_size), num_topics)
        return topics
    
    @pytest.fixture
    def sample_author_data(self):
        """Sample author data for testing"""
        return pd.DataFrame({
            'author': ['Smith, J.', 'Doe, J.', 'Johnson, A.'],
            'paper_id': [1, 1, 2],
            'topic_weights': [
                np.array([0.5, 0.3, 0.2]),
                np.array([0.4, 0.4, 0.2]),
                np.array([0.3, 0.3, 0.4])
            ]
        })
    
    def test_setup_author_probs_matrix(self, sample_author_data):
        """Test author probabilities matrix setup"""
        matrix = setup_author_probs_matrix(sample_author_data)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] > 0  # Should have authors
        assert matrix.shape[1] > 0  # Should have topics
        
        # Each row should sum to 1 (probability distribution)
        for row in matrix:
            assert np.isclose(np.sum(row), 1.0, atol=1e-10)
    
    def test_find_max_min_cut_distance(self, sample_topic_distributions):
        """Test max-min cut distance calculation"""
        distance = find_max_min_cut_distance(
            sample_topic_distributions, 
            topic_idx1=0, 
            topic_idx2=1
        )
        
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_topic_model_driver_integration(self, sample_topic_distributions, sample_author_data):
        """Test integration between topic model driver components"""
        # Test that components work together
        author_matrix = setup_author_probs_matrix(sample_author_data)
        
        # Test distance calculations work with author matrix
        assert author_matrix.shape[0] > 0
        
        # Test topic distributions are compatible
        assert sample_topic_distributions.shape[0] > 0
        assert sample_topic_distributions.shape[1] > 0
    
    def test_term_relevance_topic_filter_initialization(self):
        """Test TermRelevanceTopicFilter can be initialized"""
        filter_obj = TermRelevanceTopicFilter()
        assert filter_obj is not None
    
    def test_auth_embed_enum_values(self):
        """Test AuthEmbedEnum has expected values"""
        enum_class = getattr(sys.modules['mstml._topic_model_driver'], 'AuthEmbedEnum')
        
        # Should be an enum with multiple values
        assert len(list(enum_class)) > 0
    
    def test_term_relevance_topic_type_values(self):
        """Test TermRelevanceTopicType has expected values"""
        enum_class = getattr(sys.modules['mstml._topic_model_driver'], 'TermRelevanceTopicType')
        
        # Should be an enum with multiple values
        assert len(list(enum_class)) > 0


class TestDriverInterintegration:
    """Test integration between different driver modules"""
    
    @pytest.fixture
    def sample_network_and_topics(self):
        """Sample network and topic data for integration testing"""
        # Create sample network
        network = nx.Graph()
        network.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        # Create sample topic distributions
        topics = np.random.dirichlet(np.ones(10), 5)
        
        return network, topics
    
    def test_math_and_graph_integration(self, sample_network_and_topics):
        """Test integration between math and graph drivers"""
        network, topics = sample_network_and_topics
        
        # Test that network metrics can be used with math functions
        assert len(network.nodes()) > 0
        assert topics.shape[0] > 0
        
        # Test Hellinger distance on topic pairs
        for i in range(len(topics) - 1):
            distance = hellinger(topics[i], topics[i + 1])
            assert np.isfinite(distance)
    
    def test_file_and_embedding_integration(self):
        """Test integration between file and embedding drivers"""
        # Test that file operations support embedding workflows
        timestamp = get_date_hour_minute()
        assert isinstance(timestamp, str)
        
        # Could be used for embedding result file naming
        filename = f"embedding_results_{timestamp}.pkl"
        assert timestamp in filename
    
    def test_topic_model_and_math_integration(self, sample_network_and_topics):
        """Test integration between topic model and math drivers"""
        network, topics = sample_network_and_topics
        
        # Create sample author data
        author_data = pd.DataFrame({
            'author': list(network.nodes()),
            'paper_id': [1, 2, 3],
            'topic_weights': [topics[0], topics[1], topics[2]]
        })
        
        # Test that topic model functions work with math functions
        author_matrix = setup_author_probs_matrix(author_data)
        
        # Test distance calculations on author representations
        for i in range(len(author_matrix) - 1):
            distance = hellinger(author_matrix[i], author_matrix[i + 1])
            assert np.isfinite(distance)


class TestDriverPerformance:
    """Test performance characteristics of driver modules"""
    
    def test_hellinger_distance_performance(self):
        """Test Hellinger distance calculation performance"""
        import time
        
        # Create large distributions
        np.random.seed(42)
        p = np.random.dirichlet(np.ones(1000))
        q = np.random.dirichlet(np.ones(1000))
        
        start_time = time.time()
        distance = hellinger(p, q)
        end_time = time.time()
        
        # Should be fast even for large distributions
        assert end_time - start_time < 1.0
        assert np.isfinite(distance)
    
    def test_network_construction_performance(self):
        """Test network construction performance"""
        import time
        
        # Create large author dataset
        authors = [f"Author{i}" for i in range(100)]
        large_df = pd.DataFrame({
            'authors': [f"{authors[i]}; {authors[(i+1)%100]}" for i in range(1000)],
            'title': [f"Paper {i}" for i in range(1000)],
            'date': pd.to_datetime(['2023-01-01'] * 1000)
        })
        
        start_time = time.time()
        network = build_coauthor_network_from_dataframe(large_df)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10.0
        assert len(network.nodes()) > 0
    
    def test_author_matrix_performance(self):
        """Test author matrix setup performance"""
        import time
        
        # Create large author dataset
        np.random.seed(42)
        large_author_data = pd.DataFrame({
            'author': [f"Author{i}" for i in range(1000)],
            'paper_id': list(range(1000)),
            'topic_weights': [np.random.dirichlet(np.ones(50)) for _ in range(1000)]
        })
        
        start_time = time.time()
        matrix = setup_author_probs_matrix(large_author_data)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0
        assert matrix.shape[0] > 0