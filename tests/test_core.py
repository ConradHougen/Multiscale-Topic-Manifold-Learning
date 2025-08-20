"""
Comprehensive test cases for core.py module
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mstml.core import (
    MstmlOrchestrator, 
    UnitTopicModel, 
    EmbeddingDistanceMetric,
    LowDimEmbedding,
    LDATopicModel,
    HellingerDistance,
    PHATEEmbedding
)
from mstml.dataframe_schema import MainDataSchema


class TestAbstractBaseClasses:
    """Test abstract base classes for modular components"""
    
    def test_unit_topic_model_is_abstract(self):
        """Test UnitTopicModel cannot be instantiated directly"""
        with pytest.raises(TypeError):
            UnitTopicModel()
    
    def test_embedding_distance_metric_is_abstract(self):
        """Test EmbeddingDistanceMetric cannot be instantiated directly"""
        with pytest.raises(TypeError):
            EmbeddingDistanceMetric()
    
    def test_low_dim_embedding_is_abstract(self):
        """Test LowDimEmbedding cannot be instantiated directly"""
        with pytest.raises(TypeError):
            LowDimEmbedding()


class TestLDATopicModel:
    """Test LDA topic model implementation"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            "machine learning artificial intelligence",
            "deep neural networks training",
            "natural language processing text",
            "computer vision image recognition"
        ]
    
    def test_lda_topic_model_initialization(self):
        """Test LDA topic model can be initialized"""
        model = LDATopicModel(num_topics=2, random_state=42)
        assert model.num_topics == 2
        assert model.random_state == 42
    
    @patch('mstml.core.LdaMulticore')
    def test_lda_fit_method(self, mock_lda, sample_documents):
        """Test LDA model fit method"""
        model = LDATopicModel(num_topics=2)
        mock_lda_instance = Mock()
        mock_lda.return_value = mock_lda_instance
        
        result = model.fit(sample_documents)
        assert result is model
        mock_lda.assert_called_once()
    
    def test_lda_get_topic_distributions_without_fit(self):
        """Test getting topic distributions before fitting raises error"""
        model = LDATopicModel(num_topics=2)
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_topic_distributions()


class TestHellingerDistance:
    """Test Hellinger distance metric implementation"""
    
    def test_hellinger_distance_initialization(self):
        """Test Hellinger distance can be initialized"""
        distance = HellingerDistance()
        assert distance is not None
    
    def test_hellinger_distance_compute(self):
        """Test Hellinger distance computation"""
        distance = HellingerDistance()
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        result = distance.compute(p, q)
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_hellinger_distance_identical_distributions(self):
        """Test Hellinger distance is zero for identical distributions"""
        distance = HellingerDistance()
        p = np.array([0.5, 0.3, 0.2])
        
        result = distance.compute(p, p)
        assert np.isclose(result, 0.0, atol=1e-10)
    
    def test_hellinger_distance_invalid_input(self):
        """Test Hellinger distance with invalid input"""
        distance = HellingerDistance()
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4])  # Different length
        
        with pytest.raises(ValueError):
            distance.compute(p, q)


class TestPHATEEmbedding:
    """Test PHATE embedding implementation"""
    
    @pytest.fixture
    def sample_distance_matrix(self):
        """Sample distance matrix for testing"""
        np.random.seed(42)
        n = 50
        # Create a random symmetric distance matrix
        matrix = np.random.rand(n, n)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        return matrix
    
    def test_phate_embedding_initialization(self):
        """Test PHATE embedding can be initialized"""
        embedding = PHATEEmbedding(n_components=2, k=10)
        assert embedding.n_components == 2
        assert embedding.k == 10
    
    @patch('mstml.core.PHATE_AVAILABLE', True)
    @patch('mstml.core.phate.PHATE')
    def test_phate_fit_transform(self, mock_phate_class, sample_distance_matrix):
        """Test PHATE fit_transform method"""
        mock_phate_instance = Mock()
        mock_phate_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_phate_class.return_value = mock_phate_instance
        
        embedding = PHATEEmbedding(n_components=2)
        result = embedding.fit_transform(sample_distance_matrix)
        
        assert result.shape == (50, 2)
        mock_phate_instance.fit_transform.assert_called_once()
    
    @patch('mstml.core.PHATE_AVAILABLE', False)
    def test_phate_unavailable_fallback(self, sample_distance_matrix):
        """Test PHATE fallback when library unavailable"""
        embedding = PHATEEmbedding(n_components=2)
        result = embedding.fit_transform(sample_distance_matrix)
        
        # Should fallback to MDS
        assert result.shape == (50, 2)


class TestMstmlOrchestrator:
    """Comprehensive tests for MstmlOrchestrator class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'temporal': {'smoothing_decay': 0.75},
            'topic_model': {'num_topics': 10},
            'embedding': {'n_components': 2}
        }
    
    def test_orchestrator_initialization_basic(self, temp_dir):
        """Test basic MstmlOrchestrator initialization"""
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        assert orchestrator.dataset_name == "test_dataset"
        assert orchestrator.experiment_directory == temp_dir
        assert orchestrator.config is not None
        assert orchestrator.logger is not None
    
    def test_orchestrator_initialization_with_config(self, temp_dir, sample_config):
        """Test MstmlOrchestrator initialization with custom config"""
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir,
            config=sample_config
        )
        
        assert orchestrator.config == sample_config
        assert orchestrator.smoothing_decay == 0.75
    
    @patch('mstml.core.get_dataset_directory')
    def test_orchestrator_auto_experiment_directory(self, mock_get_dataset_dir, temp_dir):
        """Test auto-generation of experiment directory"""
        mock_get_dataset_dir.return_value = temp_dir
        
        orchestrator = MstmlOrchestrator(dataset_name="test_dataset")
        
        assert "experiments" in orchestrator.experiment_directory
        assert "mstml_" in orchestrator.experiment_directory
    
    def test_orchestrator_component_configuration(self, temp_dir):
        """Test component configuration and dependency injection"""
        mock_data_loader = Mock()
        mock_text_preprocessor = Mock()
        mock_author_disambiguator = Mock()
        
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir,
            data_loader=mock_data_loader,
            text_preprocessor=mock_text_preprocessor,
            author_disambiguator=mock_author_disambiguator
        )
        
        assert orchestrator._data_loader is mock_data_loader
        assert orchestrator._text_preprocessor is mock_text_preprocessor
        assert orchestrator._author_disambiguator is mock_author_disambiguator
    
    def test_orchestrator_default_config(self, temp_dir):
        """Test default configuration values"""
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        config = orchestrator.config
        assert 'temporal' in config
        assert 'topic_model' in config
        assert 'embedding' in config
        assert 'network' in config
    
    @patch('mstml.core.os.makedirs')
    def test_orchestrator_experiment_directory_creation(self, mock_makedirs, temp_dir):
        """Test experiment directory is created"""
        MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        mock_makedirs.assert_called_with(temp_dir, exist_ok=True)
    
    def test_orchestrator_state_tracking(self, temp_dir):
        """Test experiment parameter tracking"""
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_name="test_experiment",
            experiment_directory=temp_dir
        )
        
        params = orchestrator.experiment_params
        assert params['dataset_name'] == "test_dataset"
        assert params['experiment_name'] == "test_experiment"
        assert params['experiment_directory'] == temp_dir
        assert 'steps' in params
    
    @patch('mstml.core.logging.getLogger')
    def test_orchestrator_logger_setup(self, mock_get_logger, temp_dir):
        """Test logger setup"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        assert orchestrator.logger is mock_logger
    
    def test_orchestrator_custom_logger(self, temp_dir):
        """Test custom logger injection"""
        custom_logger = Mock()
        
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir,
            logger=custom_logger
        )
        
        assert orchestrator.logger is custom_logger
    
    @patch('mstml.core.DataLoaderRegistry.create_loader')
    def test_orchestrator_load_data_workflow(self, mock_create_loader, temp_dir):
        """Test data loading workflow"""
        mock_loader = Mock()
        mock_df = pd.DataFrame({'title': ['test'], 'abstract': ['test abstract']})
        mock_loader.load_data.return_value = mock_df
        mock_create_loader.return_value = mock_loader
        
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        result = orchestrator.load_data("test.json")
        
        assert result is orchestrator
        assert orchestrator.documents_df is not None
        mock_create_loader.assert_called_once()
    
    def test_orchestrator_preprocess_text_without_data(self, temp_dir):
        """Test preprocessing text without loaded data raises error"""
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        with pytest.raises(ValueError, match="No data loaded"):
            orchestrator.preprocess_text()
    
    def test_orchestrator_create_temporal_chunks_without_data(self, temp_dir):
        """Test creating temporal chunks without data raises error"""
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        with pytest.raises(ValueError, match="No data loaded"):
            orchestrator.create_temporal_chunks()
    
    def test_orchestrator_train_ensemble_models_without_chunks(self, temp_dir):
        """Test training ensemble models without temporal chunks raises error"""
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        with pytest.raises(ValueError, match="No temporal chunks"):
            orchestrator.train_ensemble_models()


class TestOrchestratatorWorkflow:
    """Test complete workflow scenarios"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe for workflow testing"""
        return pd.DataFrame({
            'title': ['Machine Learning Research', 'Deep Learning Applications'],
            'abstract': ['This paper discusses ML algorithms', 'Study of deep neural networks'],
            'authors': ['Smith, J.', 'Doe, J.; Johnson, A.'],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'categories': ['cs.AI', 'cs.LG']
        })
    
    @patch('mstml.core.DataLoaderRegistry.create_loader')
    @patch('mstml.core.TextPreprocessor')
    @patch('mstml.core.AuthorDisambiguator')
    def test_full_workflow_integration(self, mock_disambiguator_class, mock_preprocessor_class, 
                                      mock_create_loader, temp_dir, sample_dataframe):
        """Test full workflow integration"""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.load_data.return_value = sample_dataframe
        mock_create_loader.return_value = mock_loader
        
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess.return_value = sample_dataframe
        mock_preprocessor_class.return_value = mock_preprocessor
        
        mock_disambiguator = Mock()
        mock_disambiguator.disambiguate.return_value = sample_dataframe
        mock_disambiguator_class.return_value = mock_disambiguator
        
        # Test workflow
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        # Load data
        orchestrator.load_data("test.json")
        assert orchestrator.documents_df is not None
        
        # Preprocess text  
        orchestrator.preprocess_text()
        mock_preprocessor.preprocess.assert_called_once()
        
        # Setup co-author network
        orchestrator.setup_coauthor_network()
        mock_disambiguator.disambiguate.assert_called_once()
        
        # Create temporal chunks
        orchestrator.create_temporal_chunks(months_per_chunk=1)
        assert len(orchestrator.time_chunks) > 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_orchestrator_invalid_dataset_name(self, temp_dir):
        """Test orchestrator with invalid dataset name"""
        with pytest.raises((ValueError, FileNotFoundError)):
            MstmlOrchestrator(
                dataset_name="",
                experiment_directory=temp_dir
            )
    
    def test_orchestrator_nonexistent_experiment_directory(self):
        """Test orchestrator with non-existent experiment directory"""
        nonexistent_dir = "/tmp/definitely_does_not_exist_12345"
        
        # Should create directory automatically
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=nonexistent_dir
        )
        
        assert Path(nonexistent_dir).exists()
        
        # Cleanup
        shutil.rmtree(nonexistent_dir)
    
    def test_orchestrator_invalid_config(self, temp_dir):
        """Test orchestrator with invalid configuration"""
        invalid_config = "not_a_dict"
        
        with pytest.raises(TypeError):
            MstmlOrchestrator(
                dataset_name="test_dataset",
                experiment_directory=temp_dir,
                config=invalid_config
            )


class TestPerformanceAndScalability:
    """Test performance and scalability considerations"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_orchestrator_memory_usage(self, temp_dir):
        """Test orchestrator memory usage is reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_orchestrator_initialization_time(self, temp_dir):
        """Test orchestrator initialization time is reasonable"""
        import time
        
        start_time = time.time()
        
        orchestrator = MstmlOrchestrator(
            dataset_name="test_dataset",
            experiment_directory=temp_dir
        )
        
        end_time = time.time()
        initialization_time = end_time - start_time
        
        # Initialization should take less than 5 seconds
        assert initialization_time < 5.0