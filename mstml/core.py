"""
Core MSTML (Multiscale Topic Manifold Learning) functionality.

This module provides user-friendly interfaces and integration points for the MSTML framework,
combining temporal topic modeling (GDLTM), network analysis, and hierarchical modeling.
"""

import numpy as np
import pandas as pd
import logging
import os
import re
import json  # For metadata
import datetime as dt
import pandas.api.types as ptypes
import multiprocessing as mp
import yaml
import random
import warnings
import shutil
import gensim
import pickle
import faiss
import gensim.corpora as corpora
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Optional imports with fallbacks
try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    PHATE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from datetime import datetime
from gensim.models import LdaModel, LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import entropy
from collections import OrderedDict
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from typing import Optional, Union, Dict, Any, List, Tuple
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from .dataframe_schema import FieldDef, MainDataSchema
from .data_loaders import get_dataset_directory, JsonDataLoader, DataLoader
from .data_loader_registry import DataLoaderRegistry
from .author_disambiguation import AuthorDisambiguator
from .text_preprocessing import TextPreprocessor
from ._embedding_driver import *
from ._file_driver import *
from ._graph_driver import *
from ._graph_driver import (
    build_coauthor_network_from_dataframe,
    build_temporal_coauthor_networks_from_dataframe, 
    compose_networks_from_dict
)
from ._math_driver import hellinger, diffuse_distribution, rescale_parameter
from ._topic_model_driver import setup_author_probs_matrix, find_max_min_cut_distance, expand_doc_topic_distns, compute_author_barycenters
from .fast_encode_tree import fast_encode_tree_structure


"""============================================================================
JSON Serialization Utilities
============================================================================"""
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        return super().default(obj)

def safe_json_convert(obj):
    """Recursively convert NumPy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, tuple):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.complex64, np.complex128)):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    else:
        return obj


"""============================================================================
Abstract Base Classes for Modular MSTML Components
============================================================================"""
class UnitTopicModel(ABC):
    """
    Abstract base class for topic models used in temporal ensemble.
    
    This interface allows MSTML to work with different topic modeling approaches
    (LDA, BERTopic, GPTopic, etc.) in a consistent manner.
    """
    
    @abstractmethod
    def fit(self, documents: List[str], vocabulary: Optional[Dict[str, int]] = None) -> 'UnitTopicModel':
        """
        Fit the topic model to a corpus of documents.
        
        Args:
            documents: List of preprocessed document strings
            vocabulary: Optional vocabulary mapping {word: index}
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def get_topic_word_distributions(self) -> np.ndarray:
        """
        Get topic-word distributions (phi matrix).
        
        Returns:
            Array of shape (n_topics, n_words) with topic-word probabilities
        """
        pass
    
    @abstractmethod
    def get_document_topic_distributions(self) -> np.ndarray:
        """
        Get document-topic distributions (theta matrix).
        
        Returns:
            Array of shape (n_documents, n_topics) with document-topic probabilities
        """
        pass
    
    @abstractmethod
    def get_num_topics(self) -> int:
        """
        Get the number of topics in the model.
        
        Returns:
            Number of topics
        """
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters for serialization/logging.
        
        Returns:
            Dictionary of model parameters
        """
        pass


class EmbeddingDistanceMetric(ABC):
    """
    Abstract base class for distance metrics used in manifold learning.
    
    This interface allows MSTML to work with different distance measures
    (Hellinger, cosine, euclidean, etc.) consistently.
    """
    
    @abstractmethod
    def compute_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix between vectors.
        
        Args:
            vectors: Array of shape (n_samples, n_features)
        
        Returns:
            Symmetric distance matrix of shape (n_samples, n_samples)
        """
        pass
    
    @abstractmethod
    def compute_pairwise_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute distances between two sets of vectors.
        
        Args:
            X: First set of vectors, shape (n_samples_X, n_features)
            Y: Second set of vectors, shape (n_samples_Y, n_features)
        
        Returns:
            Distance matrix of shape (n_samples_X, n_samples_Y)
        """
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Get the name of the distance metric.
        
        Returns:
            String identifier for the metric
        """
        pass


class LowDimEmbedding(ABC):
    """
    Abstract base class for dimensionality reduction/manifold learning methods.
    
    This interface allows MSTML to work with different embedding techniques
    (PHATE, UMAP, t-SNE, etc.) consistently.
    """
    
    @abstractmethod
    def fit_transform(self, X: np.ndarray, 
                     distance_metric: Optional[EmbeddingDistanceMetric] = None) -> np.ndarray:
        """
        Fit the embedding model and transform the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            distance_metric: Optional custom distance metric
        
        Returns:
            Embedded data of shape (n_samples, n_components)
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted embedding model.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Embedded data of shape (n_samples, n_components)
        """
        pass
    
    @abstractmethod
    def get_embedding_params(self) -> Dict[str, Any]:
        """
        Get embedding-specific parameters for serialization/logging.
        
        Returns:
            Dictionary of embedding parameters
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """
        Get the name of the embedding method.
        
        Returns:
            String identifier for the method
        """
        pass


"""============================================================================
Concrete Implementations of Default MSTML Components
============================================================================"""
class LDATopicModel(UnitTopicModel):
    """
    LDA topic model implementation using Gensim.
    
    Default topic model for MSTML temporal ensemble. Uses Gensim's LdaModel
    which is optimized for topic modeling and provides better performance
    and additional features compared to sklearn.
    """
    
    def __init__(self, n_topics: int = 10, alpha: Union[str, float] = 'symmetric', 
                 eta: Union[str, float] = None, random_state: int = 42, 
                 npasses: int = 5, ngibbs: int = 50, **kwargs):
        self.n_topics = n_topics
        self.alpha = alpha  # Document-topic density (higher = more topics per doc)
        self.eta = eta      # Topic-word density (higher = more words per topic) 
        self.random_state = random_state
        self.npasses = npasses  # Number of passes through corpus during training
        self.ngibbs = ngibbs  # Number of iterations per pass
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.kwargs = kwargs
    
    def fit(self, documents: List[str], vocabulary: Optional[Dict[str, int]] = None) -> 'LDATopicModel':
        try:
            # Check if gensim is available
            corpora.Dictionary([])
        except ImportError:
            raise ImportError("Gensim not installed. Install with: pip install gensim")
        
        # Tokenize documents (assuming they're already preprocessed)
        tokenized_docs = [doc.split() for doc in documents]
        
        # Create or use provided dictionary
        if vocabulary is not None:
            # Convert vocabulary dict to gensim format
            self.dictionary = corpora.Dictionary()
            self.dictionary.token2id = vocabulary
            self.dictionary.id2token = {v: k for k, v in vocabulary.items()}
        else:
            # Create dictionary from documents
            self.dictionary = corpora.Dictionary(tokenized_docs)
        
        # Convert documents to bag-of-words corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        # Train LDA model
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            alpha=self.alpha,
            eta=self.eta,
            random_state=self.random_state,
            passes=self.npasses,
            iterations=self.ngibbs,
            **self.kwargs
        )
        
        return self
    
    def get_topic_word_distributions(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get topic-word matrix (phi)
        vocab_size = len(self.dictionary)
        topic_word_matrix = np.zeros((self.n_topics, vocab_size))
        
        for topic_id in range(self.n_topics):
            topic_terms = self.model.get_topic_terms(topic_id, topn=vocab_size)
            for word_id, prob in topic_terms:
                topic_word_matrix[topic_id, word_id] = prob
        
        return topic_word_matrix
    
    def get_document_topic_distributions(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get document-topic matrix (theta)
        doc_topic_matrix = np.zeros((len(self.corpus), self.n_topics))
        
        for doc_id, doc_bow in enumerate(self.corpus):
            doc_topics = self.model.get_document_topics(doc_bow, minimum_probability=0.0)
            for topic_id, prob in doc_topics:
                doc_topic_matrix[doc_id, topic_id] = prob
        
        return doc_topic_matrix
    
    def get_num_topics(self) -> int:
        return self.n_topics
    
    def get_model_params(self) -> Dict[str, Any]:
        return {
            'model_type': 'LDA_Gensim',
            'n_topics': self.n_topics,
            'alpha': self.alpha,
            'eta': self.eta,
            'random_state': self.random_state,
            'npasses': self.npasses,
            'ngibbs': self.ngibbs,
            **self.kwargs
        }
    
    def get_topic_terms(self, topic_id: int, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Get top terms for a specific topic (Gensim-specific feature).
        
        Args:
            topic_id: Topic index
            topn: Number of top terms to return
        
        Returns:
            List of (term, probability) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return [(self.dictionary[word_id], prob) 
                for word_id, prob in self.model.get_topic_terms(topic_id, topn=topn)]
    
    def get_coherence_score(self, coherence_measure: str = 'c_v') -> float:
        """
        Calculate topic coherence score (Gensim-specific feature).
        
        Args:
            coherence_measure: Coherence measure ('c_v', 'c_npmi', 'c_uci', 'u_mass')
        
        Returns:
            Coherence score
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            
            # Recreate tokenized documents for coherence calculation
            tokenized_docs = [[self.dictionary[word_id] for word_id, _ in doc] 
                            for doc in self.corpus]
            
            coherence_model = CoherenceModel(
                model=self.model,
                texts=tokenized_docs,
                dictionary=self.dictionary,
                coherence=coherence_measure
            )
            
            return coherence_model.get_coherence()
        except ImportError:
            raise ImportError("Gensim CoherenceModel not available in this version")


class HellingerDistance(EmbeddingDistanceMetric):
    """
    Hellinger distance metric for probability distributions.
    
    Default distance metric for MSTML topic manifold learning.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Hellinger distance metric.
        
        Args:
            **kwargs: Additional parameters (currently unused but provided for interface consistency)
        """
        super().__init__()
        # Store any additional parameters for future extensions
        self.params = kwargs
    
    def compute_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute Hellinger distance matrix using optimized vectorized operations.
        
        For probability distributions P and Q:
        H(P,Q) = √(0.5 * Σ(√P - √Q)²)
        
        Uses scipy.spatial.distance.pdist equivalent vectorized approach for maximum performance.
        Automatically switches to memory-efficient mode for large datasets.
        """
        n_samples = vectors.shape[0]
        
        # For very large datasets (>10K samples), use memory-efficient chunked computation
        # to avoid creating large (n, n, d) tensors that may exceed memory
        memory_threshold = 10000
        
        if n_samples > memory_threshold:
            return self._compute_distance_matrix_chunked(vectors)
        
        # Ensure vectors are probability distributions (vectorized normalization)
        vectors = vectors / vectors.sum(axis=1, keepdims=True)
        
        # Compute square roots once
        sqrt_vectors = np.sqrt(vectors)
        
        # Vectorized pairwise distance computation using broadcasting
        # This creates (n, n, d) tensor where diff[i,j,:] = sqrt_vectors[i] - sqrt_vectors[j]
        diff = sqrt_vectors[:, np.newaxis, :] - sqrt_vectors[np.newaxis, :, :]
        
        # Compute squared differences and sum along feature dimension, then apply sqrt(0.5 * ...)
        # This is equivalent to the Hellinger distance formula
        distance_matrix = np.sqrt(0.5 * np.sum(diff ** 2, axis=2))
        
        return distance_matrix
        
    def _compute_distance_matrix_chunked(self, vectors: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
        """
        Memory-efficient chunked computation for large distance matrices.
        
        Processes the distance matrix in chunks to avoid memory overflow.
        """
        n_samples = vectors.shape[0]
        
        # Ensure vectors are probability distributions (vectorized normalization)
        vectors = vectors / vectors.sum(axis=1, keepdims=True)
        sqrt_vectors = np.sqrt(vectors)
        
        # Initialize distance matrix
        distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        # Process in chunks to manage memory usage
        for i in range(0, n_samples, chunk_size):
            i_end = min(i + chunk_size, n_samples)
            for j in range(0, n_samples, chunk_size):
                j_end = min(j + chunk_size, n_samples)
                
                # Compute chunk of distance matrix
                chunk_i = sqrt_vectors[i:i_end]
                chunk_j = sqrt_vectors[j:j_end]
                
                # Vectorized computation for this chunk
                diff = chunk_i[:, np.newaxis, :] - chunk_j[np.newaxis, :, :]
                chunk_distances = np.sqrt(0.5 * np.sum(diff ** 2, axis=2))
                
                distance_matrix[i:i_end, j:j_end] = chunk_distances
        
        return distance_matrix
    
    def compute_pairwise_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Hellinger distances between two sets of vectors.
        
        Optimized vectorized implementation for cross-set distance computation.
        
        Args:
            X: First set of vectors (n_samples_X, n_features)  
            Y: Second set of vectors (n_samples_Y, n_features)
            
        Returns:
            Distance matrix (n_samples_X, n_samples_Y)
        """
        # Ensure vectors are probability distributions (vectorized normalization)
        X = X / X.sum(axis=1, keepdims=True)
        Y = Y / Y.sum(axis=1, keepdims=True)
        
        # Compute square roots once
        sqrt_X = np.sqrt(X)
        sqrt_Y = np.sqrt(Y)
        
        # Vectorized pairwise distance computation using broadcasting
        # sqrt_X[:, np.newaxis, :] has shape (n_X, 1, d)
        # sqrt_Y[np.newaxis, :, :] has shape (1, n_Y, d)  
        # Broadcasting creates (n_X, n_Y, d) difference tensor
        diff = sqrt_X[:, np.newaxis, :] - sqrt_Y[np.newaxis, :, :]
        
        # Compute Hellinger distances: √(0.5 * Σ(√P - √Q)²)
        distances = np.sqrt(0.5 * np.sum(diff ** 2, axis=2))
        
        return distances
    
    def get_metric_name(self) -> str:
        return "hellinger"


class PHATEEmbedding(LowDimEmbedding):
    """
    PHATE embedding implementation.
    
    Default dimensionality reduction method for MSTML topic visualization.
    """
    
    def __init__(self, n_components: int = 2, knn_neighbors: int = 10,
                 gamma: float = 1.0, t = 'auto', **kwargs):
        self.n_components = n_components
        self.knn_neighbors = knn_neighbors
        self.gamma = gamma
        self.t = t
        self.kwargs = kwargs
        self.phate_model = None
        self.fitted = False
    
    def fit_transform(self, X: np.ndarray, 
                     distance_metric: Optional[EmbeddingDistanceMetric] = None) -> np.ndarray:
        if not PHATE_AVAILABLE:
            raise ImportError("PHATE not installed. Install with: pip install phate")
        
        # Use custom distance metric if provided
        if distance_metric is not None:
            # Compute distance matrix and use precomputed metric
            distance_matrix = distance_metric.compute_distance_matrix(X)
            self.phate_model = phate.PHATE(
                n_components=self.n_components,
                knn=self.knn_neighbors,
                gamma=self.gamma,
                t=self.t,
                metric='precomputed',
                **self.kwargs
            )
            embedding = self.phate_model.fit_transform(distance_matrix)
        else:
            # Use default euclidean metric
            self.phate_model = phate.PHATE(
                n_components=self.n_components,
                knn=self.knn_neighbors,
                gamma=self.gamma,
                t=self.t,
                **self.kwargs
            )
            embedding = self.phate_model.fit_transform(X)
        
        self.fitted = True
        return embedding
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or self.phate_model is None:
            raise ValueError("Model not fitted. Call fit_transform() first.")
        return self.phate_model.transform(X)
    
    def get_embedding_params(self) -> Dict[str, Any]:
        return {
            'method': 'PHATE',
            'n_components': self.n_components,
            'knn_neighbors': self.knn_neighbors,
            'gamma': self.gamma,
            't': self.t,
            **self.kwargs
        }
    
    def get_method_name(self) -> str:
        return "PHATE"


"""============================================================================
Factory Methods for Creating Model Instances
============================================================================"""
def create_topic_model(model_type: str = 'LDA', **kwargs) -> UnitTopicModel:
    """
    Factory function to create topic model instances.
    
    Args:
        model_type: Type of topic model ('LDA', 'BERTopic', 'GPTopic', etc.)
        **kwargs: Model-specific parameters
    
    Returns:
        UnitTopicModel instance
    """
    if model_type.upper() == 'LDA':
        return LDATopicModel(**kwargs)
    else:
        raise ValueError(f"Unknown topic model type: {model_type}")


def create_distance_metric(metric_type: str = 'hellinger', **kwargs) -> EmbeddingDistanceMetric:
    """
    Factory function to create distance metric instances.
    
    Args:
        metric_type: Type of distance metric ('hellinger', 'cosine', 'euclidean', etc.)
        **kwargs: Metric-specific parameters
    
    Returns:
        EmbeddingDistanceMetric instance
    """
    if metric_type.lower() == 'hellinger':
        return HellingerDistance(**kwargs)
    else:
        raise ValueError(f"Unknown distance metric type: {metric_type}")


def create_embedding_method(method_type: str = 'PHATE', **kwargs) -> LowDimEmbedding:
    """
    Factory function to create embedding method instances.
    
    Args:
        method_type: Type of embedding method ('PHATE', 'UMAP', 'TSNE', etc.)
        **kwargs: Method-specific parameters
    
    Returns:
        LowDimEmbedding instance
    """
    if method_type.upper() == 'PHATE':
        return PHATEEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding method type: {method_type}")


# Backwards compatibility aliases for existing code
def create_phate_embedding(*args, **kwargs):
    """Backwards compatibility alias for create_topic_embedding."""
    # This would be implemented at the orchestrator level
    pass


"""============================================================================
class MstmlOrchestrator

Primary user interface for MSTML and related functionality

Modular Design:
- Topic Models: Abstract interface supporting LDA, BERTopic, GPTopic, etc.
- Distance Metrics: Abstract interface supporting Hellinger, cosine, euclidean, etc.  
- Embedding Methods: Abstract interface supporting PHATE, UMAP, t-SNE, etc.
- Default: LDA + Hellinger + PHATE (can be configured via configure_components())
============================================================================"""
class MstmlOrchestrator:
    """
    This class provides MSTML Orchestration and an upper-level interface for applying MSTML and related models.

    High-level functionality includes:
    1. Data loading and automated management of data files and directories.
    2. Text preprocessing and application of default and custom vocabulary filtering.
    3. Author/entity disambiguation and co-author network(s) management.
    4. Application of temporal model ensemble, with support for LDA or custom topic models.
    5. Meso-scale topic embedding vector neighborhood masking prior to dendrogram learning.
    6. Fast hierarchical dendrogram learning with link probability parameter estimation (MSTML core: fast tree encoding).
    7. Longitudinal topic alignment and visualization using Hellinger-PHATE or alternative embeddings (GDLTM core).
    8. Author/document representation learning and interdisciplinarity ranking for anomaly detection (MSTML core).
    9. Author and author community topic drift mapping for local subnetworks and topic embeddings (MSTML core).
    10. Link prediction in both co-author graphs and hypergraphs (MSTML derivative work).
    
    Example Usage:
        # Basic workflow with automatic experiment directory
        orchestrator = MstmlOrchestrator("my_research")
        orchestrator.load_data("data.json", dataset_name="my_research")
        orchestrator.preprocess_text()
        orchestrator.setup_coauthor_network()
        orchestrator.create_temporal_chunks(months_per_chunk=6)
        orchestrator.train_ensemble_models()
        orchestrator.build_topic_manifold()
        orchestrator.compute_author_embeddings()
        orchestrator.score_interdisciplinarity()
        
        # Finalize and save all results
        exp_dir = orchestrator.finalize_experiment()
        print(f"Results saved to: {exp_dir}")
        
        # Custom experiment directory
        orchestrator = MstmlOrchestrator(
            experiment_name="topic_analysis"
            experiment_directory="/path/to/my_analysis_v2",
        )
        
        # Restore previous state
        restored = MstmlOrchestrator.load_orchestrator_state("path/to/state.pkl")
    """
    def __init__(self,
                 dataset_name: str,
                 experiment_name: str = "mstml",
                 experiment_directory: Optional[str] = None,
                 data_loader: Optional[DataLoader] = None,
                 text_preprocessor: Optional['TextPreprocessor'] = None,
                 author_disambiguator: Optional[AuthorDisambiguator] = None,
                 config: Optional[dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize MSTML Orchestrator with data directory and optional components.
        
        Args:
            dataset_name: Name of dataset to load, e.g. "arxiv"
            experiment_name: Prefix that describes the current experiment type (default: "mstml")
            experiment_directory: Path to experiment directory for results, auto-generated if None
            data_loader: Pre-configured DataLoader instance, created with defaults if None
            text_preprocessor: Pre-configured TextPreprocessor instance, created with defaults if None  
            author_disambiguator: Pre-configured AuthorDisambiguator instance, created with defaults if None
            config: Configuration dictionary for model parameters
            logger: Custom logger, creates default if None
        """
        # Core configuration
        self.dataset_name = dataset_name
        self.dataset_directory = get_dataset_directory(dataset_name)
        self.config = config or self._get_default_config()
        self.logger = logger or self._setup_logger()
        
        # Initialize experiment parameters for tracking
        self.experiment_params = {
            'dataset_name': dataset_name,
            'experiment_name': experiment_name,
            'experiment_directory': None,  # Will be set after directory creation
            'steps': {}
        }
        
        # Experiment directory setup
        self.experiment_name = experiment_name
        if experiment_directory is None:
            # Auto-generate experiment directory with timestamp
            timestamp = get_date_hour_minute()
            exp_dir_name = f"{experiment_name}_{timestamp}"
            # Convert to Path object if it's a string
            data_path = Path(self.dataset_directory) if isinstance(self.dataset_directory, str) else self.dataset_directory
            # Place experiments at project root level (same level as data/)
            project_root = data_path.parent.parent  # data_path is data/dataset_name, so go up two levels
            self.experiment_directory = str(project_root / "experiments" / exp_dir_name)
        else:
            self.experiment_directory = experiment_directory
        
        # Create experiment directory
        os.makedirs(self.experiment_directory, exist_ok=True)
        self.experiment_params['experiment_directory'] = self.experiment_directory
        self.logger.info(f"Experiment directory: {self.experiment_directory}")
        
        # Component configuration - accept user-provided components or create defaults
        self._data_loader = data_loader
        self._text_preprocessor = text_preprocessor  
        self._author_disambiguator = author_disambiguator
        self._pending_data_filters = None  # Store filters until DataLoader is created
        
        # Data management
        self.schema = None  
        self.documents_df = None
        self.authors_df = None
        self.coauthor_network = None
        
        # Text processing components
        self.vocabulary = None
        self.global_lda_model = None  # For term relevancy filtering

        # Temporal ensemble components
        self.time_chunks = []
        self.chunk_topic_models = []
        self.chunk_topics = []
        self.gamma_temporal_decay = self.config.get('temporal', {}).get('gamma_temporal_decay', 0.75)
        
        # Modular components (dependency injection)
        self.topic_model_factory = create_topic_model
        self.embedding_method = None
        self.distance_metric = None
        
        # Set default components
        self._set_default_components()
        
        # Topic manifold and embedding components
        self.topic_vectors = None  # All phi vectors from ensemble
        self.topic_knn_graph = None
        self.topic_dendrogram_linkage = None  # Raw scipy linkage matrix (Z)
        self.topic_dendrogram = None          # Enhanced dendrogram with link probabilities
        self.internal_node_probabilities = {}
        self.topic_embedding = None
        
        # Author/document representations
        self.author_ct_distns = {}  # psi(u) vectors
        self.document_ct_distns = {}
        self.interdisciplinarity_scores = {}

        # Visualization components
        self.embed_params = {}
        self.embed_figure = None
        
        self.logger.info("MstmlOrchestrator initialized")

    # ============================================================================
    # Data Getter/Setter Methods
    # ============================================================================
    def get_df(self) -> pd.DataFrame:
        """Get the current dataframe managed by orchestrator"""
        return self.documents_df

    def set_df(self, df: pd.DataFrame) -> None:
        """Set the current dataframe managed by orchestrator"""
        self.documents_df = df.copy()

    # ============================================================================
    # Component Management - Getter/Setter Methods
    # ============================================================================
    
    def get_data_loader(self) -> Optional[DataLoader]:
        """Get the current DataLoader instance."""
        return self._data_loader
    
    def set_data_loader(self, data_loader: DataLoader) -> 'MstmlOrchestrator':
        """
        Set a custom DataLoader instance.
        
        Args:
            data_loader: Configured DataLoader instance
            
        Returns:
            Self for method chaining
            
        Example:
            
            # Create and configure DataLoader
            loader = JsonDataLoader.with_config({
                'data_filters': {
                    'date_range': {'start': '2020-01-01', 'end': '2023-12-31'},
                    'categories': ['cs.AI', 'cs.LG', 'stat.ML']
                }
            })
            
            orchestrator.set_data_loader(loader)
        """
        self._data_loader = data_loader
        self.logger.info("DataLoader updated")
        return self
    
    def get_text_preprocessor(self) -> Optional['TextPreprocessor']:
        """Get the current TextPreprocessor instance.""" 
        return self._text_preprocessor
    
    def set_text_preprocessor(self, text_preprocessor: 'TextPreprocessor') -> 'MstmlOrchestrator':
        """
        Set a custom TextPreprocessor instance.
        
        Args:
            text_preprocessor: Configured TextPreprocessor instance
            
        Returns:
            Self for method chaining
        """
        self._text_preprocessor = text_preprocessor
        self.logger.info("TextPreprocessor updated")
        return self
    
    def get_author_disambiguator(self) -> Optional[AuthorDisambiguator]:
        """Get the current AuthorDisambiguator instance."""
        return self._author_disambiguator
    
    def set_author_disambiguator(self, author_disambiguator: AuthorDisambiguator) -> 'MstmlOrchestrator':
        """
        Set a custom AuthorDisambiguator instance.
        
        Args:
            author_disambiguator: Configured AuthorDisambiguator instance
            
        Returns:
            Self for method chaining
        """
        self._author_disambiguator = author_disambiguator
        self.logger.info("AuthorDisambiguator updated")
        return self
    
    def configure_data_filters(self, 
                              date_range: Optional[Dict[str, str]] = None,
                              categories: Optional[List[str]] = None,
                              custom_filter: Optional[callable] = None,
                              **kwargs) -> 'MstmlOrchestrator':
        """
        Configure data filters for the DataLoader. This is a convenient way to set
        common filtering options without manually creating a DataLoader.
        
        Args:
            date_range: Date range filter with 'start' and/or 'end' keys
                       Accepts flexible date formats:
                       - '2020-01-01' (ISO format)
                       - '2020/01/01' (slash format)
                       - '01-01-2020' (US format)
                       - 'Jan 1, 2020' (text format)
            categories: List of category labels to include (for arXiv-style data)
                       Example: ['cs.AI', 'cs.LG', 'stat.ML']
            custom_filter: Custom filter function that takes a DataFrame and returns a boolean mask
            **kwargs: Additional filter configurations
            
        Returns:
            Self for method chaining
            
        Example:
            # Date range filtering
            orchestrator.configure_data_filters(
                date_range={'start': '2020-01-01', 'end': '2023-12-31'}
            )
            
            # Category filtering
            orchestrator.configure_data_filters(
                categories=['cs.AI', 'cs.LG', 'stat.ML']
            )
            
            # Combined filtering
            orchestrator.configure_data_filters(
                date_range={'start': '2020-01-01'},
                categories=['cs.AI', 'cs.LG']
            )
            
            # Custom filtering
            def recent_papers(df):
                return df['date'] >= pd.Timestamp('2022-01-01')
                
            orchestrator.configure_data_filters(
                custom_filter=recent_papers
            )
        """
        if self._data_loader is None:
            # Create a default DataLoader with the filters
            filters = {}
            if date_range:
                filters['date_range'] = self._normalize_date_range(date_range)
            if categories:
                filters['categories'] = categories
            if custom_filter:
                filters['custom'] = {'function': custom_filter}
            filters.update(kwargs)
            
            # We'll need to create the DataLoader when load_data is called
            # For now, store the filter configuration
            self._pending_data_filters = filters
        else:
            # Update existing DataLoader's configuration
            current_filters = self._data_loader.data_filters.copy()
            if date_range:
                current_filters['date_range'] = self._normalize_date_range(date_range)
            if categories:
                current_filters['categories'] = categories
            if custom_filter:
                current_filters['custom'] = {'function': custom_filter}
            current_filters.update(kwargs)
            
            self._data_loader.data_filters = current_filters
            
        self.logger.info("Data filters configured")
        return self
    
    def _normalize_date_range(self, date_range: Dict[str, str]) -> Dict[str, str]:
        """
        Normalize date range to ISO format with flexible input parsing.
        
        Args:
            date_range: Dict with 'start' and/or 'end' keys
            
        Returns:
            Dict with normalized ISO format dates
        """
        normalized = {}
        for key, date_str in date_range.items():
            if key not in ['start', 'end']:
                continue
                
            try:
                # Use pandas to parse flexible date formats
                parsed_date = pd.to_datetime(date_str)
                normalized[key] = parsed_date.strftime('%Y-%m-%d')
            except Exception as e:
                raise ValueError(
                    f"Could not parse date '{date_str}' for {key}. "
                    f"Supported formats: '2020-01-01', '2020/01/01', '01-01-2020', 'Jan 1, 2020'. "
                    f"Error: {e}"
                )
        
        return normalized

    def set_experiment_directory(self, experiment_directory: str, create_if_missing: bool = True):
        """
        Set a custom experiment directory for saving results.
        
        Args:
            experiment_directory: Path to experiment directory
            create_if_missing: Whether to create the directory if it doesn't exist
        """
        self.experiment_directory = experiment_directory
        
        if create_if_missing:
            os.makedirs(self.experiment_directory, exist_ok=True)
        
        self.logger.info(f"Experiment directory set to: {self.experiment_directory}")

    def save_orchestrator_state(self, filename: Optional[str] = None):
        """
        Save the entire orchestrator state to experiment directory for later restoration.
        
        Args:
            filename: Custom filename, auto-generated if None
        """
        if filename is None:
            timestamp = get_date_hour_minute()
            filename = f"mstml_orchestrator_state_{timestamp}.pkl"
        
        state_path = os.path.join(self.experiment_directory, filename)
        write_pickle(state_path, self)
        self.logger.info(f"Orchestrator state saved to {state_path}")
        return state_path

    @staticmethod
    def load_orchestrator_state(state_path: str) -> 'MstmlOrchestrator':
        """
        Load a previously saved orchestrator state.
        
        Args:
            state_path: Path to saved orchestrator state file
            
        Returns:
            Restored MstmlOrchestrator instance
        """
        orchestrator = read_pickle(state_path)
        orchestrator.logger.info(f"Orchestrator state loaded from {state_path}")
        return orchestrator

    def _set_default_components(self):
        """Initialize default components based on configuration."""
        # Initialize default distance metric
        distance_type = self.config['distance_metric']['type']
        distance_params = self.config['distance_metric'].get('params', {})
        self.distance_metric = create_distance_metric(distance_type, **distance_params)
        
        # Initialize default embedding method
        embedding_type = self.config['topic_embedding']['type']
        embedding_params = self.config['topic_embedding'].get('params', {})
        self.embedding_method = create_embedding_method(embedding_type, **embedding_params)
        
        self.logger.info(f"Default components initialized: {distance_type} distance, {embedding_type} embedding")

    # ========================================================================================
    # 1. DATA LOADING AND MANAGEMENT
    # ========================================================================================
    
    def load_data(self,
                  original_data_fnames: Optional[Union[str, List[str]]] = None,
                  author_disambiguation: Optional[bool] = None,
                  overwrite: Optional[bool] = None,
                  data_filters: Optional[Dict[str, any]] = None) -> 'MstmlOrchestrator':
        """
        Load and validate document corpus data using DataLoader pipeline.
        
        This method now works with the new component system - you can either:
        1. Provide a pre-configured DataLoader via constructor or set_data_loader()
        2. Use this method with parameters to create a DataLoader automatically
        3. Use configure_data_filters() before calling this method
        
        Args:
            original_data_fnames: Filename(s) of source data in data/<dataset_name>/original
                                 Can be a single string, list of strings, or None for auto-discovery
                                 If None, all supported files in original/ directory will be loaded
            author_disambiguation: Whether to apply author disambiguation (overrides component setting)
            overwrite: Whether to overwrite existing processed data (overrides component setting)
            data_filters: Dict of filters to apply before expensive operations (merges with component filters)
                         Example: {
                             'date_range': {'start': '2020-01-01', 'end': '2023-12-31'},
                             'categories': ['cs.AI', 'cs.LG', 'stat.ML']
                         }
        
        Returns:
            Self for method chaining
            
        Example:
            # Simple usage - configure filters first
            orchestrator = MstmlOrchestrator("arxiv")
            orchestrator.configure_data_filters(
                date_range={'start': '2020-01-01', 'end': '2023-12-31'},
                categories=['cs.AI', 'cs.LG']
            )
            orchestrator.load_data()
            
            # Override settings  
            orchestrator.load_data(
                author_disambiguation=False,
                data_filters={'categories': ['cs.AI']}
            )
        """
        # Skip loading if already loaded (unless overwrite)
        if self.documents_df is not None and (overwrite is False or (overwrite is None and not getattr(self, '_default_overwrite', False))):
            self.logger.info("Data already loaded. Use overwrite=True to reload.")
            return self
            
        self.logger.info(f"Loading uncleaned data from {self.dataset_directory}/original/")
        
        # Handle filename input or auto-discovery
        if original_data_fnames is None:
            # Auto-discover all supported files in original directory
            filenames = self._discover_data_files()
            self.logger.info(f"Auto-discovered {len(filenames)} file(s): {filenames}")
        elif isinstance(original_data_fnames, str):
            filenames = [original_data_fnames]
            self.logger.info(f"Loading specified file: {filenames[0]}")
        else:
            filenames = original_data_fnames
            self.logger.info(f"Loading {len(filenames)} specified file(s): {filenames}")
        
        # Validate that we have files to load
        if not filenames:
            raise ValueError("No files to load. Either specify filenames or ensure supported files exist in original/ directory.")
        
        # Handle DataLoader creation or configuration
        if self._data_loader is not None:
            # Use existing DataLoader but may need to update configuration
            self.logger.info("Using pre-configured DataLoader")
            
            # Check if it was created with with_config() and needs initialization
            if hasattr(self._data_loader, '_stored_config'):
                self._data_loader._apply_stored_config(filenames, self.dataset_name)
            
            # Apply parameter overrides if provided
            if author_disambiguation is not None:
                if author_disambiguation and self._data_loader.author_disambiguator is None:
                    self._data_loader.author_disambiguator = self._author_disambiguator or AuthorDisambiguator()
                elif not author_disambiguation:
                    self._data_loader.author_disambiguator = None
                    
            if overwrite is not None:
                self._data_loader._overwrite = overwrite
                
            # Merge data filters  
            if data_filters or self._pending_data_filters:
                current_filters = self._data_loader.data_filters.copy()
                if self._pending_data_filters:
                    current_filters.update(self._pending_data_filters)
                if data_filters:
                    current_filters.update(data_filters)
                self._data_loader.data_filters = current_filters
                
            self.data_loader = self._data_loader
            
        else:
            # Create new DataLoader based on file extensions
            # Collect all file extensions
            file_extensions = set()
            for fname in filenames:
                ext = fname.split('.')[-1].lower()
                file_extensions.add(ext)
            
            # Check if all files are compatible using registry
            if DataLoaderRegistry.are_extensions_compatible(file_extensions):
                # Get the appropriate loader class from registry
                loader_class = DataLoaderRegistry.get_loader_for_extensions(file_extensions)
                
                # Determine settings
                use_author_disambiguation = author_disambiguation if author_disambiguation is not None else True
                use_overwrite = overwrite if overwrite is not None else False
                
                # Merge data filters
                final_filters = {}
                if self._pending_data_filters:
                    final_filters.update(self._pending_data_filters)
                if data_filters:
                    final_filters.update(data_filters)
                
                # Create disambiguator
                disambiguator = self._author_disambiguator or AuthorDisambiguator() if use_author_disambiguation else None
                
                self.data_loader = loader_class(
                    input_path=filenames,
                    dataset_name=self.dataset_name,
                    overwrite=use_overwrite,
                    author_disambiguator=disambiguator,
                    data_filters=final_filters or None
                )
                
                # Store reference for future use
                self._data_loader = self.data_loader
                
            else:
                # Get supported extensions for error message
                supported_extensions = DataLoaderRegistry.get_supported_extensions()
                if len(file_extensions) == 1:
                    # Single unsupported extension
                    ext = list(file_extensions)[0]
                    raise ValueError(
                        f"Unsupported data format: .{ext}. "
                        f"Supported formats: {sorted(supported_extensions)}"
                    )
                else:
                    # Mixed incompatible formats
                    raise ValueError(
                        f"Incompatible file formats detected: {sorted(file_extensions)}. "
                        f"All files must be handled by the same loader. "
                        f"Supported formats: {sorted(supported_extensions)}"
                    )
        
        # Clear pending filters since they've been applied
        self._pending_data_filters = None
        
        # Run the complete data loading pipeline
        self.data_loader.run()
        
        # Get the processed dataframes
        self.documents_df = self.data_loader.get_clean_df()
        self.logger.info(f"Loaded {len(self.documents_df)} documents")
        
        return self
    
    def load_raw_data(self,
                      original_data_fnames: Optional[Union[str, List[str]]] = None,
                      overwrite: Optional[bool] = None,
                      input_schema_map: Optional[Dict[str, str]] = None) -> 'MstmlOrchestrator':
        """
        Load raw document data without applying any filters.
        
        This method loads the data into a DataFrame but skips all filtering steps,
        making it easier to inspect the raw data and debug filtering issues.
        
        Args:
            original_data_fnames: Filename(s) of source data in data/<dataset_name>/original
            overwrite: Whether to overwrite existing processed data
            input_schema_map: Dict mapping data field names to schema field names
                             Example for ArXiv: {'abstract': 'raw_text', 'update_date': 'date'}
        
        Returns:
            Self for method chaining
            
        Example:
            # Load raw data first to inspect
            orchestrator = MstmlOrchestrator("arxiv")
            orchestrator.load_raw_data()
            
            # Check what's in the data
            print(f"Total documents: {len(orchestrator.documents_df)}")
            print(f"Date range: {orchestrator.documents_df['date'].min()} to {orchestrator.documents_df['date'].max()}")
            
            # For ArXiv data, use field mapping:
            orchestrator.load_raw_data(
                input_schema_map={'abstract': 'raw_text', 'update_date': 'date'}
            )
        """
        # Skip loading if already loaded (unless overwrite)
        if self.documents_df is not None and (overwrite is False or (overwrite is None and not getattr(self, '_default_overwrite', False))):
            self.logger.info("Data already loaded. Use overwrite=True to reload.")
            return self
            
        self.logger.info(f"Loading raw data from {self.dataset_directory}/original/ (no filters)")
        
        # Handle filename input or auto-discovery
        if original_data_fnames is None:
            filenames = self._discover_data_files()
            self.logger.info(f"Auto-discovered {len(filenames)} file(s): {filenames}")
        elif isinstance(original_data_fnames, str):
            filenames = [original_data_fnames]
            self.logger.info(f"Loading specified file: {filenames[0]}")
        else:
            filenames = original_data_fnames
            self.logger.info(f"Loading {len(filenames)} specified file(s): {filenames}")
        
        if not filenames:
            raise ValueError("No files to load. Either specify filenames or ensure supported files exist in original/ directory.")
        
        # Create DataLoader WITHOUT any filters
        file_extensions = set()
        for fname in filenames:
            ext = fname.split('.')[-1].lower()
            file_extensions.add(ext)
        
        # Get the appropriate loader class
        if len(file_extensions) == 1:
            ext = list(file_extensions)[0]
            loader_class = DataLoaderRegistry.get_loader_class(ext)
            
            if loader_class is None:
                supported_extensions = DataLoaderRegistry.get_supported_extensions()
                raise ValueError(f"Unsupported file extension: '.{ext}'. Supported: {sorted(supported_extensions)}")
        elif DataLoaderRegistry.are_extensions_compatible(file_extensions):
            loader_class = DataLoaderRegistry.get_loader_for_extensions(file_extensions)
        else:
            supported_extensions = DataLoaderRegistry.get_supported_extensions()
            raise ValueError(f"Incompatible file formats: {sorted(file_extensions)}. Supported: {sorted(supported_extensions)}")
        
        # Create loader with NO processing components
        self.data_loader = loader_class(
            input_path=filenames,
            dataset_name=self.dataset_name,
            overwrite=overwrite,
            author_disambiguator=None,  # No author disambiguation!
            data_filters=None,  # No filters!
            input_schema_map=input_schema_map  # Field mapping for data format
        )
        
        self._data_loader = self.data_loader
        
        # ONLY run the minimal data loading steps (not the full pipeline)
        # Step 1: Load raw data from files
        self.data_loader._prepare_environment()
        self.data_loader._prepare_input()
        self.data_loader._load_raw_data()  # This sets self.data_loader.raw_data internally
        
        # Step 2: Convert to DataFrame with schema (no filtering, no author processing)
        self.data_loader._convert_to_schema()
        
        # Step 3: Basic validation and flagging only
        self.data_loader._validate_and_flag()
        
        # Get the raw dataframes (before any processing)
        self.documents_df = self.data_loader.get_clean_df()
        self.logger.info(f"Loaded {len(self.documents_df)} raw documents (no filters, no author disambiguation)")
        
        return self
    
    def apply_data_filters(self,
                          date_range: Optional[Dict[str, str]] = None,
                          categories: Optional[List[str]] = None,
                          custom_filter: Optional[callable] = None,
                          **kwargs) -> 'MstmlOrchestrator':
        """
        Apply data filters to already loaded data.
        
        This method allows you to apply filters step by step and see the impact
        of each filter on your dataset size.
        
        Args:
            date_range: Dict with 'start' and/or 'end' keys (ISO date strings)
            categories: List of category strings to keep
            custom_filter: Custom filter function that takes a DataFrame and returns a filtered DataFrame
            **kwargs: Additional filter parameters
        
        Returns:
            Self for method chaining
            
        Example:
            # Load raw data first
            orchestrator.load_raw_data()
            print(f"Raw data: {len(orchestrator.documents_df)} documents")
            
            # Apply date filter only
            orchestrator.apply_data_filters(date_range={'start': '2020-01-01', 'end': '2023-12-31'})
            print(f"After date filter: {len(orchestrator.documents_df)} documents")
            
            # Apply category filter
            orchestrator.apply_data_filters(categories=['cs.AI', 'cs.LG'])
            print(f"After category filter: {len(orchestrator.documents_df)} documents")
        """
        if self.documents_df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        initial_count = len(self.documents_df)
        self.logger.info(f"Applying filters to {initial_count} documents")
        
        # Build filter configuration - start with stored filters
        filters = {}
        
        # First, check for filters stored from configure_data_filters()
        if self._pending_data_filters:
            filters.update(self._pending_data_filters)
            self.logger.info(f"Using pending filters: {list(self._pending_data_filters.keys())}")
        elif self._data_loader and hasattr(self._data_loader, 'data_filters') and self._data_loader.data_filters:
            filters.update(self._data_loader.data_filters)
            self.logger.info(f"Using DataLoader filters: {list(self._data_loader.data_filters.keys())}")
        
        # Then, add any additional filters passed as parameters (these take precedence)
        if date_range:
            filters['date_range'] = self._normalize_date_range(date_range)
        if categories:
            filters['categories'] = categories
        if custom_filter:
            filters['custom'] = {'function': custom_filter}
        filters.update(kwargs)
        
        if not filters:
            self.logger.warning("No filters configured. Call configure_data_filters() first or pass filter parameters.")
            return self
        
        # Track hyperparameters for this step
        filter_params = filters.copy()
        self._track_step_params('data_filtering', filter_params)
        
        # Apply each filter and report results
        current_df = self.documents_df.copy()
        
        self.logger.info(f"Applying filters: {list(filters.keys())}")
        
        for filter_name, filter_config in filters.items():
            pre_filter_count = len(current_df)
            self.logger.info(f"Applying filter '{filter_name}' with config: {filter_config}")
            
            if hasattr(self.data_loader, '_apply_single_filter'):
                try:
                    current_df = self.data_loader._apply_single_filter(filter_name, filter_config, current_df)
                except Exception as e:
                    log_print(f"Filter '{filter_name}' failed: {e}", level="error", logger=self.logger)
                    continue
            else:
                self.logger.warning(f"Filter '{filter_name}' not supported by current DataLoader")
                continue
                
            post_filter_count = len(current_df)
            removed_count = pre_filter_count - post_filter_count
            
            self.logger.info(f"Filter '{filter_name}': {post_filter_count}/{pre_filter_count} documents retained ({removed_count} removed)")
            
            # Debug: Show sample data after filtering
            if post_filter_count > 0 and filter_name == 'categories':
                sample_categories = current_df['categories'].iloc[0] if 'categories' in current_df.columns else 'N/A'
                log_print(f"Sample categories after filter: {sample_categories}", level="debug", logger=self.logger)
            elif post_filter_count > 0 and filter_name == 'date_range':
                sample_date = current_df['date'].iloc[0] if 'date' in current_df.columns else 'N/A'
                log_print(f"Sample date after filter: {sample_date}", level="debug", logger=self.logger)
        
        # Update the main dataframe
        self.documents_df = current_df
        final_count = len(self.documents_df)
        total_removed = initial_count - final_count
        
        self.logger.info(f"Total filtering result: {final_count}/{initial_count} documents retained ({total_removed} removed)")
        
        # Clear pending filters since they've been applied
        if self._pending_data_filters:
            self._pending_data_filters = None
        
        return self
    
    def _track_step_params(self, step_name: str, params: dict):
        """Track hyperparameters for a processing step."""
        self.experiment_params['steps'][step_name] = {
            'timestamp': get_date_hour_minute(),
            'parameters': params.copy()
        }
        
        # Save updated params to experiment directory
        params_path = os.path.join(self.experiment_directory, 'experiment_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.experiment_params, f, indent=2, default=str)
    
    def _copy_essential_files_to_experiment(self):
        """Copy essential files to experiment directory for reproducibility."""
        if not hasattr(self, 'data_loader') or not self.data_loader:
            self.logger.warning("No data loader available - cannot copy essential files")
            return
            
        clean_dir = self.data_loader.dataset_dirs["clean"]
        essential_files = [
            'main_df.pkl',
            'id2word.pkl', 
            'author_to_authorId.pkl',
            'authorId_to_author.pkl',
            'authorId_to_df_row.pkl'
        ]
        
        copied_files = []
        for filename in essential_files:
            src_path = os.path.join(clean_dir, filename)
            dst_path = os.path.join(self.experiment_directory, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_files.append(filename)
            else:
                self.logger.warning(f"Essential file not found: {filename}")
        
        self.logger.info(f"Copied {len(copied_files)} essential files to experiment directory: {copied_files}")
    
    def _update_main_dataframe(self):
        """Update main_df.pkl in both clean directory and experiment directory when dataframe is modified."""
        if not hasattr(self, 'data_loader') or not self.data_loader:
            self.logger.warning("No data loader available - cannot update main_df.pkl")
            return
            
        if self.documents_df is None:
            self.logger.warning("No documents dataframe available - cannot update main_df.pkl")
            return
        
        # Update in clean directory
        clean_dir = self.data_loader.dataset_dirs["clean"]
        clean_path = os.path.join(clean_dir, 'main_df.pkl')
        write_pickle(clean_path, self.documents_df)
        
        # Update in experiment directory
        exp_path = os.path.join(self.experiment_directory, 'main_df.pkl')
        write_pickle(exp_path, self.documents_df)
        
        self.logger.info(f"Updated main_df.pkl in both clean directory and experiment directory ({len(self.documents_df)} rows)")
    
    def apply_author_disambiguation(self) -> 'MstmlOrchestrator':
        """
        Apply author disambiguation to already loaded and filtered data.
        
        This method should be called after load_raw_data() and apply_data_filters()
        to perform author name disambiguation and ID assignment.
        
        Returns:
            Self for method chaining
            
        Example:
            # Load raw data, filter, then apply author disambiguation
            orchestrator.load_raw_data()
            orchestrator.apply_data_filters(date_range={'start': '2020-01-01', 'end': '2023-12-31'})
            orchestrator.apply_author_disambiguation()
        """
        if self.documents_df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        # Get or create author disambiguator
        disambiguator = self._author_disambiguator
        if disambiguator is None:
            disambiguator = AuthorDisambiguator()
            self._author_disambiguator = disambiguator
        
        # Track hyperparameters for this step
        disambig_params = {
            'similarity_threshold': getattr(disambiguator, 'similarity_threshold', 'default'),
            'max_authors_per_doc': getattr(disambiguator, 'max_authors_per_doc', 'default'),
            'ngram_range': getattr(disambiguator, 'ngram_range', 'default')
        }
        self._track_step_params('author_disambiguation', disambig_params)
        
        initial_count = len(self.documents_df)
        self.logger.info(f"Applying author disambiguation to {initial_count} documents")
        
        # Apply author disambiguation to the current dataframe
        processed_series = disambiguator.update_dataframe(
            self.documents_df, 
            MainDataSchema.AUTHOR_NAMES.colname
        )
        
        # Update the dataframe with author IDs
        self.documents_df[MainDataSchema.AUTHOR_IDS.colname] = processed_series
        
        final_count = len(self.documents_df)
        self.logger.info(f"Author disambiguation completed on {final_count} documents")
        
        # Update main_df.pkl with the new author IDs column
        self._update_main_dataframe()
        
        return self

    def _discover_data_files(self) -> List[str]:
        """
        Auto-discover all supported data files in the original directory.
        
        Returns:
            List of filenames found in the original directory
            
        Raises:
            FileNotFoundError: If original directory doesn't exist
            ValueError: If no supported files are found
        """
        original_dir = Path(self.dataset_directory) / "original"
        
        if not original_dir.exists():
            raise FileNotFoundError(f"Original data directory not found: {original_dir}")
        
        # Use registry to discover supported files
        discovered_files = DataLoaderRegistry.discover_supported_files(original_dir)
        
        if not discovered_files:
            available_files = [f.name for f in original_dir.iterdir() if f.is_file()]
            supported_extensions = DataLoaderRegistry.get_supported_extensions()
            raise ValueError(
                f"No supported data files found in {original_dir}. "
                f"Supported extensions: {sorted(supported_extensions)}. "
                f"Available files: {available_files}"
            )
        
        self.logger.info(f"Discovered {len(discovered_files)} supported file(s) in {original_dir}")
        return discovered_files
    
    def setup_coauthor_network(self, 
                              min_collaborations: Optional[int] = None,
                              temporal: Optional[bool] = None,
                              degree_limits: Optional[tuple] = None) -> 'MstmlOrchestrator':
        """
        Create co-author network from loaded document data using _graph_driver functions.
        Authors should already be disambiguated from the load_data step.
        
        Args:
            min_collaborations: Minimum collaborations to include edge (overrides config)
            temporal: Whether to build temporal networks and compose them (overrides config)
            degree_limits: Optional (min_degree, max_degree) tuple to filter nodes (overrides config)
        
        Returns:
            Self for method chaining
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded. Call load_data() first.")
        
        # Get config defaults and apply parameter overrides
        network_config = self.config['coauthor_network']
        min_collaborations = min_collaborations if min_collaborations is not None else network_config['min_collaborations']
        temporal = temporal if temporal is not None else network_config['temporal']
        degree_limits = degree_limits if degree_limits is not None else network_config['degree_limits']
        
        self.logger.info("Setting up co-author network using graph driver functions")
        
        if temporal and MainDataSchema.DATE.colname in self.documents_df.columns:
            # Set up save directories and dataset name
            dataset_name = getattr(self.data_loader, '_dataset_name', 'unknown')
            networks_dir = self.data_loader.dataset_dirs["networks"]
            
            # Build temporal networks and compose them
            self.logger.info("Building temporal co-author networks")
            year_networks = build_temporal_coauthor_networks_from_dataframe(
                self.documents_df,
                author_names_col=MainDataSchema.AUTHOR_NAMES.colname,
                author_ids_col=MainDataSchema.AUTHOR_IDS.colname,
                date_col=MainDataSchema.DATE.colname,
                min_collaborations=min_collaborations,
                save_dir=networks_dir,
                dataset_name=dataset_name
            )
            
            # Create composed network filename
            if year_networks:
                years = sorted(year_networks.keys())
                composed_filename = f"composed_{dataset_name}_{years[0]}-{years[-1]}.graphml"
                composed_path = os.path.join(networks_dir, composed_filename)
            else:
                composed_path = None
            
            # Compose yearly networks into single network
            self.coauthor_network = compose_networks_from_dict(
                year_networks, 
                degree_limits, 
                save_path=composed_path
            )
            
            self.logger.info(f"Composed networks from {len(year_networks)} time periods")
            
        else:
            # Build single co-author network
            self.coauthor_network = build_coauthor_network_from_dataframe(
                self.documents_df,
                author_names_col=MainDataSchema.AUTHOR_NAMES.colname,
                author_ids_col=MainDataSchema.AUTHOR_IDS.colname,
                min_collaborations=min_collaborations
            )
            
            # Apply degree limits if specified
            if degree_limits:
                if not isinstance(degree_limits, tuple) or len(degree_limits) != 2:
                    raise ValueError("degree_limits should be a 2-tuple of (min, max) degrees")
                
                nodes_to_remove = [
                    node for node, degree in dict(self.coauthor_network.degree()).items() 
                    if degree < degree_limits[0] or degree > degree_limits[1]
                ]
                self.coauthor_network.remove_nodes_from(nodes_to_remove)
                self.logger.info(f"Removed {len(nodes_to_remove)} nodes due to degree limits {degree_limits}")
            
            # Save single network
            dataset_name = getattr(self.data_loader, '_dataset_name', 'unknown')
            networks_dir = self.data_loader.dataset_dirs["networks"]
            os.makedirs(networks_dir, exist_ok=True)
            
            network_filename = f"{dataset_name}_coauthor_network.graphml"
            network_path = os.path.join(networks_dir, network_filename)
            nx.write_graphml(self.coauthor_network, network_path)
            self.logger.info(f"Saved co-author network to {network_path}")
        
        # Add author metadata to nodes if authors_df is available
        if self.authors_df is not None:
            for _, author_row in self.authors_df.iterrows():
                author_id = str(author_row['author_id'])
                if author_id in self.coauthor_network.nodes:
                    self.coauthor_network.nodes[author_id].update({
                        'name': author_row['name'],
                        'publication_count': author_row['publication_count']
                    })
        
        self.logger.info(
            f"Created co-author network with {self.coauthor_network.number_of_nodes()} nodes "
            f"and {self.coauthor_network.number_of_edges()} edges "
            f"(min_collaborations={min_collaborations})"
        )
        
        return self
    
    # ========================================================================================
    # 2. TEXT PREPROCESSING AND VOCABULARY FILTERING
    # ========================================================================================
    
    def preprocess_text(self,
                       allowed_categories: Optional[List[str]] = None,
                       low_thresh: Optional[int] = None,
                       high_frac: Optional[float] = None,
                       extra_stopwords: Optional[List[str]] = None,
                       train_lda: Optional[bool] = None,
                       num_topics: Optional[int] = None,
                       lda_passes: Optional[int] = None,
                       lambda_param: Optional[float] = None,
                       top_n_terms: Optional[int] = None) -> 'MstmlOrchestrator':
        """
        Apply comprehensive text preprocessing and vocabulary filtering using TextPreprocessor.
        
        All parameters are optional and will use values from the configuration file if not provided.
        
        Args:
            allowed_categories: List of allowed categories for filtering
            low_thresh: Minimum document frequency threshold (overrides config)
            high_frac: Maximum document frequency fraction (overrides config)
            extra_stopwords: Additional stopwords to remove
            train_lda: Whether to train LDA model for relevancy filtering (overrides config)
            num_topics: Number of topics for LDA model (overrides config)
            lda_passes: Number of passes for LDA training (overrides config)
            lambda_param: Lambda parameter for relevancy scoring (overrides config)
            top_n_terms: Number of top relevant terms to keep (overrides config)
        
        Returns:
            Self for method chaining
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded. Call load_data() first.")
        
        # Get config defaults and apply parameter overrides
        vocab_config = self.config['text_processing']['vocabulary_filtering']
        lda_config = self.config['text_processing']['lda_relevancy']  
        relevancy_config = self.config['analysis']['term_relevancy']
        
        # Apply config defaults or parameter overrides
        low_thresh = low_thresh if low_thresh is not None else vocab_config['low_thresh']
        high_frac = high_frac if high_frac is not None else vocab_config['high_frac']
        train_lda = train_lda if train_lda is not None else lda_config['train_lda']
        num_topics = num_topics if num_topics is not None else lda_config['num_topics']
        lda_passes = lda_passes if lda_passes is not None else lda_config['lda_passes']
        lambda_param = lambda_param if lambda_param is not None else relevancy_config['lambda']
        top_n_terms = top_n_terms if top_n_terms is not None else relevancy_config['top_terms']
        
        # Track hyperparameters for this step
        step_params = {
            'allowed_categories': allowed_categories,
            'low_thresh': low_thresh,
            'high_frac': high_frac,
            'extra_stopwords': extra_stopwords,
            'train_lda': train_lda,
            'num_topics': num_topics,
            'lda_passes': lda_passes,
            'lambda_param': lambda_param,
            'top_n_terms': top_n_terms
        }
        self._track_step_params('text_preprocessing', step_params)
        
        self.logger.info("Starting comprehensive text preprocessing pipeline")
        
        # Initialize TextPreprocessor if not already set
        if self._text_preprocessor is None:
            self.text_preprocessor = TextPreprocessor()
        else:
            self.text_preprocessor = self._text_preprocessor
        
        # Apply the complete preprocessing pipeline
        processed_docs = self.text_preprocessor.update_dataframe(
            self.documents_df,
            text_column=MainDataSchema.RAW_TEXT.colname,
            allowed_categories=allowed_categories,
            low_thresh=low_thresh,
            high_frac=high_frac,
            extra_stopwords=extra_stopwords,
            train_lda=train_lda,
            num_topics=num_topics,
            lda_passes=lda_passes,
            lambda_param=lambda_param,
            top_n_terms=top_n_terms
        )
        
        # Store results in orchestrator and update dataframe
        # Note: preprocessed text stored in dataframe to avoid memory duplication
        self.vocabulary = self.text_preprocessor.get_dictionary()
        self.bow_corpus = self.text_preprocessor.get_corpus()
        
        # Update the main dataframe with preprocessed text
        self.documents_df[MainDataSchema.PREPROCESSED_TEXT.colname] = processed_docs
        
        # Get preprocessing statistics
        stats = self.text_preprocessor.get_stats_log()
        
        self.logger.info(
            f"Text preprocessing complete. "
            f"Initial vocab: {stats.get('vocab_initial', 'N/A')}, "
            f"After filtering: {stats.get('vocab_filtered', 'N/A')}, "
            f"Final vocab: {stats.get('vocab_final', len(self.vocabulary))}"
        )
        
        # Save essential preprocessing results and copy to experiment directory
        self._save_preprocessing_results()
        self._copy_essential_files_to_experiment()
        
        # Update main_df.pkl with the new preprocessed_text column
        self._update_main_dataframe()
        
        return self
    
    def _save_preprocessing_results(self):
        """Save only essential preprocessing results to data/clean/ directory."""
        if not hasattr(self, 'data_loader') or not self.data_loader:
            self.logger.warning("No data loader available - cannot save preprocessing results")
            return
            
        clean_dir = self.data_loader.dataset_dirs["clean"]
        
        # Save only the vocabulary dictionary (essential for reproducibility)
        dict_path = os.path.join(clean_dir, 'id2word.pkl')
        write_pickle(dict_path, self.vocabulary)
        self.logger.info(f"Saved vocabulary dictionary with {len(self.vocabulary)} terms to id2word.pkl")
    
    
    # ========================================================================================
    # 3. TEMPORAL ENSEMBLE TOPIC MODELING
    # ========================================================================================
    
    def create_temporal_chunks(self, 
                             months_per_chunk: Optional[int] = None,
                             temporal_smoothing_decay: Optional[float] = None,
                             save_statistics: Optional[bool] = None,
                             create_plot: Optional[bool] = None,
                             save_plot: Optional[bool] = None) -> 'MstmlOrchestrator':
        """
        Split corpus into temporal chunks for ensemble learning.
        
        Args:
            months_per_chunk: Number of months per temporal chunk (overrides config)
            temporal_smoothing_decay: Exponential decay factor for temporal smoothing (0-1), overrides config if provided
            save_statistics: Whether to save chunk statistics to file (overrides config)
            create_plot: Whether to create a visualization of chunk sizes (overrides config)
            save_plot: Whether to save the plot to file (overrides config)
        
        Returns:
            Self for method chaining
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded. Call load_data() first.")
        
        # Get config defaults and apply parameter overrides
        temporal_config = self.config['temporal']
        months_per_chunk = months_per_chunk if months_per_chunk is not None else temporal_config['months_per_chunk']
        save_statistics = save_statistics if save_statistics is not None else temporal_config['save_statistics']
        create_plot = create_plot if create_plot is not None else temporal_config['create_plot']
        save_plot = save_plot if save_plot is not None else temporal_config['save_plot']
        
        self.logger.info(f"Creating temporal chunks with {months_per_chunk} months per chunk")
        
        # Update temporal smoothing decay if provided
        if temporal_smoothing_decay is not None:
            if not (0 < temporal_smoothing_decay < 1):
                raise ValueError(f"temporal_smoothing_decay must be in (0,1), got {temporal_smoothing_decay}")
            self.config['temporal']['gamma_temporal_decay'] = temporal_smoothing_decay
            self.logger.info(f"Updated temporal smoothing decay to {temporal_smoothing_decay}")
        
        # Ensure date column is datetime
        if 'date' not in self.documents_df.columns:
            raise ValueError("Documents dataframe must have a 'date' column")
        
        df = self.documents_df.copy()
        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by the specified interval and create chunks
        chunks = [group for _, group in df.groupby(pd.Grouper(key='date', freq=f'{months_per_chunk}ME'))]
        
        # Convert to the expected format
        self.time_chunks = []
        for chunk_id, chunk_df in enumerate(chunks):
            if not chunk_df.empty:  # Only include non-empty chunks
                chunk_info = {
                    'chunk_id': chunk_id,
                    'start_date': chunk_df['date'].min(),
                    'end_date': chunk_df['date'].max(),
                    'document_indices': chunk_df.index.tolist(),
                    'num_documents': len(chunk_df)
                }
                self.time_chunks.append(chunk_info)
        
        self.logger.info(f"Created {len(self.time_chunks)} temporal chunks")
        
        # Log chunk statistics
        if self.time_chunks:
            chunk_sizes = [chunk['num_documents'] for chunk in self.time_chunks]
            self.logger.info(
                f"Chunk sizes - min: {min(chunk_sizes)}, max: {max(chunk_sizes)}, "
                f"mean: {np.mean(chunk_sizes):.1f}"
            )
            
            # Save statistics to file
            if save_statistics:
                self._save_chunk_statistics(months_per_chunk, chunk_sizes)
            
            # Create and optionally save plot
            if create_plot:
                self._create_chunk_plot(months_per_chunk, chunk_sizes, save_plot)
        
        return self
    
    def _save_chunk_statistics(self, months_per_chunk: int, chunk_sizes: List[int]):
        """Save temporal chunk statistics to JSON file."""
        # Use experiment directory for all results
        save_dir = self.experiment_directory
        
        stats = {
            'months_per_chunk': months_per_chunk,
            'num_chunks': len(chunk_sizes),
            'chunk_sizes': chunk_sizes,
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'mean_size': float(np.mean(chunk_sizes)),
            'std_size': float(np.std(chunk_sizes)),
            'total_documents': sum(chunk_sizes),
            'chunk_date_ranges': [
                {
                    'chunk_id': chunk['chunk_id'],
                    'start_date': chunk['start_date'].strftime('%Y-%m-%d'),
                    'end_date': chunk['end_date'].strftime('%Y-%m-%d'),
                    'num_documents': chunk['num_documents']
                }
                for chunk in self.time_chunks
            ]
        }
        
        stats_path = os.path.join(save_dir, f'temporal_chunks_{months_per_chunk}month_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved temporal chunk statistics to {stats_path}")
    
    def _create_chunk_plot(self, months_per_chunk: int, chunk_sizes: List[int], save_plot: bool):
        """Create visualization of temporal chunk sizes."""
        # Create month-year labels for each chunk based on the first date in each chunk
        chunk_labels = [chunk['start_date'].strftime('%b-%Y') for chunk in self.time_chunks]
        
        # Calculate tick intervals to label up to 20 bars
        if len(chunk_labels) <= 20:
            tick_indices = np.arange(len(chunk_labels))  # Label all bars if there are 20 or fewer
        else:
            tick_interval = max(1, len(chunk_labels) // 20)  # Calculate interval to label up to 20 bars
            tick_indices = np.arange(0, len(chunk_labels), tick_interval)  # Select indices to label
        
        # Plotting
        plt.figure(figsize=(12, 6))
        bars = plt.bar(list(range(len(chunk_sizes))), chunk_sizes)  # Plot with the indices of the chunks
        plt.xlabel("Time Chunk")
        plt.ylabel("Number of Documents")
        plt.title(f"Number of Documents in {months_per_chunk}-Month Chunks")
        
        # Customize x-axis ticks with formatted chunk labels at selected positions
        plt.xticks(tick_indices, [chunk_labels[i] for i in tick_indices], rotation=45, ha="right")
        
        # Add text annotations above selected bars at tick positions
        for i in tick_indices:
            plt.text(i, chunk_sizes[i] + max(chunk_sizes) * 0.02,  # Position text slightly above the bar
                     str(chunk_sizes[i]), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the figure
        if save_plot:
            # Use experiment directory for all results
            plot_path = os.path.join(self.experiment_directory, f'document_counts_{months_per_chunk}month_chunks.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            self.logger.info(f"Saved temporal chunk plot to {plot_path}")
        
        plt.show()
    
    def _apply_temporal_smoothing(self) -> 'MstmlOrchestrator':
        """
        Apply gamma-weighted temporal smoothing across chunks with random sampling.
        
        Implements the reference algorithm: for each chunk, includes all base documents
        plus gamma-weighted random samples from other chunks based on temporal distance.
        Uses smoothing_decay parameter from configuration.
        
        Returns:
            Self for method chaining
        """
        if not self.time_chunks:
            raise ValueError("No temporal chunks created. Call create_temporal_chunks() first.")
        
        # Get smoothing decay from configuration
        smoothing_decay = self.config['temporal']['gamma_temporal_decay']
        
        if not (0 < smoothing_decay < 1):
            raise ValueError(f"Smoothing decay parameter must be in (0,1), got {smoothing_decay}")
        
        self.gamma_temporal_decay = smoothing_decay
        log_print(f"Applying temporal smoothing with decay γ={smoothing_decay}", level="info", logger=self.logger)
        
        # Create indices mapping as in reference
        chunk_list = list(range(len(self.time_chunks)))
        inds_by_chunk = {}
        
        for chunk_idx, chunk in enumerate(self.time_chunks):
            inds_by_chunk[chunk_idx] = list(chunk['document_indices'])
        
        # Apply gamma-weighted smoothing (following reference implementation)
        slice_df_inds = {}
        smoothed_chunks = []
        
        for c_base in chunk_list:
            # Base chunk indices (always included fully)
            c_base_data_words_idx = set(inds_by_chunk[c_base])
            c_offset_data_words_idx = set()
            
            # Sample from other chunks with gamma weighting
            for c_offset in chunk_list:
                if c_offset == c_base:
                    continue
                    
                # Calculate weight based on temporal distance (as in reference)
                weight = np.float32(smoothing_decay ** abs(c_offset - c_base))
                
                # Random sample with decay-proportional size (as in reference)
                offset_chunk_size = len(inds_by_chunk[c_offset])
                sample_size = int(weight * offset_chunk_size)
                
                if sample_size > 0 and offset_chunk_size > 0:
                    rsamp = random.sample(list(inds_by_chunk[c_offset]), 
                                        min(sample_size, offset_chunk_size))
                    c_offset_data_words_idx.update(rsamp)
            
            # Combine base and offset indices (as in reference)
            s_data_words_idx = np.concatenate((list(c_base_data_words_idx), 
                                             list(c_offset_data_words_idx)))
            slice_df_inds[c_base] = s_data_words_idx
            
            # Create smoothed chunk info
            original_chunk = self.time_chunks[c_base]
            smoothed_chunk = {
                'chunk_id': original_chunk['chunk_id'],
                'start_date': original_chunk['start_date'],
                'end_date': original_chunk['end_date'],
                'document_indices': list(s_data_words_idx),
                'num_documents': len(s_data_words_idx),
                'original_num_documents': len(c_base_data_words_idx),
                'base_documents': len(c_base_data_words_idx),
                'sampled_documents': len(c_offset_data_words_idx),
                'gamma_temporal_decay': smoothing_decay
            }
            smoothed_chunks.append(smoothed_chunk)
        
        # Update time chunks with smoothed versions
        self.time_chunks = smoothed_chunks
        self.slice_df_indices = slice_df_inds  # Store for corpus recreation
        
        # Log smoothing statistics
        base_sizes = [chunk['base_documents'] for chunk in self.time_chunks]
        total_sizes = [chunk['num_documents'] for chunk in self.time_chunks]
        sampled_sizes = [chunk['sampled_documents'] for chunk in self.time_chunks]
        
        log_print(
            f"Temporal smoothing complete. Average sizes: "
            f"base={np.mean(base_sizes):.1f}, "
            f"sampled={np.mean(sampled_sizes):.1f}, "
            f"total={np.mean(total_sizes):.1f}",
            level="info", logger=self.logger
        )
        
        return self
    
    def train_ensemble_models(self,
                              base_model: Optional[str] = None,
                              topics_per_chunk: Optional[int] = None,
                              docs_per_topic: Optional[int] = None,
                              model_params: Optional[dict] = None) -> 'MstmlOrchestrator':
        """
        Train ensemble of topic models on temporal chunks.
        
        Args:
            base_model: Base topic model type ('LDA', 'T-LDA', etc.) (overrides config)
            topics_per_chunk: Number of topics per chunk model (overrides automatic calculation)
            docs_per_topic: Target number of documents per topic for automatic calculation (overrides config)
            model_params: Additional model parameters
        
        Returns:
            Self for method chaining
        """
        
        if not self.time_chunks:
            raise ValueError("No temporal chunks created. Call create_temporal_chunks() first.")
        
        if (self.documents_df is None or 
            MainDataSchema.PREPROCESSED_TEXT.colname not in self.documents_df.columns or
            self.bow_corpus is None):
            raise ValueError("No preprocessed corpus available. Call preprocess_text() first.")
        
        # Get config defaults and apply parameter overrides
        topic_config = self.config['topic_model']
        base_model = base_model if base_model is not None else topic_config['type']
        docs_per_topic = docs_per_topic if docs_per_topic is not None else topic_config['docs_per_topic']
        
        self.logger.info(f"Training {base_model} ensemble models")
        
        # Apply temporal smoothing before training models
        self._apply_temporal_smoothing()
        
        # Set default model parameters
        if model_params is None:
            params_section = topic_config.get('params', {})
            model_params = params_section.get(base_model.lower(), {})
        
        self.chunk_topic_models = []
        self.chunk_topics = []
        
        for chunk_idx, chunk_info in enumerate(self.time_chunks):
            chunk_doc_indices = chunk_info['document_indices']

            # Create chunk-specific BOW corpus following reference implementation
            # Get preprocessed text for all documents in this chunk (including smoothed)
            chunk_df = self.documents_df.loc[chunk_doc_indices]
            
            # Extract preprocessed text and create BOW corpus (as in reference)
            preprocessed_col = MainDataSchema.PREPROCESSED_TEXT.colname
            s_data_words = chunk_df[preprocessed_col].tolist()
            
            # Create BOW corpus for this chunk using the dictionary (as in reference)
            if self.vocabulary is None:
                raise AttributeError("vocabulary not initialized. Call preprocess_text() first.")
            chunk_bow_corpus = [self.vocabulary.doc2bow(text) for text in s_data_words]
            valid_indices = list(chunk_doc_indices)
            
            if not chunk_bow_corpus:
                log_print(f"No valid documents found for chunk {chunk_idx + 1}", level="warning", logger=self.logger)
                continue
            
            # Update chunk size to reflect actual valid documents
            chunk_size = len(chunk_bow_corpus)
            
            # Determine number of topics for this chunk
            if topics_per_chunk is None:
                # Use original affine function: minimum of 4 topics per chunk, 
                # unless the chunk has fewer than 4 documents
                k_topics = max(min(4, chunk_size), chunk_size // docs_per_topic)
            else:
                k_topics = topics_per_chunk
            
            self.logger.info(
                f"Training chunk {chunk_idx + 1}/{len(self.time_chunks)}: "
                f"{chunk_size} docs, {k_topics} topics"
            )
            
            if base_model.upper() == 'LDA':
                # Get alpha and eta parameters
                alpha_param = model_params.get('alpha', 'auto')
                eta_param = model_params.get('eta', 'auto')
                
                # Use LdaModel for auto-tuning, LdaMulticore otherwise
                if alpha_param == 'auto' or eta_param == 'auto':
                    self.logger.info("Using LdaModel for auto-tuning alpha/eta parameters")
                    chunk_model = LdaModel(
                        corpus=chunk_bow_corpus,
                        id2word=self.vocabulary,
                        num_topics=k_topics,
                        passes=model_params.get('npasses', 5),
                        iterations=model_params.get('ngibbs', 50),
                        random_state=42,
                        alpha=alpha_param,
                        eta=eta_param
                    )
                else:
                    # Use faster LdaMulticore when not using auto-tuning
                    chunk_model = LdaMulticore(
                        corpus=chunk_bow_corpus,
                        id2word=self.vocabulary,
                        num_topics=k_topics,
                        passes=model_params.get('npasses', 5),
                        iterations=model_params.get('ngibbs', 50),
                        random_state=42,
                        alpha=alpha_param,
                        eta=eta_param
                    )
                
                # Extract topic-word distributions φ(k)
                topic_word_distributions = chunk_model.get_topics()  # Shape: (num_topics, vocab_size)
                
                # Extract document-topic distributions for this chunk
                doc_topic_distributions = []
                for doc_bow in chunk_bow_corpus:
                    doc_topics = chunk_model.get_document_topics(doc_bow, minimum_probability=0.0)
                    doc_topic_array = np.zeros(k_topics)
                    for topic_id, prob in doc_topics:
                        doc_topic_array[topic_id] = prob
                    doc_topic_distributions.append(doc_topic_array)
                
                # Calculate perplexity metrics
                log_perplexity = chunk_model.log_perplexity(chunk_bow_corpus)
                actual_perplexity = np.exp(-1.0 * log_perplexity)
                
                chunk_model_info = {
                    'model': chunk_model,
                    'model_type': base_model,
                    'chunk_id': chunk_idx,
                    'num_topics': k_topics,
                    'num_documents': chunk_size,
                    'topic_word_distributions': topic_word_distributions,
                    'document_topic_distributions': np.array(doc_topic_distributions),
                    'document_indices': chunk_doc_indices,
                    'log_perplexity': log_perplexity,  # Gensim's log perplexity (negative value, closer to 0 is better)
                    'perplexity': actual_perplexity,   # Converted perplexity (positive value, lower is better)
                    'coherence': None  # Can be computed later if needed
                }
                
            else:
                raise ValueError(f"Unsupported base model: {base_model}")
            
            self.chunk_topic_models.append(chunk_model_info)
            
            # Store individual topic vectors for manifold construction
            for topic_idx in range(k_topics):
                topic_vector = topic_word_distributions[topic_idx]
                topic_info = {
                    'chunk_id': chunk_idx,
                    'topic_id': topic_idx,
                    'global_topic_id': len(self.chunk_topics),  # Global index across all chunks
                    'vector': topic_vector,
                    'chunk_size': chunk_size
                }
                self.chunk_topics.append(topic_info)
        
        # Collect all topic vectors for manifold learning
        self.topic_vectors = np.array([topic['vector'] for topic in self.chunk_topics])
        
        total_topics = len(self.chunk_topics)
        avg_log_perplexity = np.mean([model['log_perplexity'] for model in self.chunk_topic_models])
        
        # Correct perplexity calculation: exp(-mean(log_perplexities))
        # This is equivalent to geometric mean of individual perplexities
        avg_perplexity = np.exp(-1.0 * avg_log_perplexity)
        
        self.logger.info(
            f"Ensemble training complete: {len(self.chunk_topic_models)} models, "
            f"{total_topics} total topics, "
            f"average log-perplexity: {avg_log_perplexity:.2f} (closer to 0 is better), "
            f"average perplexity: {avg_perplexity:.2f} (lower is better)"
        )
        
        # Build the document-topic distribution and author processing pipeline
        self._build_author_document_distributions()
        
        return self
    
    def _build_author_document_distributions(self) -> None:
        """
        Build document-topic and author-topic distributions with diffusion processing.
        
        This implements the core document → expanded → author → diffusion pipeline:
        1. Collect document-topic distributions from all chunk models
        2. Expand distributions to full chunk topic space
        3. Compute author barycenters with inverse author count weighting
        4. Create diffusion KNN graph and apply diffusion process
        """
        self.logger.info("Building author-document distribution pipeline")
        
        # Step 1: Collect document-topic distributions from chunk models
        doc_topic_distns = {}
        inds_by_chunk = {}
        ntopics_by_chunk = {}
        
        for model_info in self.chunk_topic_models:
            chunk_id = model_info['chunk_id'] 
            doc_indices = model_info['document_indices']
            doc_topic_matrix = model_info['document_topic_distributions']
            
            inds_by_chunk[chunk_id] = doc_indices
            ntopics_by_chunk[chunk_id] = model_info['num_topics']
            
            # Map document indices to their topic distributions
            for i, doc_idx in enumerate(doc_indices):
                doc_topic_distns[doc_idx] = doc_topic_matrix[i]
        
        # Step 2: Expand document-topic distributions to full chunk topic space
        self.logger.info("Expanding document-topic distributions to full chunk topic space")
        num_chunks = len(self.chunk_topic_models)
        
        self.expanded_doc_topic_distns = expand_doc_topic_distns(
            doc_topic_distns, inds_by_chunk, num_chunks, ntopics_by_chunk
        )
        
        # Step 3: Compute author barycenters
        self.logger.info("Computing author barycenters with inverse author count weighting")
        author_column = MainDataSchema.AUTHOR_IDS.colname
        
        self.author_topic_barycenters, self.authId_to_docs, self.authId2docweights = compute_author_barycenters(
            self.expanded_doc_topic_distns, self.documents_df, author_column
        )
        
        if not self.author_topic_barycenters:
            raise ValueError("No author barycenters computed. Check author data in documents_df.")
        
        # Step 4: Create diffusion KNN graph and apply diffusion
        self._create_diffusion_knn_graph()
        
        self.logger.info(
            f"Author-document pipeline complete: "
            f"{len(self.expanded_doc_topic_distns)} documents, "
            f"{len(self.author_topic_barycenters)} authors, "
            f"diffusion applied with {self.config['author_doc_embeddings']['ct_distn_diffusion_knnk']} neighbors"
        )
    
    def _create_diffusion_knn_graph(self) -> None:
        """
        Create KNN graph for chunk topic diffusion and apply diffusion process.
        Uses the small KNN parameter for local diffusion of author/document distributions.
        """
        
        diffusion_knnk = self.config['author_doc_embeddings']['ct_distn_diffusion_knnk']
        self.logger.info(f"Creating diffusion KNN graph with k={diffusion_knnk}")
        
        # Use topic vectors (chunk_topic_wfs) for KNN graph construction
        if self.topic_vectors is None:
            raise ValueError("Topic vectors not available for KNN graph construction")
        
        # Try FAISS first, fallback to scipy if not available
        try:
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS not available")
                
            self.logger.info("Using FAISS for diffusion KNN graph construction")
            # Apply sqrt transformation for Hellinger distance in FAISS
            sqrt_topic_vectors = np.sqrt(self.topic_vectors).astype(np.float32)
            
            # Build FAISS index for KNN search
            dimension = sqrt_topic_vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(sqrt_topic_vectors)
            
            # Perform KNN search
            distances, indices = index.search(sqrt_topic_vectors, diffusion_knnk + 1)  # +1 to include self
            
            # Normalize distances for Hellinger (divide by sqrt(2))
            distances = distances / np.sqrt(2.0).astype(np.float32)
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"FAISS not available for diffusion KNN: {e}. Using scipy fallback")
            # Fallback to scipy for small datasets
            from scipy.spatial.distance import pdist, squareform
            
            # Apply sqrt transformation for Hellinger distance
            sqrt_topic_vectors = np.sqrt(self.topic_vectors).astype(np.float32)
            
            # Compute pairwise Hellinger distances using scipy
            hellinger_distances = pdist(sqrt_topic_vectors, metric='euclidean') / np.sqrt(2.0)
            distance_matrix = squareform(hellinger_distances)
            
            # Get KNN indices for each topic
            indices = []
            distances = []
            for i in range(len(sqrt_topic_vectors)):
                row_distances = distance_matrix[i]
                # Get indices of k+1 nearest neighbors (including self)
                nearest_indices = np.argsort(row_distances)[:diffusion_knnk + 1]
                nearest_distances = row_distances[nearest_indices]
                indices.append(nearest_indices)
                distances.append(nearest_distances)
            
            indices = np.array(indices)
            distances = np.array(distances)
        
        # Construct NetworkX graph
        self.knn_chunk_topic_graph = nx.Graph()
        num_topics = len(self.topic_vectors)
        
        # Add all nodes
        self.knn_chunk_topic_graph.add_nodes_from(range(num_topics))
        
        # Add edges (skip first index which is self)
        for i in range(num_topics):
            max_neighbors = min(diffusion_knnk + 1, indices.shape[1])
            for j in range(1, max_neighbors):  # Start from 1 to skip self
                if j < indices.shape[1]:
                    neighbor_idx = indices[i, j]
                    distance = float(distances[i, j])
                    self.knn_chunk_topic_graph.add_edge(i, neighbor_idx, weight=distance)
        
        # Apply diffusion to author and document distributions  
        self.logger.info("Applying diffusion to author and document distributions")
        
        # Get diffusion parameters from config
        num_iterations = self.config['author_doc_embeddings']['num_diffusion_iterations']
        diffusion_rate = self.config['author_doc_embeddings']['diffusion_rate']
        
        self.author_ct_distns = {}
        for author_id, barycenter in self.author_topic_barycenters.items():
            self.author_ct_distns[author_id] = diffuse_distribution(
                self.knn_chunk_topic_graph, barycenter, num_iterations, diffusion_rate
            )
        
        self.doc_ct_distns = {}
        for doc_id, distribution in self.expanded_doc_topic_distns.items():
            self.doc_ct_distns[doc_id] = diffuse_distribution(
                self.knn_chunk_topic_graph, distribution, num_iterations, diffusion_rate
            )
    
    # ========================================================================================
    # 4. TOPIC MANIFOLD LEARNING AND DENDROGRAM CONSTRUCTION
    # ========================================================================================
    
    def build_topic_manifold(self,
                           distance_metric: Optional[str] = None,
                           knn_neighbors: Optional[int] = None,
                           use_faiss: Optional[bool] = None) -> 'MstmlOrchestrator':
        """
        Build topic manifold using information-geometric distances.
        
        Uses the large KNN parameter for manifold construction and hierarchical clustering.
        
        Args:
            distance_metric: Distance metric ('hellinger', 'euclidean') (overrides config)
            knn_neighbors: Number of nearest neighbors (overrides config)
            use_faiss: Use FAISS for fast approximate nearest neighbors (overrides config)
        
        Returns:
            Self for method chaining
        """
        
        if self.topic_vectors is None:
            raise ValueError("No topic vectors available. Call train_ensemble_models() first.")
        
        # Get config defaults and apply parameter overrides
        distance_config = self.config['distance_metric']
        mesoscale_config = self.config['mesoscale_filtering']
        distance_metric = distance_metric if distance_metric is not None else distance_config['type']
        knn_neighbors = knn_neighbors if knn_neighbors is not None else mesoscale_config['knnk_for_tpc_dendro']
        use_faiss = use_faiss if use_faiss is not None else mesoscale_config['faiss_acceleration']['enabled']
        
        self.logger.info(f"Building topic manifold with {distance_metric} distance and k={knn_neighbors}")
        
        n_topics = len(self.topic_vectors)
        
        # Use FAISS for large datasets if requested and available
        faiss_utilized = False
        min_vectors = mesoscale_config['faiss_acceleration']['min_vectors_for_faiss']
        
        if use_faiss and n_topics >= min_vectors:
            self.logger.info(f"Attempting FAISS: use_faiss={use_faiss}, n_topics={n_topics}, min_vectors={min_vectors}")
            try:
                if not FAISS_AVAILABLE:
                    raise ImportError("FAISS not available")
                
                self.logger.info(f"Using FAISS for {n_topics} topic vectors")
                
                if distance_metric.lower() == 'hellinger':
                    # For Hellinger distance, we need to use square root transformation
                    sqrt_vectors = np.sqrt(self.topic_vectors).astype(np.float32)
                    dimension = sqrt_vectors.shape[1]
                    
                    # Create FAISS index
                    index = faiss.IndexFlatL2(dimension)
                    index.add(sqrt_vectors)
                    
                    # Search for k+1 nearest neighbors (including self)
                    distances, indices = index.search(sqrt_vectors, knn_neighbors + 1)
                    
                    # Convert L2 distances to Hellinger distances
                    distances = distances / np.sqrt(2.0)
                    
                elif distance_metric.lower() == 'euclidean':
                    vectors = self.topic_vectors.astype(np.float32)
                    dimension = vectors.shape[1]
                    
                    index = faiss.IndexFlatL2(dimension)
                    index.add(vectors)
                    
                    distances, indices = index.search(vectors, knn_neighbors + 1)
                    distances = np.sqrt(distances)  # FAISS returns squared distances
                    
                else:
                    raise ValueError(f"FAISS acceleration not supported for {distance_metric} distance")
                
                # Build kNN graph from FAISS results
                self.topic_knn_graph = nx.Graph()
                self.topic_knn_graph.add_nodes_from(range(n_topics))
                
                for i in range(n_topics):
                    for j in range(1, knn_neighbors + 1):  # Skip self (index 0)
                        neighbor_idx = indices[i, j]
                        distance = distances[i, j]
                        self.topic_knn_graph.add_edge(i, neighbor_idx, weight=distance)
                
                faiss_utilized = True
                
            except (ImportError, Exception) as e:
                self.logger.warning(f"FAISS failed ({str(e)}), falling back to scipy")
                faiss_utilized = False
        
        if not faiss_utilized:
            # Use scipy for smaller datasets or when FAISS is not available
            self.logger.info(f"Using scipy for {n_topics} topic vectors (faiss_utilized={faiss_utilized})")
            
            # Compute pairwise distances
            if distance_metric.lower() == 'hellinger':
                distances = pdist(self.topic_vectors, metric=hellinger)
            elif distance_metric.lower() == 'euclidean':
                distances = pdist(self.topic_vectors, metric='euclidean')
            elif distance_metric.lower() == 'cosine':
                distances = pdist(self.topic_vectors, metric='cosine')
            else:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")
            
            # Convert to square matrix
            distance_matrix = squareform(distances)
            
            # Build kNN graph
            self.topic_knn_graph = nx.Graph()
            self.topic_knn_graph.add_nodes_from(range(n_topics))
            
            for i in range(n_topics):
                # Get k nearest neighbors (excluding self)
                neighbor_distances = [(j, distance_matrix[i, j]) for j in range(n_topics) if i != j]
                neighbor_distances.sort(key=lambda x: x[1])
                
                # Add edges to k nearest neighbors
                for j, dist in neighbor_distances[:knn_neighbors]:
                    self.topic_knn_graph.add_edge(i, j, weight=dist)
        
        # Ensure graph was created successfully
        if self.topic_knn_graph is None:
            raise RuntimeError("Failed to create topic k-NN graph.")
        
        # Add metadata to nodes
        for topic_idx, topic_info in enumerate(self.chunk_topics):
            self.topic_knn_graph.nodes[topic_idx].update({
                'chunk_id': topic_info['chunk_id'],
                'topic_id': topic_info['topic_id'],
                'chunk_size': topic_info['chunk_size']
            })
        
        self.logger.info(
            f"Topic manifold built: {self.topic_knn_graph.number_of_nodes()} nodes, "
            f"{self.topic_knn_graph.number_of_edges()} edges (k={knn_neighbors})"
        )
        
        return self
    
    def construct_topic_dendrogram(self,
                                 linkage_method: Optional[str] = None,
                                 distance_metric: Optional[str] = None,
                                 knn_neighbors: Optional[int] = None) -> 'MstmlOrchestrator':
        """
        Construct hierarchical topic dendrogram using agglomerative clustering.
        
        Based on the reference implementation with proper handling of Ward's linkage
        with Hellinger distances using sqrt transformation to Euclidean space.
        
        Args:
            linkage_method: Hierarchical clustering linkage ('ward', 'average', 'complete', 'single') (overrides config)
            distance_metric: Distance metric ('hellinger', 'euclidean', 'cosine') (overrides config)
            knn_neighbors: Number of nearest neighbors for sparse distance matrix (overrides config)
        
        Returns:
            Self for method chaining
        """
        
        if self.topic_vectors is None:
            raise ValueError("No topic vectors available. Call train_ensemble_models() first.")
        
        # Get config defaults and apply parameter overrides
        clustering_config = self.config['clustering']
        distance_config = self.config['distance_metric']
        mesoscale_config = self.config['mesoscale_filtering']
        linkage_method = linkage_method if linkage_method is not None else clustering_config['linkage_method']
        distance_metric = distance_metric if distance_metric is not None else distance_config['type']
        knn_neighbors = knn_neighbors if knn_neighbors is not None else mesoscale_config['knnk_for_tpc_dendro']
        
        self.logger.info(f"Constructing topic dendrogram with {linkage_method} linkage and {distance_metric} distance")
        
        topic_matrix = np.array(self.topic_vectors).astype(np.float32)
        n_topics = len(topic_matrix)
        
        # Handle Ward's linkage with Hellinger distance
        if linkage_method.lower() == 'ward' and distance_metric.lower() == 'hellinger':
            self.logger.info("Using sqrt transformation for Ward's linkage with Hellinger distance")
            
            # Use k-NN graph approach
            if n_topics >= 100 and hasattr(self, 'topic_knn_graph') and self.topic_knn_graph is not None:
                # Extract distances from existing k-NN graph
                pairs = []
                for i, j, data in self.topic_knn_graph.edges(data=True):
                    # Convert Hellinger to squared Euclidean for Ward's
                    hellinger_dist = data['weight']
                    euclidean_dist_sq = (hellinger_dist * np.sqrt(2)) ** 2
                    pairs.append((i, j, euclidean_dist_sq))
                
                # Create sparse distance matrix
                distance_matrix = np.full((n_topics, n_topics), 1.0, dtype=np.float32)
                np.fill_diagonal(distance_matrix, 0)
                
                for i, j, dist in pairs:
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
                
                # Convert to condensed form and compute linkage
                condensed_distances = squareform(distance_matrix)
                self.topic_dendrogram_linkage = linkage(condensed_distances, method='ward')
                
            else:
                # Direct computation for smaller datasets
                sqrt_vectors = np.sqrt(topic_matrix)
                euclidean_distances = pdist(sqrt_vectors, metric='euclidean')
                # Square the distances for Ward's linkage
                squared_distances = euclidean_distances ** 2
                self.topic_dendrogram_linkage = linkage(squared_distances, method='ward')
        
        elif linkage_method.lower() != 'ward':
            # For non-Ward linkage methods
            if distance_metric.lower() == 'hellinger':
                distances = pdist(topic_matrix, metric=hellinger)
            elif distance_metric.lower() == 'euclidean':
                distances = pdist(topic_matrix, metric='euclidean')
            elif distance_metric.lower() == 'cosine':
                distances = pdist(topic_matrix, metric='cosine')
            else:
                self.logger.warning(f"Unknown distance metric {distance_metric}, using euclidean")
                distances = pdist(topic_matrix, metric='euclidean')
            
            self.topic_dendrogram_linkage = linkage(distances, method=linkage_method)
        
        else:
            # Ward with non-Hellinger metrics
            if distance_metric.lower() != 'euclidean':
                self.logger.warning("Ward linkage works best with Euclidean distances")
            
            distances = pdist(topic_matrix, metric=distance_metric.lower())
            self.topic_dendrogram_linkage = linkage(distances, method='ward')
        
        # Store original min/max heights for rescaling
        self.min_cut_height = np.min(self.topic_dendrogram_linkage[:, 2])
        self.max_cut_height = np.max(self.topic_dendrogram_linkage[:, 2])
        
        # Create enhanced dendrogram with link probabilities using fast_encode_tree_structure
        self._create_enhanced_dendrogram()
        
        # Store dendrogram metadata
        self.dendrogram_info = {
            'linkage_method': linkage_method,
            'distance_metric': distance_metric,
            'num_topics': n_topics,
            'min_cut_height': self.min_cut_height,
            'max_cut_height': self.max_cut_height,
            'knn_neighbors': knn_neighbors
        }
        
        self.logger.info(
            f"Topic dendrogram constructed: {n_topics} leaves, "
            f"cut height range [{self.min_cut_height:.3f}, {self.max_cut_height:.3f}]"
        )
        
        return self
    
    def _create_enhanced_dendrogram(self):
        """Create enhanced dendrogram with link probabilities using fast_encode_tree_structure."""
        if self.topic_dendrogram_linkage is None:
            raise ValueError("No linkage matrix available")
        
        # Check if we have required components for enhanced dendrogram
        if (self.coauthor_network is None or 
            not hasattr(self, 'author_ct_distns') or 
            self.author_ct_distns is None):
            
            self.logger.warning("Co-author network or author distributions not available.")
            self.topic_dendrogram = None
            return
        
        try:
            self.logger.info("Creating enhanced dendrogram with link probabilities")
            
            # Call fast_encode_tree_structure with the linkage matrix (Z), author distributions, and network
            encoded_root, author_index_map = fast_encode_tree_structure(
                self.topic_dendrogram_linkage,
                self.author_ct_distns,
                self.coauthor_network
            )
            
            # Store the enhanced dendrogram and mapping
            self.topic_dendrogram = encoded_root
            self.author_index_map = author_index_map
            
            # Compute min/max cut distances for the enhanced dendrogram
            self.max_cut_distance, self.min_cut_distance = find_max_min_cut_distance(encoded_root)
            
            self.logger.info(
                f"Dendrogram created with cut distance range "
                f"[{self.min_cut_distance:.3f}, {self.max_cut_distance:.3f}]"
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create dendrogram: {e}")
            self.topic_dendrogram = None
    
    def estimate_node_probabilities(self,
                                  use_author_embeddings: bool = True) -> 'MstmlOrchestrator':
        """
        Estimate internal node probabilities for hierarchical random graph model.
        
        Uses MLE estimator: p̂_m ≈ E[E_m] / (E[L_m] × E[R_m])
        where E_m is expected edges, L_m and R_m are left/right subtree sizes.
        
        Args:
            use_author_embeddings: Use author-topic distributions for estimation
        
        Returns:
            Self for method chaining
        """
        
        if self.topic_dendrogram_linkage is None:
            raise ValueError("No topic dendrogram available. Call construct_topic_dendrogram() first.")
        
        if self.coauthor_network is None:
            raise ValueError("No co-author network available. Call setup_coauthor_network() first.")
        
        self.logger.info("Estimating dendrogram node probabilities")
        
        # Ensure author distributions are available
        if use_author_embeddings:
            if not hasattr(self, 'author_ct_distns') or self.author_ct_distns is None:
                raise ValueError("No author distributions available. Call train_ensemble_models() first.")
        
        # Encode tree structure for fast probability computation
        # Create author index mapping
        author_index_map = {}
        for idx, author_id in enumerate(self.coauthor_network.nodes()):
            author_index_map[author_id] = idx
        
        # Use fast tree encoding from the driver
        try:
            encoded_root, _ = fast_encode_tree_structure(
                self.topic_dendrogram_linkage, 
                self.author_ct_distns or {},
                self.coauthor_network
            )
            
            # Extract node probabilities from encoded tree
            self.internal_node_probabilities = self._extract_node_probabilities(encoded_root)
            
        except Exception as e:
            self.logger.warning(f"Fast tree encoding failed: {e}, using simplified estimation")
            
            # Fallback to simplified probability estimation
            n_internal_nodes = len(self.topic_dendrogram_linkage)
            self.internal_node_probabilities = {}
            
            for node_idx in range(n_internal_nodes):
                # Get internal node information from linkage matrix
                left_child = int(self.topic_dendrogram_linkage[node_idx, 0])
                right_child = int(self.topic_dendrogram_linkage[node_idx, 1])
                height = self.topic_dendrogram_linkage[node_idx, 2]
                
                # Estimate subtree sizes (simplified)
                left_size = self._estimate_subtree_size(left_child, n_internal_nodes)
                right_size = self._estimate_subtree_size(right_child, n_internal_nodes)
                
                # Estimate expected edges based on author network density
                if self.coauthor_network.number_of_edges() > 0:
                    network_density = (2 * self.coauthor_network.number_of_edges() / 
                                     (self.coauthor_network.number_of_nodes() * 
                                      (self.coauthor_network.number_of_nodes() - 1)))
                else:
                    network_density = 0.01  # Default small value
                
                # Simple MLE estimate
                expected_edges = network_density * left_size * right_size
                probability = min(1.0, expected_edges / (left_size * right_size)) if left_size * right_size > 0 else 0.0
                
                internal_node_id = node_idx + len(self.topic_vectors)  # Offset by number of leaves
                self.internal_node_probabilities[internal_node_id] = {
                    'probability': probability,
                    'left_size': left_size,
                    'right_size': right_size,
                    'expected_edges': expected_edges,
                    'height': height
                }
        
        num_estimated = len(self.internal_node_probabilities)
        avg_probability = np.mean([info['probability'] for info in self.internal_node_probabilities.values()])
        
        self.logger.info(
            f"Estimated probabilities for {num_estimated} internal nodes, "
            f"average probability: {avg_probability:.4f}"
        )
        
        return self
    
    def _extract_node_probabilities(self, encoded_root):
        """Extract node probabilities from encoded tree structure."""
        probabilities = {}
        
        def traverse_tree(node):
            if hasattr(node, 'id') and hasattr(node, 'probability'):
                probabilities[node.id] = {
                    'probability': getattr(node, 'probability', 0.0),
                    'left_size': getattr(node, 'left_size', 1),
                    'right_size': getattr(node, 'right_size', 1),
                    'expected_edges': getattr(node, 'expected_edges', 0.0),
                    'height': getattr(node, 'height', 0.0)
                }
            
            if hasattr(node, 'left') and node.left:
                traverse_tree(node.left)
            if hasattr(node, 'right') and node.right:
                traverse_tree(node.right)
        
        if encoded_root:
            traverse_tree(encoded_root)
        
        return probabilities
    
    def _estimate_subtree_size(self, node_id, n_internal_nodes):
        """Estimate the size of a subtree rooted at node_id."""
        if node_id < len(self.topic_vectors):  # Leaf node
            return 1
        else:  # Internal node
            # Simplified estimation based on tree structure
            return max(1, int(np.log2(n_internal_nodes - node_id + len(self.topic_vectors)) + 1))
    
    # ========================================================================================
    # 5. AUTHOR REPRESENTATION LEARNING
    # ========================================================================================
    
    
    def compute_interdisciplinarity_scores_docs(self,
                                              entropy_weighting: Optional[bool] = None,
                                              author_publication_weighting: Optional[bool] = None,
                                              topn: Optional[int] = None) -> 'MstmlOrchestrator':
        """
        Compute interdisciplinarity scores for documents.
        
        Score = H(Ω(j)) × Σ w_u for document j with topic distribution Ω(j)
        where w_u = √N_D(u) weights authors by publication count.
        
        Args:
            entropy_weighting: Weight by topic distribution entropy (overrides config)
            author_publication_weighting: Weight by author publication counts (overrides config)
            topn: Return top N ranked interdisciplinary documents (None for all)
        
        Returns:
            Self for method chaining
        """
        
        if not hasattr(self, 'doc_ct_distns') or self.doc_ct_distns is None:
            raise ValueError("No document diffused distributions available. Call train_ensemble_models() first.")
        
        # Get config defaults and apply parameter overrides
        interdisciplinarity_config = self.config['analysis']['interdisciplinarity']
        entropy_weighting = entropy_weighting if entropy_weighting is not None else interdisciplinarity_config['entropy_weighting']
        author_publication_weighting = author_publication_weighting if author_publication_weighting is not None else interdisciplinarity_config['author_publication_weighting']
        
        self.logger.info(f"Computing interdisciplinarity scores for documents{' (top ' + str(topn) + ')' if topn else ''}")
        
        doc_scores = {}
        
        for doc_id in self.doc_ct_distns:
            # Use the diffused document-topic distribution directly
            doc_topic_dist = self.doc_ct_distns[doc_id]
            
            # Get document metadata for author weighting if needed
            if doc_id in self.documents_df.index:
                row = self.documents_df.loc[doc_id]
                authors = row.get('authors', [])
                if isinstance(authors, str):
                    authors = [authors]
                
                # Calculate author publication weight if requested
                author_weight_factor = 1.0
                if author_publication_weighting and authors:
                    total_pub_weight = 0.0
                    for author in authors:
                        author_id = str(author)
                        if hasattr(self, 'coauthor_network') and self.coauthor_network and author_id in self.coauthor_network.nodes:
                            pub_count = self.coauthor_network.nodes[author_id].get('publication_count', 1)
                            total_pub_weight += np.sqrt(pub_count)
                        else:
                            total_pub_weight += 1.0
                    author_weight_factor = total_pub_weight / len(authors) if len(authors) > 0 else 1.0
            
            # Compute interdisciplinarity score
            score = 0.0
            
            if entropy_weighting:
                # Calculate entropy of topic distribution
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                doc_topic_dist_safe = doc_topic_dist + epsilon
                doc_topic_dist_safe /= doc_topic_dist_safe.sum()  # Renormalize
                
                topic_entropy = entropy(doc_topic_dist_safe, base=2)
                score = topic_entropy
            else:
                # Simple diversity measure (1 - max probability)
                score = 1.0 - np.max(doc_topic_dist)
            
            # Weight by author publication mass if requested
            if author_publication_weighting:
                score *= author_weight_factor
            
            doc_scores[doc_id] = score
        
        # Sort and potentially truncate results
        sorted_docs = OrderedDict(
            sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if topn is not None:
            sorted_docs = OrderedDict(list(sorted_docs.items())[:topn])
        
        self.interdisciplinarity_scores['documents'] = sorted_docs
        
        avg_score = np.mean(list(doc_scores.values()))
        max_score = max(doc_scores.values())
        
        self.logger.info(
            f"Computed interdisciplinarity scores for {len(doc_scores)} documents. "
            f"Average: {avg_score:.3f}, Max: {max_score:.3f}"
        )
        
        return self
    
    def compute_interdisciplinarity_scores_authors(self,
                                                  entropy_weighting: Optional[bool] = None,
                                                  publication_count_weighting: Optional[bool] = None,
                                                  top_topics_threshold: Optional[int] = None,
                                                  topn: Optional[int] = None) -> 'MstmlOrchestrator':
        """
        Compute interdisciplinarity scores for authors.
        
        Score based on author topic distribution entropy and publication patterns.
        
        Args:
            entropy_weighting: Weight by topic distribution entropy (overrides config)
            publication_count_weighting: Weight by author publication counts (overrides config)
            top_topics_threshold: Number of top topics to retain per author (overrides config)
            topn: Return top N ranked interdisciplinary authors (None for all)
        
        Returns:
            Self for method chaining
        """
        
        if not hasattr(self, 'author_ct_distns') or self.author_ct_distns is None:
            raise ValueError("No author diffused distributions available. Call train_ensemble_models() first.")
        
        # Get config defaults and apply parameter overrides
        interdisciplinarity_config = self.config['analysis']['interdisciplinarity']
        entropy_weighting = entropy_weighting if entropy_weighting is not None else interdisciplinarity_config['entropy_weighting']
        publication_count_weighting = publication_count_weighting if publication_count_weighting is not None else interdisciplinarity_config['publication_count_weighting']
        top_topics_threshold = top_topics_threshold if top_topics_threshold is not None else interdisciplinarity_config['top_topics_threshold']
        
        self.logger.info(f"Computing interdisciplinarity scores for authors{' (top ' + str(topn) + ')' if topn else ''}")
        
        author_scores = {}
        
        for author_id, topic_dist in self.author_ct_distns.items():
            # Filter to top topics if specified
            if top_topics_threshold > 0 and top_topics_threshold < len(topic_dist):
                # Keep only top K topics, zero out the rest
                top_indices = np.argsort(topic_dist)[-top_topics_threshold:]
                filtered_dist = np.zeros_like(topic_dist)
                filtered_dist[top_indices] = topic_dist[top_indices]
                
                # Renormalize
                if filtered_dist.sum() > 0:
                    filtered_dist /= filtered_dist.sum()
                else:
                    filtered_dist = topic_dist  # Fallback to original
            else:
                filtered_dist = topic_dist
            
            # Compute interdisciplinarity score
            score = 0.0
            
            if entropy_weighting:
                # Calculate entropy of topic distribution
                epsilon = 1e-10
                dist_safe = filtered_dist + epsilon
                dist_safe /= dist_safe.sum()  # Renormalize
                
                topic_entropy = entropy(dist_safe, base=2)
                score = topic_entropy
            else:
                # Simple diversity measure (1 - max probability)
                score = 1.0 - np.max(filtered_dist)
            
            # Weight by publication count if requested
            if publication_count_weighting:
                pub_count = self.coauthor_network.nodes[author_id].get('publication_count', 1)
                score *= np.log1p(pub_count)  # Log(1 + count) for smoother scaling
            
            author_scores[author_id] = score
        
        # Sort and potentially truncate results
        sorted_authors = OrderedDict(
            sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if topn is not None:
            sorted_authors = OrderedDict(list(sorted_authors.items())[:topn])
        
        self.interdisciplinarity_scores['authors'] = sorted_authors
        
        avg_score = np.mean(list(author_scores.values()))
        max_score = max(author_scores.values())
        
        self.logger.info(
            f"Computed interdisciplinarity scores for {len(author_scores)} authors. "
            f"Average: {avg_score:.3f}, Max: {max_score:.3f}"
        )
        
        return self
    
    # ========================================================================================
    # 6. LONGITUDINAL ANALYSIS AND VISUALIZATION
    # ========================================================================================
    
    def create_topic_embedding(self,
                             method: str = 'phate',
                             n_components: int = 3,
                             knn_neighbors: int = 5,
                             phate_gamma: float = 0,
                             t: str = "auto",
                             cut_height: float = 0.7,
                             distance_metric: str = 'hellinger',
                             linkage_method: str = 'ward',
                             dendrogram_knn_neighbors: int = 50,
                             **method_kwargs) -> 'MstmlOrchestrator':
        """
        Create topic space embedding for visualization using various manifold learning methods.
        
        Based on reference implementation with proper Hellinger distance handling,
        dendrogram integration, and support for 2D/3D embeddings.
        
        Args:
            method: Embedding method ('phate', 'pca', 'umap', 'tsne')
            n_components: Number of embedding dimensions (2 or 3)
            knn_neighbors: Number of nearest neighbors for PHATE/UMAP
            phate_gamma: PHATE gamma parameter for kernel bandwidth (0 = auto)
            t: Number of diffusion steps for PHATE ('auto' or integer)
            cut_height: Dendrogram cut height for meta-topic clustering (0-1 range)
            distance_metric: Distance metric ('hellinger', 'euclidean', 'cosine')
            linkage_method: Hierarchical clustering linkage method ('ward', 'average', 'complete', 'single')
            dendrogram_knn_neighbors: Number of nearest neighbors for dendrogram construction
            **method_kwargs: Additional parameters for specific embedding methods
        
        Returns:
            Self for method chaining
        """
        
        if self.topic_vectors is None:
            raise ValueError("No topic vectors available. Call train_ensemble_models() first.")
        
        # Validate linkage method
        VALID_LINKAGE_METHODS = ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median']
        if linkage_method not in VALID_LINKAGE_METHODS:
            raise ValueError(f"linkage_method must be one of {VALID_LINKAGE_METHODS}")
        
        self.logger.info(f"Creating {n_components}D {method.upper()} embedding")
        
        # Ensure dendrogram exists with user-specified parameters
        if not hasattr(self, 'topic_dendrogram_linkage') or self.topic_dendrogram_linkage is None:
            self.logger.info(f"Constructing topic dendrogram for embedding (linkage={linkage_method}, knn={dendrogram_knn_neighbors})")
            self.construct_topic_dendrogram(
                linkage_method=linkage_method,
                distance_metric=distance_metric,
                knn_neighbors=dendrogram_knn_neighbors
            )
        
        # Convert topic vectors to numpy array with proper dtype
        topic_matrix = np.array(self.topic_vectors, dtype=np.float32)
        n_topics = topic_matrix.shape[0]
        
        # Validate parameters
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_components >= n_topics:
            log_print(
                f"n_components ({n_components}) >= n_topics ({n_topics}). "
                f"Reducing to {n_topics - 1} components.",
                level="warning", logger=self.logger
            )
            n_components = n_topics - 1
        if knn_neighbors >= n_topics:
            log_print(
                f"knn_neighbors ({knn_neighbors}) >= n_topics ({n_topics}). "
                f"Reducing to {n_topics - 1} neighbors.",
                level="warning", logger=self.logger
            )
            knn_neighbors = n_topics - 1
        
        # Apply dendrogram cut height if specified
        if cut_height is not None:
            cut_result = self.cut_topic_dendrogram(
                cut_height=cut_height,
                min_cluster_size=1,  # Allow all clusters for embedding purposes
                validate_height=True
            )
        
        # Compute embedding based on method
        method_lower = method.lower()
        
        if method_lower == 'phate' and PHATE_AVAILABLE:
            # PHATE embedding with proper distance handling
            try:
                if distance_metric.lower() == 'hellinger':
                    # Compute Hellinger distance matrix using pdist for efficiency
                    hellinger_distances = pdist(topic_matrix, metric=hellinger)
                    distance_matrix = squareform(hellinger_distances)
                    
                    # Use knn_dist='precomputed' instead of metric='precomputed' (from reference)
                    phate_op = phate.PHATE(
                        n_components=n_components,
                        knn=knn_neighbors,
                        gamma=phate_gamma,
                        t=t,
                        knn_dist='precomputed',
                        random_state=42,
                        **method_kwargs
                    )
                    
                    self.topic_embedding = phate_op.fit_transform(distance_matrix)
                    
                else:
                    # For other distance metrics, use them directly if supported by PHATE
                    metric_name = distance_metric.lower()
                    if metric_name not in ['euclidean', 'cosine', 'l1', 'l2', 'manhattan']:
                        # Fallback to euclidean for unsupported metrics
                        log_print(
                            f"Distance metric '{distance_metric}' not directly supported by PHATE, "
                            f"falling back to euclidean distance", 
                            level="warning", logger=self.logger
                        )
                        metric_name = 'euclidean'
                    
                    phate_op = phate.PHATE(
                        n_components=n_components,
                        knn=knn_neighbors,
                        gamma=phate_gamma,
                        t=t,
                        metric=metric_name,
                        random_state=42,
                        **method_kwargs
                    )
                    
                    self.topic_embedding = phate_op.fit_transform(topic_matrix)
                
                self.phate_operator = phate_op  # Store for t_value access
                
            except Exception as e:
                log_print(
                    f"PHATE embedding failed: {e}. Falling back to PCA.",
                    level="warning", logger=self.logger
                )
                method_lower = 'pca'  # Fallback to PCA
            
        elif method_lower == 'umap':
            # UMAP embedding
            try:
                
                # Validate distance metric for UMAP
                metric_name = distance_metric.lower()
                if metric_name == 'hellinger':
                    # UMAP doesn't support Hellinger directly, use precomputed
                    hellinger_distances = pdist(topic_matrix, metric=hellinger)
                    distance_matrix = squareform(hellinger_distances)
                    
                    umap_op = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=knn_neighbors,
                        metric='precomputed',
                        random_state=42,
                        **method_kwargs
                    )
                    self.topic_embedding = umap_op.fit_transform(distance_matrix)
                else:
                    # Validate supported metrics for UMAP
                    supported_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine', 'correlation']
                    if metric_name not in supported_metrics:
                        log_print(
                            f"Distance metric '{distance_metric}' not supported by UMAP, "
                            f"falling back to euclidean distance", 
                            level="warning", logger=self.logger
                        )
                        metric_name = 'euclidean'
                    
                    umap_op = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=knn_neighbors,
                        metric=metric_name,
                        random_state=42,
                        **method_kwargs
                    )
                    self.topic_embedding = umap_op.fit_transform(topic_matrix)
                    
                method_lower = 'umap'
                
            except ImportError:
                log_print("UMAP not available, falling back to PCA", level="warning", logger=self.logger)
                method_lower = 'pca'
            except Exception as e:
                log_print(f"UMAP embedding failed: {e}. Falling back to PCA.", level="warning", logger=self.logger)
                method_lower = 'pca'
                
        elif method_lower == 'tsne':
            # t-SNE embedding
            try:
                # Validate parameters for t-SNE
                if n_components > 3:
                    log_print(
                        f"t-SNE with {n_components} components may not be effective. "
                        f"Consider using n_components <= 3 for t-SNE.",
                        level="warning", logger=self.logger
                    )
                
                # Adjust perplexity for small datasets
                tsne_kwargs = method_kwargs.copy()
                if 'perplexity' not in tsne_kwargs:
                    # Default perplexity should be smaller than n_samples
                    default_perplexity = min(30, (n_topics - 1) / 3)
                    if default_perplexity < 5:
                        default_perplexity = min(5, n_topics - 1)
                    tsne_kwargs['perplexity'] = default_perplexity
                    
                    if tsne_kwargs['perplexity'] != 30:
                        log_print(
                            f"Adjusted t-SNE perplexity to {tsne_kwargs['perplexity']} for dataset with {n_topics} topics",
                            level="info", logger=self.logger
                        )
                
                # For t-SNE, handle distance metrics carefully
                metric_name = distance_metric.lower()
                if metric_name == 'hellinger':
                    # Use precomputed distance matrix for Hellinger
                    hellinger_distances = pdist(topic_matrix, metric=hellinger)
                    distance_matrix = squareform(hellinger_distances)
                    
                    tsne_op = TSNE(
                        n_components=n_components,
                        metric='precomputed',
                        random_state=42,
                        **tsne_kwargs
                    )
                    self.topic_embedding = tsne_op.fit_transform(distance_matrix)
                else:
                    # Validate supported metrics for t-SNE
                    supported_metrics = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
                    if metric_name not in supported_metrics:
                        log_print(
                            f"Distance metric '{distance_metric}' not supported by t-SNE, "
                            f"falling back to euclidean distance", 
                            level="warning", logger=self.logger
                        )
                        metric_name = 'euclidean'
                    
                    tsne_op = TSNE(
                        n_components=n_components,
                        metric=metric_name,
                        random_state=42,
                        **tsne_kwargs
                    )
                    self.topic_embedding = tsne_op.fit_transform(topic_matrix)
                    
                method_lower = 'tsne'
                
            except Exception as e:
                log_print(f"t-SNE embedding failed: {e}. Falling back to PCA.", level="warning", logger=self.logger)
                method_lower = 'pca'
        
        # Fallback to PCA (or explicit PCA request)
        if method_lower == 'pca' or not hasattr(self, 'topic_embedding'):
            pca = PCA(n_components=n_components, random_state=42, **method_kwargs)
            self.topic_embedding = pca.fit_transform(topic_matrix)
            log_print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}", level="info", logger=self.logger)
            method_lower = 'pca'
        
        # Store embedding parameters
        self.embed_params = {
            'method': method_lower.upper(),
            'n_components': n_components,
            'knn_neighbors': knn_neighbors,
            'phate_gamma': phate_gamma,
            't': t,
            'cut_height': cut_height,
            'distance_metric': distance_metric,
            'method_kwargs': method_kwargs,
            'n_topics': n_topics
        }
        
        log_print(
            f"Created {method_lower.upper()} embedding: "
            f"{n_topics} topics -> {n_components}D space",
            level="info", logger=self.logger
        )
        
        return self
    
    def create_phate_embedding(self, **kwargs) -> 'MstmlOrchestrator':
        """
        Backward compatibility method for create_phate_embedding.
        Calls create_topic_embedding with method='phate'.
        """
        warnings.warn(
            "create_phate_embedding is deprecated. Use create_topic_embedding(method='phate') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.create_topic_embedding(method='phate', **kwargs)
    
    def display_topic_embedding(self,
                              color_by: str = 'meta_topic',
                              title: Optional[str] = None,
                              save_path: Optional[str] = None,
                              figure_size: Optional[tuple] = None,
                              elevation: Optional[int] = None,
                              azimuth: Optional[int] = None,
                              show_colorbar: Optional[bool] = None,
                              show_histogram: Optional[bool] = None,
                              meta_topic_names: Optional[List[str]] = None) -> None:
        """
        Display sophisticated topic embedding visualization based on reference implementation.
        
        Creates publication-quality plots with proper coloring, 3D support, topic labeling,
        and optional histogram subplot showing topic frequency distribution.
        
        Args:
            color_by: Color scheme ('meta_topic', 'time', 'none')
            title: Custom title for the plot
            save_path: Optional path to save the figure
            figure_size: Figure size as (width, height) tuple (overrides config)
            elevation: 3D plot elevation angle (ignored for 2D) (overrides config)
            azimuth: 3D plot azimuth angle (ignored for 2D) (overrides config)
            show_colorbar: Whether to display colorbar (overrides config)
            show_histogram: Whether to show topic frequency histogram (overrides config)
            meta_topic_names: Optional list of names for meta-topics
        """
        if self.topic_embedding is None:
            raise ValueError("Topic embedding not created. Call create_topic_embedding() first.")
        
        # Get config defaults and apply parameter overrides
        output_config = self.config['output']
        figure_size = figure_size if figure_size is not None else tuple(output_config['figure_size'])
        elevation = elevation if elevation is not None else output_config['elevation']
        azimuth = azimuth if azimuth is not None else output_config['azimuth']
        show_colorbar = show_colorbar if show_colorbar is not None else output_config['show_colorbar']
        show_histogram = show_histogram if show_histogram is not None else output_config['show_histogram']
        
        params = self.embed_params
        method_name = params.get('method', 'EMBEDDING')
        n_components = params.get('n_components', 2)
        cut_height = params.get('cut_height')
        
        # Determine if we have meta-topic clustering
        has_meta_topics = (hasattr(self, 'topic_cluster_labels') and 
                          self.topic_cluster_labels is not None and
                          color_by == 'meta_topic')
        
        # Create figure layout based on options
        if show_histogram and has_meta_topics:
            fig = plt.figure(figsize=figure_size)
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
            ax1 = fig.add_subplot(gs[0], projection='3d' if n_components == 3 else None)
            ax2 = fig.add_subplot(gs[1])
        else:
            fig, ax1 = plt.subplots(figsize=figure_size, 
                                   subplot_kw={'projection': '3d'} if n_components == 3 else {})
            ax2 = None
        
        # Set 3D viewing angles
        if n_components == 3:
            ax1.view_init(elev=elevation, azim=azimuth)
            ax1.set_facecolor('none')
        
        # Determine coloring scheme
        if has_meta_topics:
            # Use discrete colormap for meta-topics
            unique_topics = np.unique(self.topic_cluster_labels)
            num_topics = len(unique_topics)
            
            # Create discrete colormap with boundaries
            cmap = plt.get_cmap('rainbow', num_topics)
            boundaries = np.arange(min(self.topic_cluster_labels), 
                                 max(self.topic_cluster_labels) + 2) - 0.5
            norm = mcolors.BoundaryNorm(boundaries, ncolors=num_topics, clip=True)
            
            colors = self.topic_cluster_labels
            colorbar_label = 'Meta-Topic'
            
        elif color_by == 'time' and hasattr(self, 'chunk_topics'):
            # Color by time chunks
            time_labels = []
            for topic_info in self.chunk_topics:
                chunk_id = topic_info['chunk_id']
                chunk_name = f"Chunk {chunk_id}"
                if hasattr(self, 'time_chunks') and chunk_id < len(self.time_chunks):
                    chunk_name = self.time_chunks[chunk_id].get('name', chunk_name)
                time_labels.append(chunk_id)
            
            cmap = plt.cm.viridis
            norm = None
            colors = time_labels
            colorbar_label = 'Time Chunk'
            
        else:
            # Default single color
            cmap = None
            norm = None
            colors = 'blue'
            colorbar_label = None
        
        # Create the main scatter plot
        if n_components == 3:
            scatter = ax1.scatter(self.topic_embedding[:, 0], 
                                self.topic_embedding[:, 1], 
                                self.topic_embedding[:, 2],
                                c=colors, cmap=cmap, norm=norm, 
                                marker='o', s=10, edgecolor='k', alpha=0.7)
        else:
            scatter = ax1.scatter(self.topic_embedding[:, 0], 
                                self.topic_embedding[:, 1],
                                c=colors, cmap=cmap, norm=norm, 
                                marker='o', s=50, edgecolor='k', alpha=0.7)
        
        # Add topic representative point labels
        if has_meta_topics:
            self._add_topic_representative_labels(ax1, n_components, meta_topic_names)
        
        # Set axis labels and title
        if n_components == 3:
            ax1.set_xlabel(f'{method_name} 1')
            ax1.set_ylabel(f'{method_name} 2') 
            ax1.set_zlabel(f'{method_name} 3')
            # Remove tick labels for cleaner look
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])
        else:
            ax1.set_xlabel(f'{method_name} 1')
            ax1.set_ylabel(f'{method_name} 2')
        
        if title:
            ax1.set_title(title)
        
        # Add colorbar if requested and applicable
        if show_colorbar and colorbar_label and cmap:
            if has_meta_topics:
                # Create discrete colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                
                # Position colorbar appropriately
                cbar_ax = fig.add_axes([0.8, 0.5, 0.02, 0.25])
                cbar = fig.colorbar(sm, cax=cbar_ax, 
                                  ticks=np.arange(min(self.topic_cluster_labels), 
                                                max(self.topic_cluster_labels) + 1))
                
                # Add meta-topic names if provided
                if meta_topic_names and len(meta_topic_names) == num_topics:
                    cbar.ax.set_yticklabels(meta_topic_names)
                
                # Add black dividers between color segments
                for boundary in boundaries[1:-1]:
                    cbar.ax.hlines(boundary, *cbar.ax.get_xlim(), 
                                 color='black', linewidth=1.5)
            else:
                plt.colorbar(scatter, ax=ax1, label=colorbar_label)
        
        # Add histogram subplot if requested
        if show_histogram and ax2 is not None and has_meta_topics:
            topic_label_counts = np.bincount(self.topic_cluster_labels)[min(self.topic_cluster_labels):]
            
            # Set colors for histogram bars based on topic labels
            hist_colors = [cmap(norm(label)) for label in unique_topics]
            ax2.bar(unique_topics, topic_label_counts, color=hist_colors, edgecolor='black')
            
            ax2.set_xlabel('Meta-Topic Labels')
            ax2.set_ylabel('Frequency')
            ax2.set_xticks(np.arange(min(self.topic_cluster_labels), 
                                   max(self.topic_cluster_labels) + 1, 
                                   step=max(1, num_topics // 15)))
        
        self.embed_figure = fig
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"{method_name} embedding saved to {save_path}")
        
        plt.show()
        self.logger.info(f"{method_name} embedding displayed")
    
    def _add_topic_representative_labels(self, ax, n_components: int, meta_topic_names: Optional[List[str]] = None):
        """
        Add intelligent topic representative point labels based on reference algorithm.
        
        Implements the iterative optimization algorithm from reference code to select
        representative points that balance centroid proximity and pairwise distances.
        """
        if not hasattr(self, 'topic_cluster_labels') or self.topic_cluster_labels is None:
            return
        
        unique_topics = np.unique(self.topic_cluster_labels)
        
        # Compute topic centroids
        topic_centroids = {}
        for topic in unique_topics:
            topic_indices = np.where(self.topic_cluster_labels == topic)[0]
            topic_points = self.topic_embedding[topic_indices]
            topic_centroids[topic] = np.mean(topic_points, axis=0)
        
        # Find embedding center
        embed_center = np.mean(self.topic_embedding, axis=0)
        
        # Initialize representative points: closest to each topic's centroid
        topic_representatives = {}
        for topic in unique_topics:
            topic_indices = np.where(self.topic_cluster_labels == topic)[0]
            topic_points = self.topic_embedding[topic_indices]
            distances_to_centroid = np.linalg.norm(topic_points - topic_centroids[topic], axis=1)
            topic_representatives[topic] = topic_indices[np.argmin(distances_to_centroid)]
        
        # Iterative optimization
        max_iterations = 100
        alpha = 0.25  # Weight balance parameter
        converged = False
        iteration = 0
        
        while not converged and iteration < max_iterations:
            converged = True
            
            for topic in unique_topics:
                current_rep_index = topic_representatives[topic]
                topic_indices = np.where(self.topic_cluster_labels == topic)[0]
                best_rep_index = current_rep_index
                max_weighted_score = -np.inf
                
                # Evaluate candidate points
                for candidate_index in topic_indices:
                    candidate_point = self.topic_embedding[candidate_index]
                    
                    # Midpoint between candidate and topic centroid
                    midpoint = (candidate_point + topic_centroids[topic]) / 2
                    
                    # Pairwise distance score to other representatives
                    pairwise_distance_sum = sum(
                        np.linalg.norm(midpoint - self.topic_embedding[topic_representatives[other_topic]])
                        for other_topic in unique_topics if other_topic != topic
                    )
                    
                    # Proximity to topic centroid
                    centroid_proximity = -np.linalg.norm(candidate_point - topic_centroids[topic])
                    
                    # Weighted score
                    weighted_score = alpha * pairwise_distance_sum + (1 - alpha) * centroid_proximity
                    
                    if weighted_score > max_weighted_score:
                        best_rep_index = candidate_index
                        max_weighted_score = weighted_score
                
                # Update if changed
                if best_rep_index != current_rep_index:
                    topic_representatives[topic] = best_rep_index
                    converged = False
            
            iteration += 1
        
        # Add labels to the plot
        for topic, index in topic_representatives.items():
            point = self.topic_embedding[index]
            
            # Determine label text
            if meta_topic_names and len(meta_topic_names) >= topic:
                label_text = meta_topic_names[topic - 1] if topic > 0 else f'<{topic}>'
            else:
                label_text = f'<{topic}>'
            
            # Add text annotation
            if n_components == 3:
                ax.text(point[0], point[1], point[2], label_text,
                       size=12, zorder=3, color='white',
                       bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))
            else:
                ax.annotate(label_text, (point[0], point[1]),
                           xytext=(5, 5), textcoords='offset points',
                           size=12, color='white',
                           bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'),
                           ha='left')
    
    def save_topic_embedding(self,
                           filename: Optional[str] = None,
                           format: str = 'pdf',
                           dpi: int = 300,
                           save_subplots: bool = True,
                           elevation: Optional[int] = None,
                           azimuth: Optional[int] = None) -> Dict[str, str]:
        """
        Save the topic embedding figure with rich filename generation matching reference code.
        
        Generates filenames that include key parameters like method, cut_height, knn_neighbors,
        and viewing angles. Supports saving individual subplots and combined figures.
        
        Args:
            filename: Custom base filename (without extension). Auto-generated if None.
            format: Image format ('pdf', 'png', 'svg', 'eps')
            dpi: Resolution for raster formats
            save_subplots: Whether to save individual subplots separately
            elevation: 3D elevation angle for filename (uses display value if None)
            azimuth: 3D azimuth angle for filename (uses display value if None)
        
        Returns:
            Dictionary mapping save types to file paths
        """
        if self.embed_figure is None:
            raise ValueError("No topic embedding figure to save. Call display_topic_embedding() first.")
        
        # Get parameters for filename generation
        params = self.embed_params
        method_name = params.get('method', 'embedding').lower()
        cut_height = params.get('cut_height', 0.0)
        knn_neighbors = params.get('knn_neighbors', 5)
        n_components = params.get('n_components', 2)
        
        # Get t_value from PHATE operator if available
        t_value = 'auto'
        if hasattr(self, 'phate_operator') and hasattr(self.phate_operator, 't'):
            t_value = getattr(self.phate_operator, 't', 'auto')
        
        # Use provided angles or defaults
        el = elevation if elevation is not None else 25
        az = azimuth if azimuth is not None else 60
        
        saved_files = {}
        
        # Generate rich filename based on reference pattern
        if filename is None:
            # Base filename pattern: method + knn + hellinger + dendro + cut_height + t_value + angles
            base_filename = f"{method_name}{knn_neighbors}_hellinger_dendro_{cut_height:.2f}_cut_topic_cluster_t{t_value}"
            
            # Add 3D viewing angles for 3D plots
            if n_components == 3:
                base_filename += f"_el{el}_az{az}"
        else:
            base_filename = filename
        
        # Ensure experiment directory exists
        if hasattr(self, 'experiment_directory') and self.experiment_directory:
            save_dir = self.experiment_directory
        else:
            save_dir = os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        
        # Save complete figure
        complete_path = os.path.join(save_dir, f"zzz_{base_filename}.{format}")
        self.embed_figure.savefig(complete_path, format=format, dpi=dpi, bbox_inches='tight')
        saved_files['complete'] = complete_path
        
        # Save individual subplots if requested (matching reference pattern)
        if save_subplots and hasattr(self.embed_figure, 'axes'):
            axes = self.embed_figure.axes
            
            # Save main embedding plot (first axis)
            if len(axes) >= 1:
                ax1 = axes[0]
                extent1 = ax1.get_tightbbox(self.embed_figure.canvas.get_renderer()).transformed(
                    self.embed_figure.dpi_scale_trans.inverted())
                
                if n_components == 3:
                    # 3D plot with special positioning adjustments
                    expanded_extent1 = extent1.expanded(1.2, 0.85)
                    expanded_extent1 = expanded_extent1.translated(
                        0.72 + 0.02 * abs(az - 45), 
                        -0.25 + 0.01 * abs(el - 45)
                    )
                    embedding_path = os.path.join(save_dir, f"zzs_{base_filename}.{format}")
                else:
                    # 2D plot
                    expanded_extent1 = extent1.expanded(1.05, 1.05)
                    embedding_path = os.path.join(save_dir, f"zzs_{base_filename}.{format}")
                
                self.embed_figure.savefig(embedding_path, format=format, dpi=dpi, 
                                        bbox_inches=expanded_extent1)
                saved_files['embedding'] = embedding_path
            
            # Save histogram subplot if present (second axis)
            if len(axes) >= 2:
                ax2 = axes[1]
                extent2 = ax2.get_tightbbox(self.embed_figure.canvas.get_renderer()).transformed(
                    self.embed_figure.dpi_scale_trans.inverted())
                
                expanded_extent2 = extent2.expanded(1.05, 1.05)
                expanded_extent2 = expanded_extent2.translated(0, 0.05)
                
                histogram_path = os.path.join(save_dir, f"zzh_{base_filename}.{format}")
                self.embed_figure.savefig(histogram_path, format=format, dpi=dpi, 
                                        bbox_inches=expanded_extent2)
                saved_files['histogram'] = histogram_path
        
        # Log all saved files
        method_display = params.get('method', 'Embedding')
        self.logger.info(f"{method_display} topic embedding saved:")
        for save_type, path in saved_files.items():
            self.logger.info(f"  {save_type}: {path}")
        
        return saved_files
    
    def identify_topic_trajectories(self,
                                  trajectory_algorithm: str = 'shortest_path',
                                  min_trajectory_length: int = 3) -> 'MstmlOrchestrator':
        """
        Identify and track topic trajectories over time using geometric methods.
        
        Uses Dijkstra's shortest path algorithm on topic similarity graph
        to identify natural progressions in topic evolution.
        
        Args:
            trajectory_algorithm: Algorithm for trajectory identification
            min_trajectory_length: Minimum length for valid trajectories
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement trajectory identification
        # - Apply shortest path algorithms on topic similarity graph
        # - Identify coherent topic evolution patterns
        # - Store trajectory metadata and visualization data
        self.logger.info("Identifying topic trajectories")
        return self
    
    # ========================================================================================
    # 7. LINK PREDICTION AND NETWORK ANALYSIS
    # ========================================================================================
    
    def predict_coauthor_links(self,
                             prediction_method: str = 'hrg_likelihood',
                             evaluation_metrics: list = None) -> 'MstmlOrchestrator':
        """
        Predict future or missing co-author links using topic-based methods.
        
        Args:
            prediction_method: Link prediction algorithm
            evaluation_metrics: Metrics for evaluation (AUC, precision, recall)
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement link prediction
        # - Use HRG model probabilities for link likelihood estimation
        # - Apply topic similarity measures for prediction
        # - Evaluate using standard link prediction metrics
        self.logger.info(f"Predicting co-author links using {prediction_method}")
        return self
    
    # ========================================================================================
    # 8. MULTI-SCALE ANALYSIS AND RESULTS
    # ========================================================================================
    
    def cut_topic_dendrogram(self,
                           cut_height: Optional[float] = None,
                           min_cluster_size: int = 2,
                           validate_height: bool = True) -> dict:
        """
        Cut topic dendrogram at specified height to create meta-topic clusters.
        
        Cuts the hierarchical clustering dendrogram at the specified height to group
        topics into meta-topic clusters. Provides detailed analysis of resulting clusters
        including statistics, assignments, and filtering by minimum cluster size.
        
        Args:
            cut_height: Height at which to cut dendrogram (0-1 range is auto-scaled). 
                       If None, uses median height between min/max.
            min_cluster_size: Minimum size for valid clusters (clusters smaller than this are filtered out)
            validate_height: Whether to validate cut_height against dendrogram bounds
        
        Returns:
            Dictionary containing:
                - cut_height: Original cut height parameter
                - actual_cut_height: Rescaled cut height used for cutting
                - num_clusters: Number of valid clusters (meeting min_cluster_size)
                - num_total_clusters: Total clusters before filtering
                - cluster_assignments: Dict mapping topic_id -> cluster_id for valid clusters
                - cluster_sizes: Dict mapping cluster_id -> cluster_size for valid clusters  
                - cluster_statistics: Dict with centroid, variance, etc. for each cluster
                - filtered_out_clusters: Number of clusters removed by size filtering
        
        Side Effects:
            Sets self.topic_cluster_labels with the cluster assignment array
        """
        if self.topic_dendrogram_linkage is None:
            raise ValueError("Topic dendrogram not constructed. Call construct_topic_dendrogram() first.")
        
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1")
        
        # Get dendrogram height bounds from actual dendrogram structure
        if validate_height and hasattr(self, 'topic_dendrogram_linkage'):
            # Extract actual min/max heights from dendrogram linkage matrix
            if hasattr(self, 'min_cut_height') and hasattr(self, 'max_cut_height'):
                min_height = self.min_cut_height
                max_height = self.max_cut_height
            else:
                # Fallback: compute from linkage matrix
                min_height = 0.0
                max_height = float(np.max(self.topic_dendrogram_linkage[:, 2]))
            
            if cut_height is None:
                cut_height = (min_height + max_height) / 2.0
            elif cut_height < min_height or cut_height > max_height:
                log_print(
                    f"Cut height {cut_height} outside valid range [{min_height:.3f}, {max_height:.3f}]. "
                    f"Clamping to valid range.", level="warning", logger=self.logger
                )
                cut_height = max(min_height, min(cut_height, max_height))
        elif cut_height is None:
            cut_height = 0.5  # Default fallback
        
        # Rescale cut_height from [0,1] to actual dendrogram range if needed
        actual_cut_height = cut_height
        if (hasattr(self, 'min_cut_height') and hasattr(self, 'max_cut_height') and 
            0 <= cut_height <= 1):
            actual_cut_height = rescale_parameter(cut_height, self.min_cut_height, self.max_cut_height)
        
        # Cut dendrogram at specified height to create meta-topics
        cluster_labels = fcluster(self.topic_dendrogram_linkage, actual_cut_height, criterion='distance')
        
        # Analyze clusters
        unique_clusters = np.unique(cluster_labels)
        cluster_assignments = {i: int(cluster_labels[i]) for i in range(len(cluster_labels))}
        cluster_sizes = {cluster_id: int(np.sum(cluster_labels == cluster_id)) for cluster_id in unique_clusters}
        
        # Filter clusters by minimum size requirement
        valid_clusters = {cid: size for cid, size in cluster_sizes.items() if size >= min_cluster_size}
        filtered_assignments = {topic_id: cluster_id for topic_id, cluster_id in cluster_assignments.items() 
                              if cluster_id in valid_clusters}
        
        # Store cluster labels in orchestrator for other methods to use
        self.topic_cluster_labels = cluster_labels
        
        # Compute cluster-level statistics
        cluster_statistics = {}
        if hasattr(self, 'topic_vectors') and self.topic_vectors is not None:
            topic_matrix = np.array(self.topic_vectors)
            for cluster_id in valid_clusters:
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_topics = topic_matrix[cluster_indices]
                
                cluster_statistics[cluster_id] = {
                    'centroid': np.mean(cluster_topics, axis=0),
                    'std': np.std(cluster_topics, axis=0),
                    'intra_cluster_variance': float(np.mean(np.var(cluster_topics, axis=0))),
                    'topic_indices': cluster_indices.tolist()
                }
        
        result = {
            'cut_height': cut_height,
            'actual_cut_height': actual_cut_height,
            'min_cluster_size': min_cluster_size,
            'num_clusters': len(valid_clusters),
            'num_total_clusters': len(unique_clusters),
            'cluster_assignments': filtered_assignments,
            'cluster_sizes': valid_clusters,
            'cluster_statistics': cluster_statistics,
            'filtered_out_clusters': len(unique_clusters) - len(valid_clusters)
        }
        
        log_print(
            f"Cut topic dendrogram at height {cut_height:.3f} (actual: {actual_cut_height:.3f}), "
            f"found {len(valid_clusters)} meta-topics ({len(unique_clusters)} total clusters, "
            f"{len(unique_clusters) - len(valid_clusters)} filtered out by size)",
            level="info", logger=self.logger
        )
        return result
    
    def export_results(self,
                      output_directory: Optional[str] = None,
                      formats: Optional[List[str]] = None) -> 'MstmlOrchestrator':
        """
        Export analysis results in multiple formats to experiment directory.
        
        Args:
            output_directory: Directory for output files (defaults to experiment_directory)
            formats: List of output formats ('json', 'csv', 'pkl')
        
        Returns:
            Self for method chaining
        """
        if output_directory is None:
            output_directory = self.experiment_directory
            
        if formats is None:
            formats = ['json', 'pkl']
            
        os.makedirs(output_directory, exist_ok=True)
        self.logger.info(f"Exporting results to {output_directory} in formats: {formats}")
        
        # Export topic vectors if available
        if self.topic_vectors is not None:
            if 'pkl' in formats:
                write_pickle(os.path.join(output_directory, 'topic_vectors.pkl'), self.topic_vectors)
            if 'json' in formats:
                topic_vectors_list = [vec.tolist() for vec in self.topic_vectors]
                with open(os.path.join(output_directory, 'topic_vectors.json'), 'w') as f:
                    json.dump(topic_vectors_list, f, indent=2, cls=NumpyEncoder)
        
        # Export topic embedding if available
        if self.topic_embedding is not None:
            if 'pkl' in formats:
                write_pickle(os.path.join(output_directory, 'topic_embedding.pkl'), self.topic_embedding)
            if 'json' in formats:
                with open(os.path.join(output_directory, 'topic_embedding.json'), 'w') as f:
                    json.dump(self.topic_embedding.tolist(), f, indent=2, cls=NumpyEncoder)
        
        # Export author embeddings if available
        if hasattr(self, 'author_ct_distns') and self.author_ct_distns is not None:
            if 'pkl' in formats:
                write_pickle(os.path.join(output_directory, 'author_embeddings.pkl'), self.author_ct_distns)
            if 'json' in formats:
                # Convert numpy arrays to lists for JSON serialization
                json_embeddings = {
                    author: safe_json_convert(embedding)
                    for author, embedding in self.author_ct_distns.items()
                }
                with open(os.path.join(output_directory, 'author_embeddings.json'), 'w') as f:
                    json.dump(json_embeddings, f, indent=2, cls=NumpyEncoder)
        
        # Export interdisciplinarity scores if available
        if self.interdisciplinarity_scores:
            if 'pkl' in formats:
                write_pickle(os.path.join(output_directory, 'interdisciplinarity_scores.pkl'), self.interdisciplinarity_scores)
            if 'json' in formats:
                # Convert NumPy types for JSON serialization
                safe_scores = safe_json_convert(self.interdisciplinarity_scores)
                with open(os.path.join(output_directory, 'interdisciplinarity_scores.json'), 'w') as f:
                    json.dump(safe_scores, f, indent=2, cls=NumpyEncoder)
        
        # Export time chunks information if available
        if self.time_chunks:
            if 'pkl' in formats:
                write_pickle(os.path.join(output_directory, 'time_chunks.pkl'), self.time_chunks)
            if 'json' in formats:
                # Convert datetime objects and NumPy types to JSON-serializable format
                json_chunks = []
                for chunk in self.time_chunks:
                    json_chunk = {}
                    for key, value in chunk.items():
                        if key in ['start_date', 'end_date'] and hasattr(value, 'strftime'):
                            json_chunk[key] = value.strftime('%Y-%m-%d')
                        else:
                            json_chunk[key] = safe_json_convert(value)
                    json_chunks.append(json_chunk)
                with open(os.path.join(output_directory, 'time_chunks.json'), 'w') as f:
                    json.dump(json_chunks, f, indent=2, cls=NumpyEncoder)
        
        # Export configuration and metadata
        export_metadata = {
            'experiment_directory': self.experiment_directory,
            'dataset_directory': str(self.dataset_directory),
            'config': safe_json_convert(self.config),
            'analysis_components': {
                'has_topic_vectors': self.topic_vectors is not None,
                'has_topic_embedding': self.topic_embedding is not None,
                'has_author_embeddings': hasattr(self, 'author_ct_distns') and self.author_ct_distns is not None,
                'has_interdisciplinarity_scores': bool(self.interdisciplinarity_scores),
                'num_time_chunks': int(len(self.time_chunks)),  # Ensure Python int
                'coauthor_network_nodes': int(self.coauthor_network.number_of_nodes()) if self.coauthor_network else 0,
                'coauthor_network_edges': int(self.coauthor_network.number_of_edges()) if self.coauthor_network else 0
            },
            'export_timestamp': dt.datetime.now().isoformat()
        }
        
        # Safe convert the entire metadata structure
        safe_metadata = safe_json_convert(export_metadata)
        
        with open(os.path.join(output_directory, 'analysis_metadata.json'), 'w') as f:
            json.dump(safe_metadata, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Results export completed - {len(formats)} formats exported")
        return self
    
    def finalize_experiment(self, 
                          export_results: bool = True, 
                          save_state: bool = True,
                          formats: Optional[List[str]] = None) -> str:
        """
        Finalize the experiment by saving all results and orchestrator state.
        
        Args:
            export_results: Whether to export analysis results
            save_state: Whether to save orchestrator state
            formats: Export formats for results
            
        Returns:
            Path to experiment directory
        """
        self.logger.info("Finalizing experiment...")
        
        if export_results:
            self.export_results(formats=formats)
        
        if save_state:
            self.save_orchestrator_state()
        
        self.logger.info(f"Experiment finalized in: {self.experiment_directory}")
        return self.experiment_directory
    
    # ========================================================================================
    # 9. UTILITY AND CONFIGURATION METHODS
    # ========================================================================================
    
    def _get_default_config(self) -> dict:
        """Load default configuration from YAML file."""
        config_path = Path(__file__).parent / 'config.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.error(f"Config file not found at {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            raise
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger for MSTML operations."""
        logger = logging.getLogger('MSTML')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_status(self) -> dict:
        """Get current status of orchestrator components."""
        return {
            'data_loaded': self.documents_df is not None,
            'coauthor_network_built': self.coauthor_network is not None,
            'text_preprocessed': (self.documents_df is not None and 
                                MainDataSchema.PREPROCESSED_TEXT.colname in self.documents_df.columns and
                                self.bow_corpus is not None),
            'ensemble_trained': len(self.chunk_topic_models) > 0,
            'dendrogram_built': self.topic_dendrogram_linkage is not None,
            'author_embeddings_computed': self.author_embeddings is not None,
            'topic_embedding_created': self.topic_embedding is not None,
            'num_documents': len(self.documents_df) if self.documents_df is not None else 0,
            'num_authors': len(self.authors_df) if self.authors_df is not None else 0,
            'num_topics': len(self.topic_vectors) if self.topic_vectors is not None else 0,
            'num_time_chunks': len(self.time_chunks)
        }



