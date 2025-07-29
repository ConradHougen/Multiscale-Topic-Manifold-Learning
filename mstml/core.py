"""
Core MSTML (Multi-Scale Topic Manifold Learning) functionality.

This module provides user-friendly interfaces and integration points for the MSTML framework,
combining temporal topic modeling (GDLTM), network analysis, and hierarchical modeling.
"""

import numpy as np
import pandas as pd
import logging
import os
import json  # For metadata
import datetime as dt
import pandas.api.types as ptypes
from abc import ABC, abstractmethod

from typing import Optional, Union, Dict, Any, List, Tuple

from .dataframe_schema import FieldDef, MainDataSchema
from .data_loaders import get_project_root_directory, get_data_directory, JsonDataLoader
from .utils import *
from .gdltm_utils import *
from .mstml_utils import *

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
    
    def __init__(self, n_topics: int = 10, alpha: Union[str, float] = 'auto', 
                 eta: Union[str, float] = 'auto', random_state: int = 42, 
                 passes: int = 10, iterations: int = 50, **kwargs):
        self.n_topics = n_topics
        self.alpha = alpha  # Document-topic density (higher = more topics per doc)
        self.eta = eta      # Topic-word density (higher = more words per topic) 
        self.random_state = random_state
        self.passes = passes  # Number of passes through corpus during training
        self.iterations = iterations  # Number of iterations per pass
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.kwargs = kwargs
    
    def fit(self, documents: List[str], vocabulary: Optional[Dict[str, int]] = None) -> 'LDATopicModel':
        try:
            from gensim import corpora
            from gensim.models import LdaModel
            import gensim
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
            passes=self.passes,
            iterations=self.iterations,
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
            'passes': self.passes,
            'iterations': self.iterations,
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
            from gensim.models import CoherenceModel
            
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
    
    def compute_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute Hellinger distance matrix.
        
        For probability distributions P and Q:
        H(P,Q) = (1/√2) * ||√P - √Q||_2
        """
        # Ensure vectors are probability distributions
        vectors = vectors / vectors.sum(axis=1, keepdims=True)
        sqrt_vectors = np.sqrt(vectors)
        
        n_samples = vectors.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.linalg.norm(sqrt_vectors[i] - sqrt_vectors[j]) / np.sqrt(2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def compute_pairwise_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Ensure vectors are probability distributions
        X = X / X.sum(axis=1, keepdims=True)
        Y = Y / Y.sum(axis=1, keepdims=True)
        
        sqrt_X = np.sqrt(X)
        sqrt_Y = np.sqrt(Y)
        
        distances = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                distances[i, j] = np.linalg.norm(sqrt_X[i] - sqrt_Y[j]) / np.sqrt(2)
        
        return distances
    
    def get_metric_name(self) -> str:
        return "hellinger"


class PHATEEmbedding(LowDimEmbedding):
    """
    PHATE embedding implementation.
    
    Default dimensionality reduction method for MSTML topic visualization.
    """
    
    def __init__(self, n_components: int = 2, knn_neighbors: int = 5, 
                 gamma: float = 1.0, t: int = 10, **kwargs):
        self.n_components = n_components
        self.knn_neighbors = knn_neighbors
        self.gamma = gamma
        self.t = t
        self.kwargs = kwargs
        self.phate_model = None
        self.fitted = False
    
    def fit_transform(self, X: np.ndarray, 
                     distance_metric: Optional[EmbeddingDistanceMetric] = None) -> np.ndarray:
        try:
            import phate
        except ImportError:
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
    """
    def __init__(self, 
                 data_directory: Optional[str] = None,
                 config: Optional[dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize MSTML Orchestrator with data directory and configuration.
        
        Args:
            data_directory: Path to data directory, defaults to project data/
            config: Configuration dictionary for model parameters
            logger: Custom logger, creates default if None
        """
        # Core configuration
        self.data_directory = data_directory or get_data_directory()
        self.config = config or self._get_default_config()
        self.logger = logger or self._setup_logger()
        
        # Data management
        self.schema = None
        self.data_loader = None
        self.documents_df = None
        self.authors_df = None
        self.coauthor_network = None
        
        # Text processing components
        self.vocabulary = None
        self.global_lda_model = None  # For term relevancy filtering
        self.preprocessed_corpus = None
        
        # Temporal ensemble components
        self.time_chunks = []
        self.chunk_topic_models = []
        self.chunk_topics = []
        self.smoothing_decay = self.config.get('temporal_smoothing_decay', 0.75)
        
        # Modular components (dependency injection)
        self.topic_model_factory = create_topic_model
        self.embedding_method = None
        self.distance_metric = None
        
        # Set default components
        self._set_default_components()
        
        # Topic manifold and embedding components
        self.topic_vectors = None  # All phi vectors from ensemble
        self.topic_knn_graph = None
        self.topic_dendrogram = None
        self.internal_node_probabilities = {}
        self.topic_embedding = None
        
        # Author representation components
        self.author_topic_distributions = None  # psi(u) vectors
        self.author_embeddings = None
        self.interdisciplinarity_scores = {}
        
        # Analysis results
        self.topic_trajectories = []
        self.community_drift_analysis = {}
        self.link_predictions = {}
        
        # Visualization components
        self.embed_params = {}
        self.embed_figure = None
        
        self.logger.info("MstmlOrchestrator initialized")

    # ========================================================================================
    # 1. DATA LOADING AND MANAGEMENT
    # ========================================================================================
    
    def load_data(self, 
                  data_source: str,
                  schema_config: Optional[dict] = None,
                  force_reload: bool = False) -> 'MstmlOrchestrator':
        """
        Load and validate document corpus data.
        
        Args:
            data_source: Path to data file or identifier
            schema_config: Custom schema configuration
            force_reload: Force reload even if data exists
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement data loading logic
        # - Initialize appropriate DataLoader based on file type
        # - Set up MainDataSchema with field definitions
        # - Load and validate documents dataframe
        # - Extract author information and create authors dataframe
        self.logger.info(f"Loading data from {data_source}")
        return self
    
    def setup_coauthor_network(self, 
                              author_disambiguation: bool = True,
                              min_collaborations: int = 1) -> 'MstmlOrchestrator':
        """
        Create co-author network from loaded document data.
        
        Args:
            author_disambiguation: Apply fuzzy author name matching
            min_collaborations: Minimum collaborations to include edge
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement co-author network construction
        # - Apply author disambiguation if requested
        # - Build undirected co-author graph G = (V, E)
        # - Store network structure and metadata
        self.logger.info("Setting up co-author network")
        return self
    
    # ========================================================================================
    # 2. TEXT PREPROCESSING AND VOCABULARY FILTERING
    # ========================================================================================
    
    def preprocess_text(self,
                       remove_stopwords: bool = True,
                       apply_stemming: bool = True,
                       min_term_freq: int = 2,
                       max_term_freq: float = 0.8,
                       custom_filters: Optional[list] = None) -> 'MstmlOrchestrator':
        """
        Apply text preprocessing and vocabulary filtering.
        
        Args:
            remove_stopwords: Remove common stop words
            apply_stemming: Apply stemming/lemmatization
            min_term_freq: Minimum term frequency threshold
            max_term_freq: Maximum term frequency threshold (fraction)
            custom_filters: Additional custom preprocessing filters
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement text preprocessing pipeline
        # - Apply standard NLP preprocessing (stopwords, stemming, etc.)
        # - Filter vocabulary by frequency thresholds
        # - Create vocabulary mapping and document term matrices
        self.logger.info("Preprocessing text and filtering vocabulary")
        return self
    
    def apply_term_relevancy_filtering(self,
                                     lambda_param: float = 0.4,
                                     top_terms_per_topic: int = 50) -> 'MstmlOrchestrator':
        """
        Apply global LDA-based term relevancy filtering.
        
        Uses global LDA model to compute term relevancy scores:
        r(w,k|λ) = λ*log P(w|k) + (1-λ)*log[P(w|k)/P(w)]
        
        Args:
            lambda_param: Relevancy weighting parameter (0-1)
            top_terms_per_topic: Number of top relevant terms to retain
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement term relevancy filtering
        # - Train global LDA model on full corpus
        # - Compute term relevancy scores for each topic
        # - Filter vocabulary to most relevant terms
        # - Update document representations
        self.logger.info(f"Applying term relevancy filtering with λ={lambda_param}")
        return self
    
    # ========================================================================================
    # 3. TEMPORAL ENSEMBLE TOPIC MODELING
    # ========================================================================================
    
    def create_temporal_chunks(self,
                             chunk_size: Optional[int] = None,
                             chunk_duration: Optional[str] = None,
                             overlap_factor: float = 0.0) -> 'MstmlOrchestrator':
        """
        Split corpus into temporal chunks for ensemble learning.
        
        Args:
            chunk_size: Number of documents per chunk
            chunk_duration: Time duration per chunk (e.g., '3M' for 3 months)
            overlap_factor: Fraction of overlap between adjacent chunks
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement temporal chunking
        # - Split corpus by time or document count
        # - Apply optional overlap between chunks
        # - Store chunk metadata and document assignments
        self.logger.info("Creating temporal chunks")
        return self
    
    def apply_temporal_smoothing(self,
                               decay_parameter: float = 0.75) -> 'MstmlOrchestrator':
        """
        Apply exponential decay smoothing across temporal chunks.
        
        Uses exponential weights w_t = γ^|τ-t| for subsampling documents
        across time chunks to ensure topic continuity.
        
        Args:
            decay_parameter: Exponential decay factor γ ∈ (0,1)
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement temporal smoothing
        # - Apply exponential decay weighting to documents
        # - Create smoothed sub-corpora for each time chunk
        # - Update chunk document assignments
        self.smoothing_decay = decay_parameter
        self.logger.info(f"Applying temporal smoothing with γ={decay_parameter}")
        return self
    
    def train_ensemble_models(self,
                            base_model: str = 'LDA',
                            topics_per_chunk: Optional[int] = None,
                            model_params: Optional[dict] = None) -> 'MstmlOrchestrator':
        """
        Train ensemble of topic models on temporal chunks.
        
        Args:
            base_model: Base topic model type ('LDA', 'T-LDA', etc.)
            topics_per_chunk: Number of topics per chunk model
            model_params: Additional model parameters
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement ensemble model training
        # - Train separate topic models on each temporal chunk
        # - Extract topic-word distributions φ(k) for each model
        # - Store model outputs and metadata
        # - Scale K as affine function of documents per chunk if not specified
        self.logger.info(f"Training {base_model} ensemble models")
        return self
    
    # ========================================================================================
    # 4. TOPIC MANIFOLD LEARNING AND DENDROGRAM CONSTRUCTION
    # ========================================================================================
    
    def build_topic_manifold(self,
                           distance_metric: str = 'hellinger',
                           knn_neighbors: int = 10,
                           use_faiss: bool = True) -> 'MstmlOrchestrator':
        """
        Build topic manifold using information-geometric distances.
        
        Args:
            distance_metric: Distance metric ('hellinger', 'euclidean')
            knn_neighbors: Number of nearest neighbors for graph construction
            use_faiss: Use FAISS for fast approximate nearest neighbors
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement topic manifold construction
        # - Collect all topic vectors {φ(k)} from ensemble models
        # - Build k-nearest neighbors graph using Hellinger distance
        # - Use FAISS for scalable approximate NN if requested
        # - Store topic graph structure for downstream analysis
        self.logger.info(f"Building topic manifold with {distance_metric} distance")
        return self
    
    def construct_topic_dendrogram(self,
                                 linkage_method: str = 'ward',
                                 height_normalization: bool = True) -> 'MstmlOrchestrator':
        """
        Construct hierarchical topic dendrogram using agglomerative clustering.
        
        Args:
            linkage_method: Hierarchical clustering linkage ('ward', 'average', 'complete')
            height_normalization: Normalize dendrogram heights to [0,1]
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement dendrogram construction
        # - Apply agglomerative clustering to topic vectors
        # - Use Ward's linkage: d(Ci,Cj) = |Ci|×|Cj|/(|Ci|+|Cj|) × ||μi-μj||²
        # - Build binary tree structure with internal nodes
        # - Normalize heights for consistent interpretation
        self.logger.info(f"Constructing topic dendrogram with {linkage_method} linkage")
        return self
    
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
        # TODO: Implement HRG probability estimation
        # - Compute expected edges E_m for each internal node
        # - Estimate left/right subtree sizes L_m, R_m
        # - Calculate MLE probabilities p_m = E_m / (L_m × R_m)
        # - Store probabilities for link evaluation and prediction
        self.logger.info("Estimating dendrogram node probabilities")
        return self
    
    # ========================================================================================
    # 5. AUTHOR REPRESENTATION LEARNING
    # ========================================================================================
    
    def compute_author_embeddings(self,
                                weighting_scheme: str = 'inverse_coauthors',
                                apply_diffusion: bool = True,
                                diffusion_steps: int = 1) -> 'MstmlOrchestrator':
        """
        Compute author embeddings in topic space.
        
        Creates author-topic barycenters: ξ(u) = Σ (1/|V(j)|) θ(j)
        for author u across their documents C(u).
        
        Args:
            weighting_scheme: Document weighting ('inverse_coauthors', 'uniform')
            apply_diffusion: Apply diffusion process to redistribute probability mass
            diffusion_steps: Number of diffusion steps on topic k-NN graph
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement author embedding computation
        # - Create weighted averages of document-topic distributions
        # - Weight by inverse number of co-authors per document
        # - Apply diffusion process using topic k-NN graph if requested
        # - Normalize to create proper probability distributions ψ(u)
        self.logger.info("Computing author embeddings in topic space")
        return self
    
    def compute_interdisciplinarity_scores_docs(self,
                                              entropy_weighting: bool = True,
                                              author_publication_weighting: bool = True,
                                              topn: Optional[int] = None) -> 'MstmlOrchestrator':
        """
        Compute interdisciplinarity scores for documents.
        
        Score = H(Ω(j)) × Σ w_u for document j with topic distribution Ω(j)
        where w_u = √N_D(u) weights authors by publication count.
        
        Args:
            entropy_weighting: Weight by topic distribution entropy
            author_publication_weighting: Weight by author publication counts
            topn: Return top N ranked interdisciplinary documents (None for all)
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement document interdisciplinarity scoring
        # - Compute document-level topic distributions from author contributions
        # - Calculate entropy H(Ω(j)) for each document
        # - Weight by author publication counts if requested
        # - Rank and return top N documents if topn specified
        # - Store scores for anomaly detection and analysis
        self.logger.info(f"Computing interdisciplinarity scores for documents{' (top ' + str(topn) + ')' if topn else ''}")
        return self
    
    def compute_interdisciplinarity_scores_authors(self,
                                                  entropy_weighting: bool = True,
                                                  publication_count_weighting: bool = True,
                                                  top_topics_threshold: int = 5,
                                                  topn: Optional[int] = None) -> 'MstmlOrchestrator':
        """
        Compute interdisciplinarity scores for authors.
        
        Score based on author topic distribution entropy and publication patterns.
        
        Args:
            entropy_weighting: Weight by topic distribution entropy
            publication_count_weighting: Weight by author publication counts
            top_topics_threshold: Number of top topics to retain per author
            topn: Return top N ranked interdisciplinary authors (None for all)
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement author interdisciplinarity scoring
        # - Compute author-level topic distributions ψ(u)
        # - Calculate entropy H(ψ(u)) for each author
        # - Weight by publication counts if requested
        # - Filter to top topics per author if specified
        # - Rank and return top N authors if topn specified
        # - Store scores for anomaly detection and analysis
        self.logger.info(f"Computing interdisciplinarity scores for authors{' (top ' + str(topn) + ')' if topn else ''}")
        return self
    
    # ========================================================================================
    # 6. LONGITUDINAL ANALYSIS AND VISUALIZATION
    # ========================================================================================
    
    def create_phate_embedding(self,
                             n_components: int = 2,
                             knn_neighbors: int = 5,
                             gamma: float = 1.0,
                             t: int = 10,
                             color_by: str = 'time',
                             show_colorbar: bool = True,
                             interactive: bool = False,
                             figure_size: tuple = (10, 8)) -> 'MstmlOrchestrator':
        """
        Create PHATE embedding for topic trajectory visualization.
        
        PHATE preserves local and global distances through density-adaptive
        diffusion process, ideal for visualizing topic drift over time.
        
        Args:
            n_components: Number of embedding dimensions
            knn_neighbors: Number of nearest neighbors for PHATE
            gamma: PHATE gamma parameter for kernel bandwidth
            t: Number of diffusion steps
            color_by: Color coding scheme ('time', 'meta_topic', 'none')
            show_colorbar: Whether to display colorbar
            interactive: Whether to create interactive plot (plotly vs matplotlib)
            figure_size: Figure size as (width, height) tuple
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement PHATE embedding
        # - Apply PHATE to topic vectors using Hellinger distance
        # - Preserve temporal structure and topic relationships
        # - Store embedding coordinates and visualization parameters
        # - Support time-based and meta-topic color coding
        # - Handle colorbar and interactivity options
        self.embed_params = {
            'n_components': n_components,
            'knn_neighbors': knn_neighbors,
            'gamma': gamma,
            't': t,
            'color_by': color_by,
            'show_colorbar': show_colorbar,
            'interactive': interactive,
            'figure_size': figure_size
        }
        self.logger.info(f"Creating {n_components}D PHATE embedding with {color_by} coloring")
        return self
    
    def display_topic_embedding(self,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Display the topic embedding visualization.
        
        Args:
            title: Custom title for the plot
            save_path: Optional path to save the figure
        """
        if self.topic_embedding is None:
            raise ValueError("Topic embedding not created. Call create_topic_embedding() first.")
        
        import matplotlib.pyplot as plt
        
        params = self.embed_params
        method_name = params.get('method', 'Embedding')
        fig, ax = plt.subplots(figsize=params.get('figure_size', (10, 8)))
        
        # Create scatter plot based on color scheme
        if params.get('color_by') == 'time':
            # TODO: Color by time chunk assignment
            scatter = ax.scatter(self.topic_embedding[:, 0], self.topic_embedding[:, 1],
                               c=range(len(self.topic_embedding)), cmap='viridis', alpha=0.7)
            if params.get('show_colorbar', True):
                plt.colorbar(scatter, ax=ax, label='Time')
        elif params.get('color_by') == 'meta_topic':
            # TODO: Color by meta-topic cluster assignment from dendrogram
            scatter = ax.scatter(self.topic_embedding[:, 0], self.topic_embedding[:, 1],
                               c='blue', alpha=0.7)  # Placeholder
        else:
            scatter = ax.scatter(self.topic_embedding[:, 0], self.topic_embedding[:, 1],
                               c='blue', alpha=0.7)
        
        ax.set_xlabel(f'{method_name} 1')
        ax.set_ylabel(f'{method_name} 2')
        ax.set_title(title or f'{method_name} Embedding of Topic Manifold')
        
        if params.get('interactive', False):
            # TODO: Convert to plotly for interactivity
            self.logger.warning("Interactive plotting not yet implemented, showing static plot")
        
        self.embed_figure = fig
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"{method_name} embedding saved to {save_path}")
        
        plt.show()
        self.logger.info(f"{method_name} embedding displayed")
    
    def save_topic_embedding(self,
                           filename: Optional[str] = None,
                           format: str = 'png',
                           dpi: int = 300) -> str:
        """
        Save the topic embedding figure to the experiment directory.
        
        Args:
            filename: Custom filename (without extension). Auto-generated if None.
            format: Image format ('png', 'pdf', 'svg', 'eps')
            dpi: Resolution for raster formats
        
        Returns:
            Path to saved file
        """
        if self.embed_figure is None:
            raise ValueError("No topic embedding figure to save. Call display_topic_embedding() first.")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            method_name = self.embed_params.get('method', 'embedding').lower()
            color_suffix = f"_{self.embed_params.get('color_by', 'default')}"
            filename = f"{method_name}_topic_embedding{color_suffix}_{timestamp}"
        
        # Construct full path using data directory as base
        save_path = os.path.join(self.data_directory, f"{filename}.{format}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save figure
        self.embed_figure.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
        
        method_name = self.embed_params.get('method', 'Embedding')
        self.logger.info(f"{method_name} topic embedding saved to {save_path}")
        return save_path
    
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
        
        Args:
            cut_height: Height at which to cut dendrogram. If None, uses median height.
            min_cluster_size: Minimum size for valid clusters
            validate_height: Whether to validate cut_height against dendrogram bounds
        
        Returns:
            Dictionary containing cluster assignments and analysis results
        """
        if self.topic_dendrogram is None:
            raise ValueError("Topic dendrogram not constructed. Call construct_topic_dendrogram() first.")
        
        # Get dendrogram height bounds if validation requested
        if validate_height and hasattr(self, 'topic_dendrogram'):
            # TODO: Extract actual min/max heights from dendrogram structure
            min_height = 0.0  # Placeholder - should be actual minimum
            max_height = 1.0  # Placeholder - should be actual maximum
            
            if cut_height is None:
                cut_height = (min_height + max_height) / 2.0
            elif cut_height < min_height or cut_height > max_height:
                self.logger.warning(
                    f"Cut height {cut_height} outside valid range [{min_height:.3f}, {max_height:.3f}]. "
                    f"Clamping to valid range."
                )
                cut_height = max(min_height, min(cut_height, max_height))
        elif cut_height is None:
            cut_height = 0.5  # Default fallback
        
        # TODO: Implement dendrogram cutting and cluster analysis
        # - Cut dendrogram at specified height to create meta-topics
        # - Filter clusters by minimum size requirement
        # - Analyze topic and author distributions within each cluster
        # - Compute cluster-level statistics and relationships
        # - Return cluster assignments and metadata
        
        result = {
            'cut_height': cut_height,
            'min_cluster_size': min_cluster_size,
            'num_clusters': 0,  # Placeholder
            'cluster_assignments': {},  # topic_id -> cluster_id mapping
            'cluster_sizes': {},  # cluster_id -> size
            'cluster_statistics': {}  # cluster-level analysis
        }
        
        self.logger.info(f"Cut topic dendrogram at height {cut_height:.3f}")
        return result
    
    def export_results(self,
                      output_directory: str,
                      formats: list = ['json', 'csv', 'pkl']) -> 'MstmlOrchestrator':
        """
        Export analysis results in multiple formats.
        
        Args:
            output_directory: Directory for output files
            formats: List of output formats
        
        Returns:
            Self for method chaining
        """
        # TODO: Implement results export
        # - Export topic models, embeddings, and analysis results
        # - Support multiple formats (JSON, CSV, pickle)
        # - Include metadata and configuration information
        self.logger.info(f"Exporting results to {output_directory}")
        return self
    
    # ========================================================================================
    # 9. UTILITY AND CONFIGURATION METHODS
    # ========================================================================================
    
    def _get_default_config(self) -> dict:
        """Get default configuration parameters."""
        return {
            # Temporal ensemble parameters  
            'temporal_smoothing_decay': 0.75,
            
            # Default topic model parameters (Gensim LDA)
            'topic_model_type': 'LDA',
            'lda_alpha': 'auto',
            'lda_eta': 'auto', 
            'lda_passes': 10,
            'lda_iterations': 50,
            
            # Default distance metric parameters (Hellinger)
            'distance_metric_type': 'hellinger',
            
            # Default embedding parameters (PHATE)
            'embedding_method_type': 'PHATE',
            'phate_n_components': 2,
            'phate_knn': 5,
            'phate_gamma': 1.0,
            'phate_t': 10,
            
            # Analysis parameters
            'hellinger_knn_neighbors': 10,
            'ward_linkage_normalize_heights': True,
            'term_relevancy_lambda': 0.4,
            'interdisciplinarity_top_topics': 5,
            'link_anomaly_threshold': 0.1
        }
    
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
            'text_preprocessed': self.preprocessed_corpus is not None,
            'ensemble_trained': len(self.chunk_topic_models) > 0,
            'dendrogram_built': self.topic_dendrogram is not None,
            'author_embeddings_computed': self.author_embeddings is not None,
            'topic_embedding_created': self.topic_embedding is not None,
            'num_documents': len(self.documents_df) if self.documents_df is not None else 0,
            'num_authors': len(self.authors_df) if self.authors_df is not None else 0,
            'num_topics': len(self.topic_vectors) if self.topic_vectors is not None else 0,
            'num_time_chunks': len(self.time_chunks)
        }



