# Multiscale Topic Manifold Learning (MSTML) [Version 0.01 - Beta Version]

This is the public release version of the code and examples related to the papers: "Multiscale Topic Manifold Learning for Understanding Interdisciplinary Co-Authorship" and "A Multiscale Geometric Method for Capturing Relational Topic Alignment".

## Overview

MSTML provides a scalable method for predicting collaborative behaviors using textual data and probabilistic information geometry of author-topic interests. It combines a relational topic modeling (RTM) approach with time alignment and multiscale learning. The framework combines:

- **GDLTM**: Geometry-Driven Longitudinal Topic Model for analyzing topic evolution over time. This is based on the paper "A Geometry-Driven Longitudinal Topic Model" (Wang, Hougen, et. al.; Harvard Data Science Review, 2021).
- **HRG**: Hierarchical Random Graph models for network structure analysis and link prediction. This is based on "Hierarchical structure and the prediction of missing links in networks" (Clauset, Moore, and Newman; Nature 453, 98-101, 2008)
- **Text Processing**: Comprehensive preprocessing utilities for academic documents
- **Network Analysis**: Tools for analyzing collaboration patterns and community structure

## Installation

MSTML uses a custom build system that automatically handles dependencies, compilation, and installation:

```bash
# Clone the repository
git clone https://github.com/your-repo/Multiscale-Topic-Manifold-Learning.git
cd Multiscale-Topic-Manifold-Learning

# Run the automated build and installation
python build.py
```

The `build.py` script will:
- Create and configure a conda environment
- Install all required dependencies
- Compile Cython extensions for optimal performance
- Install the MSTML package in development mode

### Build Requirements
- **Python 3.7+**
- **C Compiler** (GCC, Clang, or MSVC)
- **Conda or Miniconda** (recommended for dependency management)

### Performance Note
MSTML includes Cython extensions for optimal performance. The `fast_encode_tree` module provides 5-50x speedup for hierarchical operations. If compilation fails, the framework will fall back to pure Python implementations with a performance warning.

## Quick Start

### Basic Usage

```python
from mstml.core import MstmlOrchestrator

# Create orchestrator for your dataset
orchestrator = MstmlOrchestrator("arxiv")

# Configure data filters for AI/ML papers from 2020-2023
orchestrator.configure_data_filters(
    date_range={'start': '2020-01-01', 'end': '2023-12-31'},
    categories=['cs.AI', 'cs.LG', 'stat.ML', 'cs.CL']
)

# Run complete analysis pipeline
orchestrator.load_data()
orchestrator.preprocess_text()
orchestrator.setup_coauthor_network()
orchestrator.create_temporal_chunks(months_per_chunk=6)
orchestrator.train_ensemble_models()
orchestrator.build_topic_manifold()
orchestrator.compute_author_embeddings()

# Compute interdisciplinarity scores
orchestrator.compute_interdisciplinarity_scores_docs()
orchestrator.compute_interdisciplinarity_scores_authors()

# Create and save topic space embedding
orchestrator.create_topic_embedding(method='phate', cut_height=0.5)
orchestrator.display_topic_embedding(save_path='topic_embedding.png')

# Save results
results_path = orchestrator.finalize_experiment()
print(f"Analysis complete! Results saved to: {results_path}")
```

### Advanced Configuration

```python
from mstml.core import MstmlOrchestrator
from mstml.text_preprocessing import TextPreprocessor
from mstml.author_disambiguation import AuthorDisambiguator

# Configure components with custom hyperparameters
text_preprocessor = TextPreprocessor(
    custom_stopwords=['arxiv', 'paper', 'abstract']
)

author_disambiguator = AuthorDisambiguator(
    similarity_threshold=0.85,  # More aggressive name merging
    max_authors_per_doc=15     # Skip documents with too many authors
)

# Create orchestrator with pre-configured components
orchestrator = MstmlOrchestrator(
    dataset_name="arxiv",
    experiment_name="ai_trends_analysis",
    text_preprocessor=text_preprocessor,
    author_disambiguator=author_disambiguator
)

# Advanced text preprocessing options
orchestrator.preprocess_text(
    low_thresh=3,              # Remove rare terms (< 3 documents)
    high_frac=0.98,            # Remove common terms (> 98% documents)
    num_topics=80,             # LDA topics for relevancy filtering
    top_n_terms=2500           # Final vocabulary size
)

# Advanced topic embedding and interdisciplinarity analysis
orchestrator.build_topic_manifold()
orchestrator.compute_author_embeddings()

# Compute interdisciplinarity scores with custom parameters
orchestrator.compute_interdisciplinarity_scores_docs(
    entropy_weighting=True,
    author_publication_weighting=True,
    topn=100  # Get top 100 most interdisciplinary documents
)

orchestrator.compute_interdisciplinarity_scores_authors(
    entropy_weighting=True,
    publication_count_weighting=True,
    top_topics_threshold=5,    # Consider only top 5 topics per author
    topn=50                    # Get top 50 most interdisciplinary authors
)

# Create topic embeddings with different methods
# PHATE embedding (default, preserves local and global structure)
orchestrator.create_topic_embedding(
    method='phate',
    n_components=2,
    cut_height=0.5,           # Cut dendrogram to create meta-topics
    color_by='meta_topic'
)

# Alternative: UMAP embedding (faster, good for large datasets)
orchestrator.create_topic_embedding(
    method='umap',
    n_components=2,
    knn_neighbors=15,
    min_dist=0.1
)

# Alternative: t-SNE embedding (good for local structure)
orchestrator.create_topic_embedding(
    method='tsne',
    n_components=2,
    perplexity=30
)

# Display and save visualizations
orchestrator.display_topic_embedding(
    title='Topic Space Embedding - Meta-Topic Clusters',
    save_path='topic_space_embedding.png'
)
```


## Repository Structure

```
Multiscale-Topic-Manifold-Learning/
├── mstml/                      # Core MSTML package
│   ├── __init__.py            # Main package imports
│   ├── core.py                # MstmlOrchestrator and core functionality
│   ├── data_loaders.py        # Data loading and preprocessing pipeline
│   ├── dataframe_schema.py    # Data extraction and field definitions
│   ├── author_disambiguation.py # Author name disambiguation and linking
│   ├── text_preprocessing.py   # Text processing and vocabulary filtering
│   ├── data_loader_registry.py # Extensible file format support
│   ├── model_evaluation.py    # Model evaluation and metrics
│   ├── fast_encode_tree/      # High-performance Cython extensions
│   │   ├── __init__.py       # Fast tree encoding for HRG
│   │   ├── fast_encode_tree.pyx # Cython implementation
│   │   └── fast_encode_tree_py.py # Python fallback
│   ├── _embedding_driver.py   # Embedding methods (PHATE, UMAP, etc.)
│   ├── _graph_driver.py       # Network analysis and graph operations
│   ├── _math_driver.py        # Mathematical utilities and distance metrics
│   ├── _topic_model_driver.py # Topic modeling utilities (LDA, etc.)
│   └── _file_driver.py        # File I/O and utility functions
├── notebooks/                 # Example notebooks and tutorials
│   ├── 01_gdltm_geometric_longitudinal_topic_model.ipynb
│   ├── 02_mstml_multimodal_topic_drift.ipynb
│   ├── 03_mstml_vocabulary_filtering_and_tuning.ipynb
│   └── 04_mstml_network_link_prediction.ipynb
├── data/                      # Data directory (created automatically)
│   ├── experiments/          # Experiment results and outputs
│   └── [dataset_name]/       # Dataset-specific directories
│       ├── original/         # Raw input data files
│       ├── clean/           # Processed dataframes and metadata
│       └── networks/        # Network data and analysis results
├── tests/                     # Unit tests and test utilities
├── papers/                    # Research papers and documentation
├── build.py                   # Automated build and installation script
├── setup.py                   # Package installation configuration
├── pyproject.toml            # Modern Python project configuration
├── requirements.txt          # Python dependencies
├── conda_requirements.txt    # Conda-specific dependencies
└── README.md                 # This file
```

## Key Features

### 1. Geometry-Driven Longitudinal Topic Model (GDLTM)
- Extracts topics from document collections across time slices
- Supports multiple embedding methods (PHATE, UMAP, t-SNE, PCA) for topic space visualization
- Uses Hellinger distances between topics for geometric analysis
- Discovers topic trajectories using shortest path algorithms
- Provides hierarchical clustering of topics with dendrogram cut heights for meta-topic creation
- Computes interdisciplinarity scores for both documents and authors

### 2. Hierarchical Random Graph (HRG)
- Fits hierarchical models to network data
- Predicts missing links with probabilistic scores
- Extracts community structure from network hierarchy
- Supports ensemble methods for improved robustness

### 3. Text Processing Pipeline
- Academic document preprocessing
- Tokenization, cleaning, and normalization
- Stopword removal and lemmatization
- Frequency-based filtering and n-gram extraction

### 4. Network Analysis Tools
- Collaboration network creation from author data
- Network metric computation (centrality, clustering, etc.)
- Community detection and analysis
- Visualization utilities

## Usage Examples

See the `notebooks/` directory for comprehensive examples:

- `01_gdltm_geometric_longitudinal_topic_model.ipynb`: GDLTM topic evolution analysis
- `02_mstml_multimodal_topic_drift.ipynb`: Multiscale topic drift detection
- `03_mstml_vocabulary_filtering_and_tuning.ipynb`: Text preprocessing and vocabulary optimization
- `04_mstml_network_link_prediction.ipynb`: Co-author network analysis and link prediction

## Data Format

The framework creates preprocessed data in the following format:
- `main_df.pkl`: Main dataframe with document metadata
- `data_words.pkl`: Tokenized documents
- `id2word.pkl`: Vocabulary mapping
- `authorId_to_author.pkl`: Author ID mappings
- `author_to_authorId.pkl`: Reverse author mappings

## Citation

If you use this code in your research, please cite:

```bibtex
@unpublished{hougen2025mstml_camsap,
  author  = {Conrad D. Hougen and Karl T. Pazdernik and Alfred O. Hero},
  title   = {A Multiscale Geometric Method for Capturing Relational Topic Alignment},
  note    = {Preprint}
}

@unpublished{hougen2025mstml,
  author  = {Conrad D. Hougen and Karl T. Pazdernik and Alfred O. Hero},
  title   = {Multiscale Topic Manifold Learning for Understanding Interdisciplinarity in Co-Authored Articles},
  note    = {Preprint}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Planned Features

The following features are planned for future releases but are not yet completely implemented:

- **Abstract Manifold Learning Support**: Generalized framework for integrating any manifold learning or low-dimensional embedding technique, with standardized interfaces for custom distance metrics and dimensionality reduction methods beyond the current PHATE, UMAP, t-SNE, and PCA implementations.

- **Interactive Topic Visualization Widgets**: Web-based interactive widgets for labeling, exploring, and displaying topics within embeddings, including real-time clustering, topic annotation tools, and dynamic filtering capabilities for improved topic interpretation and analysis.

- **Link Prediction**: Enhanced algorithms for predicting missing or future collaboration links in co-author networks, incorporating both topological features and semantic similarity from topic distributions to improve prediction accuracy.

- **Hyperlink Prediction in Co-Author Hypergraphs**: Advanced prediction methods for higher-order collaboration patterns in co-author hypergraphs, where hyperedges represent multi-author collaborations, enabling prediction of complex team formation dynamics beyond pairwise relationships.

- **Automated Topic Labeling with LLMs**: Integration of large language models for automated labeling and interpretation of multiscale chunk topic clusters, leveraging contextual understanding to generate meaningful topic names and descriptions from topic-word distributions and representative documents.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
