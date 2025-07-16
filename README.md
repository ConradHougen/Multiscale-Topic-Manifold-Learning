# Multi-Scale Topic Manifold Learning (MSTML)

This is the public release version of the code and examples related to the papers: "Multi-Scale Topic Manifold Learning for Understanding Interdisciplinary Collaborations in Co-Author Networks" and "A Multiscale Geometric Method for Capturing Relational Topic Alignment".

## Overview

MSTML provides a scalable method for predicting collaborative behaviors using textual data and probabilistic information geometry of author-topic interests. It combines a relational topic modeling (RTM) approach with time alignment and multiscale learning. The framework combines:

- **GDLTM**: Geometry-Driven Longitudinal Topic Model for analyzing topic evolution over time. This is based on the paper "A Geometry-Driven Longitudinal Topic Model" (Wang, Hougen, et. al.; Harvard Data Science Review, 2021).
- **HRG**: Hierarchical Random Graph models for network structure analysis and link prediction. This is based on "Hierarchical structure and the prediction of missing links in networks" (Clauset, Moore, and Newman; Nature 453, 98-101, 2008)
- **Text Processing**: Comprehensive preprocessing utilities for academic documents
- **Network Analysis**: Tools for analyzing collaboration patterns and community structure

## Installation

### Quick Installation

#### Windows

#### Linux/macOS


### Build Requirements
- **Python 3.7+**
- **C Compiler** (GCC, Clang, or MSVC)
- **NumPy and Cython** (for compiling performance extensions)

### Performance Note
MSTML includes Cython extensions for optimal performance. The `fast_encode_tree` module provides 5-50x speedup for hierarchical operations. If compilation fails, the framework will fall back to pure Python implementations with a performance warning.

## Quick Start


## Repository Structure

```
Multi-Scale-Topic-Manifold-Learning/
├── source/                 # Core MSTML modules
│   ├── __init__.py        # Main package imports
│   ├── core.py            # Core MSTML classes and functionality
│   ├── gdltm.py           # Geometry-Driven Longitudinal Topic Model
│   ├── hrg.py             # Hierarchical Random Graph implementation
│   └── utils.py           # Utility functions and helpers
├── preprocessing/          # Data preprocessing utilities
│   ├── __init__.py
│   └── text_processing.py # Text processing and cleaning
├── notebooks/             # Example notebooks and tutorials
│   └── 01_basic_usage_example.ipynb
├── data/                  # Data directory structure
│   └── arxiv/            # Example dataset organization
├── setup.py              # Package installation script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Features

### 1. Geometry-Driven Longitudinal Topic Model (GDLTM)
- Extracts topics from document collections across time slices
- Uses PHATE embeddings of Hellinger distances between topics
- Discovers topic trajectories using shortest path algorithms
- Provides hierarchical clustering of topics based on geometric structure

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

- `01_basic_usage_example.ipynb`: Introduction to core functionality

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
@article{your_paper,
  title={Multi-Scale Topic Manifold Learning for Understanding Interdisciplinary Collaborations in Co-Author Networks},
  author={Your Authors},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
