# Multi-Scale Topic Manifold Learning (MSTML)

This is the public release version of the code and examples related to the paper "Multi-Scale Topic Manifold Learning for Understanding Interdisciplinary Collaborations in Co-Author Networks".

## Overview

MSTML provides a scalable method for predicting collaborative behaviors using textual data and probabilistic information geometry of author-topical interests. The framework combines:

- **GDLTM**: Geometry-Driven Longitudinal Topic Model for analyzing topic evolution over time
- **HRG**: Hierarchical Random Graph models for network structure analysis and link prediction
- **Text Processing**: Comprehensive preprocessing utilities for academic documents
- **Network Analysis**: Tools for analyzing collaboration patterns and community structure

## Installation

### Quick Installation

#### Windows (Recommended)
```cmd
git clone https://github.com/ConradHougen/Multi-Scale-Topic-Manifold-Learning.git
cd Multi-Scale-Topic-Manifold-Learning

# Double-click setup-dev.bat or run:
setup-dev.bat

# Or use the Python build script:
python build.py dev
```

#### Linux/macOS
```bash
git clone https://github.com/ConradHougen/Multi-Scale-Topic-Manifold-Learning.git
cd Multi-Scale-Topic-Manifold-Learning

# Using make (if available)
make dev

# Or using Python build script (works everywhere)
python build.py dev
```

#### Manual Setup (All Platforms)
```bash
python setup_mstml.py
pip install -r requirements.txt
python setup.py build_ext --inplace
```

#### If You Get Build Errors
If you encounter "Preparing metadata (pyproject.toml) ... error" or similar issues:

```bash
# Try the simple build method (recommended)
python build_simple.py

# Or install dependencies first, then build
pip install numpy cython setuptools
python setup.py build_ext --inplace
```

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Build Requirements
- **Python 3.7+**
- **C Compiler** (GCC, Clang, or MSVC)
- **NumPy and Cython** (for compiling performance extensions)

For detailed build instructions, see [BUILD.md](BUILD.md).

### Performance Note
MSTML includes Cython extensions for optimal performance. The `fast_encode_tree` module provides 5-50x speedup for hierarchical operations. If compilation fails, the framework will fall back to pure Python implementations with a performance warning.

## Quick Start

```python
import sys
sys.path.append('source')

from source import Mstml, MstmlParams
from source.gdltm import Gdltm, GdltmParams
from source.hrg import HierarchicalRandomGraph
from preprocessing.text_processing import create_academic_text_processor

# Text processing
processor = create_academic_text_processor()
processed_docs = processor.process_documents(your_documents)

# Topic modeling with GDLTM
gdltm_params = GdltmParams(dset="your_dataset", dsub="subset", ntopics=10)
gdltm = Gdltm(gdltm_params)
gdltm.run_full_pipeline()

# Network analysis with HRG
hrg = HierarchicalRandomGraph(your_network)
hrg.fit()
predictions = hrg.predict_links()
```

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
- Additional notebooks demonstrating specific use cases

## Data Format

The framework expects preprocessed data in the following format:
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