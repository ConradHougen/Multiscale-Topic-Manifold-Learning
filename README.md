# Multiscale Topic Manifold Learning (MSTML)

Code for the papers:
- *"A Multiscale Geometric Method for Capturing Relational Topic Alignment"* (CAMSAP 2025)
- *"Network Models for Learning Uncertain and Multimodal Data"* (dissertation)

---

## Overview

MSTML provides a scalable method for predicting collaborative behaviors using textual data and probabilistic information geometry of author-topic interests. It combines:

- **GDLTM**: Geometry-Driven Longitudinal Topic Model for analyzing topic evolution over time, based on *"A Geometry-Driven Longitudinal Topic Model"* (Wang, Hougen et al.; Harvard Data Science Review, 2021).
- **HRG**: Hierarchical Random Graph models for network structure analysis and link prediction, based on *"Hierarchical structure and the prediction of missing links in networks"* (Clauset, Moore & Newman; Nature 453, 2008).
- **Text Processing**: Preprocessing utilities for academic document corpora.
- **Network Analysis**: Tools for analyzing collaboration patterns and community structure.

---

## Reproducing CAMSAP 2025 Results

The `legacy/` directory is a self-contained pipeline that exactly reproduces the CAMSAP 2025 paper results. See [`legacy/README.md`](legacy/README.md) for full instructions.

**Quick start (from the repo root):**

```powershell
# Windows
.\run_experiments.ps1
```

```bash
# Linux / macOS
bash run_experiments.sh
```

This runs the full pipeline twice — once reproducing the original AToMS-LP results (including the known FAISS Hellinger bug), and once with the corrected implementation — and writes outputs to `experiments/arxiv/`.

---

## Installation

The project uses **uv** for environment management. Requires **Python 3.10+**.

```bash
# Install core dependencies
uv sync

# Optional extras
uv sync --extra notebook   # word clouds, interactive plots
uv sync --extra neural     # BERTopic / sentence-transformers
uv sync --extra dev        # pytest, coverage
uv sync --all-extras       # everything
```

---

## Quick Start (`mstml/` library)

```python
import os
from mstml.core import MstmlOrchestrator

orch = MstmlOrchestrator(
    dataset_name="arxiv",
    experiment_name="gdltm",
    experiment_directory=os.path.join(os.pardir, 'experiments', 'gdltm_test1')
)

orch.configure_data_filters(
    date_range={'start': '2012-01-01', 'end': '2023-12-31'},
    categories=['stat.AP', 'stat.CO', 'stat.ME', 'stat.OT', 'stat.TH', 'cs.LG']
)

arxiv_schema_map = {
    'abstract': 'raw_text',
    'update_date': 'date',
    'authors_parsed': 'authors'
}

orch.load_raw_data(input_schema_map=arxiv_schema_map, overwrite=False)
orch.apply_data_filters()
orch.preprocess_text()
orch.apply_author_disambiguation()
orch.setup_coauthor_network(temporal=True)
orch.create_temporal_chunks(months_per_chunk=1, temporal_smoothing_decay=0.75)
orch.train_chunk_models(overwrite=False)
orch.build_author_document_distributions()
orch.apply_diffusion()
orch.build_topic_manifold()
orch.create_topic_embedding(cut_height=0.7)
orch.display_topic_embedding(color_by='meta_topic')
results_path = orch.finalize_experiment()
print(f"Results saved to: {results_path}")
```

---

## Repository Structure

```
Multiscale-Topic-Manifold-Learning/
├── legacy/                     # Self-contained CAMSAP 2025 reproduction pipeline
│   ├── README.md               ← full instructions, math, test reference
│   ├── run_pipeline.py         ← CLI entrypoint (6 resumable stages)
│   ├── config.yaml             ← secondary hyperparameter defaults
│   ├── _math.py / _preprocessing.py / _topic_models.py / ...
│   ├── comparison/             ← MSTML vs BERTopic comparison experiments
│   └── tests/                  ← 201 unit + integration tests
├── mstml/                      # Core MSTML library (in development)
│   ├── core.py                 ← MstmlOrchestrator
│   ├── text_preprocessing.py
│   ├── author_disambiguation.py
│   ├── data_loaders.py
│   ├── model_evaluation.py
│   └── fast_encode_tree/       ← Cython HRG tree encoding
├── tests/                      # mstml/ library tests
├── notebooks.future/           # Example notebooks (in development)
├── papers/                     # Research papers
├── data/                       # Data directory (not committed)
│   └── arxiv/original/         ← place arXiv snapshot here
├── experiments/                # Pipeline outputs (not committed)
├── run_experiments.ps1         ← Windows: run both CAMSAP 2025 variants
├── run_experiments.sh          ← Linux/macOS: run both CAMSAP 2025 variants
├── pyproject.toml              ← dependencies and build config
└── uv.lock                     ← pinned dependency versions
```

---

## Key Features

### 1. Geometry-Driven Longitudinal Topic Model (GDLTM)
- Extracts topics from document collections across time slices
- Supports PHATE, UMAP, t-SNE, and PCA embeddings for topic space visualisation
- Uses Hellinger distances between topics for geometric analysis
- Hierarchical clustering with dendrogram cut heights for meta-topic creation
- Interdisciplinarity scores for documents and authors

### 2. Hierarchical Random Graph (HRG)
- Fits hierarchical models to network data
- Predicts missing links with probabilistic scores
- Extracts community structure from network hierarchy

### 3. Text Processing Pipeline
- Academic document preprocessing: tokenisation, cleaning, normalisation
- Stopword removal, lemmatisation, frequency-based filtering

### 4. Network Analysis Tools
- Collaboration network construction from author data
- Community detection, centrality metrics, visualisation

---

## Citation

```bibtex
@inproceedings{hougen2025mstml_camsap,
  author    = {Conrad D. Hougen and Karl T. Pazdernik and Alfred O. Hero},
  title     = {A Multiscale Geometric Method for Capturing Relational Topic Alignment},
  booktitle = {Proceedings of the 2025 IEEE International Workshop on Computational
               Advances in Multi-Sensor Adaptive Processing (CAMSAP)},
  year      = {2025},
  publisher = {IEEE}
}

@phdthesis{hougen2025dissertation,
  author = {Conrad D. Hougen},
  title  = {Network Models for Learning Uncertain and Multimodal Data},
  school = {},
  year   = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Planned Features

- **Abstract Manifold Learning Support**: Generalised framework for integrating any manifold learning or low-dimensional embedding technique with standardised interfaces for custom distance metrics.
- **Interactive Topic Visualisation Widgets**: Web-based widgets for labelling and exploring topics within embeddings.
- **Enhanced Link Prediction**: Improved algorithms for predicting missing or future collaboration links, incorporating both topological and semantic similarity features.
- **Hyperlink Prediction in Co-Author Hypergraphs**: Higher-order collaboration prediction beyond pairwise relationships.
- **Automated Topic Labelling with LLMs**: Integration of large language models for automated meta-topic naming from topic-word distributions.
