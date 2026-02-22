"""
_config.py — Configuration dataclasses and YAML loader for the legacy pipeline.

Two-tier configuration:
  PrimaryConfig  — scientifically meaningful, user-facing (CLI / API)
  PipelineConfig — stable secondary defaults loaded from config.yaml

Usage
-----
    from legacy._config import PrimaryConfig, load_pipeline_config

    primary = PrimaryConfig(
        input_file="arxiv-metadata-oai-snapshot.json",
        output_dir="./out",
        categories=["cs.LG", "stat.ML"],
        year_start=2012,
        year_end=2023,
    )
    cfg = load_pipeline_config()          # reads legacy/config.yaml
    cfg = load_pipeline_config("my.yaml") # or any override path
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(d: dict, *keys, default=None):
    """Safely navigate nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ---------------------------------------------------------------------------
# Sub-dataclasses (one per config.yaml section)
# ---------------------------------------------------------------------------

@dataclass
class PipelineSection:
    months_per_chunk: int = 1
    max_authors_per_doc: int = 20
    author_disambig_threshold: float = 0.90


@dataclass
class VocabLDASection:
    lam: float = 0.6          # term relevance lambda (stored as 'lambda' in YAML)
    top_n: int = 2000
    num_topics: int = 50


@dataclass
class TextProcessingSection:
    vocab_low_freq_thresh: int = 1
    vocab_high_freq_frac: float = 0.995
    vocab_lda: VocabLDASection = field(default_factory=VocabLDASection)


@dataclass
class LDASection:
    docs_per_topic: int = 100
    ngibbs: int = 50
    npasses: int = 5
    smoothing_gamma: float = 0.75   # Exponential decay weight for neighbouring chunks


@dataclass
class BERTopicSection:
    embedding_model: str = "all-MiniLM-L6-v2"
    extra: dict = field(default_factory=dict)  # forward-compat kwargs


@dataclass
class TopicModelsSection:
    lda: LDASection = field(default_factory=LDASection)
    bertopic: BERTopicSection = field(default_factory=BERTopicSection)
    extra: dict = field(default_factory=dict)  # unknown model names → **kwargs

    def get_kwargs(self, model_name: str) -> dict:
        """Return **kwargs for a given model name (falls back to extra)."""
        name = model_name.lower()
        if name == "lda":
            d = vars(self.lda)
        elif name == "bertopic":
            d = {k: v for k, v in vars(self.bertopic).items() if k != "extra"}
            d.update(self.bertopic.extra)
        else:
            d = self.extra.get(name, {})
        return d


@dataclass
class DiffusionSection:
    knn: int = 5
    rate: float = 0.7
    num_iterations: int = 1


@dataclass
class DendrogramSection:
    knn: int = 100


@dataclass
class PHATESection:
    n_components: int = 3
    gamma: float = 0.0
    knn: int = 5
    t: Any = "auto"   # int or the string "auto"


@dataclass
class UMAPSection:
    n_components: int = 3
    n_neighbors: int = 15
    extra: dict = field(default_factory=dict)


@dataclass
class TSNESection:
    n_components: int = 3
    perplexity: float = 30.0
    extra: dict = field(default_factory=dict)


@dataclass
class PCASection:
    n_components: int = 3
    extra: dict = field(default_factory=dict)


@dataclass
class EmbeddingsSection:
    phate: PHATESection = field(default_factory=PHATESection)
    umap: UMAPSection = field(default_factory=UMAPSection)
    tsne: TSNESection = field(default_factory=TSNESection)
    pca: PCASection = field(default_factory=PCASection)
    extra: dict = field(default_factory=dict)  # unknown method names → **kwargs

    def get_kwargs(self, method_name: str) -> dict:
        """Return **kwargs for a given embedding method (excludes 'extra' sub-key)."""
        name = method_name.lower().replace("-", "").replace("_", "")
        if name == "phate":
            d = vars(self.phate)
        elif name == "umap":
            d = {k: v for k, v in vars(self.umap).items() if k != "extra"}
            d.update(self.umap.extra)
        elif name in ("tsne", "tsne"):
            d = {k: v for k, v in vars(self.tsne).items() if k != "extra"}
            d.update(self.tsne.extra)
        elif name == "pca":
            d = {k: v for k, v in vars(self.pca).items() if k != "extra"}
            d.update(self.pca.extra)
        else:
            d = self.extra.get(name, {})
        return d


@dataclass
class VisualizationSection:
    figure_dpi: int = 300


# ---------------------------------------------------------------------------
# PipelineConfig — root object
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    pipeline: PipelineSection = field(default_factory=PipelineSection)
    text_processing: TextProcessingSection = field(default_factory=TextProcessingSection)
    topic_models: TopicModelsSection = field(default_factory=TopicModelsSection)
    diffusion: DiffusionSection = field(default_factory=DiffusionSection)
    dendrogram: DendrogramSection = field(default_factory=DendrogramSection)
    embeddings: EmbeddingsSection = field(default_factory=EmbeddingsSection)
    visualization: VisualizationSection = field(default_factory=VisualizationSection)


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    """
    Load PipelineConfig from a YAML file.

    If *path* is None, reads ``<legacy_dir>/config.yaml`` next to this file.
    Unknown keys under ``topic_models`` or ``embeddings`` are stored in the
    ``extra`` dict of their parent section and forwarded as **kwargs when the
    corresponding model/method is instantiated.
    """
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        return PipelineConfig()

    raw = _load_yaml(path)

    # ── pipeline ──────────────────────────────────────────────────────────
    p = raw.get("pipeline", {})
    pipeline = PipelineSection(
        months_per_chunk=p.get("months_per_chunk", 1),
        max_authors_per_doc=p.get("max_authors_per_doc", 20),
        author_disambig_threshold=p.get("author_disambig_threshold", 0.90),
    )

    # ── text_processing ───────────────────────────────────────────────────
    tp = raw.get("text_processing", {})
    vlda = tp.get("vocab_lda", {})
    text_processing = TextProcessingSection(
        vocab_low_freq_thresh=tp.get("vocab_low_freq_thresh", 1),
        vocab_high_freq_frac=tp.get("vocab_high_freq_frac", 0.995),
        vocab_lda=VocabLDASection(
            lam=vlda.get("lambda", 0.6),
            top_n=vlda.get("top_n", 2000),
            num_topics=vlda.get("num_topics", 50),
        ),
    )

    # ── topic_models ──────────────────────────────────────────────────────
    tm_raw = raw.get("topic_models", {})
    lda_raw = tm_raw.get("lda", {})
    bert_raw = tm_raw.get("bertopic", {})
    known_models = {"lda", "bertopic"}
    topic_models = TopicModelsSection(
        lda=LDASection(
            docs_per_topic=lda_raw.get("docs_per_topic", 100),
            ngibbs=lda_raw.get("ngibbs", 50),
            npasses=lda_raw.get("npasses", 5),
            smoothing_gamma=lda_raw.get("smoothing_gamma", 0.75),
        ),
        bertopic=BERTopicSection(
            embedding_model=bert_raw.get("embedding_model", "all-MiniLM-L6-v2"),
            extra={k: v for k, v in bert_raw.items() if k != "embedding_model"},
        ),
        extra={k: v for k, v in tm_raw.items() if k not in known_models},
    )

    # ── diffusion ─────────────────────────────────────────────────────────
    diff = raw.get("diffusion", {})
    diffusion = DiffusionSection(
        knn=diff.get("knn", 5),
        rate=diff.get("rate", 0.7),
        num_iterations=diff.get("num_iterations", 1),
    )

    # ── dendrogram ────────────────────────────────────────────────────────
    dend = raw.get("dendrogram", {})
    dendrogram = DendrogramSection(knn=dend.get("knn", 100))

    # ── embeddings ────────────────────────────────────────────────────────
    emb_raw = raw.get("embeddings", {})
    ph = emb_raw.get("phate", {})
    um = emb_raw.get("umap", {})
    ts = emb_raw.get("tsne", {})
    pc = emb_raw.get("pca", {})
    known_methods = {"phate", "umap", "tsne", "pca"}

    t_val = ph.get("t", "auto")
    if isinstance(t_val, str) and t_val != "auto":
        t_val = int(t_val)

    embeddings = EmbeddingsSection(
        phate=PHATESection(
            n_components=ph.get("n_components", 3),
            gamma=ph.get("gamma", 0.0),
            knn=ph.get("knn", 5),
            t=t_val,
        ),
        umap=UMAPSection(
            n_components=um.get("n_components", 3),
            n_neighbors=um.get("n_neighbors", 15),
            extra={k: v for k, v in um.items() if k not in ("n_components", "n_neighbors")},
        ),
        tsne=TSNESection(
            n_components=ts.get("n_components", 3),
            perplexity=ts.get("perplexity", 30.0),
            extra={k: v for k, v in ts.items() if k not in ("n_components", "perplexity")},
        ),
        pca=PCASection(
            n_components=pc.get("n_components", 3),
            extra={k: v for k, v in pc.items() if k != "n_components"},
        ),
        extra={k: v for k, v in emb_raw.items() if k not in known_methods},
    )

    # ── visualization ─────────────────────────────────────────────────────
    vis = raw.get("visualization", {})
    visualization = VisualizationSection(figure_dpi=vis.get("figure_dpi", 300))

    return PipelineConfig(
        pipeline=pipeline,
        text_processing=text_processing,
        topic_models=topic_models,
        diffusion=diffusion,
        dendrogram=dendrogram,
        embeddings=embeddings,
        visualization=visualization,
    )


# ---------------------------------------------------------------------------
# PrimaryConfig — user-facing, CLI-level settings
# ---------------------------------------------------------------------------

@dataclass
class PrimaryConfig:
    """
    Scientifically meaningful hyperparameters exposed to the user.

    These are the parameters that change results and should be chosen by the
    researcher.  Secondary/stable defaults live in PipelineConfig / config.yaml.
    """
    input_file: str = ""
    output_dir: str = ""
    categories: list[str] = field(default_factory=list)
    year_start: int = 2012
    year_end: int = 2023
    topic_model: str = "lda"         # 'lda' | 'bertopic' | any registered name
    distance_metric: str = "hellinger"  # 'hellinger' | 'cosine' | 'euclidean'
    embedding_method: str = "phate"     # 'phate' | 'umap' | 'tsne' | 'pca'
    linkage_method: str = "ward"        # 'ward' | 'complete' | 'average' | 'single'
    cut_height: float = 0.68            # normalized [0, 1] dendrogram cut

    def validate(self) -> None:
        """Raise ValueError for obviously wrong settings."""
        if not self.input_file:
            raise ValueError("input_file is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if not self.categories:
            raise ValueError("categories must be a non-empty list")
        if self.year_start > self.year_end:
            raise ValueError(f"year_start ({self.year_start}) > year_end ({self.year_end})")
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"input_file not found: {self.input_file}")
        if not 0.0 <= self.cut_height <= 1.0:
            raise ValueError(f"cut_height must be in [0, 1], got {self.cut_height}")
        valid_metrics = {"hellinger", "cosine", "euclidean"}
        if self.distance_metric.lower() not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")
        valid_embeddings = {"phate", "umap", "tsne", "pca"}
        if self.embedding_method.lower() not in valid_embeddings:
            raise ValueError(f"embedding_method must be one of {valid_embeddings}")
        valid_linkage = {"ward", "complete", "average", "single"}
        if self.linkage_method.lower() not in valid_linkage:
            raise ValueError(f"linkage_method must be one of {valid_linkage}")
