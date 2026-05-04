"""
_topic_models.py — Step 2: Temporal chunking, topic model ensemble, and distribution expansion.

Provides a registry-based topic model interface so that swapping LDA ↔ BERTopic
requires only changing the ``topic_model`` primary config parameter.

Protocol
--------
  TopicModel.fit(chunk_df, gensim_dict) → None
  TopicModel.get_topic_vectors()        → ndarray (n_topics, vocab_size)
  TopicModel.get_doc_topic_distributions(chunk_df) → dict[int, ndarray]

Pipeline functions
------------------
  create_temporal_chunks   → list of per-month DataFrames
  train_ensemble           → (models, topic_vectors, ntopics_by_chunk, inds_by_chunk)
  compute_doc_topic_distributions → doc_id → theta
  expand_distributions     → doc_id → zero-padded global theta

Ported from:
  AToMS-LP/AToMS_HRG_Ensemble_Interdisciplinarity.ipynb
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaMulticore

from ._config import PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — structural interface for topic models
# ---------------------------------------------------------------------------

@runtime_checkable
class TopicModel(Protocol):
    """Minimal interface that every registered topic model must satisfy."""

    def fit(self, chunk_df: pd.DataFrame, gensim_dict: corpora.Dictionary) -> None:
        """Train the model on ``chunk_df``."""
        ...

    def get_topic_vectors(self) -> np.ndarray:
        """Return the phi matrix (n_topics, vocab_size), i.e. p(word | topic)."""
        ...

    def get_doc_topic_distributions(
        self,
        chunk_df: pd.DataFrame,
    ) -> "dict[int, np.ndarray]":
        """Return theta for each document in ``chunk_df``, keyed by doc index."""
        ...


# ---------------------------------------------------------------------------
# LDA implementation
# ---------------------------------------------------------------------------

class LDATopicModel:
    """Wraps gensim LdaMulticore as a TopicModel.

    Args:
        docs_per_topic:   Target documents per topic for automatic num_topics calculation.
        ngibbs:           Gibbs sampling iterations per pass.
        npasses:          Full corpus passes.
        n_workers:        LDA worker processes (0 = cpu_count − 1).
        random_state:     Seed for reproducibility.
        **kwargs:         Forwarded to LdaMulticore (ignored if unrecognised).
    """

    def __init__(
        self,
        docs_per_topic: int = 100,
        ngibbs: int = 50,
        npasses: int = 5,
        smoothing_gamma: float = 0.75,  # consumed by train_ensemble, not LDA itself
        n_workers: int = 0,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.docs_per_topic  = docs_per_topic
        self.ngibbs          = ngibbs
        self.npasses         = npasses
        self.smoothing_gamma = smoothing_gamma
        self.n_workers       = n_workers if n_workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        self.random_state    = random_state
        self._lda: LdaMulticore | None = None
        self._id2word: corpora.Dictionary | None = None

    # ------------------------------------------------------------------

    def fit(self, chunk_df: pd.DataFrame, gensim_dict: corpora.Dictionary) -> None:
        docs = chunk_df["text_processed"].tolist()
        corpus = [gensim_dict.doc2bow(doc) for doc in docs]
        n = len(chunk_df)
        num_topics = max(4, min(n, n // max(1, self.docs_per_topic)))

        self._id2word = gensim_dict
        self._lda = LdaMulticore(
            corpus=corpus,
            num_topics=num_topics,
            id2word=gensim_dict,
            workers=max(1, self.n_workers - 1),
            iterations=self.ngibbs,
            passes=self.npasses,
            random_state=self.random_state,
        )

    def get_topic_vectors(self) -> np.ndarray:
        assert self._lda is not None, "Call fit() first."
        vocab_size = len(self._id2word)
        n_topics   = self._lda.num_topics
        phi = np.zeros((n_topics, vocab_size), dtype=np.float32)
        for t in range(n_topics):
            for wid, prob in self._lda.get_topic_terms(t, topn=vocab_size):
                phi[t, wid] = prob
        return phi

    def get_doc_topic_distributions(
        self, chunk_df: pd.DataFrame
    ) -> "dict[int, np.ndarray]":
        assert self._lda is not None, "Call fit() first."
        n_topics = self._lda.num_topics
        result: dict[int, np.ndarray] = {}
        for doc_id, row in chunk_df.iterrows():
            bow = self._id2word.doc2bow(row["text_processed"])
            topic_dist = dict(self._lda.get_document_topics(bow, minimum_probability=0.0))
            theta = np.array(
                [topic_dist.get(t, 0.0) for t in range(n_topics)],
                dtype=np.float32,
            )
            result[doc_id] = theta
        return result


# ---------------------------------------------------------------------------
# BERTopic implementation (optional — requires bertopic + sentence-transformers)
# ---------------------------------------------------------------------------

class BERTopicTopicModel:
    """Wraps BERTopic as a TopicModel.

    Args:
        embedding_model: sentence-transformers model name.
        **kwargs:        Forwarded to BERTopic (e.g. nr_topics, min_topic_size).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> None:
        self.embedding_model = embedding_model
        self._kwargs = kwargs
        self._bertopic = None
        self._id2word: corpora.Dictionary | None = None

    def fit(self, chunk_df: pd.DataFrame, gensim_dict: corpora.Dictionary) -> None:
        try:
            from bertopic import BERTopic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "BERTopic is not installed. Run: pip install bertopic sentence-transformers"
            ) from exc
        docs = chunk_df["abstract"].fillna("").tolist()
        self._bertopic = BERTopic(
            embedding_model=self.embedding_model,
            **self._kwargs,
        )
        self._topics, _ = self._bertopic.fit_transform(docs)
        self._id2word = gensim_dict
        self._chunk_df = chunk_df

    def get_topic_vectors(self) -> np.ndarray:
        assert self._bertopic is not None, "Call fit() first."
        vocab = list(self._id2word.values())
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        vocab_size = len(vocab)
        topic_ids = [t for t in self._bertopic.get_topic_info()["Topic"] if t >= 0]
        n_topics = len(topic_ids)
        phi = np.zeros((n_topics, vocab_size), dtype=np.float32)
        for row_idx, tid in enumerate(sorted(topic_ids)):
            for word, score in self._bertopic.get_topic(tid):
                if word in word_to_idx:
                    phi[row_idx, word_to_idx[word]] = max(score, 0.0)
        # Row-normalise to probability distributions
        row_sums = phi.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return phi / row_sums

    def get_doc_topic_distributions(
        self, chunk_df: pd.DataFrame
    ) -> "dict[int, np.ndarray]":
        assert self._bertopic is not None, "Call fit() first."
        topic_ids = [t for t in self._bertopic.get_topic_info()["Topic"] if t >= 0]
        topic_id_to_row = {tid: i for i, tid in enumerate(sorted(topic_ids))}
        n_topics = len(topic_ids)
        result: dict[int, np.ndarray] = {}
        for i, (doc_id, _) in enumerate(chunk_df.iterrows()):
            theta = np.zeros(n_topics, dtype=np.float32)
            if i < len(self._topics):
                tid = self._topics[i]
                if tid in topic_id_to_row:
                    theta[topic_id_to_row[tid]] = 1.0  # hard assignment
            result[doc_id] = theta
        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOPIC_MODEL_REGISTRY: dict[str, type] = {
    "lda":      LDATopicModel,
    "bertopic": BERTopicTopicModel,
}


def make_topic_model(model_name: str, **kwargs) -> TopicModel:
    """Instantiate a topic model by name from the registry.

    Args:
        model_name: Key in TOPIC_MODEL_REGISTRY (case-insensitive).
        **kwargs:   Forwarded to the model constructor.

    Returns:
        Uninitialised TopicModel instance.

    Raises:
        ValueError: If model_name is not registered.
    """
    key = model_name.lower()
    if key not in TOPIC_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown topic model '{model_name}'. "
            f"Available: {sorted(TOPIC_MODEL_REGISTRY)}"
        )
    return TOPIC_MODEL_REGISTRY[key](**kwargs)


# ---------------------------------------------------------------------------
# Temporal chunking
# ---------------------------------------------------------------------------

def create_temporal_chunks(
    df: pd.DataFrame,
    months_per_chunk: int = 1,
) -> list[pd.DataFrame]:
    """Split df into non-overlapping temporal windows.

    Args:
        df:                DataFrame with a ``date`` column (datetime).
        months_per_chunk:  Window size in calendar months.

    Returns:
        List of DataFrames ordered by time, empty chunks omitted.
    """
    chunks = []
    freq = f"{months_per_chunk}MS"  # month-start frequency
    grouped = df.groupby(pd.Grouper(key="date", freq=freq))
    for _, chunk_df in grouped:
        if len(chunk_df) > 0:
            chunks.append(chunk_df)
    log.info("Created %d temporal chunks (months_per_chunk=%d).", len(chunks), months_per_chunk)
    return chunks


# ---------------------------------------------------------------------------
# Ensemble training with exponential smoothing
# ---------------------------------------------------------------------------

def _smooth_chunk(
    chunks: list[pd.DataFrame],
    cidx: int,
    gamma: float,
    random_state: int,
) -> pd.DataFrame:
    """Augment chunk ``cidx`` with subsampled neighbours (exponential decay)."""
    base = chunks[cidx]
    parts = [base]
    for other_cidx, other_df in enumerate(chunks):
        if other_cidx == cidx:
            continue
        offset = abs(other_cidx - cidx)
        frac = gamma ** offset
        if frac < 0.01:
            continue
        n_sample = max(1, int(len(other_df) * frac))
        sampled = other_df.sample(
            n=min(n_sample, len(other_df)),
            random_state=random_state + cidx,
        )
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def train_ensemble(
    chunks: list[pd.DataFrame],
    gensim_dict: corpora.Dictionary,
    topic_model: str,
    cfg: PipelineConfig,
    output_dir: str | Path | None = None,
    n_workers: int = 0,
) -> tuple[list[TopicModel], np.ndarray, dict[int, int], dict[int, list[int]]]:
    """Train one topic model per temporal chunk and collect results.

    Each chunk is first augmented with exponentially-decayed samples from
    neighbouring chunks (``smoothing_gamma``) before training.

    Args:
        chunks:       List of per-chunk DataFrames from ``create_temporal_chunks``.
        gensim_dict:  Reduced gensim Dictionary from ``preprocess_text``.
        topic_model:  Model name from TOPIC_MODEL_REGISTRY.
        cfg:          PipelineConfig carrying topic_model kwargs and gamma.
        output_dir:   Optional directory to save per-chunk model pickles.
        n_workers:    Worker count forwarded to the model (0 = auto).

    Returns:
        (models, topic_vectors, ntopics_by_chunk, inds_by_chunk)
        - models:           Fitted TopicModel per chunk.
        - topic_vectors:    (total_topics, vocab_size) stacked phi matrix.
        - ntopics_by_chunk: dict chunk_idx → int number of topics.
        - inds_by_chunk:    dict chunk_idx → list of doc-index ints in that chunk.
    """
    model_kwargs = cfg.topic_models.get_kwargs(topic_model)
    if topic_model.lower() == "lda":
        gamma = cfg.topic_models.lda.smoothing_gamma
    else:
        gamma = 0.0  # no smoothing for non-LDA models

    if n_workers > 0:
        model_kwargs.setdefault("n_workers", n_workers)

    output_dir_path = Path(output_dir) if output_dir else None
    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    models: list[TopicModel] = []
    phi_list: list[np.ndarray] = []
    ntopics_by_chunk: dict[int, int] = {}
    inds_by_chunk: dict[int, list[int]] = {}

    for cidx, chunk_df in enumerate(chunks):
        log.info(
            "Training chunk %d/%d (%d docs) …", cidx + 1, len(chunks), len(chunk_df)
        )
        augmented = (
            _smooth_chunk(chunks, cidx, gamma, random_state=42)
            if gamma > 0.0
            else chunk_df
        )
        model = make_topic_model(topic_model, **model_kwargs)
        model.fit(augmented, gensim_dict)

        phi = model.get_topic_vectors()   # (n_topics, vocab_size)
        phi_list.append(phi)
        ntopics_by_chunk[cidx] = phi.shape[0]
        inds_by_chunk[cidx]    = list(chunk_df.index)
        models.append(model)

        if output_dir_path:
            model_path = output_dir_path / f"model_chunk_{cidx:04d}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

    topic_vectors = np.vstack(phi_list)
    log.info(
        "Ensemble: %d chunks, %d total topics, vocab_size=%d.",
        len(chunks), topic_vectors.shape[0], topic_vectors.shape[1],
    )
    return models, topic_vectors, ntopics_by_chunk, inds_by_chunk


# ---------------------------------------------------------------------------
# Document-topic distributions
# ---------------------------------------------------------------------------

def compute_doc_topic_distributions(
    chunks: list[pd.DataFrame],
    models: list[TopicModel],
    inds_by_chunk: "dict[int, list[int]]",
) -> "dict[int, np.ndarray]":
    """Collect per-document chunk-local theta from each model.

    Args:
        chunks:        Original (un-augmented) chunk DataFrames.
        models:        Fitted TopicModel per chunk, same order as ``chunks``.
        inds_by_chunk: Maps chunk_idx → list of doc indices in that chunk.

    Returns:
        Mapping doc_index → theta (chunk-local, sums to 1).
    """
    doc_topic_distns: dict[int, np.ndarray] = {}
    for cidx, (chunk_df, model) in enumerate(zip(chunks, models)):
        distns = model.get_doc_topic_distributions(chunk_df)
        doc_topic_distns.update(distns)
    return doc_topic_distns


def expand_distributions(
    doc_topic_distns: "dict[int, np.ndarray]",
    inds_by_chunk: "dict[int, list[int]]",
    ntopics_by_chunk: "dict[int, int]",
) -> "dict[int, np.ndarray]":
    """Expand chunk-local theta vectors to the global joint topic space.

    Each document's theta is zero-padded so that it spans all topics across
    all chunks: position = chunk_start_index + local_topic_index.

    Args:
        doc_topic_distns:  doc_id → chunk-local theta.
        inds_by_chunk:     chunk_idx → list[doc_id] in that chunk.
        ntopics_by_chunk:  chunk_idx → number of topics.

    Returns:
        Expanded doc_id → global theta (length = total_topics).
    """
    num_chunks   = len(ntopics_by_chunk)
    total_topics = sum(ntopics_by_chunk[c] for c in range(num_chunks))

    chunk_start: dict[int, int] = {}
    offset = 0
    for c in range(num_chunks):
        chunk_start[c] = offset
        offset += ntopics_by_chunk[c]

    doc_to_chunk: dict[int, int] = {
        doc_id: cidx
        for cidx, doc_ids in inds_by_chunk.items()
        for doc_id in doc_ids
    }

    expanded: dict[int, np.ndarray] = {}
    for doc_id, dist in doc_topic_distns.items():
        cidx = doc_to_chunk.get(doc_id)
        if cidx is None:
            continue
        start    = chunk_start[cidx]
        n_topics = ntopics_by_chunk[cidx]
        full     = np.zeros(total_topics, dtype=np.float32)
        full[start: start + n_topics] = dist
        expanded[doc_id] = full

    return expanded
