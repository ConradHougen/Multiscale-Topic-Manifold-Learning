"""
_bertopic_embed.py — BERTopic topic extraction for comparison experiments.

Trains one BERTopic model per temporal chunk (matching the MSTML ensemble
structure) and returns per-chunk topic word-frequency vectors aligned to the
shared gensim vocabulary.

Ported from:
  AToMS-LP/topic_alignment_comparison.ipynb
  AToMS-LP/topic_coherence_comparison.ipynb
  AToMS-LP/topic_diversity_comparison.ipynb
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import gensim.corpora as corpora

log = logging.getLogger(__name__)


def train_bertopic_ensemble(
    chunks: "list[pd.DataFrame]",
    gensim_dict: corpora.Dictionary,
    embedding_model: str = "all-MiniLM-L6-v2",
    **bertopic_kwargs,
) -> "tuple[list, list[np.ndarray]]":
    """Train one BERTopic model per temporal chunk.

    Args:
        chunks:           Per-chunk DataFrames with an ``abstract`` column.
        gensim_dict:      Shared gensim Dictionary (defines the vocabulary).
        embedding_model:  sentence-transformers model name.
        **bertopic_kwargs: Forwarded to BERTopic (e.g. min_topic_size).

    Returns:
        (models, topic_vectors_per_chunk)
        - models:                  List of fitted BERTopic instances.
        - topic_vectors_per_chunk: List of (n_topics, vocab_size) float arrays.
    """
    try:
        from bertopic import BERTopic  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "BERTopic is not installed. Run: pip install bertopic sentence-transformers"
        ) from exc

    vocab      = list(gensim_dict.values())
    word2idx   = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    models: list = []
    phi_list: list[np.ndarray] = []

    for cidx, chunk_df in enumerate(chunks):
        docs = chunk_df["abstract"].fillna("").tolist()
        log.info("BERTopic chunk %d/%d (%d docs) …", cidx + 1, len(chunks), len(docs))

        model = BERTopic(embedding_model=embedding_model, **bertopic_kwargs)
        topics, _ = model.fit_transform(docs)

        # Build phi matrix aligned to gensim vocab
        topic_ids = sorted(t for t in model.get_topic_info()["Topic"] if t >= 0)
        n_topics  = len(topic_ids) if topic_ids else 1
        phi = np.zeros((n_topics, vocab_size), dtype=np.float32)
        for row_idx, tid in enumerate(topic_ids):
            for word, score in model.get_topic(tid):
                if word in word2idx:
                    phi[row_idx, word2idx[word]] = max(float(score), 0.0)
        # Row-normalise
        row_sums = phi.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        phi /= row_sums

        phi_list.append(phi)
        models.append(model)

    log.info("BERTopic ensemble: %d chunks trained.", len(chunks))
    return models, phi_list


def get_bertopic_top_words(
    models: list,
    top_n: int = 10,
) -> "list[list[list[str]]]":
    """Extract top-N words per topic for each BERTopic model.

    Returns:
        Outer list = chunks; inner list = topics; innermost = word list.
    """
    result: list[list[list[str]]] = []
    for model in models:
        chunk_topics: list[list[str]] = []
        for tid in sorted(t for t in model.get_topic_info()["Topic"] if t >= 0):
            words = [w for w, _ in model.get_topic(tid)][:top_n]
            chunk_topics.append(words)
        result.append(chunk_topics)
    return result
