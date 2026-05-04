"""
_coherence.py — Topic coherence metrics via gensim CoherenceModel.

Computes C_v, C_uci, and C_npmi for any set of topic word lists.

Ported from:
  AToMS-LP/topic_coherence_comparison.ipynb
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def compute_coherence(
    topic_word_lists: "list[list[str]]",
    tokenized_docs: "list[list[str]]",
    gensim_dict=None,
    measures: "Optional[list[str]]" = None,
    top_n: int = 10,
) -> "dict[str, float]":
    """Compute multiple coherence scores for a set of topics.

    Args:
        topic_word_lists: List of per-topic word lists (one list per topic).
        tokenized_docs:   List of tokenised documents (list of word lists).
        gensim_dict:      Optional pre-built gensim Dictionary.  Built from
                          tokenized_docs if not provided.
        measures:         Coherence measures to compute.
                          Defaults to ['c_v', 'c_npmi', 'c_uci'].
        top_n:            Number of top words to use per topic.

    Returns:
        Dict metric_name → mean coherence score across topics.
        Returns NaN for any metric that fails.
    """
    try:
        from gensim.models import CoherenceModel
        import gensim.corpora as corpora
    except ImportError as exc:
        raise ImportError("gensim is required for coherence computation.") from exc

    if measures is None:
        measures = ["c_v", "c_npmi", "c_uci"]

    if gensim_dict is None:
        gensim_dict = corpora.Dictionary(tokenized_docs)

    # Trim topic word lists to top_n
    trimmed = [words[:top_n] for words in topic_word_lists]
    # Filter out empty topics
    trimmed = [words for words in trimmed if words]
    if not trimmed:
        return {m: float("nan") for m in measures}

    results: dict[str, float] = {}
    for measure in measures:
        try:
            cm = CoherenceModel(
                topics=trimmed,
                texts=tokenized_docs,
                dictionary=gensim_dict,
                coherence=measure,
            )
            results[measure] = float(cm.get_coherence())
        except Exception as exc:
            log.warning("Coherence metric %s failed: %s", measure, exc)
            results[measure] = float("nan")

    return results


def compute_ensemble_coherence(
    topic_word_lists_per_chunk: "list[list[list[str]]]",
    tokenized_docs_per_chunk: "list[list[list[str]]]",
    measures: "Optional[list[str]]" = None,
    top_n: int = 10,
) -> "dict[str, float]":
    """Average coherence scores across all chunks of an ensemble.

    Args:
        topic_word_lists_per_chunk: Outer = chunks, inner = topics, innermost = words.
        tokenized_docs_per_chunk:   Matching per-chunk tokenised documents.
        measures:                   Coherence measures (see ``compute_coherence``).
        top_n:                      Top words per topic.

    Returns:
        Dict metric_name → mean score across chunks (NaN chunks excluded).
    """
    if measures is None:
        measures = ["c_v", "c_npmi", "c_uci"]

    chunk_results: list[dict[str, float]] = []
    for chunk_words, chunk_docs in zip(topic_word_lists_per_chunk, tokenized_docs_per_chunk):
        r = compute_coherence(chunk_words, chunk_docs, measures=measures, top_n=top_n)
        chunk_results.append(r)

    aggregated: dict[str, float] = {}
    for m in measures:
        vals = [r[m] for r in chunk_results if not np.isnan(r.get(m, float("nan")))]
        aggregated[m] = float(np.mean(vals)) if vals else float("nan")
    return aggregated
