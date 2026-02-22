"""
legacy/tests/conftest.py — Shared fixtures for the legacy pipeline test suite.

Synthetic data is used throughout — no arXiv download or NLTK network access
is required.  The fixtures are sized to keep the full test run under ~60 s.

Dataset constants
-----------------
  N_VOCAB         = 60     distinct vocabulary tokens
  N_CHUNKS        = 4      temporal windows
  N_TOPICS_CHUNK  = 5      topics per chunk  → 20 total topics
  N_DOCS          = 40     documents (10 per chunk)
  N_AUTHORS       = 6      unique author IDs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable so `from legacy.xxx import ...` works.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Dataset constants ────────────────────────────────────────────────────────
N_VOCAB        = 60
N_CHUNKS       = 4
N_TOPICS_CHUNK = 5
N_TOPICS_TOTAL = N_CHUNKS * N_TOPICS_CHUNK   # 20
N_DOCS         = 40
N_DOCS_CHUNK   = N_DOCS // N_CHUNKS          # 10
N_AUTHORS      = 6
AUTHOR_IDS     = list(range(100, 100 + N_AUTHORS))
VOCAB          = [f"word{i}" for i in range(N_VOCAB)]

RNG = np.random.default_rng(42)


# ── Low-level building blocks ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def vocab() -> list[str]:
    return VOCAB


@pytest.fixture(scope="session")
def topic_vectors(rng) -> np.ndarray:
    """(N_TOPICS_TOTAL, N_VOCAB) Dirichlet samples — one row per chunk-topic."""
    return rng.dirichlet(np.ones(N_VOCAB) * 0.5, size=N_TOPICS_TOTAL).astype(np.float32)


@pytest.fixture(scope="session")
def topic_vecs_per_chunk(topic_vectors) -> list[np.ndarray]:
    """topic_vectors split into per-chunk lists."""
    return [
        topic_vectors[c * N_TOPICS_CHUNK:(c + 1) * N_TOPICS_CHUNK]
        for c in range(N_CHUNKS)
    ]


@pytest.fixture(scope="session")
def ntopics_by_chunk() -> dict[int, int]:
    return {c: N_TOPICS_CHUNK for c in range(N_CHUNKS)}


@pytest.fixture(scope="session")
def inds_by_chunk() -> dict[int, list[int]]:
    return {
        c: list(range(c * N_DOCS_CHUNK, (c + 1) * N_DOCS_CHUNK))
        for c in range(N_CHUNKS)
    }


@pytest.fixture(scope="session")
def synthetic_df(rng) -> pd.DataFrame:
    """Minimal DataFrame with text_processed, author_ids, date columns."""
    rows = []
    for doc_id in range(N_DOCS):
        chunk_idx = doc_id // N_DOCS_CHUNK
        # 2–4 random words from vocab per doc
        n_words = rng.integers(6, 15)
        text = rng.choice(VOCAB, size=n_words, replace=True).tolist()
        # 1–3 random authors
        n_auth = rng.integers(1, 4)
        auth_ids = rng.choice(AUTHOR_IDS, size=n_auth, replace=False).tolist()
        rows.append({
            "id":             str(doc_id),
            "abstract":       " ".join(text),
            "text_processed": text,
            "author_ids":     auth_ids,
            "categories":     "cs.LG",
            "date":           pd.Timestamp("2018-01-01") + pd.DateOffset(months=chunk_idx),
        })
    return pd.DataFrame(rows, index=list(range(N_DOCS)))


@pytest.fixture(scope="session")
def expanded_distns(rng, inds_by_chunk, ntopics_by_chunk) -> dict[int, np.ndarray]:
    """Synthetic expanded doc-topic distributions (global topic space)."""
    result: dict[int, np.ndarray] = {}
    total = N_TOPICS_TOTAL
    for cidx, doc_ids in inds_by_chunk.items():
        start = cidx * N_TOPICS_CHUNK
        for doc_id in doc_ids:
            vec = np.zeros(total, dtype=np.float32)
            theta = rng.dirichlet(np.ones(N_TOPICS_CHUNK)).astype(np.float32)
            vec[start: start + N_TOPICS_CHUNK] = theta
            result[doc_id] = vec
    return result


@pytest.fixture(scope="session")
def author_ct_distns(rng) -> dict[int, np.ndarray]:
    """Synthetic diffused author chunk-topic distributions."""
    return {
        aid: rng.dirichlet(np.ones(N_TOPICS_TOTAL)).astype(np.float32)
        for aid in AUTHOR_IDS
    }


@pytest.fixture(scope="session")
def coauthor_graph(synthetic_df):
    """NetworkX co-author graph built from synthetic_df."""
    from legacy._distributions import build_coauthor_graph
    return build_coauthor_graph(synthetic_df)


@pytest.fixture(scope="session")
def dendrogram_and_heights(topic_vectors):
    """Scipy linkage matrix Z plus (min_h, max_h) for the synthetic topic set."""
    from legacy._manifold import build_topic_dendrogram
    Z, min_h, max_h = build_topic_dendrogram(
        topic_vectors,
        knn=min(10, N_TOPICS_TOTAL - 1),
        linkage_method="ward",
        distance_metric="hellinger",
    )
    return Z, min_h, max_h


@pytest.fixture(scope="session")
def encoded_tree(dendrogram_and_heights, author_ct_distns, coauthor_graph):
    """Encoded HRG tree + author_index_map."""
    from legacy._scoring import encode_dendrogram_tree
    Z, _, _ = dendrogram_and_heights
    return encode_dendrogram_tree(Z, author_ct_distns, coauthor_graph)
