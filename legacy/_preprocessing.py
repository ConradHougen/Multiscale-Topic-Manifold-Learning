"""
_preprocessing.py — Step 1: arXiv JSONL loading, text preprocessing, and author disambiguation.

Pipeline
--------
load_arxiv_jsonl()      →  raw DataFrame (filtered by category, year, author count)
preprocess_text()       →  processed DataFrame + gensim Dictionary (reduced vocabulary)
disambiguate_authors()  →  DataFrame with 'author_ids' column, id↔name dicts

Ported from:
  AToMS-LP/arxiv_oai_collate_disambiguate.ipynb
  AToMS-LP/AToMS/author_disambiguation.py
  AToMS-LP/AToMS/atoms_gdltm_utils.py
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from gensim.utils import simple_preprocess
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ._config import PipelineConfig
from ._math import term_relevance

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal NLTK helpers (lazy-loaded to avoid mandatory download at import)
# ---------------------------------------------------------------------------

_lemmatizer = None
_stop_words: Optional[set] = None


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stop_words() -> set:
    global _stop_words
    if _stop_words is None:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def _lemmatize_doc(tokens: list[str]) -> list[str]:
    lem = _get_lemmatizer()
    return [lem.lemmatize(t) for t in tokens]


def _worker_lemmatize(abstract: str) -> list[str]:
    """Top-level function so it can be pickled by multiprocessing."""
    _get_lemmatizer()  # initialise inside worker
    tokens = simple_preprocess(abstract, deacc=True)
    lem = _get_lemmatizer()
    return [lem.lemmatize(t) for t in tokens]


# ---------------------------------------------------------------------------
# Step 1a: Load arXiv JSONL
# ---------------------------------------------------------------------------

def load_arxiv_jsonl(
    path: str | Path,
    categories: list[str],
    year_start: int,
    year_end: int,
    max_authors: int = 20,
) -> pd.DataFrame:
    """Load and filter the arXiv metadata JSONL snapshot.

    Args:
        path:         Path to ``arxiv-metadata-oai-snapshot.json`` (one JSON object per line).
        categories:   arXiv category codes to keep (e.g. ``['cs.LG', 'stat.ML']``).
                      A paper is kept if **any** of its categories match.
        year_start:   First year to include (inclusive), based on ``update_date``.
        year_end:     Last year to include (inclusive).
        max_authors:  Drop papers with strictly more than this many authors.

    Returns:
        DataFrame with at least the columns:
        ``['id', 'title', 'abstract', 'categories', 'authors_parsed', 'date']``.
        Index is reset to a contiguous integer range.
    """
    log.info("Loading arXiv JSONL from %s …", path)
    df = pd.read_json(path, lines=True)

    # ── Category filter ──────────────────────────────────────────────────
    cat_set = set(categories)
    df = df[
        df["categories"].apply(
            lambda x: bool(cat_set.intersection(str(x).split()))
        )
    ]

    # ── Date filter ──────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["update_date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df[
        (df["date"].dt.year >= year_start) &
        (df["date"].dt.year <= year_end)
    ]

    # ── Author-count filter ──────────────────────────────────────────────
    df = df[df["authors_parsed"].apply(len) <= max_authors]

    df = df.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d papers after filtering.", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 1b: Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    n_workers: int = 0,
) -> tuple[pd.DataFrame, corpora.Dictionary]:
    """Tokenise, lemmatise, filter stopwords, build vocabulary, and reduce it
    via LDA-based term relevance.

    Steps
    -----
    1. ``gensim.simple_preprocess`` on the ``abstract`` column.
    2. Multiprocess lemmatisation (WordNetLemmatizer).
    3. Stopword removal.
    4. Build a gensim Dictionary and apply frequency thresholds.
    5. Train a 50-topic LDA on the full corpus for vocabulary reduction.
    6. Keep only the top-N terms per topic by LDAvis relevance score.
    7. Filter the processed documents to the reduced vocabulary.
    8. Store the result in ``df['text_processed']`` (list of tokens per row).

    Args:
        df:         DataFrame from ``load_arxiv_jsonl``.
        cfg:        PipelineConfig with ``text_processing`` sub-config.
        n_workers:  Number of parallel worker processes for lemmatisation.
                    0 = use ``os.cpu_count() − 1``.

    Returns:
        (updated_df, id2word) — df has a new ``text_processed`` column;
        id2word is the reduced gensim Dictionary.
    """
    import os

    tp = cfg.text_processing
    vl = tp.vocab_lda

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    abstracts = df["abstract"].fillna("").tolist()
    log.info("Lemmatising %d documents with %d workers …", len(abstracts), n_workers)

    if n_workers == 1:
        tokenised = [_worker_lemmatize(a) for a in abstracts]
    else:
        with mp.Pool(processes=n_workers) as pool:
            tokenised = pool.map(_worker_lemmatize, abstracts)

    # ── Stopword removal ─────────────────────────────────────────────────
    stop = _get_stop_words()
    tokenised = [[w for w in doc if w not in stop] for doc in tokenised]

    # ── Build initial dictionary ─────────────────────────────────────────
    id2word = corpora.Dictionary(tokenised)
    id2word.filter_extremes(
        no_below=tp.vocab_low_freq_thresh,
        no_above=tp.vocab_high_freq_frac,
    )
    id2word.compactify()
    log.info("Initial vocabulary size: %d tokens.", len(id2word))

    # ── LDA-based vocabulary reduction ───────────────────────────────────
    corpus = [id2word.doc2bow(doc) for doc in tokenised]
    log.info(
        "Training %d-topic LDA for vocabulary reduction (lambda=%.2f, top_n=%d) …",
        vl.num_topics, vl.lam, vl.top_n,
    )
    lda_vocab = LdaMulticore(
        corpus=corpus,
        num_topics=vl.num_topics,
        id2word=id2word,
        workers=max(1, n_workers - 1),
        passes=3,
        random_state=42,
    )

    # Collect top-n terms per topic by relevance
    vocab_size = len(id2word)
    overall_counts = np.zeros(vocab_size, dtype=np.float64)
    for bow in corpus:
        for wid, cnt in bow:
            overall_counts[wid] += cnt
    p_w = overall_counts / max(overall_counts.sum(), 1e-12)

    keep_ids: set[int] = set()
    for t in range(vl.num_topics):
        phi = np.zeros(vocab_size, dtype=np.float64)
        for wid, prob in lda_vocab.get_topic_terms(t, topn=vocab_size):
            phi[wid] = prob
        scores = term_relevance(phi, p_w, lam=vl.lam)
        top_ids = np.argsort(scores)[-vl.top_n:]
        keep_ids.update(top_ids.tolist())

    bad_ids = [wid for wid in id2word.keys() if wid not in keep_ids]
    id2word.filter_tokens(bad_ids=bad_ids)
    id2word.compactify()
    log.info("Reduced vocabulary size: %d tokens.", len(id2word))

    # ── Filter documents to reduced vocab ────────────────────────────────
    keep_tokens = set(id2word.values())
    processed = [[w for w in doc if w in keep_tokens] for doc in tokenised]

    df = df.copy()
    df["text_processed"] = processed
    return df, id2word


# ---------------------------------------------------------------------------
# Step 1c: Author disambiguation
# ---------------------------------------------------------------------------

def _normalise_name(parsed_author: list) -> str:
    """Convert [last, first, suffix] → 'First Last' (ASCII, lowercase-cleaned)."""
    last  = (parsed_author[0] if len(parsed_author) > 0 else "").strip()
    first = (parsed_author[1] if len(parsed_author) > 1 else "").strip()
    full  = f"{first} {last}".strip()
    # Normalise Unicode to ASCII for consistent comparison
    full = unicodedata.normalize("NFKD", full).encode("ascii", errors="ignore").decode()
    full = re.sub(r"\s+", " ", full).strip()
    return full


def _ngrams(string: str, n: int = 3) -> list[str]:
    """Character n-grams after ASCII conversion and lowercasing."""
    s = string.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return ["".join(s[i:i + n]) for i in range(len(s) - n + 1)]


def _build_fuzzy_matches(
    unique_names: list[str],
    threshold: float,
) -> dict[str, set[str]]:
    """Return dict of name → set of similar names (char-trigram TF-IDF cosine ≥ threshold).

    Names are partitioned by their first 2 characters before comparison so that
    the quadratic similarity computation stays tractable.
    """
    # Group by 2-char prefix
    prefix_buckets: dict[str, list[str]] = defaultdict(list)
    for name in unique_names:
        key = name[:2].lower() if len(name) >= 2 else name.lower()
        prefix_buckets[key].append(name)

    fuzzy: dict[str, set[str]] = defaultdict(set)

    for bucket_names in prefix_buckets.values():
        if len(bucket_names) < 2:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
        try:
            X = vec.fit_transform(bucket_names)
        except ValueError:
            continue
        sim = cosine_similarity(X)
        rows, cols = np.where(sim >= threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i != j:
                fuzzy[bucket_names[i]].add(bucket_names[j])

    return fuzzy


def _union_find_ids(
    all_names: list[str],
    fuzzy: dict[str, set[str]],
) -> tuple[dict[str, int], dict[int, list[str]]]:
    """Assign sequential integer IDs via union-find over the fuzzy-match graph."""
    parent: dict[str, str] = {n: n for n in all_names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for name, matches in fuzzy.items():
        for match in matches:
            if name in parent and match in parent:
                union(name, match)

    groups: dict[str, list[str]] = defaultdict(list)
    for name in all_names:
        groups[find(name)].append(name)

    name_to_id: dict[str, int] = {}
    id_to_names: dict[int, list[str]] = {}
    for auth_id, group in enumerate(groups.values()):
        for name in group:
            name_to_id[name] = auth_id
        id_to_names[auth_id] = sorted(group)

    return name_to_id, id_to_names


def disambiguate_authors(
    df: pd.DataFrame,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, list[str]]]:
    """Disambiguate author names using char-trigram TF-IDF cosine similarity.

    Algorithm
    ---------
    1. Parse ``authors_parsed`` column (list of [last, first, suffix]) and
       normalise each name to 'First Last' ASCII form.
    2. Group unique names by 2-character prefix.
    3. Within each prefix bucket, compute pairwise TF-IDF char-trigram cosine
       similarity.
    4. Connect names with similarity ≥ ``threshold`` via union-find.
    5. Assign sequential integer IDs to each connected component.
    6. Add ``author_ids`` column to df (list of int IDs per paper).

    Args:
        df:        DataFrame from ``preprocess_text``.
        threshold: Cosine similarity cutoff for fuzzy name matching (default 0.90).

    Returns:
        (df_with_ids, name_to_id, id_to_names)
        - df_with_ids: original df with a new ``author_ids`` column (list[int]).
        - name_to_id: mapping from normalised author name → integer ID.
        - id_to_names: mapping from integer ID → sorted list of name variants.
    """
    log.info("Disambiguating authors (threshold=%.2f) …", threshold)

    # ── Collect all author names ─────────────────────────────────────────
    all_doc_names: list[list[str]] = []
    for parsed_list in df["authors_parsed"]:
        names = [_normalise_name(a) for a in (parsed_list or [])]
        names = [n for n in names if n]  # drop empty
        all_doc_names.append(names)

    unique_names = sorted(
        {n for names in all_doc_names for n in names}
    )
    log.info("Unique author names: %d", len(unique_names))

    # ── Fuzzy matching ───────────────────────────────────────────────────
    fuzzy = _build_fuzzy_matches(unique_names, threshold)
    log.info("Fuzzy-match pairs computed.")

    # ── ID assignment via union-find ─────────────────────────────────────
    name_to_id, id_to_names = _union_find_ids(unique_names, fuzzy)
    log.info("Unique author IDs assigned: %d", len(id_to_names))

    # ── Add author_ids column ────────────────────────────────────────────
    df = df.copy()
    df["author_ids"] = [
        [name_to_id[n] for n in names if n in name_to_id]
        for names in all_doc_names
    ]

    return df, name_to_id, id_to_names
