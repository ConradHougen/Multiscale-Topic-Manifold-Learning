"""
run_comparison.py — CLI for MSTML vs BERTopic comparison experiments.

Loads a completed MSTML legacy pipeline run, trains a BERTopic ensemble,
and computes alignment, coherence, and diversity metrics for both systems.
Outputs: topic_comparison_results.csv + bar-chart PDFs.

Usage
-----
    python -m legacy.comparison.run_comparison \\
        --pipeline_dir  ./results \\
        --output_dir    ./comparison_results \\
        --embedding_model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("legacy.comparison")


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Per-chunk word lists from MSTML LDA models
# ---------------------------------------------------------------------------

def _mstml_top_words(topic_vecs_per_chunk, vocabulary: list[str], top_n: int = 10):
    """Extract top-n words per topic for each MSTML chunk."""
    result = []
    for phi in topic_vecs_per_chunk:
        chunk_words = []
        for row in phi:
            top_ids = np.argsort(row)[-top_n:][::-1]
            chunk_words.append([vocabulary[i] for i in top_ids if i < len(vocabulary)])
        result.append(chunk_words)
    return result


# ---------------------------------------------------------------------------
# Main comparison routine
# ---------------------------------------------------------------------------

def run_comparison(args) -> pd.DataFrame:
    from ._alignment  import compute_alignment_scores
    from ._coherence  import compute_ensemble_coherence
    from ._diversity  import compute_ensemble_diversity
    from ._bertopic_embed import train_bertopic_ensemble, get_bertopic_top_words

    pipe_dir = Path(args.pipeline_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load MSTML artefacts ─────────────────────────────────────────────
    log.info("Loading MSTML pipeline artefacts from %s …", pipe_dir)
    df              = _load(pipe_dir / "main_df.pkl")
    id2word         = _load(pipe_dir / "id2word.pkl")
    topic_vectors   = _load(pipe_dir / "topic_vectors.pkl")   # (total, vocab)
    ntopics_by_chunk = _load(pipe_dir / "ntopics_by_chunk.pkl")
    inds_by_chunk   = _load(pipe_dir / "inds_by_chunk.pkl")

    vocabulary = list(id2word.values())

    # Reconstruct per-chunk topic vector lists
    num_chunks = len(ntopics_by_chunk)
    mstml_vecs: list[np.ndarray] = []
    offset = 0
    for cidx in range(num_chunks):
        n = ntopics_by_chunk[cidx]
        mstml_vecs.append(topic_vectors[offset: offset + n])
        offset += n

    # Reconstruct per-chunk tokenised docs
    from .._topic_models import create_temporal_chunks
    chunks = create_temporal_chunks(df, months_per_chunk=1)  # use default
    doc_chunks_tokens = [
        chunk["text_processed"].tolist() for chunk in chunks
    ]

    mstml_words = _mstml_top_words(mstml_vecs, vocabulary, top_n=args.top_n)

    # ── Train BERTopic ensemble ──────────────────────────────────────────
    log.info("Training BERTopic ensemble …")
    bt_models, bt_vecs = train_bertopic_ensemble(
        chunks, id2word, embedding_model=args.embedding_model
    )
    bt_words = get_bertopic_top_words(bt_models, top_n=args.top_n)

    # ── Alignment ────────────────────────────────────────────────────────
    log.info("Computing alignment scores …")
    mstml_align = compute_alignment_scores(mstml_vecs, distance_metric="hellinger", knn=1)
    bt_align    = compute_alignment_scores(bt_vecs,   distance_metric="cosine",    knn=1)

    # ── Coherence ────────────────────────────────────────────────────────
    log.info("Computing coherence scores …")
    mstml_coh = compute_ensemble_coherence(mstml_words, doc_chunks_tokens, top_n=args.top_n)
    bt_coh    = compute_ensemble_coherence(bt_words,   doc_chunks_tokens, top_n=args.top_n)

    # ── Diversity ────────────────────────────────────────────────────────
    log.info("Computing diversity scores …")
    mstml_div = compute_ensemble_diversity(mstml_words, mstml_vecs, k=args.top_n)
    bt_div    = compute_ensemble_diversity(bt_words,   bt_vecs,    k=args.top_n)

    # ── Compile results ──────────────────────────────────────────────────
    rows = []
    metrics = {
        "alignment_mean_distance": ("mean_distance", mstml_align, bt_align),
        "alignment_skewness":      ("skewness",      mstml_align, bt_align),
        "diversity_at_k":          ("diversity_at_k", mstml_div, bt_div),
        "mean_hellinger":          ("mean_hellinger", mstml_div, bt_div),
        "mean_cosine":             ("mean_cosine",    mstml_div, bt_div),
        "coherence_c_v":           ("c_v",            mstml_coh, bt_coh),
        "coherence_c_npmi":        ("c_npmi",         mstml_coh, bt_coh),
        "coherence_c_uci":         ("c_uci",          mstml_coh, bt_coh),
    }
    for label, (key, md, bd) in metrics.items():
        rows.append({
            "metric":  label,
            "MSTML":   md.get(key, float("nan")),
            "BERTopic": bd.get(key, float("nan")),
        })

    results_df = pd.DataFrame(rows)
    csv_path   = out_dir / "topic_comparison_results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("Results saved to %s", csv_path)

    # ── Bar chart ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 5))
        if len(rows) == 1:
            axes = [axes]
        for ax, row in zip(axes, rows):
            ax.bar(["MSTML", "BERTopic"], [row["MSTML"], row["BERTopic"]],
                   color=["steelblue", "coral"])
            ax.set_title(row["metric"], fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
        plt.tight_layout()
        fig.savefig(out_dir / "topic_comparison_bars.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Bar chart saved.")
    except Exception as exc:
        log.warning("Could not generate bar chart: %s", exc)

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m legacy.comparison.run_comparison",
        description="MSTML vs BERTopic comparison experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pipeline_dir",   required=True,
                   help="Directory containing completed MSTML pipeline outputs.")
    p.add_argument("--output_dir",     required=True,
                   help="Directory to write comparison results.")
    p.add_argument("--embedding_model", default="all-MiniLM-L6-v2",
                   help="sentence-transformers model for BERTopic.")
    p.add_argument("--top_n", default=10, type=int,
                   help="Top-N words per topic for coherence and diversity metrics.")
    return p


def main(argv=None):
    _setup_logging()
    args = build_parser().parse_args(argv)
    df   = run_comparison(args)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
