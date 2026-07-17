"""
run_pipeline.py — CLI entrypoint for the MSTML legacy pipeline.

Usage (CAMSAP 2025 exact reproduction)
---------------------------------------
    python -m legacy.run_pipeline \\
        --input_file  arxiv-metadata-oai-snapshot.json \\
        --output_dir  ./results \\
        --categories  cs.LG stat.AP stat.CO stat.ME stat.ML stat.OT stat.TH \\
        --year_start  2012 \\
        --year_end    2023 \\
        --distance_metric  hellinger \\
        --embedding_method phate \\
        --linkage_method   ward \\
        --cut_height       0.68

All secondary hyperparameters (knn values, LDA params, diffusion params) are
read from ``legacy/config.yaml`` and can be overridden with ``--config``.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path


def _setup_logging(level: str = "INFO", output_dir: "Path | None" = None) -> None:
    fmt      = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt  = "%H:%M:%S"
    level_v  = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level_v)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)

    if output_dir is not None:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"pipeline_{ts}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)
        print(f"  Log: {log_path}", flush=True)


log = logging.getLogger("legacy.pipeline")


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("Saved -> %s", path)


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_preprocess(args, cfg, out: Path) -> dict:
    from ._preprocessing import load_arxiv_jsonl, preprocess_text, disambiguate_authors

    df = load_arxiv_jsonl(
        args.input_file,
        categories=args.categories,
        year_start=args.year_start,
        year_end=args.year_end,
        max_authors=cfg.pipeline.max_authors_per_doc,
    )
    df, id2word = preprocess_text(df, cfg)
    df, name_to_id, id_to_names = disambiguate_authors(
        df, threshold=cfg.pipeline.author_disambig_threshold
    )

    _save(df,          out / "main_df.pkl")
    _save(id2word,     out / "id2word.pkl")
    _save(name_to_id,  out / "name_to_id.pkl")
    _save(id_to_names, out / "id_to_names.pkl")

    return {"df": df, "id2word": id2word, "name_to_id": name_to_id, "id_to_names": id_to_names}


def stage_ensemble(args, cfg, state: dict, out: Path) -> dict:
    from ._topic_models import (
        create_temporal_chunks,
        train_ensemble,
        compute_doc_topic_distributions,
        expand_distributions,
    )

    df     = state["df"]
    id2word = state["id2word"]

    chunks = create_temporal_chunks(df, cfg.pipeline.months_per_chunk)
    models_dir = out / "topic_models"
    models, topic_vectors, ntopics_by_chunk, inds_by_chunk = train_ensemble(
        chunks, id2word, args.topic_model, cfg, output_dir=models_dir
    )

    doc_topic_distns = compute_doc_topic_distributions(chunks, models, inds_by_chunk)
    expanded         = expand_distributions(doc_topic_distns, inds_by_chunk, ntopics_by_chunk)

    _save(topic_vectors,      out / "topic_vectors.pkl")
    _save(ntopics_by_chunk,   out / "ntopics_by_chunk.pkl")
    _save(inds_by_chunk,      out / "inds_by_chunk.pkl")
    _save(doc_topic_distns,   out / "doc_topic_distns.pkl")
    _save(expanded,           out / "expanded_doc_topic_distns.pkl")

    return {**state, "chunks": chunks, "topic_vectors": topic_vectors,
            "ntopics_by_chunk": ntopics_by_chunk, "inds_by_chunk": inds_by_chunk,
            "expanded": expanded}


def stage_distributions(args, cfg, state: dict, out: Path) -> dict:
    from ._distributions import (
        build_coauthor_graph,
        compute_author_barycenters,
        build_diffusion_graph,
        diffuse_distributions,
    )

    df       = state["df"]
    expanded = state["expanded"]
    tvecs    = state["topic_vectors"]

    coauthor_graph = build_coauthor_graph(df)
    author_barycenters, authId_to_docs, authId_to_weights = compute_author_barycenters(
        expanded, df
    )
    knn_graph = build_diffusion_graph(
        tvecs,
        knn=cfg.diffusion.knn,
        distance_metric=args.distance_metric,
        legacy_bug=args.reproduce_legacy_bug,
    )

    diff_cfg = cfg.diffusion
    author_ct = diffuse_distributions(
        knn_graph, author_barycenters,
        num_iterations=diff_cfg.num_iterations,
        diffusion_rate=diff_cfg.rate,
    )
    doc_ct = diffuse_distributions(
        knn_graph, expanded,
        num_iterations=diff_cfg.num_iterations,
        diffusion_rate=diff_cfg.rate,
    )

    _save(coauthor_graph,   out / "coauthor_graph.pkl")
    _save(author_ct,        out / "author_ct_distns.pkl")
    _save(doc_ct,           out / "doc_ct_distns.pkl")
    _save(knn_graph,        out / "knn_graph.pkl")
    _save(authId_to_docs,   out / "authId_to_docs.pkl")

    return {**state, "coauthor_graph": coauthor_graph, "author_ct": author_ct,
            "doc_ct": doc_ct, "authId_to_docs": authId_to_docs}


def stage_manifold(args, cfg, state: dict, out: Path) -> dict:
    from ._manifold import (
        compute_pairwise_distances,
        build_topic_dendrogram,
        cut_dendrogram,
        compute_embedding,
        project_distributions_onto_embedding,
    )
    import numpy as np

    tvecs = state["topic_vectors"]

    Z, min_h, max_h = build_topic_dendrogram(
        tvecs,
        knn=cfg.dendrogram.knn,
        linkage_method=args.linkage_method,
        distance_metric=args.distance_metric,
        legacy_bug=args.reproduce_legacy_bug,
    )
    cluster_labels = cut_dendrogram(Z, args.cut_height, min_h, max_h)

    dist_mat = compute_pairwise_distances(tvecs, args.distance_metric)

    emb_kwargs = cfg.embeddings.get_kwargs(args.embedding_method)
    emb_knn    = emb_kwargs.pop("knn", cfg.embeddings.phate.knn)
    embedding, phate_op = compute_embedding(
        dist_mat, args.embedding_method, knn=emb_knn, **emb_kwargs
    )

    # Time labels: chunk index per topic
    ntopics = state["ntopics_by_chunk"]
    time_labels = np.array([
        cidx
        for cidx, n in sorted(ntopics.items())
        for _ in range(n)
    ])

    _save(Z,              out / "dendrogram_Z.pkl")
    _save((min_h, max_h), out / "dendrogram_heights.pkl")
    _save(cluster_labels, out / "cluster_labels.pkl")
    _save(dist_mat,       out / "distance_matrix.pkl")
    _save(embedding,      out / "embedding.pkl")
    _save(time_labels,    out / "time_labels.pkl")

    result = {**state, "Z": Z, "min_h": min_h, "max_h": max_h,
              "cluster_labels": cluster_labels, "dist_mat": dist_mat,
              "embedding": embedding, "time_labels": time_labels,
              "phate_op": phate_op}

    # Map author and doc distributions into embedding space via barycentric
    # interpolation: weighted sum of topic PHATE coordinates using each
    # author/doc's topic distribution as weights.  This matches how
    # AToMS_HRG_Longitudinal_Analysis.ipynb overlays author trajectories on the
    # PHATE plot (it never calls phate_operator.transform() on new data —
    # transform() is unsupported when PHATE was fit on a precomputed distance
    # matrix).
    if phate_op is not None:
        def _barycentric_embed(distns_dict: dict) -> np.ndarray:
            mat = np.array(list(distns_dict.values()), dtype=np.float32)
            row_sums = mat.sum(axis=1, keepdims=True)
            mat = mat / np.where(row_sums > 0, row_sums, 1.0)
            return mat @ embedding  # (n_points, n_components)

        log.info("[4/6] Projecting author distributions onto embedding ...")
        author_ids  = list(state["author_ct"].keys())
        author_emb  = _barycentric_embed(state["author_ct"])
        _save({"ids": author_ids, "embedding": author_emb}, out / "author_embedding.pkl")
        result["author_emb"] = author_emb
        result["author_emb_ids"] = author_ids

        log.info("[4/6] Projecting document distributions onto embedding ...")
        doc_ids  = list(state["doc_ct"].keys())
        doc_emb  = _barycentric_embed(state["doc_ct"])
        _save({"ids": doc_ids, "embedding": doc_emb}, out / "doc_embedding.pkl")
        result["doc_emb"] = doc_emb
        result["doc_emb_ids"] = doc_ids
    else:
        log.info("[4/6] Skipping author/doc projection (only supported for PHATE).")

    return result


def stage_scoring(args, cfg, state: dict, out: Path) -> dict:
    from ._scoring import run_scoring
    import pandas as pd

    results = run_scoring(
        Z=state["Z"],
        min_height=state["min_h"],
        max_height=state["max_h"],
        cut_height=args.cut_height,
        author_ct_distns=state["author_ct"],
        coauthor_graph=state["coauthor_graph"],
        df=state["df"],
    )

    _save(results["author_index_map"],   out / "author_index_map.pkl")
    _save(results["author_meta_distns"], out / "author_meta_distns.pkl")
    _save(results["doc_scores"],         out / "doc_scores.pkl")
    _save(results["link_scores"],        out / "link_scores.pkl")
    _save(results["author_ranking"],     out / "author_ranking.pkl")

    # Save human-readable CSV rankings
    import pandas as pd
    pd.DataFrame(
        list(results["author_ranking"].items()), columns=["author_id", "entropy"]
    ).to_csv(out / "author_ranking.csv", index=False)

    return {**state, **results}


def stage_visualize(args, cfg, state: dict, out: Path) -> None:
    from ._visualization import (
        plot_phate_embedding,
        plot_dendrogram_figure,
        generate_meta_topic_wordclouds,
        plot_interdisciplinarity_bars,
        plot_coauthor_network,
    )
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for script usage

    prefix = f"{args.distance_metric}_{args.embedding_method}_"
    vocab  = list(state["id2word"].values()) if state.get("id2word") else []

    plot_phate_embedding(
        state["embedding"], state["cluster_labels"], state["time_labels"],
        out, prefix=prefix, cfg=cfg,
    )
    plot_dendrogram_figure(state["Z"], state["cluster_labels"], out, prefix=prefix, cfg=cfg)

    if vocab and state.get("topic_vectors") is not None:
        generate_meta_topic_wordclouds(
            state["topic_vectors"], state["cluster_labels"],
            vocab, output_dir=out, prefix=prefix, cfg=cfg,
        )

    plot_interdisciplinarity_bars(
        state["doc_scores"], state["author_ranking"],
        out, prefix=prefix, cfg=cfg,
    )
    plot_coauthor_network(
        state["coauthor_graph"], state["link_scores"],
        state["author_ranking"], out, prefix=prefix, cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m legacy.run_pipeline",
        description="MSTML legacy pipeline — reproduce CAMSAP 2025 results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Required primary hyperparameters ─────────────────────────────────
    p.add_argument("--input_file",  required=True,
                   help="Path to arxiv-metadata-oai-snapshot.json (JSONL).")
    p.add_argument("--output_dir",  required=True,
                   help="Directory for all outputs (created if missing).")
    p.add_argument("--categories",  required=True, nargs="+",
                   help="arXiv category codes to include (space-separated).")
    p.add_argument("--year_start",  required=True, type=int,
                   help="First year (inclusive).")
    p.add_argument("--year_end",    required=True, type=int,
                   help="Last year (inclusive).")
    # ── Optional primary hyperparameters ─────────────────────────────────
    p.add_argument("--topic_model",      default="lda",
                   choices=["lda", "bertopic"],
                   help="Topic model to use.")
    p.add_argument("--distance_metric",  default="hellinger",
                   choices=["hellinger", "cosine", "euclidean"],
                   help="Distance metric for topic vectors.")
    p.add_argument("--embedding_method", default="phate",
                   choices=["phate", "umap", "tsne", "pca"],
                   help="Manifold embedding method.")
    p.add_argument("--linkage_method",   default="ward",
                   choices=["ward", "complete", "average", "single"],
                   help="Scipy hierarchical linkage method.")
    p.add_argument("--cut_height",       default=0.68, type=float,
                   help="Normalised dendrogram cut height in [0, 1].")
    # ── Config / control ─────────────────────────────────────────────────
    p.add_argument("--config", default=None,
                   help="Path to a YAML config override (default: legacy/config.yaml).")
    p.add_argument("--skip_preprocess",  action="store_true",
                   help="Skip preprocessing and load saved artifacts from output_dir.")
    p.add_argument("--skip_ensemble",    action="store_true")
    p.add_argument("--skip_distributions", action="store_true")
    p.add_argument("--skip_manifold",    action="store_true")
    p.add_argument("--skip_scoring",     action="store_true")
    p.add_argument("--skip_visualize",   action="store_true")
    p.add_argument("--reproduce_legacy_bug", action="store_true",
                   help="Use the original AToMS-LP FAISS Hellinger conversion "
                        "(sq_l2 / sqrt(2)) instead of the correct sqrt(sq_l2 / 2). "
                        "Set this flag to bitwise-reproduce thesis/CAMSAP 2025 results. "
                        "Has no effect for non-Hellinger distance metrics.")
    p.add_argument("--log_level",        default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None):
    from ._config import PrimaryConfig, load_pipeline_config

    parser = build_parser()
    args   = parser.parse_args(argv)
    _setup_logging(args.log_level)

    # Validate
    primary = PrimaryConfig(
        input_file=args.input_file,
        output_dir=args.output_dir,
        categories=args.categories,
        year_start=args.year_start,
        year_end=args.year_end,
        topic_model=args.topic_model,
        distance_metric=args.distance_metric,
        embedding_method=args.embedding_method,
        linkage_method=args.linkage_method,
        cut_height=args.cut_height,
    )
    if not args.skip_preprocess:
        primary.validate()

    cfg = load_pipeline_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _setup_logging(args.log_level, output_dir=out)

    log.info("=== MSTML legacy pipeline ===")
    log.info("  categories     : %s", args.categories)
    log.info("  years          : %d – %d", args.year_start, args.year_end)
    log.info("  topic_model    : %s", args.topic_model)
    log.info("  distance_metric: %s", args.distance_metric)
    log.info("  embedding      : %s", args.embedding_method)
    log.info("  linkage        : %s", args.linkage_method)
    log.info("  cut_height     : %.3f", args.cut_height)

    try:
        state: dict = {}

        # ── Stage 1: Preprocessing ───────────────────────────────────────────
        if args.skip_preprocess:
            log.info("[SKIP] Loading preprocessing artifacts …")
            state["df"]        = _load(out / "main_df.pkl")
            state["id2word"]   = _load(out / "id2word.pkl")
        else:
            log.info("[1/6] Preprocessing …")
            state = stage_preprocess(args, cfg, out)

        # ── Stage 2: Ensemble topic models ───────────────────────────────────
        if args.skip_ensemble:
            log.info("[SKIP] Loading ensemble artifacts …")
            state["topic_vectors"]    = _load(out / "topic_vectors.pkl")
            state["ntopics_by_chunk"] = _load(out / "ntopics_by_chunk.pkl")
            state["inds_by_chunk"]    = _load(out / "inds_by_chunk.pkl")
            state["expanded"]         = _load(out / "expanded_doc_topic_distns.pkl")
        else:
            log.info("[2/6] Training ensemble …")
            state = stage_ensemble(args, cfg, state, out)

        # Free objects not needed in stages 3–6
        for _k in ("chunks",):
            state.pop(_k, None)
        gc.collect()

        # ── Stage 3: Distributions ───────────────────────────────────────────
        if args.skip_distributions:
            log.info("[SKIP] Loading distribution artifacts …")
            state["coauthor_graph"] = _load(out / "coauthor_graph.pkl")
            state["author_ct"]      = _load(out / "author_ct_distns.pkl")
            state["doc_ct"]         = _load(out / "doc_ct_distns.pkl")
            state["authId_to_docs"] = _load(out / "authId_to_docs.pkl")
        else:
            log.info("[3/6] Computing distributions …")
            state = stage_distributions(args, cfg, state, out)

        # Free objects not needed in stages 4–6 (~9.8 GB for expanded)
        for _k in ("expanded", "inds_by_chunk"):
            state.pop(_k, None)
        gc.collect()

        # ── Stage 4: Manifold ────────────────────────────────────────────────
        if args.skip_manifold:
            log.info("[SKIP] Loading manifold artifacts …")
            state["Z"]              = _load(out / "dendrogram_Z.pkl")
            state["min_h"], state["max_h"] = _load(out / "dendrogram_heights.pkl")
            state["cluster_labels"] = _load(out / "cluster_labels.pkl")
            state["dist_mat"]       = _load(out / "distance_matrix.pkl")
            state["embedding"]      = _load(out / "embedding.pkl")
            state["time_labels"]    = _load(out / "time_labels.pkl")
            state["phate_op"]       = None  # operator not serialised; projection unavailable
            for _key, _fname in [("author_emb", "author_embedding.pkl"),
                                  ("doc_emb", "doc_embedding.pkl")]:
                _p = out / _fname
                if _p.exists():
                    _blob = _load(_p)
                    state[_key] = _blob["embedding"]
                    state[f"{_key}_ids"] = _blob["ids"]
        else:
            log.info("[4/6] Building dendrogram and embedding …")
            state = stage_manifold(args, cfg, state, out)

        # Free large objects not needed in stages 5–6
        # dist_mat: 12732×12732 float32 (~648 MB); doc_ct: diffused doc dists (~9.8 GB)
        for _k in ("dist_mat", "phate_op", "doc_ct"):
            state.pop(_k, None)
        gc.collect()

        # ── Stage 5: Scoring ─────────────────────────────────────────────────
        if args.skip_scoring:
            log.info("[SKIP] Loading scoring artifacts …")
            state["doc_scores"]     = _load(out / "doc_scores.pkl")
            state["link_scores"]    = _load(out / "link_scores.pkl")
            state["author_ranking"] = _load(out / "author_ranking.pkl")
            state["author_meta_distns"] = _load(out / "author_meta_distns.pkl")
        else:
            log.info("[5/6] Scoring interdisciplinarity …")
            state = stage_scoring(args, cfg, state, out)

        # Free objects not needed in stage 6
        for _k in ("author_ct",):
            state.pop(_k, None)
        gc.collect()

        # ── Stage 6: Visualisation ───────────────────────────────────────────
        if not args.skip_visualize:
            log.info("[6/6] Generating figures …")
            stage_visualize(args, cfg, state, out)

        log.info("Pipeline complete. Outputs in: %s", out.resolve())

    except Exception:
        log.exception("Pipeline failed with unhandled exception:")
        raise


if __name__ == "__main__":
    main()
