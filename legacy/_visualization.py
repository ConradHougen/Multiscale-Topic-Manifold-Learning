"""
_visualization.py — Static matplotlib plots for the legacy pipeline.

All functions write figures to ``output_dir/figures/`` and return the
matplotlib Figure object for optional further use.

Functions
---------
  plot_phate_embedding          — topic scatter, rainbow (cluster) + viridis (time)
  plot_dendrogram_figure        — scipy dendrogram with cluster colouring
  generate_meta_topic_wordclouds — per-cluster WordCloud saved as PDF
  plot_interdisciplinarity_bars — top-N doc and author interdisciplinarity scores
  plot_coauthor_network         — spring-layout network with score-coloured nodes/edges

Ported from:
  AToMS-LP/AToMS_HRG_Longitudinal_Analysis.ipynb
  AToMS-LP/AToMS/atoms_hrg_library.py (find_centroids_and_create_wordclouds,
    plot_wordcloud_for_topic, plot_phate_embedding_with_filtered_chunks)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from ._config import PipelineConfig
from ._math import term_relevance

log = logging.getLogger(__name__)

try:
    from wordcloud import WordCloud  # type: ignore[import]
    _WC_OK = True
except ImportError:
    _WC_OK = False

try:
    import networkx as nx  # type: ignore[import]
    _NX_OK = True
except ImportError:
    _NX_OK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _figure_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir) / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# PHATE embedding plots
# ---------------------------------------------------------------------------

def plot_phate_embedding(
    embedding: ndarray,
    cluster_labels: ndarray,
    time_labels: ndarray,
    output_dir: str | Path,
    prefix: str = "",
    cfg: Optional[PipelineConfig] = None,
    elev: float = 20.0,
    azim: float = 45.0,
    time_chunk_start_times=None,
) -> tuple[plt.Figure, plt.Figure]:
    """Generate two PHATE scatter plots: cluster-coloured and time-coloured.

    For 3-D embeddings a 3-D projection is used; 2-D falls back to a flat scatter.

    Args:
        embedding:              (n_topics, n_components) array.
        cluster_labels:         (n_topics,) integer cluster IDs (1-indexed).
        time_labels:            (n_topics,) integer chunk indices.
        output_dir:             Directory where figures are saved.
        prefix:                 Filename prefix (e.g. experiment name).
        cfg:                    PipelineConfig for DPI.
        elev, azim:             3-D view angles.
        time_chunk_start_times: Optional array of datetime objects for colorbar labels.

    Returns:
        (fig_cluster, fig_time)
    """
    dpi = cfg.visualization.figure_dpi if cfg else 300
    fig_dir = _figure_dir(output_dir)
    n_3d = embedding.shape[1] >= 3

    # ── Cluster-coloured ─────────────────────────────────────────────────
    n_clusters  = len(np.unique(cluster_labels))
    cmap_c      = cm.get_cmap("rainbow", n_clusters)
    boundaries  = np.arange(cluster_labels.min(), cluster_labels.max() + 2) - 0.5
    norm_c      = mcolors.BoundaryNorm(boundaries, ncolors=n_clusters, clip=True)
    colors_c    = [
        cmap_c(i / n_clusters)
        for i, lbl in enumerate(np.unique(cluster_labels))
        for _ in np.where(cluster_labels == lbl)[0]
    ]
    # build per-point colour array in label order
    uniq = {lbl: i for i, lbl in enumerate(sorted(np.unique(cluster_labels)))}
    colours_c = np.array([cmap_c(uniq[lbl] / n_clusters) for lbl in cluster_labels])

    if n_3d:
        fig_c = plt.figure(figsize=(10, 8))
        ax_c  = fig_c.add_subplot(111, projection="3d")
        ax_c.set_facecolor("none")
        ax_c.view_init(elev=elev, azim=azim)
        ax_c.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=colours_c, marker="o", s=20, edgecolor="k",
        )
        ax_c.set_xlabel("PHATE 1"); ax_c.set_ylabel("PHATE 2"); ax_c.set_zlabel("PHATE 3")
        for ax in (ax_c,):
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    else:
        fig_c, ax_c = plt.subplots(figsize=(10, 8))
        ax_c.scatter(embedding[:, 0], embedding[:, 1], c=colours_c, marker="o", s=20, edgecolor="k")
        ax_c.set_xlabel("PHATE 1"); ax_c.set_ylabel("PHATE 2")

    sm_c = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    sm_c.set_array([])
    cbar_c = fig_c.colorbar(sm_c)
    cbar_c.set_label("Topic Cluster")
    path_c = fig_dir / f"{prefix}phate_cluster.pdf"
    _save(fig_c, path_c, dpi)

    # ── Time-coloured ────────────────────────────────────────────────────
    t_min, t_max = float(time_labels.min()), float(time_labels.max())
    norm_t = (time_labels - t_min) / max(t_max - t_min, 1.0)

    if n_3d:
        fig_t = plt.figure(figsize=(10, 8))
        ax_t  = fig_t.add_subplot(111, projection="3d")
        ax_t.set_facecolor("none")
        ax_t.view_init(elev=elev, azim=azim)
        sc_t = ax_t.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=norm_t, cmap="viridis", marker="o", s=20, edgecolor="k",
        )
        ax_t.set_xlabel("PHATE 1"); ax_t.set_ylabel("PHATE 2"); ax_t.set_zlabel("PHATE 3")
        ax_t.set_xticklabels([]); ax_t.set_yticklabels([]); ax_t.set_zticklabels([])
    else:
        fig_t, ax_t = plt.subplots(figsize=(10, 8))
        sc_t = ax_t.scatter(
            embedding[:, 0], embedding[:, 1],
            c=norm_t, cmap="viridis", marker="o", s=20, edgecolor="k",
        )
        ax_t.set_xlabel("PHATE 1"); ax_t.set_ylabel("PHATE 2")

    sm_t = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm_t.set_array([])
    cbar_t = fig_t.colorbar(sm_t)
    if time_chunk_start_times is not None:
        import pandas as pd
        ticks = np.linspace(0, 1, min(10, len(time_chunk_start_times)))
        cbar_t.set_ticks(ticks)
        idx = np.linspace(0, len(time_chunk_start_times) - 1, len(ticks), dtype=int)
        labels = pd.to_datetime(time_chunk_start_times[idx]).strftime("%b %Y")
        cbar_t.set_ticklabels(labels)
    cbar_t.set_label("Time", rotation=270, labelpad=15)
    path_t = fig_dir / f"{prefix}phate_time.pdf"
    _save(fig_t, path_t, dpi)

    return fig_c, fig_t


# ---------------------------------------------------------------------------
# Dendrogram
# ---------------------------------------------------------------------------

def plot_dendrogram_figure(
    Z: ndarray,
    cluster_labels: ndarray,
    output_dir: str | Path,
    prefix: str = "",
    cfg: Optional[PipelineConfig] = None,
) -> plt.Figure:
    """Draw and save the scipy hierarchical dendrogram.

    Args:
        Z:              Linkage matrix.
        cluster_labels: (n_topics,) cluster assignments for leaf colouring.
        output_dir:     Output directory.
        prefix:         Filename prefix.
        cfg:            PipelineConfig for DPI.

    Returns:
        matplotlib Figure.
    """
    dpi     = cfg.visualization.figure_dpi if cfg else 300
    fig_dir = _figure_dir(output_dir)

    # Build link colour map
    n_clusters = len(np.unique(cluster_labels))
    cmap_d     = cm.get_cmap("rainbow", n_clusters)
    uniq       = sorted(np.unique(cluster_labels))
    leaf_colours = {
        i: mcolors.to_hex(cmap_d(uniq.index(lbl) / n_clusters))
        for i, lbl in enumerate(cluster_labels)
    }

    fig, ax = plt.subplots(figsize=(max(12, len(cluster_labels) // 5), 6))
    scipy_dendrogram(
        Z,
        ax=ax,
        leaf_font_size=4,
        color_threshold=Z[-1, 2] * 0.7,
        above_threshold_color="grey",
    )
    ax.set_xlabel("Topic index")
    ax.set_ylabel("Linkage distance")
    ax.set_title("Topic dendrogram")
    path = fig_dir / f"{prefix}dendrogram.pdf"
    _save(fig, path, dpi)
    return fig


# ---------------------------------------------------------------------------
# Meta-topic word clouds
# ---------------------------------------------------------------------------

def generate_meta_topic_wordclouds(
    topic_vectors: ndarray,
    cluster_labels: ndarray,
    vocabulary: list[str],
    lambda_param: float = 0.6,
    output_dir: str | Path = ".",
    prefix: str = "",
    cfg: Optional[PipelineConfig] = None,
    top_n: int = 100,
) -> dict[int, object]:
    """Generate one WordCloud per meta-topic cluster.

    Each cloud is built from the centroid topic vector of its cluster,
    re-ranked by LDAvis-style term relevance.

    Args:
        topic_vectors:  (n_topics, vocab_size) phi matrix.
        cluster_labels: (n_topics,) integer cluster IDs (1-indexed).
        vocabulary:     List of vocabulary tokens, length vocab_size.
        lambda_param:   LDAvis relevance λ.
        output_dir:     Directory to save PDFs.
        prefix:         Filename prefix.
        cfg:            PipelineConfig for DPI.
        top_n:          Number of top terms to include in each cloud.

    Returns:
        Dict mapping cluster_id → WordCloud object (or None if wordcloud unavailable).
    """
    if not _WC_OK:
        log.warning("wordcloud not installed; skipping word cloud generation.")
        return {}

    dpi     = cfg.visualization.figure_dpi if cfg else 300
    fig_dir = _figure_dir(output_dir)

    # Overall corpus word probabilities
    overall = topic_vectors.sum(axis=0)
    p_w     = overall / max(overall.sum(), 1e-12)

    wordclouds: dict[int, object] = {}
    for cluster_id in sorted(np.unique(cluster_labels)):
        idx = np.where(cluster_labels == cluster_id)[0]
        centroid = topic_vectors[idx].mean(axis=0)

        # Term relevance scores for centroid
        scores     = term_relevance(centroid, p_w, lam=lambda_param)
        top_ids    = np.argsort(scores)[-top_n:][::-1]
        phi_normed = centroid / max(centroid.sum(), 1e-12)
        freq_dict  = {
            vocabulary[i]: float(phi_normed[i])
            for i in top_ids
            if i < len(vocabulary)
        }

        wc = WordCloud(width=800, height=400, max_words=top_n, background_color="white")
        wc.generate_from_frequencies(freq_dict)
        wordclouds[cluster_id] = wc

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title(f"Meta-topic {cluster_id}")
        path_wc = fig_dir / f"{prefix}wordcloud_cluster_{cluster_id}.pdf"
        _save(fig_wc, path_wc, dpi)
        plt.close(fig_wc)

    log.info("Generated %d meta-topic word clouds.", len(wordclouds))
    return wordclouds


# ---------------------------------------------------------------------------
# Interdisciplinarity bar charts
# ---------------------------------------------------------------------------

def plot_interdisciplinarity_bars(
    doc_scores: "dict[int, float]",
    author_scores: "dict[int, float]",
    output_dir: str | Path,
    prefix: str = "",
    cfg: Optional[PipelineConfig] = None,
    top_n: int = 30,
) -> tuple[plt.Figure, plt.Figure]:
    """Bar charts of the top-N interdisciplinarity scores.

    Args:
        doc_scores:    Ordered dict from ``score_interdisciplinarity_docs``.
        author_scores: Ordered dict from ``rank_authors_by_interdisciplinarity``.
        output_dir:    Output directory.
        prefix:        Filename prefix.
        cfg:           PipelineConfig for DPI.
        top_n:         Number of entries to show.

    Returns:
        (fig_docs, fig_authors)
    """
    dpi     = cfg.visualization.figure_dpi if cfg else 300
    fig_dir = _figure_dir(output_dir)

    def _bar(scores, title, xlabel, path):
        items  = list(scores.items())[:top_n]
        labels = [str(k) for k, _ in items]
        values = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh(range(len(values)), values, color="steelblue")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.invert_yaxis()
        _save(fig, path, dpi)
        return fig

    fig_docs = _bar(
        doc_scores, f"Top-{top_n} Interdisciplinary Documents",
        "Entropy (bits)", fig_dir / f"{prefix}interdiscip_docs.pdf",
    )
    fig_auth = _bar(
        author_scores, f"Top-{top_n} Interdisciplinary Authors",
        "Entropy (bits)", fig_dir / f"{prefix}interdiscip_authors.pdf",
    )
    return fig_docs, fig_auth


# ---------------------------------------------------------------------------
# Co-author network
# ---------------------------------------------------------------------------

def plot_coauthor_network(
    coauthor_graph,
    link_scores: "list[tuple[frozenset, float]]",
    author_scores: "dict[int, float]",
    output_dir: str | Path,
    prefix: str = "",
    cfg: Optional[PipelineConfig] = None,
    max_nodes: int = 500,
) -> plt.Figure:
    """NetworkX spring-layout co-author network coloured by interdisciplinarity.

    Node size ∝ author interdisciplinarity score.
    Edge colour ∝ link likelihood score (low = surprising/bridging, dark blue).

    Args:
        coauthor_graph: NetworkX Graph from ``build_coauthor_graph``.
        link_scores:    Sorted list from ``score_interdisciplinarity_links``.
        author_scores:  OrderedDict from ``rank_authors_by_interdisciplinarity``.
        output_dir:     Output directory.
        prefix:         Filename prefix.
        cfg:            PipelineConfig for DPI.
        max_nodes:      Cap on nodes drawn (take highest-scoring authors).

    Returns:
        matplotlib Figure.
    """
    if not _NX_OK:
        log.warning("networkx not available; skipping co-author network plot.")
        return None

    dpi     = cfg.visualization.figure_dpi if cfg else 300
    fig_dir = _figure_dir(output_dir)

    # Subset to most interdisciplinary authors for readability
    top_authors = set(list(author_scores.keys())[:max_nodes])
    sub_G = coauthor_graph.subgraph(
        [n for n in coauthor_graph.nodes() if n in top_authors]
    )

    pos = nx.spring_layout(sub_G, seed=42, k=1.5 / max(1, np.sqrt(sub_G.number_of_nodes())))

    # Node sizes
    max_score = max(author_scores.values()) if author_scores else 1.0
    node_sizes = [
        200 * author_scores.get(n, 0.0) / max(max_score, 1e-9)
        for n in sub_G.nodes()
    ]

    # Edge colours from link scores (build lookup)
    link_dict = {fs: score for fs, score in link_scores}
    edge_scores = [
        link_dict.get(frozenset({u, v}), 0.0)
        for u, v in sub_G.edges()
    ]
    if edge_scores:
        e_min, e_max = min(edge_scores), max(edge_scores)
        edge_norm = [
            (s - e_min) / max(e_max - e_min, 1e-9)
            for s in edge_scores
        ]
    else:
        edge_norm = []

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(
        sub_G, pos, ax=ax,
        node_size=node_sizes,
        node_color=list(author_scores.get(n, 0.0) for n in sub_G.nodes()),
        cmap="YlOrRd",
        alpha=0.8,
    )
    if edge_norm:
        edge_colours = [plt.cm.Blues(v) for v in edge_norm]
        nx.draw_networkx_edges(sub_G, pos, ax=ax, edge_color=edge_colours, alpha=0.5, width=0.8)
    ax.set_title("Co-author network — node=interdisciplinarity, edge=link surprise")
    ax.axis("off")
    path = fig_dir / f"{prefix}coauthor_network.pdf"
    _save(fig, path, dpi)
    return fig
