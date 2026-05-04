"""
_interactive.py — Interactive visualisations (stretch goal).

Two entry points:

  interactive_phate_wordcloud(...)
    Matplotlib figure with mplcursors: click a topic point → word cloud inset
    for that chunk-topic.

  interactive_coauthor_network(...)
    Plotly/Dash network graph: click a node → author profile panel;
    click an edge → link score + shared meta-topics.

Both functions gracefully degrade if optional dependencies (mplcursors, plotly,
dash) are not installed.

Ported from:
  AToMS-LP/AToMS_HRG_Longitudinal_Analysis.ipynb (mplcursors block)
  AToMS-LP/AToMS/atoms_hrg_library.py (plot_phate_embedding_with_filtered_chunks,
    plot_wordcloud_for_topic)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray

from ._math import term_relevance

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactive PHATE + word cloud (mplcursors)
# ---------------------------------------------------------------------------

def interactive_phate_wordcloud(
    embedding: ndarray,
    cluster_labels: ndarray,
    topic_vectors: ndarray,
    vocabulary: list[str],
    chunk_labels: ndarray,
    ntopics_by_chunk: "dict[int, int]",
    lambda_param: float = 0.6,
    top_n: int = 20,
    title: str = "PHATE embedding — click a point for its word cloud",
) -> "object":
    """Interactive PHATE scatter with click-to-wordcloud (mplcursors).

    When the user clicks on a topic point in the scatter, a word cloud inset
    for that chunk-topic is displayed using LDAvis-style term relevance.

    Args:
        embedding:        (n_topics, n_components) array; first 2 dims are plotted.
        cluster_labels:   (n_topics,) integer cluster IDs.
        topic_vectors:    (n_topics, vocab_size) phi matrix.
        vocabulary:       Vocabulary token list.
        chunk_labels:     (n_topics,) integer chunk indices.
        ntopics_by_chunk: chunk_idx → number of topics.
        lambda_param:     LDAvis relevance λ.
        top_n:            Number of top terms for the word cloud.
        title:            Figure title.

    Returns:
        matplotlib Figure, or None if mplcursors is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import mplcursors  # type: ignore[import]
    except ImportError as exc:
        log.warning("mplcursors not installed; interactive PHATE unavailable. (%s)", exc)
        return None

    try:
        from wordcloud import WordCloud  # type: ignore[import]
        _wc_ok = True
    except ImportError:
        _wc_ok = False

    n_clusters   = len(np.unique(cluster_labels))
    cmap         = cm.get_cmap("rainbow", n_clusters)
    uniq         = {lbl: i for i, lbl in enumerate(sorted(np.unique(cluster_labels)))}
    colours      = np.array([cmap(uniq[lbl] / n_clusters) for lbl in cluster_labels])

    # Overall word probabilities for relevance scoring
    overall = topic_vectors.sum(axis=0)
    p_w     = overall / max(overall.sum(), 1e-12)

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colours, marker="o", s=20, edgecolor="k",
        picker=True,
    )
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Inset axis for word cloud
    inset_ax: list = [None]

    def _on_click(sel):
        idx = sel.index
        phi    = topic_vectors[idx]
        scores = term_relevance(phi, p_w, lam=lambda_param)
        top_ids = np.argsort(scores)[-top_n:][::-1]
        phi_n  = phi / max(phi.sum(), 1e-12)
        freq   = {vocabulary[i]: float(phi_n[i]) for i in top_ids if i < len(vocabulary)}

        # Remove previous inset
        if inset_ax[0] is not None:
            inset_ax[0].remove()
            inset_ax[0] = None

        ia = fig.add_axes([0.65, 0.65, 0.32, 0.30])
        ia.axis("off")
        inset_ax[0] = ia

        if _wc_ok and freq:
            wc = WordCloud(width=400, height=200, max_words=top_n, background_color="white")
            wc.generate_from_frequencies(freq)
            ia.imshow(wc, interpolation="bilinear")
        else:
            top_words = [vocabulary[i] for i in top_ids[:10] if i < len(vocabulary)]
            ia.text(0.05, 0.5, "\n".join(top_words), transform=ia.transAxes, fontsize=8,
                    va="center")
        ia.set_title(f"Topic {idx} (cluster {cluster_labels[idx]})", fontsize=8)
        fig.canvas.draw_idle()

    cursor = mplcursors.cursor(scatter, hover=False)
    cursor.connect("add", _on_click)

    log.info("Interactive PHATE figure created; use plt.show() to display.")
    return fig


# ---------------------------------------------------------------------------
# Interactive co-author network (Plotly)
# ---------------------------------------------------------------------------

def interactive_coauthor_network(
    coauthor_graph,
    author_meta_distns: "dict[int, ndarray]",
    link_scores: "list[tuple[frozenset, float]]",
    author_names: "Optional[dict[int, list[str]]]" = None,
    max_nodes: int = 300,
    title: str = "Interactive co-author network",
) -> "object":
    """Interactive co-author network using Plotly.

    Node colour/size encodes author interdisciplinarity (entropy of meta-topic
    distribution).  Edge colour encodes link surprise score (lower = more
    surprising).  Hovering a node shows the author name and top meta-topics.

    Args:
        coauthor_graph:    NetworkX Graph.
        author_meta_distns: author_id → meta-topic distribution.
        link_scores:        Sorted list from ``score_interdisciplinarity_links``.
        author_names:       Optional mapping author_id → list of name strings.
        max_nodes:          Cap on number of nodes displayed.
        title:              Plot title.

    Returns:
        plotly.graph_objects.Figure, or None if plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import]
        import networkx as nx
    except ImportError as exc:
        log.warning("plotly or networkx not installed; interactive network unavailable. (%s)", exc)
        return None

    # Compute entropy scores for sizing
    def _entropy(d: ndarray) -> float:
        d = d.astype(np.float32)
        return float(-np.sum(d * np.log2(d + 1e-10)))

    scores = {aid: _entropy(dist) for aid, dist in author_meta_distns.items()}
    top_authors = sorted(scores, key=scores.get, reverse=True)[:max_nodes]
    top_set     = set(top_authors)

    sub_G = coauthor_graph.subgraph([n for n in coauthor_graph.nodes() if n in top_set])
    pos   = nx.spring_layout(sub_G, seed=42)

    # Edge traces
    link_dict = {fs: score for fs, score in link_scores}
    edge_traces = []
    for u, v in sub_G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lscore  = link_dict.get(frozenset({u, v}), 0.0)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.0, color=f"rgba(0,0,200,{max(0.1, 1.0 - lscore):.2f})"),
                hoverinfo="none",
            )
        )

    # Node trace
    node_x = [pos[n][0] for n in sub_G.nodes()]
    node_y = [pos[n][1] for n in sub_G.nodes()]
    node_scores = [scores.get(n, 0.0) for n in sub_G.nodes()]

    def _label(n: int) -> str:
        names = (author_names or {}).get(n, [str(n)])
        name  = names[0] if names else str(n)
        dist  = author_meta_distns.get(n)
        top_t = ""
        if dist is not None:
            top_t_idx = np.argsort(dist)[-3:][::-1]
            top_t = f"<br>Top meta-topics: {list(top_t_idx)}"
        return f"{name}<br>Entropy: {scores.get(n, 0.0):.3f}{top_t}"

    node_hover = [_label(n) for n in sub_G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_hover,
        marker=dict(
            showscale=True,
            colorscale="YlOrRd",
            color=node_scores,
            size=[max(5, min(30, 5 + 20 * s / max(node_scores or [1]))) for s in node_scores],
            colorbar=dict(title="Interdisciplinarity"),
            line_width=1,
        ),
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    log.info("Interactive Plotly network figure created.")
    return fig
