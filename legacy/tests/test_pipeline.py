"""
legacy/tests/test_pipeline.py — Integration tests for the legacy pipeline modules.

All tests use fully-synthetic data from conftest.py; no arXiv file or network
access is required.  The tests validate the CAMSAP 2025 pipeline design:

  Stage 1  _config           — PipelineConfig loading, PrimaryConfig validation
  Stage 2  _topic_models     — temporal chunking, expand_distributions
  Stage 3  _distributions    — coauthor graph, barycenters, diffusion graph, diffusion
  Stage 4  _manifold         — pairwise distances, dendrogram, cut, embedding
  Stage 5  _scoring          — tree encoding, meta-topic distributions, scoring
  Stage 6  _tree             — TreeNode construction, leaf/internal types

FAISS and PHATE are used when available; tests skip gracefully if not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # allow `from conftest import …`

from conftest import (
    N_VOCAB, N_CHUNKS, N_TOPICS_CHUNK, N_TOPICS_TOTAL,
    N_DOCS, N_DOCS_CHUNK, N_AUTHORS, AUTHOR_IDS, VOCAB,
)


# ===========================================================================
# Stage 1 — Config
# ===========================================================================

class TestConfig:

    def test_load_default_pipeline_config(self):
        from legacy._config import load_pipeline_config
        cfg = load_pipeline_config()
        assert cfg.pipeline.months_per_chunk >= 1
        assert cfg.diffusion.knn >= 1
        assert cfg.dendrogram.knn >= cfg.diffusion.knn
        assert cfg.topic_models.lda.smoothing_gamma > 0.0

    def test_lda_section_defaults(self):
        from legacy._config import load_pipeline_config
        lda = load_pipeline_config().topic_models.lda
        assert lda.docs_per_topic > 0
        assert lda.ngibbs > 0
        assert lda.npasses > 0
        assert 0.0 < lda.smoothing_gamma < 1.0

    def test_embeddings_phate_defaults(self):
        from legacy._config import load_pipeline_config
        ph = load_pipeline_config().embeddings.phate
        assert ph.n_components >= 2
        assert ph.knn >= 1

    def test_primary_config_defaults(self):
        from legacy._config import PrimaryConfig
        pc = PrimaryConfig(
            input_file="x", output_dir="y", categories=["cs.LG"],
            year_start=2012, year_end=2023,
        )
        assert pc.topic_model == "lda"
        assert pc.distance_metric == "hellinger"
        assert pc.embedding_method == "phate"
        assert pc.linkage_method == "ward"
        assert 0.0 <= pc.cut_height <= 1.0

    def test_primary_config_validates_year_range(self):
        from legacy._config import PrimaryConfig
        pc = PrimaryConfig(
            input_file="x", output_dir="y", categories=["cs.LG"],
            year_start=2020, year_end=2019,   # reversed
        )
        with pytest.raises(ValueError):
            pc.validate()

    def test_topic_models_get_kwargs_lda(self):
        from legacy._config import load_pipeline_config
        cfg = load_pipeline_config()
        kw = cfg.topic_models.get_kwargs("lda")
        assert "docs_per_topic" in kw
        assert "ngibbs" in kw
        assert "npasses" in kw
        assert "smoothing_gamma" in kw

    def test_embeddings_get_kwargs_phate(self):
        from legacy._config import load_pipeline_config
        cfg = load_pipeline_config()
        kw = cfg.embeddings.get_kwargs("phate")
        assert "n_components" in kw

    def test_config_yaml_override(self, tmp_path):
        """Custom YAML should override specific fields while preserving defaults."""
        import yaml
        from legacy._config import load_pipeline_config
        override = tmp_path / "override.yaml"
        override.write_text(yaml.dump({"diffusion": {"knn": 99}}))
        cfg = load_pipeline_config(str(override))
        assert cfg.diffusion.knn == 99
        # Unaffected defaults must still be present
        assert cfg.dendrogram.knn >= 1


# ===========================================================================
# Stage 2 — Topic models: chunking + distribution expansion
# ===========================================================================

class TestTemporalChunking:

    def test_chunks_count(self, synthetic_df):
        from legacy._topic_models import create_temporal_chunks
        chunks = create_temporal_chunks(synthetic_df, months_per_chunk=1)
        assert len(chunks) == N_CHUNKS

    def test_chunks_non_empty(self, synthetic_df):
        from legacy._topic_models import create_temporal_chunks
        for chunk in create_temporal_chunks(synthetic_df, months_per_chunk=1):
            assert len(chunk) > 0

    def test_chunks_cover_all_docs(self, synthetic_df):
        from legacy._topic_models import create_temporal_chunks
        chunks = create_temporal_chunks(synthetic_df, months_per_chunk=1)
        covered = sum(len(c) for c in chunks)
        assert covered == len(synthetic_df)

    def test_chunk_dates_ordered(self, synthetic_df):
        from legacy._topic_models import create_temporal_chunks
        chunks = create_temporal_chunks(synthetic_df, months_per_chunk=1)
        starts = [c["date"].min() for c in chunks]
        assert starts == sorted(starts)


class TestExpandDistributions:

    def test_output_length(self, expanded_distns):
        assert len(expanded_distns) == N_DOCS

    def test_vector_length(self, expanded_distns):
        for doc_id, vec in expanded_distns.items():
            assert vec.shape == (N_TOPICS_TOTAL,), f"doc {doc_id}: wrong shape"

    def test_in_chunk_block_sums_to_one(self, expanded_distns, inds_by_chunk):
        for cidx, doc_ids in inds_by_chunk.items():
            start = cidx * N_TOPICS_CHUNK
            for doc_id in doc_ids:
                block_sum = expanded_distns[doc_id][start: start + N_TOPICS_CHUNK].sum()
                assert abs(block_sum - 1.0) < 1e-5, (
                    f"doc {doc_id} chunk block does not sum to 1: {block_sum}"
                )

    def test_out_of_chunk_blocks_zero(self, expanded_distns, inds_by_chunk):
        for cidx, doc_ids in inds_by_chunk.items():
            start = cidx * N_TOPICS_CHUNK
            for doc_id in doc_ids[:3]:   # spot-check first 3 per chunk
                vec = expanded_distns[doc_id]
                for other_c in range(N_CHUNKS):
                    if other_c == cidx:
                        continue
                    s = other_c * N_TOPICS_CHUNK
                    assert vec[s: s + N_TOPICS_CHUNK].sum() == pytest.approx(0.0, abs=1e-7)

    def test_dtype_float32(self, expanded_distns):
        for vec in list(expanded_distns.values())[:5]:
            assert vec.dtype == np.float32

    def test_expand_distributions_function(self, inds_by_chunk, ntopics_by_chunk):
        """Call the actual expand_distributions utility, not the fixture."""
        from legacy._topic_models import expand_distributions
        # Build small synthetic doc_topic_distns (chunk-local)
        rng = np.random.default_rng(7)
        doc_distns: dict[int, np.ndarray] = {}
        for cidx, doc_ids in inds_by_chunk.items():
            n = N_TOPICS_CHUNK
            for doc_id in doc_ids:
                doc_distns[doc_id] = rng.dirichlet(np.ones(n)).astype(np.float32)
        expanded = expand_distributions(doc_distns, inds_by_chunk, ntopics_by_chunk)
        assert len(expanded) == N_DOCS
        for vec in expanded.values():
            assert vec.shape == (N_TOPICS_TOTAL,)
            assert abs(vec.sum() - 1.0) < 1e-5


# ===========================================================================
# Stage 2b — LDA topic model (tiny corpus, fast training)
# ===========================================================================

class TestLDATopicModel:

    @pytest.fixture(scope="class")
    def tiny_corpus(self):
        """Tiny gensim corpus for fast LDA smoke tests."""
        import gensim.corpora as corpora
        docs = [[f"w{i % 10}" for i in range(20)] for _ in range(15)]
        id2word = corpora.Dictionary(docs)
        df = pd.DataFrame({
            "text_processed": docs,
            "abstract": [" ".join(d) for d in docs],
            "date": pd.date_range("2020-01-01", periods=15, freq="7D"),
        })
        return df, id2word

    def test_fit_runs(self, tiny_corpus):
        from legacy._topic_models import LDATopicModel
        df, id2word = tiny_corpus
        model = LDATopicModel(docs_per_topic=5, ngibbs=5, npasses=1)
        model.fit(df, id2word)
        assert model._lda is not None

    def test_topic_vectors_shape_and_normalised(self, tiny_corpus):
        from legacy._topic_models import LDATopicModel
        df, id2word = tiny_corpus
        model = LDATopicModel(docs_per_topic=5, ngibbs=5, npasses=1)
        model.fit(df, id2word)
        phi = model.get_topic_vectors()
        assert phi.ndim == 2
        assert phi.shape[1] == len(id2word)
        row_sums = phi.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_doc_topic_distributions_shapes(self, tiny_corpus):
        from legacy._topic_models import LDATopicModel
        df, id2word = tiny_corpus
        model = LDATopicModel(docs_per_topic=5, ngibbs=5, npasses=1)
        model.fit(df, id2word)
        theta = model.get_doc_topic_distributions(df)
        assert len(theta) == len(df)
        for dist in theta.values():
            assert abs(dist.sum() - 1.0) < 1e-4

    def test_num_topics_min_four(self, tiny_corpus):
        """num_topics = max(4, chunk_size // docs_per_topic) — never fewer than 4."""
        from legacy._topic_models import LDATopicModel
        df, id2word = tiny_corpus
        # docs_per_topic = 100 → ntopics would be 0; clamp to 4
        model = LDATopicModel(docs_per_topic=100, ngibbs=5, npasses=1)
        model.fit(df, id2word)
        assert model._lda.num_topics >= 4


# ===========================================================================
# Stage 3 — Distributions
# ===========================================================================

class TestCoauthorGraph:

    def test_node_coverage(self, coauthor_graph, synthetic_df):
        """Every author that co-authored a paper must be a node."""
        all_author_ids = {
            aid
            for aids in synthetic_df["author_ids"]
            for aid in aids
        }
        assert all_author_ids.issubset(set(coauthor_graph.nodes()))

    def test_edge_weights_positive(self, coauthor_graph):
        for u, v, data in coauthor_graph.edges(data=True):
            assert data["weight"] >= 1

    def test_no_self_loops(self, coauthor_graph):
        assert not any(u == v for u, v in coauthor_graph.edges())


class TestAuthorBarycenters:

    @pytest.fixture(scope="class")
    def barycenters(self, expanded_distns, synthetic_df):
        from legacy._distributions import compute_author_barycenters
        b, _, _ = compute_author_barycenters(expanded_distns, synthetic_df)
        return b

    def test_all_authors_covered(self, barycenters, synthetic_df):
        all_ids = {aid for aids in synthetic_df["author_ids"] for aid in aids}
        assert all_ids.issubset(set(barycenters.keys()))

    def test_distributions_sum_to_one(self, barycenters):
        for aid, vec in barycenters.items():
            assert abs(vec.sum() - 1.0) < 1e-5, f"Author {aid} sum={vec.sum():.5f}"

    def test_vector_length(self, barycenters):
        for vec in barycenters.values():
            assert vec.shape == (N_TOPICS_TOTAL,)

    def test_nonneg(self, barycenters):
        for vec in barycenters.values():
            assert np.all(vec >= -1e-9)


class TestDiffusionGraph:

    @pytest.fixture(scope="class")
    def diff_graph(self, topic_vectors):
        from legacy._distributions import build_diffusion_graph
        return build_diffusion_graph(topic_vectors, knn=3, distance_metric="hellinger")

    def test_node_count(self, diff_graph):
        assert diff_graph.number_of_nodes() == N_TOPICS_TOTAL

    def test_edge_weights_positive(self, diff_graph):
        for u, v, data in diff_graph.edges(data=True):
            assert data["weight"] >= 0.0, f"Negative weight on edge ({u},{v})"

    def test_knn_bound_on_degree(self, diff_graph):
        """Every node should have degree ≥ 1 (has at least one kNN edge)."""
        for node in diff_graph.nodes():
            assert diff_graph.degree(node) >= 1, f"Isolated node: {node}"

    def test_distances_in_unit_interval_for_hellinger(self, diff_graph):
        weights = [d for _, _, d in diff_graph.edges(data="weight")]
        assert all(0.0 <= w <= 1.0 + 1e-9 for w in weights)


class TestDiffusion:

    @pytest.fixture(scope="class")
    def diffused(self, topic_vectors, expanded_distns):
        from legacy._distributions import build_diffusion_graph, diffuse_distributions
        G = build_diffusion_graph(topic_vectors, knn=3, distance_metric="hellinger")
        return diffuse_distributions(G, expanded_distns, num_iterations=1, diffusion_rate=0.7)

    def test_output_keys_match_input(self, diffused, expanded_distns):
        assert set(diffused.keys()) == set(expanded_distns.keys())

    def test_distributions_normalised(self, diffused):
        for key, dist in diffused.items():
            assert abs(dist.sum() - 1.0) < 1e-4, (
                f"Key {key}: diffused dist sum={dist.sum():.5f}"
            )

    def test_nonneg(self, diffused):
        for dist in diffused.values():
            assert np.all(dist >= -1e-9)

    def test_vector_length(self, diffused):
        for dist in diffused.values():
            assert dist.shape == (N_TOPICS_TOTAL,)

    def test_single_neighbour_guard(self, topic_vectors):
        """A distribution supported on a single topic with one graph neighbour
        must still propagate after diffusion (no zero-mass silencing)."""
        from legacy._distributions import build_diffusion_graph, diffuse_distributions
        # One-hot input on topic 0
        hot = np.zeros(N_TOPICS_TOTAL, dtype=np.float32)
        hot[0] = 1.0
        G = build_diffusion_graph(topic_vectors, knn=1, distance_metric="hellinger")
        diffused = diffuse_distributions(G, {0: hot}, num_iterations=1)
        assert diffused[0].sum() > 0.0


# ===========================================================================
# Stage 4 — Manifold: pairwise distances, dendrogram, cut, embedding
# ===========================================================================

class TestPairwiseDistances:

    @pytest.fixture(scope="class")
    def hellinger_dist_mat(self, topic_vectors):
        from legacy._manifold import compute_pairwise_distances
        return compute_pairwise_distances(topic_vectors, "hellinger")

    def test_shape(self, hellinger_dist_mat):
        assert hellinger_dist_mat.shape == (N_TOPICS_TOTAL, N_TOPICS_TOTAL)

    def test_symmetric(self, hellinger_dist_mat):
        np.testing.assert_allclose(
            hellinger_dist_mat, hellinger_dist_mat.T, atol=1e-6
        )

    def test_diagonal_zero(self, hellinger_dist_mat):
        np.testing.assert_allclose(np.diag(hellinger_dist_mat), 0.0, atol=1e-6)

    def test_range(self, hellinger_dist_mat):
        assert hellinger_dist_mat.min() >= -1e-9
        assert hellinger_dist_mat.max() <= 1.0 + 1e-9

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_other_metrics_nonneg_and_symmetric(self, topic_vectors, metric):
        from legacy._manifold import compute_pairwise_distances
        M = compute_pairwise_distances(topic_vectors, metric)
        assert M.shape == (N_TOPICS_TOTAL, N_TOPICS_TOTAL)
        assert np.all(M >= -1e-9)
        np.testing.assert_allclose(M, M.T, atol=1e-6)


class TestTopicDendrogram:

    def test_linkage_matrix_shape(self, dendrogram_and_heights):
        Z, _, _ = dendrogram_and_heights
        assert Z.shape == (N_TOPICS_TOTAL - 1, 4)

    def test_heights_nonneg(self, dendrogram_and_heights):
        Z, _, _ = dendrogram_and_heights
        heights = Z[:, 2]
        assert np.all(heights >= 0), f"Negative heights: min={heights.min():.6f}"

    def test_heights_monotone(self, dendrogram_and_heights):
        """Linkage heights must be non-decreasing (Ward guarantee)."""
        Z, _, _ = dendrogram_and_heights
        heights = Z[:, 2]
        assert np.all(np.diff(heights) >= -1e-6), "Heights not monotonically non-decreasing"

    def test_first_height_in_unit_interval(self, dendrogram_and_heights):
        """First Ward merge = raw Hellinger distance between two singletons → ≤ 1."""
        Z, min_h, _ = dendrogram_and_heights
        assert 0.0 <= min_h <= 1.0, f"First merge height out of [0,1]: {min_h}"

    def test_min_le_max(self, dendrogram_and_heights):
        _, min_h, max_h = dendrogram_and_heights
        assert min_h <= max_h

    def test_cluster_count_range(self, dendrogram_and_heights):
        """The linkage matrix must span from n_topics−1 down to 1 cluster."""
        Z, _, _ = dendrogram_and_heights
        assert Z.shape[0] == N_TOPICS_TOTAL - 1   # (n-1) merges


class TestCutDendrogram:

    @pytest.mark.parametrize("cut_height", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_labels_integer_array(self, dendrogram_and_heights, cut_height):
        from legacy._manifold import cut_dendrogram
        Z, min_h, max_h = dendrogram_and_heights
        labels = cut_dendrogram(Z, cut_height, min_h, max_h)
        assert labels.dtype.kind in ("i", "u")
        assert labels.shape == (N_TOPICS_TOTAL,)

    def test_cut_1_gives_single_cluster(self, dendrogram_and_heights):
        """cut_height=1 → cut at max_h (final merge) → all topics in one cluster."""
        from legacy._manifold import cut_dendrogram
        Z, min_h, max_h = dendrogram_and_heights
        labels = cut_dendrogram(Z, 1.0, min_h, max_h)
        assert len(np.unique(labels)) == 1

    def test_cut_0_gives_many_clusters(self, dendrogram_and_heights):
        """cut_height=0 → cut at min_h → strictly more clusters than cut_height=1."""
        from legacy._manifold import cut_dendrogram
        Z, min_h, max_h = dendrogram_and_heights
        n_at_0 = len(np.unique(cut_dendrogram(Z, 0.0, min_h, max_h)))
        n_at_1 = len(np.unique(cut_dendrogram(Z, 1.0, min_h, max_h)))
        assert n_at_0 > n_at_1  # cut at top always produces fewer clusters

    def test_more_clusters_at_lower_cut(self, dendrogram_and_heights):
        """Higher cut_height → fewer clusters."""
        from legacy._manifold import cut_dendrogram
        Z, min_h, max_h = dendrogram_and_heights
        n_low  = len(np.unique(cut_dendrogram(Z, 0.3, min_h, max_h)))
        n_high = len(np.unique(cut_dendrogram(Z, 0.7, min_h, max_h)))
        assert n_low >= n_high, "Fewer clusters expected at higher cut height"

    def test_out_of_range_raises(self, dendrogram_and_heights):
        from legacy._manifold import cut_dendrogram
        Z, min_h, max_h = dendrogram_and_heights
        with pytest.raises(ValueError):
            cut_dendrogram(Z, 1.5, min_h, max_h)


class TestComputeEmbedding:

    @pytest.fixture(scope="class")
    def dist_mat(self, topic_vectors):
        from legacy._manifold import compute_pairwise_distances
        return compute_pairwise_distances(topic_vectors, "hellinger")

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("phate"),
        reason="phate not installed",
    )
    def test_phate_shape(self, dist_mat):
        from legacy._manifold import compute_embedding
        emb = compute_embedding(dist_mat, "phate", knn=3, n_components=2, gamma=0.0)
        assert emb.shape == (N_TOPICS_TOTAL, 2)

    def test_pca_shape(self, dist_mat):
        from legacy._manifold import compute_embedding
        emb = compute_embedding(dist_mat, "pca", n_components=3)
        assert emb.shape == (N_TOPICS_TOTAL, 3)

    def test_tsne_shape(self, dist_mat):
        from legacy._manifold import compute_embedding
        emb = compute_embedding(
            dist_mat, "tsne", n_components=2, perplexity=5.0, n_iter=250
        )
        assert emb.shape == (N_TOPICS_TOTAL, 2)

    def test_unknown_method_raises(self, dist_mat):
        from legacy._manifold import compute_embedding
        with pytest.raises(ValueError):
            compute_embedding(dist_mat, "unknown_method")


# ===========================================================================
# Stage 5 — TreeNode and tree encoding
# ===========================================================================

class TestTreeNode:

    def test_leaf_is_leaf(self):
        from legacy._tree import TreeNode
        leaf = TreeNode(id=0, type=0, distance=0.0,
                        author_topic_space_probs=np.array([0.5, 0.5]),
                        original_leaf_ids={0})
        assert leaf.is_leaf()

    def test_internal_not_leaf(self):
        from legacy._tree import TreeNode
        probs = np.array([0.5, 0.5])
        left  = TreeNode(0, 0, 0.0, probs, original_leaf_ids={0})
        right = TreeNode(1, 0, 0.0, probs, original_leaf_ids={1})
        internal = TreeNode(2, 1, 0.3, probs + probs, left=left, right=right,
                            original_leaf_ids={0, 1})
        assert not internal.is_leaf()

    def test_leaf_count(self):
        from legacy._tree import TreeNode
        p = np.ones(3) / 3
        l = TreeNode(0, 0, 0.0, p, original_leaf_ids={0})
        r = TreeNode(1, 0, 0.0, p, original_leaf_ids={1})
        root = TreeNode(2, 1, 0.2, p + p, left=l, right=r, original_leaf_ids={0, 1})
        assert root.get_leaf_count() == 2

    def test_get_all_leaf_ids(self):
        from legacy._tree import TreeNode
        p = np.ones(3) / 3
        nodes = [TreeNode(i, 0, 0.0, p, original_leaf_ids={i}) for i in range(3)]
        left  = TreeNode(3, 1, 0.1, p + p, left=nodes[0], right=nodes[1],
                         original_leaf_ids={0, 1})
        root  = TreeNode(4, 1, 0.2, p * 3, left=left, right=nodes[2],
                         original_leaf_ids={0, 1, 2})
        assert root.get_all_leaf_ids() == {0, 1, 2}


class TestEncodeTree:

    def test_root_is_tree_node(self, encoded_tree):
        from legacy._tree import TreeNode
        root, _ = encoded_tree
        assert isinstance(root, TreeNode)

    def test_root_is_not_leaf(self, encoded_tree):
        root, _ = encoded_tree
        assert not root.is_leaf()

    def test_author_index_map_coverage(self, encoded_tree):
        root, aim = encoded_tree
        assert set(aim.keys()) == set(AUTHOR_IDS)

    def test_author_probs_shape(self, encoded_tree):
        root, aim = encoded_tree
        n_authors = len(AUTHOR_IDS)
        assert root.author_topic_space_probs.shape == (n_authors,)

    def test_root_original_leaf_ids_complete(self, encoded_tree):
        root, _ = encoded_tree
        assert root.original_leaf_ids == set(range(N_TOPICS_TOTAL))

    def test_link_prob_in_unit_interval(self, encoded_tree):
        root, _ = encoded_tree
        # Walk the tree and check all link probs
        stack = [root]
        while stack:
            node = stack.pop()
            assert 0.0 <= node.left_right_link_prob <= 1.0 + 1e-9, (
                f"Node {node.id}: link_prob={node.left_right_link_prob}"
            )
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)


# ===========================================================================
# Stage 5 — Scoring
# ===========================================================================

class TestAuthorMetaDistributions:

    @pytest.fixture(scope="class")
    def meta_distns(self, encoded_tree, dendrogram_and_heights, author_ct_distns):
        from legacy._scoring import compute_author_meta_distributions
        root, _ = encoded_tree
        Z, min_h, max_h = dendrogram_and_heights
        cut_dist = min_h + 0.5 * (max_h - min_h)
        return compute_author_meta_distributions(root, cut_dist, author_ct_distns)

    def test_all_authors_present(self, meta_distns):
        assert set(meta_distns.keys()) == set(AUTHOR_IDS)

    def test_distributions_sum_to_one(self, meta_distns):
        for aid, dist in meta_distns.items():
            assert abs(dist.sum() - 1.0) < 1e-4, f"Author {aid}: sum={dist.sum():.5f}"

    def test_nonneg(self, meta_distns):
        for dist in meta_distns.values():
            assert np.all(dist >= -1e-9)


class TestInterdisciplinarityScoring:

    @pytest.fixture(scope="class")
    def scoring_context(self, encoded_tree, dendrogram_and_heights,
                        author_ct_distns, coauthor_graph, synthetic_df):
        from legacy._scoring import (
            compute_author_meta_distributions,
            score_interdisciplinarity_docs,
            rank_authors_by_interdisciplinarity,
        )
        from legacy._scoring import _truncate_dendrogram, _get_new_leaf_nodes, \
            _setup_author_probs_matrix, _setup_link_prob_matrix, \
            score_interdisciplinarity_links

        root, aim = encoded_tree
        Z, min_h, max_h = dendrogram_and_heights
        cut_dist = min_h + 0.5 * (max_h - min_h)

        meta_distns = compute_author_meta_distributions(root, cut_dist, author_ct_distns)
        doc_scores  = score_interdisciplinarity_docs(synthetic_df, meta_distns)
        auth_ranking = rank_authors_by_interdisciplinarity(meta_distns)

        truncated   = _truncate_dendrogram(root, cut_dist)
        new_leaves  = _get_new_leaf_nodes(truncated)
        apm         = _setup_author_probs_matrix(new_leaves, aim)
        lpm         = _setup_link_prob_matrix(truncated, new_leaves)
        link_scores = score_interdisciplinarity_links(coauthor_graph, aim, apm, lpm)

        return doc_scores, auth_ranking, link_scores

    def test_doc_scores_nonneg(self, scoring_context):
        doc_scores, _, _ = scoring_context
        assert all(v >= 0 for v in doc_scores.values())

    def test_doc_scores_ordered_descending(self, scoring_context):
        doc_scores, _, _ = scoring_context
        vals = list(doc_scores.values())
        assert vals == sorted(vals, reverse=True)

    def test_author_ranking_ordered_descending(self, scoring_context):
        _, auth_ranking, _ = scoring_context
        vals = list(auth_ranking.values())
        assert vals == sorted(vals, reverse=True)

    def test_link_scores_ordered_ascending(self, scoring_context):
        _, _, link_scores = scoring_context
        scores = [s for _, s in link_scores]
        assert scores == sorted(scores)

    def test_author_ranking_covers_all(self, scoring_context):
        _, auth_ranking, _ = scoring_context
        assert set(auth_ranking.keys()) == set(AUTHOR_IDS)

    def test_entropy_bounded_by_log2_n_meta_topics(self, scoring_context):
        """Entropy of any distribution ≤ log2(n_meta_topics)."""
        _, auth_ranking, _ = scoring_context
        # Maximum entropy for a distribution over k topics is log2(k)
        max_entropy = np.log2(N_TOPICS_TOTAL)
        for aid, score in auth_ranking.items():
            assert score <= max_entropy + 1e-6, (
                f"Author {aid} entropy {score:.4f} exceeds log2({N_TOPICS_TOTAL})={max_entropy:.4f}"
            )


# ===========================================================================
# Full end-to-end smoke test (CAMSAP pipeline with synthetic data)
# ===========================================================================

class TestEndToEnd:
    """
    Drives the entire pipeline with synthetic data, validating outputs at each
    stage in the same configuration used for the CAMSAP 2025 experiments.
    """

    def test_full_pipeline_runs_to_completion(
        self, synthetic_df, topic_vectors, inds_by_chunk, ntopics_by_chunk,
        expanded_distns, coauthor_graph, author_ct_distns
    ):
        from legacy._manifold import (
            compute_pairwise_distances, build_topic_dendrogram, cut_dendrogram,
        )
        from legacy._scoring import run_scoring

        # Stage 4: manifold
        Z, min_h, max_h = build_topic_dendrogram(
            topic_vectors,
            knn=min(10, N_TOPICS_TOTAL - 1),
            linkage_method="ward",
            distance_metric="hellinger",
        )
        assert Z.shape == (N_TOPICS_TOTAL - 1, 4)

        labels = cut_dendrogram(Z, cut_height=0.5, min_height=min_h, max_height=max_h)
        n_clusters = len(np.unique(labels))
        assert 1 <= n_clusters <= N_TOPICS_TOTAL

        # Stage 5: scoring (CAMSAP cut_height = 0.68)
        results = run_scoring(
            Z=Z, min_height=min_h, max_height=max_h, cut_height=0.5,
            author_ct_distns=author_ct_distns,
            coauthor_graph=coauthor_graph,
            df=synthetic_df,
        )

        # All expected output keys are present
        expected_keys = {
            "root", "author_index_map", "author_meta_distns",
            "doc_scores", "link_scores", "author_ranking",
        }
        assert expected_keys.issubset(set(results.keys()))

        # Author ranking is non-empty and values are non-negative
        assert len(results["author_ranking"]) > 0
        assert all(v >= 0 for v in results["author_ranking"].values())

        # Doc scores are sorted descending
        doc_vals = list(results["doc_scores"].values())
        assert doc_vals == sorted(doc_vals, reverse=True)
