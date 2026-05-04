"""
Integration tests for the MSTML pipeline.

These tests verify the three computational stages that were suspect after the refactor:
  - build_author_document_distributions
  - apply_diffusion
  - build_topic_manifold

They work by injecting synthetic data directly into the orchestrator (bypassing
the data-loading and preprocessing stages) and then running the real computation.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mstml.core import MstmlOrchestrator
from mstml._math_driver import hellinger


# ─────────────────────────────────────────────────────────────────────────────
# Constants for synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────
N_DOCS = 24
N_AUTHORS = 4
N_CHUNKS = 3
N_TOPICS_PER_CHUNK = 5          # 15 total topics
N_WORDS = 50
AUTHOR_IDS = [1001, 1002, 1003, 1004]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_orch(tmp_path):
    """Minimal orchestrator backed by a temporary directory."""
    return MstmlOrchestrator(
        dataset_name="test",
        experiment_name="integration",
        experiment_directory=str(tmp_path / "exp"),
    )


@pytest.fixture
def orch_with_synthetic_data(tmp_orch):
    """
    Inject fully-formed synthetic pipeline state into the orchestrator.

    Simulates the outputs of: load_raw_data → apply_data_filters →
    preprocess_text → apply_author_disambiguation → setup_coauthor_network →
    create_temporal_chunks → train_chunk_models
    """
    rng = np.random.default_rng(42)
    orch = tmp_orch

    # ── documents_df ──────────────────────────────────────────────────────────
    docs_per_chunk = N_DOCS // N_CHUNKS
    doc_ids = list(range(N_DOCS))

    # Assign 1-2 authors per document
    doc_author_ids = [
        list(rng.choice(AUTHOR_IDS, size=int(rng.integers(1, 3)), replace=False).tolist())
        for _ in doc_ids
    ]

    orch.documents_df = pd.DataFrame(
        {
            "title": [f"Doc {i}" for i in doc_ids],
            "date": pd.date_range("2017-01-01", periods=N_DOCS, freq="7D"),
            "raw_text": [f"synthetic abstract {i}" for i in doc_ids],
            "author_names": [[f"Author {aid}"] for aid in [a[0] for a in doc_author_ids]],
            "author_ids": doc_author_ids,
            "preprocessed_text": [["word"] * 5 for _ in doc_ids],
            "categories": [["cs.LG"] for _ in doc_ids],
        },
        index=doc_ids,
    )

    # ── chunk_topic_models / chunk_topics / topic_vectors ────────────────────
    chunk_topic_models = []
    chunk_topics = []
    topic_vectors = []

    for c in range(N_CHUNKS):
        chunk_doc_ids = doc_ids[c * docs_per_chunk : (c + 1) * docs_per_chunk]
        n_docs_in_chunk = len(chunk_doc_ids)

        # Doc-topic distributions (theta): each row sums to 1
        doc_topic_distns = rng.dirichlet(
            [1.0] * N_TOPICS_PER_CHUNK, size=n_docs_in_chunk
        ).astype(np.float32)

        chunk_topic_models.append(
            {
                "chunk_id": c,
                "document_indices": chunk_doc_ids,
                "document_topic_distributions": doc_topic_distns,
                "num_topics": N_TOPICS_PER_CHUNK,
            }
        )

        # Topic-word distributions (phi): each row sums to 1
        phi = rng.dirichlet([0.5] * N_WORDS, size=N_TOPICS_PER_CHUNK).astype(np.float32)
        for t in range(N_TOPICS_PER_CHUNK):
            chunk_topics.append({"chunk_id": c, "topic_id": t, "vector": phi[t]})
            topic_vectors.append(phi[t])

    orch.chunk_topic_models = chunk_topic_models
    orch.chunk_topics = chunk_topics
    orch.topic_vectors = np.array(topic_vectors, dtype=np.float32)  # (15, 50)

    # ── co-author network ─────────────────────────────────────────────────────
    G = nx.Graph()
    G.add_nodes_from(str(aid) for aid in AUTHOR_IDS)
    # Simple chain: 1001-1002-1003-1004
    for i in range(len(AUTHOR_IDS) - 1):
        G.add_edge(str(AUTHOR_IDS[i]), str(AUTHOR_IDS[i + 1]))
    orch.coauthor_network = G

    # ── mark upstream stages complete ─────────────────────────────────────────
    for stage in [
        "load_raw_data",
        "apply_data_filters",
        "preprocess_text",
        "apply_author_disambiguation",
        "setup_coauthor_network",
        "create_temporal_chunks",
        "train_chunk_models",
    ]:
        orch.method_completion[stage] = True

    return orch


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check: FAISS Hellinger distance formula
# ─────────────────────────────────────────────────────────────────────────────

class TestFaissHellingerConversion:
    """Verify that the corrected FAISS conversion matches direct Hellinger computation."""

    def test_faiss_squared_l2_to_hellinger(self, orch_with_synthetic_data):
        """
        IndexFlatL2 on sqrt(X) returns squared L2 distances.
        For probability distributions p, q:
            FAISS dist = ||√p − √q||² = 2 · H(p, q)²
        So:  H(p, q) = sqrt(FAISS_dist / 2)
        """
        import faiss

        topic_matrix = orch_with_synthetic_data.topic_vectors  # (15, 50)
        n_topics = topic_matrix.shape[0]

        sqrt_matrix = np.sqrt(topic_matrix).astype(np.float32)
        index = faiss.IndexFlatL2(sqrt_matrix.shape[1])
        index.add(sqrt_matrix)

        # Search for 2 nearest: index 0 = self (distance 0), index 1 = closest other
        faiss_sq_l2, nn_indices = index.search(sqrt_matrix, 2)

        # Corrected conversion
        faiss_hellinger = np.sqrt(faiss_sq_l2[:, 1] / 2.0)

        # Direct computation
        direct_hellinger = np.array(
            [hellinger(topic_matrix[i], topic_matrix[nn_indices[i, 1]])
             for i in range(n_topics)],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            faiss_hellinger, direct_hellinger, atol=1e-4,
            err_msg="FAISS sqrt(dist/2) does not match direct Hellinger",
        )

    def test_old_formula_would_be_wrong(self, orch_with_synthetic_data):
        """Show that the old formula (dist / sqrt(2)) gives the wrong answer."""
        import faiss

        topic_matrix = orch_with_synthetic_data.topic_vectors
        n_topics = topic_matrix.shape[0]

        sqrt_matrix = np.sqrt(topic_matrix).astype(np.float32)
        index = faiss.IndexFlatL2(sqrt_matrix.shape[1])
        index.add(sqrt_matrix)
        faiss_sq_l2, nn_indices = index.search(sqrt_matrix, 2)

        old_formula = faiss_sq_l2[:, 1] / np.sqrt(2)   # sqrt(2) * H²  (wrong)
        correct     = np.sqrt(faiss_sq_l2[:, 1] / 2.0) # H             (right)
        direct      = np.array(
            [hellinger(topic_matrix[i], topic_matrix[nn_indices[i, 1]])
             for i in range(n_topics)]
        )

        # The old formula should disagree with direct Hellinger significantly
        old_errors = np.abs(old_formula - direct)
        new_errors = np.abs(correct - direct)

        assert np.mean(new_errors) < np.mean(old_errors), (
            "New formula should be more accurate than old formula"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stage integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAuthorDocumentDistributions:

    def test_expanded_distributions_shape(self, orch_with_synthetic_data):
        orch = orch_with_synthetic_data
        orch.build_author_document_distributions(overwrite=True)

        total_topics = N_CHUNKS * N_TOPICS_PER_CHUNK  # 15

        assert len(orch.expanded_doc_topic_distns) == N_DOCS
        for doc_id, dist in orch.expanded_doc_topic_distns.items():
            assert dist.shape == (total_topics,), f"Wrong shape for doc {doc_id}"
            # Only the chunk this doc belongs to has non-zero entries;
            # the distribution within that block should sum to 1.
            assert abs(dist.sum() - 1.0) < 1e-3, f"Doc {doc_id} dist doesn't sum to 1"

    def test_author_barycenters_shape(self, orch_with_synthetic_data):
        orch = orch_with_synthetic_data
        orch.build_author_document_distributions(overwrite=True)

        total_topics = N_CHUNKS * N_TOPICS_PER_CHUNK

        assert len(orch.author_topic_barycenters) > 0
        for auth_id, bary in orch.author_topic_barycenters.items():
            assert bary.shape == (total_topics,), f"Wrong shape for author {auth_id}"
            assert bary.sum() > 0, f"Author {auth_id} barycenter is all zeros"

    def test_all_authors_covered(self, orch_with_synthetic_data):
        orch = orch_with_synthetic_data
        orch.build_author_document_distributions(overwrite=True)

        # Every author that appears in the doc corpus must have a barycenter
        all_doc_authors = set()
        for aids in orch.documents_df["author_ids"]:
            all_doc_authors.update(aids)

        for aid in all_doc_authors:
            assert aid in orch.author_topic_barycenters, (
                f"Author {aid} missing from barycenters"
            )


class TestApplyDiffusion:

    def _run_up_to_diffusion(self, orch):
        orch.build_author_document_distributions(overwrite=True)
        orch.apply_diffusion(overwrite=True)
        return orch

    def test_diffused_distributions_shape(self, orch_with_synthetic_data):
        orch = self._run_up_to_diffusion(orch_with_synthetic_data)

        total_topics = N_CHUNKS * N_TOPICS_PER_CHUNK

        assert len(orch.author_ct_distns) > 0
        for auth_id, dist in orch.author_ct_distns.items():
            assert dist.shape == (total_topics,), f"Wrong shape for author {auth_id}"

    def test_diffused_distributions_normalized(self, orch_with_synthetic_data):
        orch = self._run_up_to_diffusion(orch_with_synthetic_data)

        for auth_id, dist in orch.author_ct_distns.items():
            assert abs(dist.sum() - 1.0) < 1e-3, (
                f"Author {auth_id} diffused dist doesn't sum to 1: sum={dist.sum():.4f}"
            )

    def test_diffusion_matrix_shape(self, orch_with_synthetic_data):
        orch = self._run_up_to_diffusion(orch_with_synthetic_data)

        total_topics = N_CHUNKS * N_TOPICS_PER_CHUNK
        assert orch.diffusion_matrix is not None
        assert orch.diffusion_matrix.shape == (total_topics, total_topics)

    def test_diffusion_preserves_known_entries(self, orch_with_synthetic_data):
        """Known non-zero topic entries should be preserved after diffusion."""
        orch = orch_with_synthetic_data
        orch.build_author_document_distributions(overwrite=True)

        # Snapshot barycenters before diffusion
        pre_bary = {
            k: v.copy() for k, v in orch.author_topic_barycenters.items()
        }

        orch.apply_diffusion(overwrite=True)

        for auth_id, orig in pre_bary.items():
            diffused = orch.author_ct_distns[auth_id]
            # Where the original was non-zero, diffused must also be non-zero
            orig_nonzero = orig > 0
            assert np.all(diffused[orig_nonzero] > 0), (
                f"Author {auth_id}: diffusion zeroed out originally non-zero entries"
            )


class TestBuildTopicManifold:

    def _run_full_pipeline(self, orch, knn=5):
        orch.build_author_document_distributions(overwrite=True)
        orch.apply_diffusion(overwrite=True)
        orch.build_topic_manifold(knn_neighbors=knn)
        return orch

    def test_linkage_matrix_shape(self, orch_with_synthetic_data):
        orch = self._run_full_pipeline(orch_with_synthetic_data)

        n_topics = N_CHUNKS * N_TOPICS_PER_CHUNK  # 15
        assert orch.topic_dendrogram_linkage is not None
        # scipy linkage produces (n-1, 4) matrix
        assert orch.topic_dendrogram_linkage.shape == (n_topics - 1, 4)

    def test_dendrogram_heights_are_non_negative(self, orch_with_synthetic_data):
        """
        Ward linkage heights represent cluster-merge variance costs, not raw
        pairwise distances.  The first merge (two singletons) equals the input
        Hellinger distance and is in [0, 1]; subsequent merges can exceed 1.0
        as the cluster sizes grow.  What we can always assert is non-negativity
        and the correct monotone ordering (tested separately).
        """
        orch = self._run_full_pipeline(orch_with_synthetic_data)

        heights = orch.topic_dendrogram_linkage[:, 2]
        assert np.all(heights >= 0), (
            f"Negative heights found: min={heights.min():.6f}"
        )
        # The very first merge is between two singletons whose Ward distance
        # equals the input Hellinger distance, so it must be <= 1.0.
        assert heights[0] <= 1.0, (
            f"First (smallest) Ward merge height exceeds 1.0: {heights[0]:.4f}; "
            f"input distances must be Hellinger"
        )

    def test_heights_monotonically_non_decreasing(self, orch_with_synthetic_data):
        """Hierarchical clustering linkage heights must be non-decreasing."""
        orch = self._run_full_pipeline(orch_with_synthetic_data)

        heights = orch.topic_dendrogram_linkage[:, 2]
        assert np.all(np.diff(heights) >= -1e-6), (
            "Linkage heights are not monotonically non-decreasing"
        )

    def test_min_max_cut_heights_stored(self, orch_with_synthetic_data):
        orch = self._run_full_pipeline(orch_with_synthetic_data)

        assert hasattr(orch, "min_cut_height")
        assert hasattr(orch, "max_cut_height")
        # min_cut_height is the first Ward merge = a raw Hellinger distance → [0, 1].
        # max_cut_height (final merge) can exceed 1.0 for Ward linkage.
        assert 0 <= orch.min_cut_height <= 1.0
        assert orch.min_cut_height <= orch.max_cut_height
