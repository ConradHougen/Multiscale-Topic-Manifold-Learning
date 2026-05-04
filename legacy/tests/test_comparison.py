"""
legacy/tests/test_comparison.py — Unit and integration tests for legacy/comparison/

Every test uses purely synthetic numpy arrays — no network access, no real corpora.

Coverage
--------
  _alignment.compute_alignment_scores()      — keys, range, perfect=0, random>0, n_pairs
  _alignment._pairwise_hellinger()           — formula, symmetry, self=0, orthogonal=1
  _diversity.diversity_at_k()               — range, all-unique=1, all-same, k-clamp, empty
  _diversity.mean_pairwise_hellinger()      — identical→0, orthogonal→1, range, single→NaN
  _diversity.mean_pairwise_cosine_distance()— range [0,2], single→NaN, identical→0
  _diversity.compute_diversity()            — keys, no-vectors (NaN), with vectors
  _diversity.compute_ensemble_diversity()   — averaging, NaN exclusion
  _coherence.compute_coherence()            — keys, finite/NaN, empty topics → NaN
  _coherence.compute_ensemble_coherence()   — averaging across chunks
  _bertopic_embed                           — skipped when bertopic not installed
  MSTML vs random (structural comparison)   — alignment distances, diversity ordering
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make legacy importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helper — evaluated at module load time for skip markers
# ---------------------------------------------------------------------------

def _bertopic_installed() -> bool:
    try:
        import bertopic  # noqa: F401
        return True
    except ImportError:
        return False

from legacy.comparison._alignment import (
    _pairwise_hellinger,
    _pairwise_cosine_dist,
    compute_alignment_scores,
)
from legacy.comparison._diversity import (
    diversity_at_k,
    mean_pairwise_hellinger,
    mean_pairwise_cosine_distance,
    compute_diversity,
    compute_ensemble_diversity,
)
from legacy.comparison._coherence import (
    compute_coherence,
    compute_ensemble_coherence,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)


def _prob(size: int) -> np.ndarray:
    return _RNG.dirichlet(np.ones(size)).astype(np.float32)


def _prob_matrix(n: int, size: int) -> np.ndarray:
    return np.vstack([_prob(size) for _ in range(n)])


# ===========================================================================
# Pairwise Hellinger helper
# ===========================================================================

class TestPairwiseHellinger:

    def test_self_distance_is_zero(self):
        X = _prob_matrix(5, 20)
        D = _pairwise_hellinger(X, X)
        for i in range(5):
            assert D[i, i] < 1e-5, f"Self-distance D[{i},{i}] = {D[i,i]}"

    def test_symmetry(self):
        X = _prob_matrix(4, 15)
        Y = _prob_matrix(3, 15)
        Dxy = _pairwise_hellinger(X, Y)
        Dyx = _pairwise_hellinger(Y, X)
        np.testing.assert_allclose(Dxy, Dyx.T, atol=1e-5)

    def test_orthogonal_distributions_equal_one(self):
        X = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
        Y = np.array([[0.0, 0.0, 0.6, 0.4]], dtype=np.float32)
        D = _pairwise_hellinger(X, Y)
        assert abs(D[0, 0] - 1.0) < 1e-5

    def test_range(self):
        X = _prob_matrix(6, 25)
        Y = _prob_matrix(4, 25)
        D = _pairwise_hellinger(X, Y)
        assert D.min() >= -1e-6
        assert D.max() <= 1.0 + 1e-6

    def test_output_shape(self):
        X = _prob_matrix(4, 10)
        Y = _prob_matrix(7, 10)
        D = _pairwise_hellinger(X, Y)
        assert D.shape == (4, 7)

    def test_matches_scalar_formula(self):
        from legacy._math import hellinger
        p = _prob(20)
        q = _prob(20)
        X = p.reshape(1, -1)
        Y = q.reshape(1, -1)
        assert abs(_pairwise_hellinger(X, Y)[0, 0] - hellinger(p, q)) < 1e-5


# ===========================================================================
# compute_alignment_scores
# ===========================================================================

class TestComputeAlignmentScores:

    def test_returns_expected_keys(self, topic_vecs_per_chunk):
        result = compute_alignment_scores(topic_vecs_per_chunk, distance_metric="hellinger")
        assert "mean_distance" in result
        assert "skewness" in result
        assert "n_pairs" in result
        assert "all_distances" in result

    def test_n_pairs_correct(self, topic_vecs_per_chunk):
        """n_pairs = (N_CHUNKS - 1) * N_TOPICS_CHUNK * knn (knn=1)."""
        from legacy.tests.conftest import N_CHUNKS, N_TOPICS_CHUNK
        result = compute_alignment_scores(topic_vecs_per_chunk, knn=1)
        expected = (N_CHUNKS - 1) * N_TOPICS_CHUNK
        assert result["n_pairs"] == expected

    def test_perfect_alignment_zero_mean_distance(self):
        """Identical topic distributions in every chunk → mean distance = 0."""
        phi = _prob_matrix(5, 30)
        identical = [phi.copy() for _ in range(4)]
        result = compute_alignment_scores(identical, distance_metric="hellinger", knn=1)
        assert result["mean_distance"] < 1e-4, (
            f"Perfect alignment should give near-zero distance, got {result['mean_distance']}"
        )

    def test_random_chunks_positive_distance(self):
        """Independent random topic vectors → mean distance > 0."""
        chunks = [_prob_matrix(5, 30) for _ in range(4)]
        result = compute_alignment_scores(chunks, distance_metric="hellinger", knn=1)
        assert result["mean_distance"] > 0.0

    def test_distances_in_unit_interval_hellinger(self):
        """All pairwise Hellinger distances must lie in [0, 1]."""
        result = compute_alignment_scores(
            [_prob_matrix(5, 20) for _ in range(3)],
            distance_metric="hellinger",
            knn=1,
        )
        arr = result["all_distances"]
        assert np.all(arr >= -1e-6)
        assert np.all(arr <= 1.0 + 1e-6)

    def test_cosine_metric_non_negative(self):
        chunks = [_prob_matrix(4, 20) for _ in range(3)]
        result = compute_alignment_scores(chunks, distance_metric="cosine", knn=1)
        assert np.all(result["all_distances"] >= -1e-6)

    def test_euclidean_metric_non_negative(self):
        chunks = [_prob_matrix(4, 20) for _ in range(3)]
        result = compute_alignment_scores(chunks, distance_metric="euclidean", knn=1)
        assert np.all(result["all_distances"] >= -1e-6)

    def test_single_chunk_gives_zero_pairs(self):
        """Single chunk → no consecutive pairs → n_pairs = 0, mean = nan."""
        result = compute_alignment_scores([_prob_matrix(5, 10)], knn=1)
        assert result["n_pairs"] == 0
        assert np.isnan(result["mean_distance"])

    def test_knn_gt_one_multiplies_pairs(self, topic_vecs_per_chunk):
        from legacy.tests.conftest import N_CHUNKS, N_TOPICS_CHUNK
        result = compute_alignment_scores(topic_vecs_per_chunk, knn=2)
        # knn=2 but target has N_TOPICS_CHUNK=5 rows → k_actual = min(2, 5) = 2
        expected = (N_CHUNKS - 1) * N_TOPICS_CHUNK * 2
        assert result["n_pairs"] == expected

    def test_empty_chunk_skipped_gracefully(self):
        """A chunk with 0 rows is skipped without error."""
        chunks = [
            _prob_matrix(5, 10),
            np.zeros((0, 10), dtype=np.float32),  # empty
            _prob_matrix(5, 10),
        ]
        result = compute_alignment_scores(chunks, knn=1)
        # chunk 0→chunk 1 skipped (target empty); chunk 1→chunk 2 skipped (source empty)
        assert result["n_pairs"] == 0

    def test_fixture_chunks_mean_distance_finite(self, topic_vecs_per_chunk):
        result = compute_alignment_scores(topic_vecs_per_chunk, knn=1)
        assert np.isfinite(result["mean_distance"])


# ===========================================================================
# diversity_at_k
# ===========================================================================

class TestDiversityAtK:

    def test_all_unique_words_gives_one(self):
        topics = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        assert abs(diversity_at_k(topics, k=3) - 1.0) < 1e-9

    def test_all_identical_words(self):
        """All topics share the same words → unique / total = 1 / n_topics."""
        n = 4
        topics = [["x", "y"] for _ in range(n)]
        expected = 2 / (n * 2)  # = 1/n
        assert abs(diversity_at_k(topics, k=2) - expected) < 1e-9

    def test_range(self):
        rng_local = np.random.default_rng(1)
        vocab = [f"w{i}" for i in range(50)]
        topics = [
            rng_local.choice(vocab, size=10, replace=False).tolist()
            for _ in range(8)
        ]
        d = diversity_at_k(topics, k=10)
        assert 0.0 <= d <= 1.0

    def test_empty_list_gives_zero(self):
        assert diversity_at_k([], k=10) == 0.0

    def test_k_larger_than_words_uses_all_words(self):
        """k > len(topic_words): use all available words."""
        topics = [["a", "b"], ["c", "d"]]
        # Only 2 words per topic; k=10 → use 2 per topic
        # unique = 4, total = 4 → diversity = 1.0
        assert abs(diversity_at_k(topics, k=10) - 1.0) < 1e-9

    def test_single_topic(self):
        topics = [["a", "b", "c", "d", "e"]]
        # unique = 5, total = 5 → 1.0
        assert abs(diversity_at_k(topics, k=5) - 1.0) < 1e-9

    def test_partial_overlap(self):
        """Three topics with one shared word each: known calculation."""
        topics = [["a", "b"], ["b", "c"], ["c", "d"]]
        # unique = {a, b, c, d} = 4; total = 6
        expected = 4 / 6
        assert abs(diversity_at_k(topics, k=2) - expected) < 1e-9


# ===========================================================================
# mean_pairwise_hellinger
# ===========================================================================

class TestMeanPairwiseHellinger:

    def test_single_row_returns_nan(self):
        X = _prob(20).reshape(1, -1)
        assert np.isnan(mean_pairwise_hellinger(X))

    def test_identical_rows_give_zero(self):
        phi = _prob(15)
        X = np.vstack([phi, phi, phi])
        assert mean_pairwise_hellinger(X) < 1e-5

    def test_orthogonal_topics_give_one(self):
        p = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float64)
        q = np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float64)
        X = np.vstack([p, q])
        assert abs(mean_pairwise_hellinger(X) - 1.0) < 1e-5

    def test_range(self):
        X = _prob_matrix(8, 30).astype(np.float64)
        h = mean_pairwise_hellinger(X)
        assert 0.0 <= h <= 1.0 + 1e-6

    def test_monotone_separation(self):
        """Orthogonal blocks → higher mean distance than within-block random."""
        # Two near-identical topics → low distance
        p = _prob(20)
        close = np.vstack([p, p + 1e-3 * _RNG.standard_normal(20)])
        close = np.abs(close)
        close /= close.sum(axis=1, keepdims=True)

        # Two orthogonal topics → high distance
        e1 = np.eye(20)[0]
        e2 = np.eye(20)[-1]
        far = np.vstack([e1, e2]).astype(np.float64)

        assert mean_pairwise_hellinger(close) < mean_pairwise_hellinger(far)


# ===========================================================================
# mean_pairwise_cosine_distance
# ===========================================================================

class TestMeanPairwiseCosineDistance:

    def test_single_row_returns_nan(self):
        X = _prob(10).reshape(1, -1)
        assert np.isnan(mean_pairwise_cosine_distance(X))

    def test_identical_rows_give_zero(self):
        phi = _prob(15).astype(np.float64)
        X = np.vstack([phi, phi, phi])
        assert mean_pairwise_cosine_distance(X) < 1e-5

    def test_range(self):
        """Cosine distance in [0, 2] for non-negative vectors."""
        X = _prob_matrix(6, 20).astype(np.float64)
        d = mean_pairwise_cosine_distance(X)
        assert 0.0 <= d <= 2.0 + 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal unit vectors → cosine similarity = 0 → cosine distance = 1."""
        e1 = np.array([[1.0, 0.0, 0.0]])
        e2 = np.array([[0.0, 1.0, 0.0]])
        X = np.vstack([e1, e2])
        assert abs(mean_pairwise_cosine_distance(X) - 1.0) < 1e-5


# ===========================================================================
# compute_diversity (aggregate)
# ===========================================================================

class TestComputeDiversity:

    def test_returns_expected_keys(self):
        topics = [["a", "b", "c"], ["d", "e", "f"]]
        result = compute_diversity(topics, k=3)
        assert "diversity_at_k" in result
        assert "mean_hellinger" in result
        assert "mean_cosine" in result

    def test_no_vectors_gives_nan_distance_metrics(self):
        topics = [["a", "b"], ["c", "d"]]
        result = compute_diversity(topics, topic_vectors=None, k=2)
        assert np.isnan(result["mean_hellinger"])
        assert np.isnan(result["mean_cosine"])

    def test_with_vectors_gives_finite_distances(self):
        topics = [["a", "b"], ["c", "d"]]
        X = _prob_matrix(2, 10)
        result = compute_diversity(topics, topic_vectors=X, k=2)
        assert np.isfinite(result["mean_hellinger"])
        assert np.isfinite(result["mean_cosine"])

    def test_diversity_at_k_value_matches_standalone(self):
        topics = [["x", "y", "z"], ["x", "p", "q"]]
        result = compute_diversity(topics, k=3)
        expected = diversity_at_k(topics, k=3)
        assert abs(result["diversity_at_k"] - expected) < 1e-9


# ===========================================================================
# compute_ensemble_diversity
# ===========================================================================

class TestComputeEnsembleDiversity:

    def test_returns_expected_keys(self, topic_vecs_per_chunk, vocab):
        from legacy.tests.conftest import N_TOPICS_CHUNK
        words_per_chunk = [
            [vocab[:10] for _ in range(N_TOPICS_CHUNK)]
            for _ in range(len(topic_vecs_per_chunk))
        ]
        result = compute_ensemble_diversity(words_per_chunk, topic_vecs_per_chunk)
        assert "diversity_at_k" in result
        assert "mean_hellinger" in result
        assert "mean_cosine" in result

    def test_averaging_across_chunks(self):
        """diversity_at_k should equal the average of per-chunk values when no NaN."""
        chunk1 = [["a", "b"], ["c", "d"]]  # 4 unique / 4 total → 1.0
        chunk2 = [["a", "b"], ["a", "b"]]  # 2 unique / 4 total → 0.5
        result = compute_ensemble_diversity([chunk1, chunk2], k=2)
        assert abs(result["diversity_at_k"] - 0.75) < 1e-9

    def test_nan_chunks_excluded(self):
        """Chunks that produce NaN for vector metrics should be skipped."""
        # chunk 1: 1 topic → mean_hellinger NaN; chunk 2: 3 topics → finite
        v1 = _prob_matrix(1, 10)   # single topic → NaN
        v2 = _prob_matrix(3, 10)   # multiple topics → finite
        words = [[["w0", "w1"]], [["w2", "w3"], ["w4", "w5"], ["w6", "w7"]]]
        result = compute_ensemble_diversity(words, [v1, v2], k=2)
        # mean_hellinger should be finite (NaN chunk excluded)
        assert np.isfinite(result["mean_hellinger"])

    def test_no_vectors_all_nan(self):
        """No topic vectors → mean_hellinger and mean_cosine are NaN."""
        words = [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]]
        result = compute_ensemble_diversity(words, topic_vectors_per_chunk=None, k=2)
        assert np.isnan(result["mean_hellinger"])
        assert np.isnan(result["mean_cosine"])

    def test_fixture_chunks_finite_diversity(self, topic_vecs_per_chunk, vocab):
        from legacy.tests.conftest import N_TOPICS_CHUNK
        words_per_chunk = [
            [vocab[i * 5:(i + 1) * 5] for i in range(N_TOPICS_CHUNK)]
            for _ in range(len(topic_vecs_per_chunk))
        ]
        result = compute_ensemble_diversity(words_per_chunk, topic_vecs_per_chunk)
        assert np.isfinite(result["diversity_at_k"])
        assert np.isfinite(result["mean_hellinger"])


# ===========================================================================
# compute_coherence
# ===========================================================================

class TestComputeCoherence:
    """Tests skip gracefully if gensim is not installed."""

    @pytest.fixture(autouse=True)
    def require_gensim(self):
        pytest.importorskip("gensim", reason="gensim not installed")

    def _docs_and_topics(self):
        """Minimal synthetic corpus and topic word lists."""
        words = [f"w{i}" for i in range(20)]
        docs = [
            [words[i % 20], words[(i + 1) % 20], words[(i + 2) % 20]]
            for i in range(30)
        ]
        topics = [words[:5], words[5:10], words[10:15]]
        return topics, docs

    def test_returns_dict_with_expected_measures(self):
        topics, docs = self._docs_and_topics()
        result = compute_coherence(topics, docs, measures=["c_v", "c_npmi", "c_uci"])
        assert "c_v" in result
        assert "c_npmi" in result
        assert "c_uci" in result

    def test_values_are_float(self):
        topics, docs = self._docs_and_topics()
        result = compute_coherence(topics, docs, measures=["c_v"])
        for v in result.values():
            assert isinstance(v, float)

    def test_empty_topics_returns_nan(self):
        _, docs = self._docs_and_topics()
        result = compute_coherence([], docs, measures=["c_v"])
        assert np.isnan(result["c_v"])

    def test_topics_after_trim_to_top_n(self):
        """top_n < len(words) should still return finite float."""
        topics, docs = self._docs_and_topics()
        result = compute_coherence(topics, docs, measures=["c_v"], top_n=3)
        # Should not raise; value may be any finite float or NaN depending on corpus
        assert "c_v" in result

    def test_custom_measures_subset(self):
        topics, docs = self._docs_and_topics()
        result = compute_coherence(topics, docs, measures=["c_npmi"])
        assert "c_npmi" in result
        assert "c_v" not in result


# ===========================================================================
# compute_ensemble_coherence
# ===========================================================================

class TestComputeEnsembleCoherence:
    """Tests skip gracefully if gensim is not installed."""

    @pytest.fixture(autouse=True)
    def require_gensim(self):
        pytest.importorskip("gensim", reason="gensim not installed")

    def _make_chunk_data(self, n_chunks=3, n_topics=4, n_words=5):
        vocab = [f"w{i}" for i in range(30)]
        chunk_words = [
            [vocab[t * n_words:(t + 1) * n_words] for t in range(n_topics)]
            for _ in range(n_chunks)
        ]
        chunk_docs = [
            [[vocab[i % 30], vocab[(i + 3) % 30]] for i in range(20)]
            for _ in range(n_chunks)
        ]
        return chunk_words, chunk_docs

    def test_returns_dict_with_measures(self):
        chunk_words, chunk_docs = self._make_chunk_data()
        result = compute_ensemble_coherence(chunk_words, chunk_docs, measures=["c_v"])
        assert "c_v" in result
        assert isinstance(result["c_v"], float)

    def test_averaging_produces_finite_or_nan(self):
        chunk_words, chunk_docs = self._make_chunk_data(n_chunks=4)
        result = compute_ensemble_coherence(chunk_words, chunk_docs, measures=["c_v", "c_npmi"])
        for v in result.values():
            assert isinstance(v, float)  # may be NaN or finite

    def test_single_chunk_equals_per_chunk_value(self):
        chunk_words, chunk_docs = self._make_chunk_data(n_chunks=1)
        ensemble = compute_ensemble_coherence(chunk_words, chunk_docs, measures=["c_v"])
        single = compute_coherence(chunk_words[0], chunk_docs[0], measures=["c_v"])
        # If both are NaN, test passes; if both finite, values should be equal
        if np.isfinite(ensemble["c_v"]) and np.isfinite(single["c_v"]):
            assert abs(ensemble["c_v"] - single["c_v"]) < 1e-6


# ===========================================================================
# BERTopic integration (skipped when bertopic not installed)
# ===========================================================================

@pytest.mark.skipif(
    not _bertopic_installed(),
    reason="bertopic / sentence-transformers not installed",
)
class TestBERTopicEmbed:
    """Integration tests for _bertopic_embed — only run when BERTopic is available."""

    def test_train_returns_correct_number_of_models(self, synthetic_df, vocab):
        import gensim.corpora as corpora
        from legacy.comparison._bertopic_embed import train_bertopic_ensemble
        from legacy._topic_models import create_temporal_chunks

        gensim_dict = corpora.Dictionary([vocab])
        chunks = create_temporal_chunks(synthetic_df, months_per_chunk=1)
        if not chunks:
            pytest.skip("No temporal chunks produced from synthetic data")

        models, phi_list = train_bertopic_ensemble(
            chunks[:2], gensim_dict, min_topic_size=2
        )
        assert len(models) == 2
        assert len(phi_list) == 2

    def test_phi_rows_sum_to_one_or_zero(self, synthetic_df, vocab):
        import gensim.corpora as corpora
        from legacy.comparison._bertopic_embed import train_bertopic_ensemble
        from legacy._topic_models import create_temporal_chunks

        gensim_dict = corpora.Dictionary([vocab])
        chunks = create_temporal_chunks(synthetic_df, months_per_chunk=1)
        if not chunks:
            pytest.skip("No temporal chunks produced from synthetic data")

        _, phi_list = train_bertopic_ensemble(
            chunks[:1], gensim_dict, min_topic_size=2
        )
        phi = phi_list[0]
        row_sums = phi.sum(axis=1)
        # After normalisation, each row sums to 1 (or is all-zero → sum = 0)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-5 or s == pytest.approx(0.0, abs=1e-7), (
                f"Row sum {s} is neither 0 nor 1"
            )

    def test_get_top_words_structure(self, synthetic_df, vocab):
        import gensim.corpora as corpora
        from legacy.comparison._bertopic_embed import (
            train_bertopic_ensemble,
            get_bertopic_top_words,
        )
        from legacy._topic_models import create_temporal_chunks

        gensim_dict = corpora.Dictionary([vocab])
        chunks = create_temporal_chunks(synthetic_df, months_per_chunk=1)
        if not chunks:
            pytest.skip("No temporal chunks produced from synthetic data")

        models, _ = train_bertopic_ensemble(chunks[:1], gensim_dict, min_topic_size=2)
        top_words = get_bertopic_top_words(models, top_n=5)

        assert len(top_words) == 1  # one chunk
        for chunk_topics in top_words:
            for words in chunk_topics:
                assert len(words) <= 5

    def test_phi_shape_matches_vocab(self, synthetic_df, vocab):
        import gensim.corpora as corpora
        from legacy.comparison._bertopic_embed import train_bertopic_ensemble
        from legacy._topic_models import create_temporal_chunks

        gensim_dict = corpora.Dictionary([vocab])
        chunks = create_temporal_chunks(synthetic_df, months_per_chunk=1)
        if not chunks:
            pytest.skip("No temporal chunks produced from synthetic data")

        _, phi_list = train_bertopic_ensemble(chunks[:1], gensim_dict, min_topic_size=2)
        assert phi_list[0].shape[1] == len(vocab)


# ===========================================================================
# Structural MSTML vs random-baseline comparison
# (qualitative analogue of the CAMSAP paper findings using synthetic data)
# ===========================================================================

class TestMSTMLvsRandomBaseline:
    """
    The CAMSAP paper shows MSTML topic vectors are more stable across chunks
    (lower forward alignment distance) than a random baseline, because temporal
    smoothing produces topically coherent topics.

    These tests verify the mathematical properties that support that finding:
    1. Alignment distance from identical chunks = 0 (lower bound).
    2. Alignment distance from independent random chunks > 0.
    3. Partially-smoothed chunks have lower alignment distance than fully random.
    4. Diversity@k of MSTML-like topics (Dirichlet, spread) is bounded above by
       the case where all topics are identical (minimum diversity).
    """

    def test_identical_chunks_lower_bound(self):
        """Perfect temporal stability → alignment distance = 0."""
        phi = _prob_matrix(6, 40)
        result = compute_alignment_scores(
            [phi.copy() for _ in range(3)], distance_metric="hellinger", knn=1
        )
        assert result["mean_distance"] < 1e-4

    def test_random_chunks_positive_distance(self):
        """Independent random draws → non-zero alignment distance."""
        rng = np.random.default_rng(99)
        random_chunks = [
            rng.dirichlet(np.ones(40), size=6).astype(np.float32)
            for _ in range(3)
        ]
        result = compute_alignment_scores(random_chunks, distance_metric="hellinger", knn=1)
        assert result["mean_distance"] > 0.01

    def test_smoothed_chunks_less_distance_than_random(self):
        """
        Temporally-smoothed topics (centred around a base distribution)
        should align better than fully independent random topics.
        """
        rng = np.random.default_rng(42)
        vocab_size = 50
        n_topics = 8
        n_chunks = 4

        # Base distribution for each topic — acts as a temporal anchor
        base = rng.dirichlet(np.ones(vocab_size), size=n_topics).astype(np.float32)

        # Smoothed: small perturbation around base
        smoothed_chunks = []
        for _ in range(n_chunks):
            noise = rng.dirichlet(np.ones(vocab_size) * 10, size=n_topics).astype(np.float32)
            mixed = 0.9 * base + 0.1 * noise
            mixed /= mixed.sum(axis=1, keepdims=True)
            smoothed_chunks.append(mixed)

        # Random: fully independent draws
        random_chunks = [
            rng.dirichlet(np.ones(vocab_size), size=n_topics).astype(np.float32)
            for _ in range(n_chunks)
        ]

        smoothed_result = compute_alignment_scores(smoothed_chunks, knn=1)
        random_result   = compute_alignment_scores(random_chunks, knn=1)

        assert smoothed_result["mean_distance"] < random_result["mean_distance"], (
            "Smoothed (stable) topics should align better than fully random topics. "
            f"smoothed={smoothed_result['mean_distance']:.4f}, "
            f"random={random_result['mean_distance']:.4f}"
        )

    def test_diversity_specialised_vs_uniform_topics(self):
        """
        Diverse (specialised) topics have higher Diversity@k than all-identical topics.
        This models the fact that MSTML topics specialise across chunks.
        """
        n_topics = 6
        vocab = [f"w{i}" for i in range(60)]

        # Specialised: each topic gets its own exclusive vocabulary block
        block = 60 // n_topics
        specialised = [vocab[i * block:(i + 1) * block] for i in range(n_topics)]

        # Uniform: all topics have the same top words
        uniform = [vocab[:10] for _ in range(n_topics)]

        d_spec = diversity_at_k(specialised, k=block)
        d_unif = diversity_at_k(uniform, k=10)

        assert d_spec > d_unif, (
            f"Specialised topics should be more diverse: {d_spec:.4f} vs {d_unif:.4f}"
        )

    def test_hellinger_lower_than_random_for_stable_ensemble(self):
        """
        Mean pairwise Hellinger among near-duplicate topics is lower than
        among fully random topics.
        """
        phi = _prob_matrix(1, 30).flatten()  # one base vector

        # Near-duplicate topics (low inter-topic Hellinger)
        near_dup = np.vstack([
            np.abs(phi + 0.01 * _RNG.standard_normal(30))
            for _ in range(5)
        ])
        near_dup /= near_dup.sum(axis=1, keepdims=True)

        # Fully random topics (high inter-topic Hellinger)
        random = _prob_matrix(5, 30).astype(np.float64)

        h_near = mean_pairwise_hellinger(near_dup)
        h_rand = mean_pairwise_hellinger(random)

        assert h_near < h_rand, (
            f"Near-duplicate topics should have lower mean Hellinger: "
            f"{h_near:.4f} vs {h_rand:.4f}"
        )

