"""
legacy/tests/test_math.py — Unit tests for legacy/_math.py

Every test uses purely synthetic numpy arrays — no ML training, no I/O.

Coverage
--------
  hellinger()                    — range, symmetry, identity, orthogonality, formula
  hellinger_matrix()             — shape, diagonal, symmetry, element-wise match
  faiss_sq_l2_to_hellinger()     — correctness vs. direct formula; old formula shown wrong
  faiss_sq_l2_to_distance()      — dispatch for hellinger / cosine / euclidean
  prepare_faiss_vectors()        — sqrt transform (hellinger), L2-norm (cosine), passthrough
  entropy()                      — uniform max, one-hot min, base-2, zero-guard
  term_relevance()               — shape, finite, lambda boundary semantics
  corpus_word_probabilities()    — sums-to-one, coverage, relative frequency
  diffusion_weights()            — formula, single-neighbour degenerate case, ordering
"""

from __future__ import annotations

import numpy as np
import pytest

from legacy._math import (
    hellinger,
    hellinger_matrix,
    faiss_sq_l2_to_hellinger,
    faiss_sq_l2_to_distance,
    prepare_faiss_vectors,
    entropy,
    term_relevance,
    corpus_word_probabilities,
    diffusion_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _prob(size: int) -> np.ndarray:
    return _RNG.dirichlet(np.ones(size)).astype(np.float64)


def _sq_l2(p: np.ndarray, q: np.ndarray) -> float:
    """True squared-L2 of (√p − √q), as returned by FAISS IndexFlatL2."""
    return float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


# ===========================================================================
# Hellinger distance (scalar)
# ===========================================================================

class TestHellinger:

    def test_range(self):
        for _ in range(20):
            h = hellinger(_prob(30), _prob(30))
            assert 0.0 <= h <= 1.0 + 1e-9, f"Out of [0,1]: {h}"

    def test_symmetry(self):
        p, q = _prob(25), _prob(25)
        assert abs(hellinger(p, q) - hellinger(q, p)) < 1e-9

    def test_self_distance_is_zero(self):
        p = _prob(20)
        assert hellinger(p, p) < 1e-7

    def test_orthogonal_distributions_equal_one(self):
        p = np.array([0.5, 0.5, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.6, 0.4])
        assert abs(hellinger(p, q) - 1.0) < 1e-7

    def test_matches_definition(self):
        """H = √( ½ ∑ (√p_i − √q_i)² )"""
        p, q = _prob(30), _prob(30)
        expected = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        assert abs(hellinger(p, q) - expected) < 1e-9

    def test_triangle_inequality(self):
        """H satisfies the triangle inequality (it is a metric)."""
        p, q, r = _prob(15), _prob(15), _prob(15)
        assert hellinger(p, r) <= hellinger(p, q) + hellinger(q, r) + 1e-9


# ===========================================================================
# Hellinger distance matrix
# ===========================================================================

class TestHellingerMatrix:

    @pytest.fixture(scope="class")
    def mat_6x20(self):
        X = np.vstack([_prob(20) for _ in range(6)])
        return X, hellinger_matrix(X)

    def test_shape(self, mat_6x20):
        _, M = mat_6x20
        assert M.shape == (6, 6)

    def test_diagonal_zeros(self, mat_6x20):
        _, M = mat_6x20
        np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-6)

    def test_symmetric(self, mat_6x20):
        _, M = mat_6x20
        np.testing.assert_allclose(M, M.T, atol=1e-7)

    def test_range(self, mat_6x20):
        _, M = mat_6x20
        assert M.min() >= -1e-9
        assert M.max() <= 1.0 + 1e-9

    def test_matches_elementwise_hellinger(self, mat_6x20):
        X, M = mat_6x20
        for i in range(6):
            for j in range(6):
                assert abs(M[i, j] - hellinger(X[i], X[j])) < 1e-6

    def test_single_row_gives_zeros(self):
        X = _prob(10).reshape(1, -1)
        M = hellinger_matrix(X)
        assert M.shape == (1, 1)
        assert M[0, 0] < 1e-9


# ===========================================================================
# FAISS conversion — the core correctness property of the pipeline
# ===========================================================================

class TestFaissHellingerConversion:
    """
    FAISS IndexFlatL2 on √X returns squared-L2 distances:
        D_FAISS = ‖√p − √q‖² = 2 · H(p, q)²

    Correct conversion : H = √(D_FAISS / 2)          ← this module
    Original AToMS-LP  : D_FAISS / √2 = √2 · H²      ← WRONG
    """

    def test_corrected_formula_matches_direct(self):
        """sqrt(sq_l2 / 2) must equal the scalar hellinger() for 50 random pairs."""
        for _ in range(50):
            p, q = _prob(40), _prob(40)
            sq = _sq_l2(p, q)
            converted = faiss_sq_l2_to_hellinger(np.array([sq]))[0]
            direct    = hellinger(p, q)
            assert abs(converted - direct) < 1e-5, (
                f"FAISS conversion error: converted={converted:.6f}, direct={direct:.6f}"
            )

    def test_old_formula_is_wrong(self):
        """D/√2 (original AToMS-LP) is consistently more wrong than √(D/2)."""
        errors_old, errors_new = [], []
        for _ in range(40):
            p, q  = _prob(30), _prob(30)
            sq    = _sq_l2(p, q)
            direct = hellinger(p, q)
            errors_old.append(abs(sq / np.sqrt(2) - direct))
            errors_new.append(abs(np.sqrt(sq / 2.0) - direct))
        assert np.mean(errors_new) < np.mean(errors_old), (
            "Fixed formula should be more accurate than the original AToMS-LP formula"
        )

    def test_output_range_for_probability_vectors(self):
        sq_vals = np.array([_sq_l2(_prob(20), _prob(20)) for _ in range(30)])
        h = faiss_sq_l2_to_hellinger(sq_vals)
        assert np.all(h >= -1e-9)
        assert np.all(h <= 1.0 + 1e-9)

    def test_zero_distance_stays_zero(self):
        np.testing.assert_allclose(
            faiss_sq_l2_to_hellinger(np.array([0.0, 0.0])), 0.0, atol=1e-9
        )

    def test_dispatch_hellinger_alias(self):
        sq = np.array([0.18, 0.72])
        np.testing.assert_allclose(
            faiss_sq_l2_to_distance(sq, "hellinger"),
            np.sqrt(sq / 2.0), atol=1e-9,
        )

    def test_dispatch_cosine(self):
        """For L2-normalised query vectors, FAISS sq-L2 = 2*(1 - cosine_sim) = 2*cosine_dist."""
        sq = np.array([0.4, 1.0, 1.6])
        np.testing.assert_allclose(
            faiss_sq_l2_to_distance(sq, "cosine"),
            sq / 2.0, atol=1e-9,
        )

    def test_dispatch_euclidean(self):
        sq = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(
            faiss_sq_l2_to_distance(sq, "euclidean"),
            np.array([1.0, 2.0, 3.0]), atol=1e-9,
        )

    def test_dispatch_unknown_metric_raises(self):
        with pytest.raises((ValueError, KeyError)):
            faiss_sq_l2_to_distance(np.array([1.0]), "chebyshev")


# ===========================================================================
# prepare_faiss_vectors
# ===========================================================================

class TestPrepareVectors:

    def test_hellinger_returns_sqrt(self):
        X = np.vstack([_prob(15) for _ in range(5)]).astype(np.float32)
        Q = prepare_faiss_vectors(X, "hellinger")
        np.testing.assert_allclose(Q, np.sqrt(X), atol=1e-6)

    def test_hellinger_nonneg(self):
        X = np.vstack([_prob(15) for _ in range(5)]).astype(np.float32)
        assert np.all(prepare_faiss_vectors(X, "hellinger") >= 0)

    def test_cosine_unit_norm(self):
        rng = np.random.default_rng(3)
        X = rng.random((8, 20)).astype(np.float32)
        Q = prepare_faiss_vectors(X, "cosine")
        norms = np.linalg.norm(Q, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_euclidean_passthrough(self):
        X = np.vstack([_prob(10) for _ in range(4)]).astype(np.float32)
        Q = prepare_faiss_vectors(X, "euclidean")
        np.testing.assert_allclose(Q, X, atol=1e-7)

    def test_output_dtype_float32(self):
        X = np.vstack([_prob(10) for _ in range(3)]).astype(np.float64)
        assert prepare_faiss_vectors(X, "hellinger").dtype == np.float32


# ===========================================================================
# Shannon entropy (base-2)
# ===========================================================================

class TestEntropy:

    def test_uniform_is_maximum_over_same_support(self):
        n = 8
        uniform = np.ones(n) / n
        skewed  = _prob(n)
        assert entropy(uniform) >= entropy(skewed) - 1e-9

    def test_one_hot_is_zero(self):
        for n in (2, 5, 10):
            p = np.zeros(n)
            p[0] = 1.0
            assert entropy(p) < 1e-9

    def test_binary_fair_coin_equals_one_bit(self):
        assert abs(entropy(np.array([0.5, 0.5])) - 1.0) < 1e-9

    def test_uniform_n_value(self):
        for n in (2, 4, 8, 16):
            assert abs(entropy(np.ones(n) / n) - np.log2(n)) < 1e-9

    def test_no_nan_for_zero_entries(self):
        p = np.array([0.5, 0.5, 0.0, 0.0])
        assert np.isfinite(entropy(p))

    def test_nonneg(self):
        for _ in range(20):
            assert entropy(_prob(12)) >= -1e-9

    def test_monotone_in_uniformity(self):
        """More-uniform → higher entropy."""
        concentrated = np.array([0.9, 0.05, 0.05])
        spread       = np.array([0.5, 0.3, 0.2])
        assert entropy(spread) > entropy(concentrated)


# ===========================================================================
# Term relevance (LDAvis-style)
# ===========================================================================

class TestTermRelevance:

    def test_output_shape(self):
        r = term_relevance(_prob(50), _prob(50), lam=0.6)
        assert r.shape == (50,)

    def test_finite_output(self):
        r = term_relevance(_prob(30), _prob(30), lam=0.6)
        assert np.all(np.isfinite(r))

    def test_lambda_1_ranks_by_phi(self):
        """At λ=1, relevance ≈ log(phi), so top-phi word should rank highly."""
        phi = _prob(20)
        p_w = _prob(20)
        r   = term_relevance(phi, p_w, lam=1.0)
        # Top phi word must be in the top-3 by relevance
        top_phi = np.argmax(phi)
        top3_r  = set(np.argsort(r)[-3:])
        assert top_phi in top3_r

    def test_lambda_0_penalises_common_words(self):
        """At λ=0, relevance = log p(w|z) − log p(w).
        A word with identical p(w|z) and p(w) should score lower than
        a word with the same p(w|z) but rare corpus frequency."""
        phi = np.array([0.3, 0.3, 0.3, 0.1])
        # word 0 is equally common in corpus; word 2 is rare
        p_w = np.array([0.3, 0.1, 0.05, 0.55])
        r   = term_relevance(phi, p_w, lam=0.0)
        # word 2 (rare, topic-specific) should score better than word 0 (common)
        assert r[2] > r[0]

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
    def test_various_lambda_finite(self, lam):
        r = term_relevance(_prob(25), _prob(25), lam=lam)
        assert np.all(np.isfinite(r))


# ===========================================================================
# Corpus word probabilities
# ===========================================================================

class TestCorpusWordProbabilities:

    def test_sums_to_one(self):
        docs = [["a", "b", "a"], ["b", "c"], ["c", "c", "a"]]
        p_w  = corpus_word_probabilities(docs)
        assert abs(sum(p_w.values()) - 1.0) < 1e-9

    def test_covers_all_words(self):
        docs = [["x", "y"], ["y", "z"]]
        assert set(corpus_word_probabilities(docs)) == {"x", "y", "z"}

    def test_relative_frequencies(self):
        docs = [["a", "a", "a", "b"]]   # a appears 3× b
        p_w  = corpus_word_probabilities(docs)
        assert abs(p_w["a"] / p_w["b"] - 3.0) < 1e-9

    def test_empty_docs_ignored(self):
        docs = [["a"], [], ["a", "b"]]
        p_w  = corpus_word_probabilities(docs)
        assert abs(sum(p_w.values()) - 1.0) < 1e-9


# ===========================================================================
# Diffusion weights
# ===========================================================================

class TestDiffusionWeights:

    def test_formula_w_ij_equals_1_minus_d_over_sum(self):
        d = np.array([0.2, 0.3, 0.5])
        w = diffusion_weights(d)
        expected = 1.0 - d / d.sum()
        np.testing.assert_allclose(w, expected, atol=1e-9)

    def test_closer_neighbor_gets_higher_weight(self):
        d = np.array([0.1, 0.9])
        w = diffusion_weights(d)
        assert w[0] > w[1], "Closer neighbour should have higher weight"

    def test_single_neighbor_degenerates_to_zero(self):
        """The formula 1 − d/d = 0 for a single neighbour.
        Callers (diffuse_distributions) guard against this with w=1.0."""
        d = np.array([0.7])
        w = diffusion_weights(d)
        assert w[0] == pytest.approx(0.0, abs=1e-9)

    def test_nonneg_with_multiple_neighbors(self):
        d = np.array([0.1, 0.2, 0.3, 1.0])
        assert np.all(diffusion_weights(d) >= -1e-9)

    def test_uniform_distances_give_equal_weights(self):
        """When all k distances are equal, wᵢ = 1 − 1/k for every entry.
        (The single-neighbour case k=1 is a separate degenerate case → 0.)"""
        k = 3
        d = np.array([0.5] * k)
        w = diffusion_weights(d)
        np.testing.assert_allclose(w, 1.0 - 1.0 / k, atol=1e-9)
