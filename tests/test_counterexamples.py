"""The counterexamples that must never be reintroduced (CLAUDE.md §3).

These tests exist to make regression impossible: if someone later "proves" the
original conjecture, one of these will fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from idyn import linear as L
from idyn import spectra as SP
from idyn import systems as S


# --------------------------------------------------------------------------
# §3.1 -- the conjecture is false without a minimality condition
# --------------------------------------------------------------------------


def test_regrouping_gives_identical_observations():
    ce = S.regrouping_counterexample()
    rng = np.random.default_rng(0)
    z = rng.standard_normal((32, 4))
    z_tilde = z @ ce["P"].T
    assert np.allclose(ce["decoder_tilde"](z_tilde), ce["decoder"](z), atol=1e-12)


def test_regrouped_system_is_still_modular_with_the_same_shape():
    ce = S.regrouping_counterexample()
    Ft = ce["F_tilde"]
    assert np.allclose(Ft[:2, 2:], 0.0)
    assert np.allclose(Ft[2:, :2], 0.0)
    assert ce["system_tilde"].partition == ce["system"].partition == [2, 2]


def test_regrouping_map_is_invertible_but_not_a_block_permutation():
    """The heart of §3.1: h exists, is legal, and mixes modules."""
    ce = S.regrouping_counterexample()
    rep = L.block_permutation_report(ce["P"], [2, 2], [2, 2])
    assert rep.invertible
    assert not rep.is_block_permutation
    assert rep.on_block_fraction == pytest.approx(0.5)


def test_the_defect_is_that_blocks_are_decomposable():
    ce = S.regrouping_counterexample()
    cert = L.certify_finest_decomposition(ce["F"], [2, 2])
    assert not cert.A1  # each 2x2 diagonal block splits into two 1-D summands
    assert cert.n_summands == [2, 2]
    assert not cert.canonical


def test_refining_to_the_finest_partition_restores_uniqueness():
    """The §3.1 fix: require indecomposable blocks."""
    ce = S.regrouping_counterexample()
    cert = L.certify_finest_decomposition(ce["F"], [1, 1, 1, 1])
    assert cert.A1 and cert.A2 and cert.canonical

    H = L.intertwiner_space(ce["F"], ce["F"])
    rng = np.random.default_rng(1)
    for _ in range(100):
        A = np.tensordot(rng.standard_normal(H.shape[0]), H, axes=(0, 0))
        rep = L.block_permutation_report(A, [1, 1, 1, 1], [1, 1, 1, 1])
        if rep.invertible:
            assert rep.is_block_permutation


def test_regrouping_is_not_a_linear_artifact():
    """Four independent *nonlinear* 1-D maps regroup exactly the same way."""
    nl = S.nonlinear_regrouping_counterexample()
    rng = np.random.default_rng(2)
    z0 = S.sample_initial_conditions(4, 24, rng, radius=1.5)

    sys, sys_t, P = nl["system"], nl["system_tilde"], nl["P"]
    X = sys.simulate(z0, 10) @ nl["decoder"].W.T
    Xt = sys_t.simulate(z0 @ P.T, 10) @ nl["decoder_tilde"].W.T
    assert np.allclose(X, Xt, atol=1e-12)


# --------------------------------------------------------------------------
# Second counterexample: indecomposability ALONE is not enough.
# Not in the original brief; see theory/counterexamples.md §2.
# --------------------------------------------------------------------------


def test_indecomposable_blocks_with_shared_spectrum_still_fail():
    """J2(l) (+) J2(l): both blocks indecomposable, yet the splitting is not unique."""
    J = np.array([[0.8, 1.0], [0.0, 0.8]])
    F = np.block([[J, np.zeros((2, 2))], [np.zeros((2, 2)), J]])

    cert = L.certify_finest_decomposition(F, [2, 2])
    assert cert.A1, "each Jordan block is indecomposable"
    assert not cert.A2, "but they share an eigenvalue"
    assert not cert.canonical

    # an explicit alternative invariant splitting: span{e1, e2} is replaced by
    # span{e1 + e3, e2 + e4}, which is F-invariant and complemented
    U = np.zeros((4, 2))
    U[0, 0] = U[2, 0] = 1.0
    U[1, 1] = U[3, 1] = 1.0
    assert np.linalg.matrix_rank(np.hstack([U, F @ U])) == 2, "U is F-invariant"

    V = np.zeros((4, 2))
    V[2, 0] = V[3, 1] = 1.0
    assert np.linalg.matrix_rank(np.hstack([U, V])) == 4, "U (+) V = R^4"

    # and it is genuinely different from the coordinate splitting
    assert L.subspace_angle(U, np.eye(4)[:, :2]) > 0.1


def test_shared_spectrum_admits_module_mixing_intertwiners():
    J = np.array([[0.8, 1.0], [0.0, 0.8]])
    F = np.block([[J, np.zeros((2, 2))], [np.zeros((2, 2)), J]])
    H = L.intertwiner_space(F, F)
    # dim End(F) = 8 here, strictly larger than 2 * dim End(J) = 4
    assert H.shape[0] == 8

    rng = np.random.default_rng(3)
    mixing = 0
    for _ in range(200):
        A = np.tensordot(rng.standard_normal(8), H, axes=(0, 0))
        rep = L.block_permutation_report(A, [2, 2], [2, 2])
        if rep.invertible and not rep.is_block_permutation:
            mixing += 1
    assert mixing > 100, "generic invertible intertwiners mix the two modules"


# --------------------------------------------------------------------------
# Third counterexample: Theorem B's target conclusion is FALSE under (B1)-(B4).
# Found via the literature pass; see theory/counterexamples.md §5.
# This upgrades CLAUDE.md §3.7 from "the cocycle bound cannot prove it" to
# "there is nothing to prove -- block-diagonality does not hold".
# --------------------------------------------------------------------------


def test_triangular_conjugacy_is_exact():
    ce = S.triangular_conjugacy_counterexample()
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5000, 2)) * 1.5
    F, h = ce["system"].step, ce["h"]
    assert np.allclose(h(F(z)), F(h(z)), atol=1e-12), "h must conjugate F to itself"


def test_triangular_conjugacy_is_invertible():
    ce = S.triangular_conjugacy_counterexample()
    rng = np.random.default_rng(1)
    z = rng.standard_normal((2000, 2)) * 1.5
    assert np.allclose(ce["h_inv"](ce["h"](z)), z, atol=1e-12)


def test_triangular_conjugacy_satisfies_B1_and_B4():
    """Bounded C^1 derivative, unit Jacobian determinant, disjoint spectra."""
    ce = S.triangular_conjugacy_counterexample()
    assert ce["p"] > 1.0, "p > 1 makes dh1/dz2 continuous and vanishing at z2 = 0"

    z2 = np.linspace(-2.0, 2.0, 4001)
    m12 = ce["cross_derivative"](z2)
    assert np.all(np.isfinite(m12)) and m12.max() < 10.0, "(B1): Dh bounded on compacts"

    # Dh is unit lower-triangular, so it is invertible with bounded inverse too
    h, eps = ce["h"], 1e-6
    z = np.array([0.7, 1.1])
    J = np.stack([(h(z + e) - h(z - e)) / (2 * eps)
                  for e in (np.array([eps, 0.0]), np.array([0.0, eps]))], axis=1)
    assert np.linalg.det(J) == pytest.approx(1.0, abs=1e-6)

    lam1, lam2 = ce["lyapunov"]
    assert lam1 != lam2, "(B4) as written: spectra are disjoint"


def test_triangular_conjugacy_is_not_block_diagonal():
    """The whole point: (B1)-(B4) hold and h still mixes the modules."""
    ce = S.triangular_conjugacy_counterexample()
    assert ce["cross_derivative"](1.0) > 0.1, "M12 = dh1/dz2 does not vanish"
    # ...while the other cross-derivative does, exactly as Lemma C predicts
    rng = np.random.default_rng(2)
    z = rng.standard_normal((100, 2))
    assert np.allclose(ce["h"](z)[:, 1], z[:, 1]), "h2 = z2, so M21 = 0"


def test_lemma_C_is_not_contradicted():
    """The oriented gap Lemma C needs fails here; the one that holds kills M21."""
    ce = S.triangular_conjugacy_counterexample()
    lam1, lam2 = ce["lyapunov"]  # log mu1 < log mu2
    assert not (lam2 < lam1), "the gap needed to force M12 = 0 does not hold"
    assert lam1 < lam2, "the gap that holds is the one forcing M21 = 0, and M21 = 0"


def test_resonant_variant_is_smooth_and_still_fails():
    """C^infinity regularity does not rescue it: cross-module non-resonance is needed."""
    ce = S.triangular_conjugacy_counterexample(mu2=0.6, resonant_m=3)
    assert ce["p"] == pytest.approx(3.0), "p is an integer, so h is polynomial"
    assert ce["mu1"] == pytest.approx(0.6**3)

    rng = np.random.default_rng(3)
    z = rng.standard_normal((3000, 2)) * 1.5
    F, h = ce["system"].step, ce["h"]
    assert np.allclose(h(F(z)), F(h(z)), atol=1e-12)
    assert ce["cross_derivative"](1.0) > 0.1


def test_triangular_counterexample_rejects_bad_parameters():
    with pytest.raises(ValueError, match="0 < mu1 < mu2 < 1"):
        S.triangular_conjugacy_counterexample(mu1=0.8, mu2=0.5)
    with pytest.raises(ValueError, match="at least 2"):
        S.triangular_conjugacy_counterexample(resonant_m=1)


# --------------------------------------------------------------------------
# Route A: what "cross-module non-resonance" must actually mean.
# Both counterexamples are C^infinity (polynomial), so they survive the
# regularity strengthening that rescues Theorem B from §5 above.
# See theory/route_a_assessment.md and theory/counterexamples.md §6.
# --------------------------------------------------------------------------


def test_pairwise_nonresonance_is_not_sufficient():
    """mu1 = mu2*mu3: every pairwise log-ratio is far from an integer, yet
    h = (z1 + c z2 z3, z2, z3) is an exact polynomial conjugacy."""
    ce = S.multiindex_resonance_counterexample()
    mu1, mu2, mu3 = ce["mu"]
    assert mu1 == pytest.approx(mu2 * mu3)

    # pairwise non-resonant: all log-ratios well away from integers
    L = np.log(np.array([mu1, mu2, mu3]))
    worst = min(
        abs(L[i] / L[j] - round(L[i] / L[j]))
        for i in range(3)
        for j in range(3)
        if i != j
    )
    assert worst > 0.25, "the point is that pairwise non-resonance HOLDS here"

    rng = np.random.default_rng(0)
    z = rng.standard_normal((5000, 3)) * 1.5
    F, h = ce["system"].step, ce["h"]
    assert np.allclose(h(F(z)), F(h(z)), atol=1e-12)
    assert ce["cross_derivative"](1.0, 1.0) > 0.1, "h mixes modules"


def test_multiindex_resonance_is_detected():
    """The checker must catch what the pairwise test misses."""
    ce = S.multiindex_resonance_counterexample()
    assert not SP.is_cross_module_nonresonant(ce["lyapunov"])
    res = SP.cross_module_resonances(ce["lyapunov"])
    assert any(r.order == 2 and r.target_module == 0 for r in res)


def test_rotation_angle_gives_no_protection():
    """Phases cancel in x^2 + y^2, so the conjugacy works for every theta."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4000, 3)) * 1.5
    for theta in (0.0, 0.4, 1.1, 2.7, np.pi / 2):
        ce = S.repeated_exponent_resonance_counterexample(theta=theta)
        F, h = ce["system"].step, ce["h"]
        assert np.allclose(h(F(z)), F(h(z)), atol=1e-12), f"failed at theta={theta}"


def test_repeated_exponent_resonance_is_detected():
    ce = S.repeated_exponent_resonance_counterexample()
    assert not SP.is_cross_module_nonresonant(ce["lyapunov"])


def test_innocuous_looking_moduli_can_be_resonant():
    """(0.95, 0.9025) is a trap: 0.9025 = 0.95**2.  Guard test systems with this."""
    safe = [np.array([np.log(0.95)] * 2), np.array([np.log(0.70)] * 2)]
    trap = [np.array([np.log(0.95)] * 2), np.array([np.log(0.95**2)] * 2)]
    assert SP.is_cross_module_nonresonant(safe), "exp05's actual system is fine"
    assert not SP.is_cross_module_nonresonant(trap)


def test_exp05_system_is_nonresonant():
    """Regression guard on the system the experiments actually use."""
    sys = S.two_oscillator_system(s=(0.95, 0.70))
    spectra = [b.lyapunov_spectrum_exact() for b in sys.blocks]
    assert SP.is_cross_module_nonresonant(spectra, max_order=6)


def test_resonance_search_rejects_order_below_two():
    with pytest.raises(ValueError, match="order >= 2"):
        SP.cross_module_resonances([np.array([-1.0]), np.array([-2.0])], max_order=1)


def test_within_module_relations_are_not_flagged():
    """Only CROSS-module resonances matter; a module resonating with itself is not one."""
    # single module whose own exponents satisfy -2 = -1 + -1, but there is no
    # second module involved, so this must not be reported
    spectra = [np.array([-1.0, -2.0])]
    assert SP.cross_module_resonances(spectra) == []


# --------------------------------------------------------------------------
# §3.5 -- the linear decoder collapses h to linear before dynamics enter
# --------------------------------------------------------------------------


def test_full_column_rank_decoder_forces_h_linear():
    rng = np.random.default_rng(4)
    W = S.LinearDecoder.random(9, 4, rng).W
    A = rng.standard_normal((4, 4))  # any invertible reparameterisation
    W_tilde = W @ np.linalg.inv(A)

    z = rng.standard_normal((50, 4))
    z_tilde = z @ A.T
    assert np.allclose(z_tilde @ W_tilde.T, z @ W.T)

    # h is recovered exactly as W~^+ W, with no reference to the dynamics
    h = np.linalg.pinv(W_tilde) @ W
    assert np.allclose(h, A, atol=1e-9)
