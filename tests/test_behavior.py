"""Route B: the behavioural auxiliary variable and the canonical invariant subspace.

The mechanism these tests pin down: the u-invariant block is canonical, so a leak
of the u-varying block into it makes it u-dependent -- which is how behaviour
'sees', and rejects, the cross-derivative the cocycle cannot supply.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from idyn import behavior as B
from idyn import systems as S


def test_conditioned_sample_shapes_and_slices():
    rng = np.random.default_rng(0)
    s = B.conditioned_initial_conditions(2, 3, np.array([0, 1, 2]), 100, rng)
    assert s.Z.shape == (300, 5)
    assert s.U.shape == (300,)
    assert s.slice_a == slice(0, 2) and s.slice_b == slice(2, 5)
    assert set(np.unique(s.U)) == {0, 1, 2}


def test_variance_mode_moves_the_varying_block_not_the_invariant_one():
    rng = np.random.default_rng(1)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3]), 8000, rng, mode="variance")
    dep_a = B.block_u_dependence(s.Z[:, s.slice_a], s.U).total
    dep_b = B.block_u_dependence(s.Z[:, s.slice_b], s.U).total
    assert dep_a > 0.3, "the varying block's law must move with u"
    assert dep_b < 0.1, "the invariant block's law must not"


def test_variance_mode_shows_in_the_covariance_not_the_mean():
    rng = np.random.default_rng(2)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3]), 8000, rng, mode="variance")
    d = B.block_u_dependence(s.Z[:, s.slice_a], s.U)
    assert d.cov_variation > 5.0 * max(d.mean_variation, 1e-6), "variance modulation is a 2nd-moment effect"


def test_mean_mode_shows_in_the_mean():
    rng = np.random.default_rng(3)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3]), 8000, rng, mode="mean")
    d = B.block_u_dependence(s.Z[:, s.slice_a], s.U)
    assert d.mean_variation > 0.2


def test_invariant_block_is_u_independent_to_the_noise_floor():
    """The canonical-subspace claim rests on this: a block that ignores z^A is u-flat."""
    rng = np.random.default_rng(4)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3, 4]), 12000, rng)
    assert B.block_u_dependence(s.Z[:, s.slice_b], s.U).total < 0.06


def test_leaking_the_varying_block_makes_the_invariant_block_u_dependent():
    """h_B = z^B + eps z^A becomes u-dependent for eps != 0 -- behaviour kills M_BA."""
    rng = np.random.default_rng(5)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3, 4]), 12000, rng, mode="variance")
    za, zb = s.Z[:, s.slice_a], s.Z[:, s.slice_b]
    floor = B.block_u_dependence(zb, s.U).total

    nets = [B.block_u_dependence(zb + eps * za, s.U).total - floor for eps in (0.0, 0.1, 0.3, 0.5)]
    assert abs(nets[0]) < 0.01, "no leak -> u-flat"
    assert nets[-1] > 0.1, "a clear leak -> clearly u-dependent"
    assert all(nets[i] <= nets[i + 1] + 1e-3 for i in range(len(nets) - 1)), "monotone in the leak"


def test_variance_leak_is_quadratic_mean_leak_is_linear():
    """The Prop. 1 sensitivity split: O(eps^2) for variance vs O(eps) for mean."""
    rng = np.random.default_rng(6)
    # variance leak: net cov-dependence should grow faster than linearly
    sv = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3, 4]), 16000, rng, mode="variance")
    za, zb = sv.Z[:, sv.slice_a], sv.Z[:, sv.slice_b]
    base = B.block_u_dependence(zb, sv.U).cov_variation
    d2 = B.block_u_dependence(zb + 0.2 * za, sv.U).cov_variation - base
    d4 = B.block_u_dependence(zb + 0.4 * za, sv.U).cov_variation - base
    assert d4 / d2 > 2.5, "doubling eps more than doubles a variance leak (quadratic)"

    # mean leak: net mean-dependence grows about linearly
    sm = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3, 4]), 16000, rng, mode="mean")
    za_m, zb_m = sm.Z[:, sm.slice_a], sm.Z[:, sm.slice_b]
    basem = B.block_u_dependence(zb_m, sm.U).mean_variation
    m2 = B.block_u_dependence(zb_m + 0.05 * za_m, sm.U).mean_variation - basem
    m4 = B.block_u_dependence(zb_m + 0.10 * za_m, sm.U).mean_variation - basem
    assert 1.6 < m4 / m2 < 2.6, "doubling eps roughly doubles a mean leak (linear)"


def test_mean_only_detector_misses_a_variance_leak():
    """Prop. 1 cap, made measurable: a mean-only readout cannot see a variance leak."""
    rng = np.random.default_rng(7)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3, 4]), 16000, rng, mode="variance")
    za, zb = s.Z[:, s.slice_a], s.Z[:, s.slice_b]
    floor = B.block_u_dependence(zb, s.U)
    leaked = B.block_u_dependence(zb + 0.3 * za, s.U)
    total_catch = leaked.total - floor.total
    mean_only_catch = leaked.mean_only - floor.mean_only
    assert total_catch > 0.05, "the full detector sees the variance leak"
    assert abs(mean_only_catch) < 0.25 * total_catch, "a mean-only detector misses it"


def test_bad_arguments_are_rejected():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="mode"):
        B.conditioned_initial_conditions(2, 2, np.array([0, 1]), 10, rng, mode="cubic")
    with pytest.raises(ValueError, match="two behaviour levels"):
        B.conditioned_initial_conditions(2, 2, np.array([0]), 10, rng)
    with pytest.raises(ValueError, match="two u levels"):
        B.block_u_dependence(np.zeros((10, 2)), np.zeros(10, dtype=int))


# --------------------------------------------------------------------------
# Lemma D (identifiability.md 4.5): behaviour kills what the gap cannot
# --------------------------------------------------------------------------


def test_lemma_d_witness_is_an_exact_non_block_diagonal_conjugacy():
    """The obstruction is real: exact conjugacy, modular F, and NOT block-diagonal."""
    w = S.lemma_d_witness()
    F, h = w["F"], w["h"]
    z = np.random.default_rng(0).normal(size=(2000, 4))
    assert np.abs(h(F(z)) - F(h(z))).max() < 1e-12, "h must conjugate F to itself"

    eps = 1e-6
    step = np.array([eps, 0.0, 0.0, 0.0])
    dBdA = (h(z + step)[:, 2:] - h(z - step)[:, 2:]) / (2 * eps)
    assert np.linalg.norm(dBdA, axis=1).mean() > 0.5, "h_B must genuinely depend on z_A"


def test_lemma_d_witness_satisfies_the_gap_so_the_dynamics_are_exhausted():
    """Why it is a witness: Lemma C's hypothesis HOLDS and still cannot remove it.

    The gap kills M_AB; the surviving coupling is M_BA, which CLAUDE.md 3.7 shows
    no orientation of the gap can reach.  If this assertion ever fails the witness
    has stopped being a witness -- it would just be a system outside Lemma C.
    """
    w = S.lemma_d_witness()
    assert w["gap_holds"], "the one-sided gap must hold, else the example proves nothing"
    assert w["s_b"] < w["s_a"], "B must be the dominated module"
    assert w["resonance_residual"] < 1e-14, "the cross-module resonance must be exact"


def test_lemma_d_step3_the_gap_forces_degree_at_least_two():
    """identifiability.md 4.5 Step 3: rho_min(f_A)^|m| <= rho(f~_B) < rho_min(f_A)."""
    w = S.lemma_d_witness()
    s_a, s_b = w["s_a"], w["s_b"]
    m = np.log(s_b) / np.log(s_a)  # the only degree a semiconjugacy can have
    assert m == pytest.approx(w["psi_degree"], abs=1e-12)
    assert m >= 2.0 - 1e-12, "a one-sided gap with s<1 forces |m| >= 2"


def test_lemma_d_behaviour_detects_the_coupling_that_the_gap_cannot():
    """Step 4: psi homogeneous of degree 2 makes var(h_B) scale as sigma^4."""
    w = S.lemma_d_witness()
    h, predict = w["h"], w["var_h_b"]
    rng = np.random.default_rng(1)
    prev = -1.0
    for sigma in (0.6, 1.0, 1.6):
        zA = sigma * rng.normal(size=(80000, 2))
        zB = rng.normal(size=(80000, 2))
        hB = h(np.concatenate([zA, zB], axis=1))[:, 2:]
        assert hB.var() == pytest.approx(predict(sigma), rel=0.05)
        assert hB.var() > prev, "the law of h_B must move with u -- that is the kill"
        prev = hB.var()


def test_lemma_d_the_degree_zero_escape_is_u_invariant_but_needs_no_gap():
    """The unique way to hide from behaviour is scale-invariance, i.e. degree 0.

    Step 2 then demands 1 in spec(f~_B), i.e. no contraction in B -- which (D1)
    forbids.  Here we check only the first half: a degree-0 coupling really is
    invisible to behaviour, so the gap is doing indispensable work.
    """
    rng = np.random.default_rng(2)
    variances = []
    for sigma in (0.6, 1.0, 1.6):
        zA = sigma * rng.normal(size=(80000, 2))
        zB = rng.normal(size=(80000, 2))
        ang = zA / np.linalg.norm(zA, axis=1, keepdims=True)  # homogeneous of degree 0
        variances.append((zB + 0.7 * ang).var())
    assert max(variances) - min(variances) < 0.02, "degree 0 hides from behaviour"


# --------------------------------------------------------------------------
# CLAUDE.md §3.12: the behavioural constraint is a GAUGE quantity
#
# The raw u-dependence score and the raw training penalty both carry the units
# of the block, so "make the block u-invariant" and "make the block small" are
# not distinguished -- and the optimiser takes the cheap one.  These tests pin
# the exploit and the fix.
# --------------------------------------------------------------------------


def _leaky_sample(n_per_u=4000, seed=20):
    """A block that is a verbatim copy of the u-VARYING one: a maximal leak."""
    rng = np.random.default_rng(seed)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3]), n_per_u, rng,
                                         mode="variance")
    return s.Z[:, s.slice_a], s.U


def test_raw_u_dependence_is_defeated_by_shrinking_the_block():
    """The exploit itself: 20x smaller reads as ~50x more invariant, for free."""
    w, U = _leaky_sample()
    full = B.block_u_dependence(w, U).total
    small = B.block_u_dependence(0.05 * w, U).total
    assert full > 0.3, "the un-shrunk leak is plainly visible"
    assert small < 0.02 * full, "shrinking alone buys apparent invariance"


def test_normalised_u_dependence_is_invariant_under_any_invertible_block_map():
    """The fix: whitening makes the score a GL(d_b) invariant, the §7 gauge group.

    Both terms are orthogonal invariants after whitening -- the mean term is a
    trace and the covariance term a Frobenius norm -- so this holds for a general
    invertible A, not merely for a scalar.
    """
    w, U = _leaky_sample()
    base = B.block_u_dependence(w, U, normalize=True).total
    assert base > 0.5, "a maximal leak must still register"
    assert B.block_u_dependence(0.05 * w, U, normalize=True).total == pytest.approx(base, rel=1e-6)
    A = np.array([[2.0, -3.0], [0.5, 4.0]])
    assert B.block_u_dependence(w @ A.T, U, normalize=True).total == pytest.approx(base, rel=1e-6)


def test_normalised_u_dependence_still_separates_invariant_from_varying():
    """Gauge-invariance would be worthless if it also erased the signal."""
    rng = np.random.default_rng(21)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3]), 4000, rng,
                                         mode="variance")
    varying = B.block_u_dependence(s.Z[:, s.slice_a], s.U, normalize=True).total
    invariant = B.block_u_dependence(s.Z[:, s.slice_b], s.U, normalize=True).total
    assert varying > 5.0 * invariant


def test_training_penalty_is_paid_off_by_shrinking_unless_whitened():
    """The same exploit in the objective, which is where it actually did damage."""
    from idyn.models import LatentDynamicsModel as L

    w, U = _leaky_sample(n_per_u=500, seed=22)
    n = w.shape[0]
    z = np.zeros((n, 4, 4))
    z[:, :, :2] = w[:, None, :]
    z[:, :, 2:] = w[:, None, :]      # the pinned block IS the u-varying one
    zt = torch.tensor(z, dtype=torch.float64)
    ut = torch.tensor(U)
    blk = slice(2, 4)

    shrunk = zt.clone()
    shrunk[:, :, 2:] *= 0.05

    raw_full = float(L._behavioural_penalty(zt, ut, blk, whiten=False))
    raw_small = float(L._behavioural_penalty(shrunk, ut, blk, whiten=False))
    assert raw_small < 1e-3 * raw_full, "unwhitened, the penalty is bought with scale"

    wht_full = float(L._behavioural_penalty(zt, ut, blk, whiten=True))
    wht_small = float(L._behavioural_penalty(shrunk, ut, blk, whiten=True))
    assert wht_full > 1e-3, "whitened, a maximal leak is still penalised"
    # Invariance is exact up to the Cholesky ridge, which is relatively 400x
    # larger once the block is scaled by 0.05.  0.07% residual sensitivity
    # against the ~1000x the unwhitened penalty offers is the whole point.
    assert wht_small == pytest.approx(wht_full, rel=1e-2), "and shrinking no longer helps"


def test_whitened_penalty_is_near_zero_for_a_genuinely_invariant_block():
    """Not vacuous in the other direction: it must still be satisfiable."""
    from idyn.models import LatentDynamicsModel as L

    rng = np.random.default_rng(23)
    s = B.conditioned_initial_conditions(2, 2, np.array([0, 1, 2, 3]), 500, rng,
                                         mode="variance")
    z = np.repeat(s.Z[:, None, :], 4, axis=1)
    zt, ut = torch.tensor(z, dtype=torch.float64), torch.tensor(s.U)
    leak = float(L._behavioural_penalty(zt, ut, slice(0, 2), whiten=True))
    clean = float(L._behavioural_penalty(zt, ut, slice(2, 4), whiten=True))
    assert clean < 0.1 * leak
