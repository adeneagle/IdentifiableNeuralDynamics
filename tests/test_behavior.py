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


# --------------------------------------------------------------------------
# Lemma D open item (a): (D4) alone cannot carry the non-additive case
# --------------------------------------------------------------------------


def test_nonadditive_escape_satisfies_D4_exactly_while_M_BA_is_large():
    """A non-additive h_B that behaviour cannot see, at any leak size.

    Step 4's whole behavioural input is (D4) -- the law of h_B is u-invariant.
    This map satisfies it *exactly*, for every u, with M_BA large.  So no
    sharpening of (D4) closes open item (a); Steps 1-3 have to do the work.
    """
    rng = np.random.default_rng(30)
    s = B.conditioned_initial_conditions(2, 2, np.arange(4), 40000, rng, mode="variance")
    u_floor = B.block_u_dependence(s.Z[:, s.slice_b], s.U, normalize=True).total

    for gamma in (0.5, 1.0, 2.0):
        ce = S.nonadditive_behavioural_escape(gamma=gamma)
        w = ce["h"](s.Z)
        assert ce["cross_derivative"](s.Z).mean() > 0.5 * gamma, "the coupling is real"
        # (D4) holds to the sampling floor: behaviour sees nothing
        dep = B.block_u_dependence(w[:, 2:], s.U, normalize=True).total
        assert dep < 3.0 * u_floor, f"gamma={gamma} is invisible to behaviour (got {dep})"


def test_nonadditive_escape_satisfies_D1_and_is_not_additive():
    """It is inside Lemma D's own hypotheses, not an alignment artefact."""
    ce = S.nonadditive_behavioural_escape()
    assert ce["one_sided_gap"] > 0.0, "(D1): rho(f_B) < rho_min(f_A)"
    assert ce["rho_b"] < ce["rho_a"]
    assert not ce["additive"]
    # a genuine diffeomorphism
    rng = np.random.default_rng(31)
    z = rng.standard_normal((2000, 4))
    assert np.allclose(ce["h_inv"](ce["h"](z)), z, atol=1e-12)


def test_nonadditive_escape_is_killed_by_the_dynamics_not_by_behaviour():
    """Lemma D's conclusion is not threatened: this h is not a modular conjugacy.

    Step 1 is what excludes it -- the shape any proof of open item (a) must have.
    """
    rng = np.random.default_rng(32)
    za = rng.standard_normal((3000, 2))
    assert S.nonadditive_behavioural_escape(gamma=0.0)["dynamics_defect"](za) < 1e-12
    defects = [S.nonadditive_behavioural_escape(gamma=g)["dynamics_defect"](za)
               for g in (0.25, 0.5, 1.0)]
    assert all(d > 1e-3 for d in defects), "no autonomous ftilde_B exists for gamma != 0"
    assert defects[0] < defects[1] < defects[2], "the obstruction grows with the leak"


# --------------------------------------------------------------------------
# exp18: on a LIMIT CYCLE that escape becomes a genuine modular conjugacy
# --------------------------------------------------------------------------


def _cycle_ic(rng, n):
    r = rng.uniform(0.6, 1.4, n)
    th = rng.uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], -1)


def test_rotational_escape_IS_a_modular_conjugacy_unlike_the_fixed_point_case():
    """The Step 1 obstruction evaporates once f_A has a cycle instead of a fixed point.

    `nonadditive_behavioural_escape` is excluded because ``theta . f_A - theta``
    must be constant and a contracting fixed point forces that constant to 0.
    On a cycle with ``theta = arg(z_A)`` the increment is exactly ``omega_a``,
    so the very same shape of map conjugates modular dynamics to modular
    dynamics -- and it moves the recipient's rotation number.
    """
    ce = S.rotational_behavioural_escape()
    rng = np.random.default_rng(40)
    z = np.concatenate([_cycle_ic(rng, 4000), _cycle_ic(rng, 4000)], axis=1)

    lhs = ce["h"](ce["system"].step(z))
    rhs = ce["system_tilde"].step(ce["h"](z))
    assert np.abs(lhs - rhs).max() < 1e-12, "exact modular conjugacy"
    assert np.allclose(ce["h_inv"](ce["h"](z)), z, atol=1e-10), "a diffeomorphism"
    assert ce["is_modular_conjugacy"] and not ce["additive"]
    assert ce["omega_tilde"][1] != ce["omega"][1], "it moves the recipient's rotation"


def test_a_rotationally_symmetric_invariant_block_hides_the_regrouping():
    """Behaviour is blind to a coupling that acts by a symmetry of p_B.

    The regrouping ROTATES the invariant block by the donor's phase.  With the
    recipient's phase uniform, p_B is rotationally symmetric, so rotating it by
    any independent angle returns the same law: p(h_B | u) = p(z_B) for every u,
    whatever the donor does.  This is exp18's headline negative and the reason
    Route B does not rescue the lattice ambiguity in general.
    """
    ce = S.rotational_behavioural_escape()
    rng = np.random.default_rng(41)
    Z0, U = ce["sampler"](rng, 20000, kappa_b=0.0)
    Z = ce["system"].simulate(Z0, 30)
    H = ce["h"](Z)

    # the donor really is modulated, and the recipient really is invariant
    assert B.block_u_dependence(Z[:, -1, :2], U, normalize=True).total > 0.5
    floor = B.block_u_dependence(Z[:, -1, 2:], U, normalize=True).total
    seen = B.block_u_dependence(H[:, -1, 2:], U, normalize=True).total
    assert seen < 2.0 * floor, f"the regrouping is invisible to behaviour (got {seen} vs floor {floor})"


def test_breaking_that_symmetry_makes_the_same_regrouping_visible():
    """And it is the symmetry doing it, not anything about the dynamics.

    Concentrating the recipient's phase *without* making it u-dependent keeps it
    a legitimate invariant block and turns the detector back on, monotonically.
    """
    ce = S.rotational_behavioural_escape()
    seen = []
    for kb in (0.0, 1.0, 4.0):
        rng = np.random.default_rng(42)
        Z0, U = ce["sampler"](rng, 20000, kappa_b=kb)
        Z = ce["system"].simulate(Z0, 30)
        # the recipient stays u-invariant at every kappa: that is what makes it a
        # fair comparison rather than a leak introduced by hand
        assert B.block_u_dependence(Z[:, -1, 2:], U, normalize=True).total < 0.12
        seen.append(B.block_u_dependence(ce["h"](Z)[:, -1, 2:], U, normalize=True).total)
    assert seen[0] < seen[1] < seen[2], f"visibility rises with concentration: {seen}"
    assert seen[2] > 10.0 * seen[0], "and the two ends are not close"


def test_pooling_timesteps_blinds_the_penalty_to_a_rotating_u_dependence():
    """CLAUDE.md §3.15: the objective must score per timestep, not pooled.

    An oscillatory block's phase advances every step, so over a trial it wraps
    several times and the time-POOLED law is near-uniform for every u even when
    the per-timestep laws differ completely.  The penalty then reports itself
    satisfied on a representation that is maximally u-dependent at every instant.
    """
    import torch

    from idyn.models import LatentDynamicsModel

    ce = S.rotational_behavioural_escape()
    rng = np.random.default_rng(44)
    Z0, U = ce["sampler"](rng, 3000, kappa_b=4.0)
    Z = ce["system"].simulate(Z0, 30)
    H = ce["h"](Z)
    Ut, sl = torch.as_tensor(U), slice(2, 4)
    pen = LatentDynamicsModel._behavioural_penalty

    def score(A, per_time):
        t = torch.as_tensor(np.ascontiguousarray(A), dtype=torch.float32)
        return float(pen(t, Ut, sl, whiten=True, per_time=per_time))

    pooled_r1, pooled_r2 = score(Z, False), score(H, False)
    pert_r1, pert_r2 = score(Z, True), score(H, True)

    # pooled: the regrouped block looks nearly as invariant as the true one, and
    # both are so small that no weight makes them matter against the fit terms
    assert pooled_r2 < 1e-2, f"pooled cannot see it (got {pooled_r2})"
    # per-time: a large absolute score and a wide margin over the true block
    assert pert_r2 > 0.5, f"per-time sees it (got {pert_r2})"
    assert pert_r2 > 8.0 * pert_r1, "and separates R2 from R1"
    assert pert_r2 / pooled_r2 > 20.0, "the pooled form loses at least an order"
    assert pooled_r1 < pooled_r2 and pert_r1 < pert_r2, "both orderings still correct"


def test_radial_modulation_is_insensitive_to_the_pooling_fix():
    """The exp13 regime is not blind either way -- which is why it stands.

    A scale conditioning does not rotate, so pooling merely mixes decay stages.
    Both forms detect an injected leak by two to three orders; the absolute
    scales differ, which is why a weight still does not transfer.
    """
    import torch

    from idyn.models import LatentDynamicsModel

    rng = np.random.default_rng(45)
    sysm = S.ModularSystem([S.TwistBlock(s=0.90, omega=0.4, beta=0.0),
                            S.TwistBlock(s=0.55, omega=1.1, beta=0.0)])
    smp = B.conditioned_initial_conditions(2, 2, np.arange(4), 3000, rng, mode="variance")
    Z = sysm.simulate(smp.Z, 15)
    Ut = torch.as_tensor(smp.U)
    pen = LatentDynamicsModel._behavioural_penalty

    def score(c, per_time):
        A = np.array(Z)
        A[..., 2:] = A[..., 2:] + c * A[..., :2]
        t = torch.as_tensor(A, dtype=torch.float32)
        return float(pen(t, Ut, slice(2, 4), whiten=True, per_time=per_time))

    for per_time in (False, True):
        clean, leaked = score(0.0, per_time), score(0.5, per_time)
        assert leaked > 100.0 * clean, (
            f"a radial leak is visible with per_time={per_time} "
            f"({clean} -> {leaked})")


def test_lemma_D_variance_modulation_is_transient_on_a_limit_cycle():
    """(D3) is unavailable in the oscillatory regime: the attractor erases it.

    Lemma D modulates the varying block's *scale*.  A limit cycle pulls every
    radius to rho, so that conditioning is forgotten -- which is why exp18 tests
    the Route B mechanism with phase modulation rather than testing Lemma D.
    """
    ce = S.rotational_behavioural_escape()
    rng = np.random.default_rng(43)
    Zs, Us = [], []
    for k, s in enumerate((0.7, 1.3)):
        Zs.append(np.concatenate([_cycle_ic(rng, 8000) * s, _cycle_ic(rng, 8000)], axis=1))
        Us.append(np.full(8000, k))
    Z = ce["system"].simulate(np.concatenate(Zs), 30)
    U = np.concatenate(Us)
    assert np.isfinite(Z).all(), "annulus draws stay inside the basin"

    rad = np.abs(Z[..., 0] + 1j * Z[..., 1])
    gap0 = abs(rad[U == 0, 0].mean() - rad[U == 1, 0].mean())
    gapT = abs(rad[U == 0, -1].mean() - rad[U == 1, -1].mean())
    assert gap0 > 0.4, "the modulation was delivered"
    assert gapT < 1e-6 * gap0, f"and then erased ({gap0} -> {gapT})"

    early = B.block_u_dependence(Z[:, 0, :2], U, normalize=True).total
    late = B.block_u_dependence(Z[:, -1, :2], U, normalize=True).total
    assert late < 0.15 * early, f"(D3) does not survive the attractor ({early} -> {late})"


# --------------------------------------------------------------------------
# Lemma D' -- (D1) is far more than the proof needs (identifiability.md 4.5a)
#
# The one-sided gap's only job is to exclude a degree-0 (scale-invariant) psi,
# and that needs merely 1 not in spec(ftilde_B).  Step 3's |m| >= 2 is not
# load-bearing: Step 4's iteration runs at every degree >= 1.
# --------------------------------------------------------------------------


def test_gapless_witness_is_an_exact_conjugacy_with_a_live_cross_derivative():
    w = S.gapless_resonant_coupling()
    z = np.random.default_rng(0).normal(size=(4000, 4))
    assert np.abs(w["h"](w["F"](z)) - w["F"](w["h"](z))).max() < 1e-12
    assert w["cross_derivative"] > 0.9, "M_BA is genuinely nonzero"
    assert w["psi_degree"] == 1, "degree 1 -- what Step 3 says cannot happen with a gap"


def test_gapless_witness_fails_D1_and_every_spectral_hypothesis():
    """Identical spectra: (B4) is 0, (F3) is not ordered, (D1) fails, (D1') holds."""
    from idyn import spectra as SP

    w = S.gapless_resonant_coupling()
    assert not w["gap_holds"]
    assert SP.spectral_gap(w["spectra"]) == pytest.approx(0.0, abs=1e-15)
    assert not SP.filtration_gap(w["spectra"]).ordered
    assert w["unit_eigenvalue_distance"] > 0.1, "(D1') holds -- no unit eigenvalue"
    assert w["resonance_residual"] == pytest.approx(0.0, abs=1e-12)


def test_behaviour_kills_the_degree_1_coupling_with_no_gap():
    """The claim of Lemma D': two levels suffice at degree 1, gap or no gap.

    Asserted against a measured floor rather than against zero -- the c=0 arm is
    the control, and the treated arms must clear it (3.9: never assert inside
    the noise).
    """
    rng = np.random.default_rng(1)
    n, sig, tau = 40_000, (0.6, 1.6), 1.0

    def u_dep(c):
        ws, us = [], []
        for j, s in enumerate(sig):
            za = rng.normal(scale=s, size=(n, 2))
            zb = rng.normal(scale=tau, size=(n, 2))
            ws.append(zb + c * za)
            us.append(np.full(n, j))
        return B.block_u_dependence(np.concatenate(ws), np.concatenate(us),
                                    normalize=True).total

    floor = u_dep(0.0)
    scores = [u_dep(c) for c in (0.25, 0.5, 0.7)]
    assert floor < 0.03, f"control should sit near zero at n={n}, got {floor}"
    assert all(s > 4 * floor for s in scores), f"floor {floor}, scores {scores}"
    assert scores[0] < scores[1] < scores[2], "detection grows with the coupling"


def test_the_gapless_witness_matches_its_closed_form_variance():
    """var(h_B) per component = tau^2 + c^2 sigma^2 -- degree 1, so sigma^2 not sigma^4."""
    rng = np.random.default_rng(2)
    w = S.gapless_resonant_coupling(c=0.7)
    for sigma in (0.6, 1.0, 1.6):
        za = rng.normal(scale=sigma, size=(200_000, 2))
        zb = rng.normal(size=(200_000, 2))
        got = float((zb + 0.7 * za).var(axis=0).mean())
        assert got == pytest.approx(w["var_h_b"](sigma), rel=2e-2)


def test_degree_zero_is_the_escape_that_D1_prime_must_exclude():
    """A scale-invariant psi is invisible to behaviour -- so (D1') is not optional."""
    rng = np.random.default_rng(3)
    n, tau = 40_000, 1.0
    ws, us = [], []
    for j, s in enumerate((0.6, 1.6)):
        za = rng.normal(scale=s, size=(n, 2))
        zb = rng.normal(scale=tau, size=(n, 2))
        ws.append(zb + 0.7 * za / np.linalg.norm(za, axis=1, keepdims=True))
        us.append(np.full(n, j))
    dep = B.block_u_dependence(np.concatenate(ws), np.concatenate(us), normalize=True)
    assert dep.total < 0.03, "degree 0 hides from behaviour, as 4.5 already argued"


def test_equal_frequency_is_the_degree_1_resonance_condition():
    """Two oscillators of equal rate: a coupling exists iff omega_A = +/- omega_B.

    So either the frequencies differ and no linear psi exists at all, or they
    agree and Lemma D' removes it -- a complete answer for this class.
    """
    for omega_b, expected in ((0.70, 2), (-0.70, 2), (1.30, 0), (2.10, 0)):
        w = S.gapless_resonant_coupling(omega=0.70, omega_b=omega_b)
        assert w["sylvester_kernel_dim"] == expected, f"omega_b={omega_b}"


# --------------------------------------------------------------------------
# Lemma D'' -- several surviving degrees at once (identifiability.md 4.5b)
#
# Step 4 assumed psi homogeneous of a SINGLE degree; Step 2 permits several.
# The repair runs on second moments: Var(<t, psi(sigma zeta)>) is a polynomial
# in sigma with no constant term, so constancy across enough levels kills it.
# --------------------------------------------------------------------------


def test_multidegree_witness_is_an_exact_conjugacy_with_two_live_degrees():
    w = S.multidegree_resonant_coupling()
    z = np.random.default_rng(0).normal(size=(4000, 4))
    assert np.abs(w["h"](w["F"](z)) - w["F"](w["h"](z))).max() < 1e-12
    assert w["degrees"] == [1, 2], "the case Step 4 could not handle"
    # both resonances are structural (mu^1 = mu, mu^2 = mu^2), not numerical
    assert all(r == 0.0 for r in w["resonance_residuals"])
    m = np.linalg.norm(w["cross_derivative"](z[:, :2]), axis=(1, 2))
    assert m.mean() > 1.0, "M_BA is genuinely nonzero"
    assert w["unit_eigenvalue_distance"] > 0.1, "(D1') holds"


def test_multidegree_variance_is_exactly_the_gram_quadratic_form():
    """V_t(sigma) = s(sigma)^T C s(sigma) -- the identity the proof runs on."""
    rng = np.random.default_rng(4)
    w = S.multidegree_resonant_coupling()
    zeta = rng.normal(size=(400_000, 2))
    t = np.array([1.0, 1.0])

    x = zeta @ w["a_vec"]
    a1, a2 = w["c1"] * t[0] * x, w["c2"] * t[1] * x**2
    C = np.cov(np.stack([a1, a2]), ddof=0)

    for sigma in (0.6, 1.0, 1.6, 2.2):
        measured = float(np.var(w["psi"](sigma * zeta) @ t))
        s = np.array([sigma, sigma**2])
        assert measured == pytest.approx(float(s @ C @ s), rel=1e-10)


def test_symmetric_mu_A_kills_the_odd_coefficient_so_two_levels_suffice():
    """(R1)+(R2): Cov(A_1, A_2) is an odd moment of a symmetric law, hence 0.

    Asserted against a measured sampling floor, not against zero (3.9): the
    entry must fall like n^{-1/2}, which is what makes it noise rather than a
    small real value.
    """
    rng = np.random.default_rng(5)
    w = S.multidegree_resonant_coupling()

    def rms_offdiag(n, repeats=16):
        vals = []
        for _ in range(repeats):
            x = rng.normal(size=(n, 2)) @ w["a_vec"]
            vals.append(float(np.cov(np.stack([x, x**2]), ddof=0)[0, 1]))
        return float(np.sqrt(np.mean(np.square(vals))))

    small, large = rms_offdiag(5_000), rms_offdiag(80_000)
    # 16x the sample => the floor should drop by ~4x if it is n^{-1/2} noise
    assert small / large == pytest.approx(4.0, rel=0.6), f"{small} -> {large}"

    # and with the odd term absent V is strictly increasing, so 2 levels suffice
    info = S.required_behaviour_levels([1, 2], symmetric=True)
    assert info["two_levels_suffice"] and info["levels"] == 2
    assert info["unconditional_levels"] == 4, "worst case is |P+P|+1 = 4"


def test_level_count_reduces_to_lemma_d_prime_for_a_single_degree():
    for p in (1, 2, 3):
        info = S.required_behaviour_levels([p])
        assert info["levels"] == 2 and info["two_levels_suffice"]


def test_two_level_tie_threshold_predicts_every_measured_case():
    """(R3) is sharp: a tie exists iff |corr| clears the threshold."""
    rng = np.random.default_rng(6)
    n = 1_000_000
    sym = rng.normal(size=n)
    skew = rng.normal(size=n) * 0.30 + 1.0

    cases = [
        # (p, q, sample, expect_tie)
        (1, 2, sym, False),   # opposite parity under a symmetric law -> corr 0
        (1, 3, sym, False),   # EQUAL parity, corr = 3/sqrt(15) -- still no tie
        (2, 3, sym, False),
        (1, 2, skew, True),   # skewed: corr ~ 0.978 clears 0.969
    ]
    for p, q, sample, expect in cases:
        thr = S.two_level_tie_threshold(p, q, 0.6, 1.6)
        corr = abs(float(np.corrcoef(sample**p, sample**q)[0, 1]))
        assert (corr >= thr) == expect, f"P={{{p},{q}}}: corr {corr} vs thr {thr}"


def test_equal_parity_degrees_still_need_only_two_levels_under_a_gaussian():
    """The counting bound is conservative and (R3) says so.

    P = {1,3} shares parity, so (R2) does not apply and the count asks for 4.
    But corr(X, X^3) = 3/sqrt(15) = 0.775 is far below the 0.944 threshold, so
    two levels really do suffice.  Recorded because it is the one place the two
    criteria disagree.
    """
    assert S.required_behaviour_levels([1, 3], symmetric=True)["levels"] == 4
    thr = S.two_level_tie_threshold(1, 3, 0.6, 1.6)
    assert 3 / np.sqrt(15) < thr, f"3/sqrt(15) = {3 / np.sqrt(15)} vs {thr}"


def test_the_tie_defeats_the_second_moment_argument_not_the_lemma():
    """At a constructed two-level tie the VARIANCE matches and the law does not.

    This is the honest boundary of Lemma D'': the level count is a property of
    the argument.  (D4) asks for equality of laws, and skewness still separates
    the levels, so the coupling remains detectable.
    """
    rng = np.random.default_rng(7)
    n = 2_000_000
    x = (rng.normal(size=(n, 2)) * 0.30 + 1.0) @ np.array([1.0, 0.5])
    u1, u2 = x - x.mean(), x**2 - (x**2).mean()
    v11, v12, v22 = float(np.mean(u1 * u1)), float(np.mean(u1 * u2)), float(np.mean(u2 * u2))

    s1, s2 = 0.6, 1.6
    P_, Q_, R_ = s1 + s2, s1**2 + s1 * s2 + s2**2, (s1 + s2) * (s1**2 + s2**2)
    qa, qb, qc = v22 * R_, 2 * v12 * Q_, v11 * P_
    disc = qb**2 - 4 * qa * qc
    assert disc > 0, "the skewed law admits a tie -- that is the point"
    k = (-qb + np.sqrt(disc)) / (2 * qa)

    def y(sigma):
        return sigma * x + sigma**2 * k * x**2

    ya, yb = y(s1), y(s2)
    assert float(np.var(ya)) == pytest.approx(float(np.var(yb)), rel=1e-6)
    skew = [float(np.mean(((v - v.mean()) / v.std()) ** 3)) for v in (ya, yb)]
    assert abs(skew[0] - skew[1]) > 1.0, f"the law still moves: skew {skew}"


def test_separating_the_levels_shrinks_the_escape():
    """The threshold lives in [2 sqrt(pq)/(p+q), 1) and rises with separation.

    Design consequence: spread the two behaviour levels.  It costs nothing and
    it strictly raises the correlation a coupling needs in order to hide.
    """
    floor = 2 * np.sqrt(1 * 2) / (1 + 2)
    ts = [S.two_level_tie_threshold(1, 2, 1.0, r) for r in (1.001, 1.5, 5.0, 20.0)]
    assert ts[0] == pytest.approx(floor, rel=1e-3), "floor at coincident levels"
    assert all(a < b for a, b in zip(ts, ts[1:])), f"must rise: {ts}"
    assert ts[-1] > 0.998 and ts[-1] < 1.0

    # AM-GM: the threshold never reaches 1, for any degrees or levels
    rng = np.random.default_rng(9)
    for _ in range(2000):
        p, q = sorted(rng.integers(1, 9, size=2))
        s1, s2 = np.exp(rng.normal(size=2) * 2)
        if p == q or s1 == s2:
            continue
        assert S.two_level_tie_threshold(int(p), int(q), float(s1), float(s2)) <= 1.0

    for bad in ((2, 2, 0.6, 1.6), (1, 2, 0.6, 0.6), (1, 2, -1.0, 1.6)):
        with pytest.raises(ValueError):
            S.two_level_tie_threshold(*bad)


def test_surviving_degrees_are_finite_from_the_spectra_alone():
    """P is finite whenever f_A contracts and ftilde_B is invertible."""
    assert S.surviving_degree_bound(0.8, 0.64) == 2      # the witness: P subset {1,2}
    assert S.surviving_degree_bound(0.8, 0.10) == 10
    assert S.surviving_degree_bound(0.95, 0.5) == 13
    # B decaying slower than A: no degree >= 1 resonates, so psi = 0 outright
    assert S.surviving_degree_bound(0.5, 0.9) == 0

    for bad in ((1.2, 0.5), (0.0, 0.5), (0.8, 0.0)):
        with pytest.raises(ValueError):
            S.surviving_degree_bound(*bad)


def test_degree_zero_is_rejected_by_the_level_counter():
    """(D1') must be applied before counting -- degree 0 has no sigma-dependence."""
    with pytest.raises(ValueError):
        S.required_behaviour_levels([0, 2])
    with pytest.raises(ValueError):
        S.required_behaviour_levels([])


# --------------------------------------------------------------------------
# Lemma D''' -- nonlinear modules via Koopman eigenfunctions (4.5c)
#
# Step 1's  psi . f_A = Btilde psi  says each component of psi is a Koopman
# eigenfunction of f_A.  For linear f_A those are the monomials, recovering
# Step 2.  So Step 2 needs no hypothesis on the dynamics at all.
# --------------------------------------------------------------------------


def test_additive_h_B_forces_ftilde_B_affine():
    """Half of open item (c) is vacuous: nonlinear ftilde_B is out of the class.

    Step 1 splits  f_B(z_B) + psi(f_A z_A) = ftilde_B(z_B + psi(z_A))  only if
    ftilde_B is additive.  So this is not a gap in Lemma D.
    """
    rng = np.random.default_rng(11)
    mu, c = 0.7, 0.9
    za, zb = rng.normal(size=20_000) * 0.4, rng.normal(size=20_000) * 0.4

    def defect(f_b):
        return float(np.abs(f_b(zb) + c * mu * za - f_b(zb + c * za)).max())

    assert defect(lambda w: mu * w) < 1e-12, "affine: the split holds exactly"
    assert defect(lambda w: mu * w + 0.3 * w**2) > 0.5, "nonlinear: it fails outright"


def test_koopman_witness_is_an_exact_conjugacy_with_a_nonlinear_module():
    w = S.koopman_coupling_witness()
    rng = np.random.default_rng(12)
    z = np.column_stack([rng.normal(size=20_000) * 0.8, rng.normal(size=(20_000, 2))])
    assert np.abs(w["h"](w["F"](z)) - w["F"](w["h"](z))).max() < 1e-12
    assert w["unit_eigenvalue_distance"] > 0.1, "(D1'') holds"

    # f_A is genuinely nonlinear, not a perturbation of a linear map
    x = rng.normal(size=20_000) * 0.8
    fit = np.polyfit(x, w["f_A"](x), 1)
    resid = np.linalg.norm(w["f_A"](x) - np.polyval(fit, x)) / np.linalg.norm(w["f_A"](x))
    assert resid > 0.1, f"f_A must be far from linear, got residual {resid}"


def test_psi_components_are_koopman_eigenfunctions_of_f_A():
    """The reframed Step 2: psi_i . f_A = lambda_i psi_i, no linearity used."""
    w = S.koopman_coupling_witness(n_eig=3)
    z = np.random.default_rng(13).normal(size=20_000) * 0.8
    p0, p1 = w["phi"](z), w["phi"](w["f_A"](z))
    for j, lam in enumerate(w["koopman_eigenvalues"], start=1):
        assert np.abs(p1**j - lam * p0**j).max() < 1e-12, f"eigenfunction phi^{j}"
    # and the assembled psi is an exact semiconjugacy
    assert np.abs(w["psi"](w["f_A"](z)) - w["psi"](z) @ w["B_tilde"].T).max() < 1e-12


def test_the_limit_cycle_asymptotic_phase_is_a_koopman_eigenfunction():
    """Ties 4.5c to the torus counterexample: exp(i Theta) has eigenvalue e^{i omega}.

    True at every shear, which is why the 7.1 regrouping was shear-insensitive.
    """
    rng = np.random.default_rng(14)
    r = rng.uniform(0.5, 1.5, size=4000)
    th = rng.uniform(-np.pi, np.pi, size=4000)
    z = np.column_stack([r * np.cos(th), r * np.sin(th)])
    for beta in (0.0, 0.5, 1.2):
        blk = S.LimitCycleBlock(a=0.30, rho=1.0, omega=0.7, beta=beta)
        t0 = S.asymptotic_phase(blk, z)
        t1 = S.asymptotic_phase(blk, blk.step(z))
        err = np.abs(np.exp(1j * t1) - np.exp(1j * 0.7) * np.exp(1j * t0)).max()
        assert err < 1e-13, f"beta={beta}: {err}"


def test_psi_is_not_polynomial_so_the_finite_degree_count_does_not_apply():
    """tanh has infinitely many Taylor degrees -- Lemma D'''s P does not exist."""
    w = S.koopman_coupling_witness()
    assert w["psi_is_polynomial"] is False
    # fit a high-order polynomial to phi and check the coefficients do not
    # terminate: many orders carry real weight
    x = np.linspace(-1.0, 1.0, 4001)
    coef = np.polyfit(x, w["phi"](x), 21)
    assert sum(abs(c) > 1e-7 for c in coef) >= 8, f"expected many live orders: {coef}"


def test_k_koopman_eigenfunctions_tie_k_levels_so_no_finite_count_works():
    """(D2'') is load-bearing: the level count must grow with dim z_B.

    With k eigenfunctions there are k-1 free ratios, enough to tie k levels
    while psi stays nonzero.  Solved directly rather than by search.
    """
    rng = np.random.default_rng(15)
    zeta = rng.normal(size=400_000) * 0.5 + 0.9   # skewed, so parity does not save us

    def W(sigma, c):
        p = np.tanh(sigma * zeta)
        return float(np.var(sum(cj * p ** (j + 1) for j, cj in enumerate(c))))

    levels = (0.6, 1.0, 1.6)
    # solve the 2 tie equations for the 2 free ratios (c_1 fixed to 1 by scale)
    from scipy.optimize import fsolve

    def eqs(v):
        c = np.concatenate([[1.0], v])
        return [W(levels[i + 1], c) - W(levels[0], c) for i in range(len(levels) - 1)]

    sol = fsolve(eqs, -np.ones(2))
    c = np.concatenate([[1.0], sol])
    ws = [W(s, c) for s in levels]

    assert np.linalg.norm(c) > 0.5, "psi must be genuinely nonzero"
    assert max(ws) - min(ws) < 1e-10, f"three levels tied: {ws}"
    # a fourth level still separates -- the tie is not a degeneracy of psi
    assert abs(W(2.4, c) - ws[0]) > 1e-4


# ============================================================================
# Route B's exact reach: the compact-stabiliser dichotomy (Prop. S, section 14)
#
# The kill is not about additivity, it is about COMPACTNESS.  A coupling valued
# in a group acting non-compactly on z_B is detected; one valued in Stab(p_B) --
# always compact -- is not.  These pin both directions, plus the sharper fact
# that the *detector* is blinder than the criterion.
# ============================================================================

_N_DICH = 60_000
_SIGMAS_DICH = (0.7, 1.4)


def _dich_draw(kind, n, rng, mean_norm=0.0):
    u = rng.integers(0, 2, size=n)
    zA0 = np.asarray(_SIGMAS_DICH)[u] * rng.standard_normal(n)
    if kind == "iso":
        zB = rng.standard_normal((n, 2))
    elif kind == "aniso":
        zB = rng.standard_normal((n, 2)) * np.array([1.0, 2.0])
    elif kind == "skew":
        which = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
        ang = np.array([0.0, 1.75, 3.75])
        zB = 2.0 * np.stack([np.cos(ang), np.sin(ang)], 1)[which]
        zB = zB + 0.6 * rng.standard_normal((n, 2))
        zB = zB - zB.mean(0, keepdims=True)
        zB = np.linalg.solve(np.linalg.cholesky(np.cov(zB, rowvar=False)), zB.T).T
    else:  # pragma: no cover
        raise ValueError(kind)
    return zA0, zB + np.array([mean_norm, 0.0]), u


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def _matched(build, zA0, zB, target=0.30):
    """Bisect the coefficient so every arm displaces h_B by the same amount."""
    mv = lambda c: float(np.linalg.norm(build(c * zA0, zB) - zB, axis=1).mean()
                         / np.linalg.norm(zB, axis=1).mean())
    lo, hi = 1e-6, 1e3
    for _ in range(50):
        c = np.sqrt(lo * hi)
        lo, hi = (c, hi) if mv(c) < target else (lo, c)
    return build(np.sqrt(lo * hi) * zA0, zB)


def _dep(w, u):
    return B.block_u_dependence(w, u, normalize=True).total


def test_noncompact_coupling_groups_are_all_killed_not_just_translations():
    """Lemma D's additivity is the translation case of a much larger class.

    Scalings and shears are equally non-compact and equally detected, at matched
    displacement -- so the load-bearing hypothesis is non-compactness, not the
    additive form.
    """
    rng = np.random.default_rng(0)
    zA0, zB, u = _dich_draw("iso", _N_DICH, rng)
    floor = _dep(zB, u)
    builds = {
        "translation": lambda th, z: z + np.stack([th, np.zeros_like(th)], 1),
        "scaling": lambda th, z: np.exp(th)[:, None] * z,
        "shear": lambda th, z: np.einsum("nab,nb->na", np.stack(
            [np.stack([np.ones_like(th), th], -1),
             np.stack([np.zeros_like(th), np.ones_like(th)], -1)], -2), z),
    }
    for name, b in builds.items():
        assert _dep(_matched(b, zA0, zB), u) > 5 * floor, name


def test_a_rotation_is_invisible_against_an_isotropic_law():
    """Stab(N(0,I)) contains SO(2), so (D4) holds exactly with M_BA nonzero."""
    rng = np.random.default_rng(1)
    zA0, zB, u = _dich_draw("iso", _N_DICH, rng)
    hB = _matched(lambda th, z: np.einsum("nab,nb->na", _rot(th), z), zA0, zB)
    assert _dep(hB, u) < 2 * _dep(zB, u)


def test_the_blind_group_is_CONJUGATE_to_O2_not_equal_to_it():
    """Anisotropy does not remove the escape -- it relocates it.

    Against ``N(0, diag(1,4))`` a plain rotation is detected, while the
    conjugated rotation ``S^{1/2} R S^{-1/2}`` -- which *is* the stabiliser --
    is invisible.  Getting this backwards would read "make p_B anisotropic" as a
    fix, and it is not one.
    """
    rng = np.random.default_rng(2)
    zA0, zB, u = _dich_draw("aniso", _N_DICH, rng)
    floor = _dep(zB, u)
    Sh, Shi = np.diag([1.0, 2.0]), np.diag([1.0, 0.5])
    plain = _matched(lambda th, z: np.einsum("nab,nb->na", _rot(th), z), zA0, zB)
    conj = _matched(lambda th, z: np.einsum(
        "ab,nbc,cd,nd->na", Sh, _rot(th), Shi, z), zA0, zB)
    assert _dep(plain, u) > 5 * floor
    assert _dep(conj, u) < 2 * floor


def test_the_second_moment_detector_is_blinder_than_the_criterion():
    """Trivial stabiliser is NOT sufficient for ``block_u_dependence``.

    A centred, whitened, asymmetric ``p_B`` has trivial ``Stab_{O(2)}``, so a
    rotation genuinely breaks (D4) -- visible at the 3rd circular harmonic.  But
    a rotation fixes the mean (0) and the covariance (I), so a whitened
    second-moment detector cannot see it.  This is why section 3.12's
    gauge-invariance fix and section 13.3's rotational blindness are the same
    fact rather than two problems.
    """
    rng = np.random.default_rng(3)
    zA0, zB, u = _dich_draw("skew", 4 * _N_DICH, rng)
    np.testing.assert_allclose(zB.mean(0), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.cov(zB, rowvar=False), np.eye(2), atol=1e-9)
    hB = _matched(lambda th, z: np.einsum("nab,nb->na", _rot(th), z), zA0, zB)

    def harm(w, k):
        a = np.arctan2(w[:, 1], w[:, 0])
        v = [np.hypot(np.cos(k * a[u == j]).mean(), np.sin(k * a[u == j]).mean())
             for j in (0, 1)]
        return abs(v[0] - v[1])

    assert harm(hB, 3) > 20 * harm(zB, 3), "(D4) must genuinely fail here"
    assert _dep(hB, u) < 2 * _dep(zB, u), "and the detector must still miss it"


def test_a_nonzero_mean_is_what_restores_the_kill():
    """The one thing a whitened second-moment detector can still see.

    Design rule behind section 14.4: the pinned block needs a u-dependent mean
    direction, i.e. circular concentration above ~0.3.  exp18's fitted
    adversarial block sat at 0.270 -- just under.
    """
    rng = np.random.default_rng(4)
    ratios = []
    for m in (0.0, 1.0):
        zA0, zB, u = _dich_draw("iso", _N_DICH, rng, mean_norm=m)
        hB = _matched(lambda th, z: np.einsum("nab,nb->na", _rot(th), z), zA0, zB)
        ratios.append(_dep(hB, u) / _dep(zB, u))
    assert ratios[0] < 2.0 < 5.0 < ratios[1]
