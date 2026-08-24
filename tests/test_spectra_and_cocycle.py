"""Lyapunov spectra (§3.4) and the cocycle argument (§3.3).

The value of these tests is that both modules have *exactly known* answers:
TwistBlock has Lyapunov spectrum {log s, log s} by construction, so estimator
error is measurable rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from idyn import cocycle as CC
from idyn import spectra as SP
from idyn import systems as S


# --------------------------------------------------------------------------
# Lyapunov spectra
# --------------------------------------------------------------------------


def test_linear_system_lyapunov_spectrum_is_log_of_eigenvalue_moduli():
    blk = S.LinearBlock(np.diag([0.9, 0.5]))
    spec = SP.lyapunov_spectrum(blk, np.array([1.0, 1.0]), T=500, warmup=50)
    assert spec == pytest.approx([np.log(0.9), np.log(0.5)], abs=1e-9)


def test_twist_block_lyapunov_spectrum_matches_its_exact_value():
    blk = S.TwistBlock(s=0.85, omega=0.4, beta=0.9)
    spec = SP.lyapunov_spectrum(blk, np.array([0.9, 0.3]), T=2000, warmup=200)
    assert spec == pytest.approx(blk.lyapunov_spectrum_exact(), abs=1e-6)


def test_rotation_lyapunov_spectrum_is_degenerate_at_log_s():
    blk = S.LinearBlock(0.75 * S.rotation(0.37))
    spec = SP.lyapunov_spectrum(blk, np.array([1.0, 0.0]), T=1000, warmup=100)
    assert spec == pytest.approx([np.log(0.75)] * 2, abs=1e-9)


def test_module_spectra_and_gap_match_the_exact_values():
    rng = np.random.default_rng(0)
    sys = S.two_oscillator_system(s=(0.95, 0.70))
    z0s = S.sample_initial_conditions(4, 6, rng, radius=1.0)
    ms = SP.module_lyapunov_spectra(sys, z0s, T=800, warmup=150)
    assert ms.spectra[0] == pytest.approx([np.log(0.95)] * 2, abs=1e-6)
    assert ms.spectra[1] == pytest.approx([np.log(0.70)] * 2, abs=1e-6)
    assert ms.gap == pytest.approx(abs(np.log(0.95) - np.log(0.70)), abs=1e-6)
    assert ms.separated


def test_equal_moduli_give_zero_gap():
    rng = np.random.default_rng(1)
    sys = S.two_oscillator_system(s=(0.85, 0.85), omega=(0.4, 1.1))
    z0s = S.sample_initial_conditions(4, 4, rng, radius=1.0)
    ms = SP.module_lyapunov_spectra(sys, z0s, T=600, warmup=100)
    assert ms.gap == pytest.approx(0.0, abs=1e-6)
    assert not ms.separated


def test_lyapunov_gap_is_independent_of_the_visited_region():
    """The §3.4 point: unlike pointwise Jacobian spectra, this does not move."""
    rng = np.random.default_rng(2)
    sys = S.two_oscillator_system(s=(0.95, 0.70))
    gaps = []
    for radius in (0.5, 1.5, 2.5):
        z0s = S.sample_initial_conditions(4, 4, rng, radius=radius)
        gaps.append(SP.module_lyapunov_spectra(sys, z0s, T=600, warmup=150).gap)
    assert max(gaps) - min(gaps) < 1e-6


def test_pointwise_jacobian_spectra_do_move():
    """Companion to the above: the draft's Assumption 4 is regime-dependent."""
    sys = S.two_oscillator_system(s=(0.95, 0.70), omega=(0.40, 1.10), beta=(0.60, -0.50))
    f1, f2 = sys.blocks

    def closest(radius):
        g = np.linspace(-radius, radius, 40)
        pts = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        s1 = np.array([np.linalg.eigvals(f1.jacobian(p)) for p in pts])
        s2 = np.array([np.linalg.eigvals(f2.jacobian(p)) for p in pts])
        return np.abs(s1[:, None, :, None] - s2[None, :, None, :]).min()

    assert closest(0.8) > 1e-2
    assert closest(2.4) < 1e-2


def test_jacobian_product_logs_match_a_direct_product_when_it_does_not_underflow():
    blk = S.TwistBlock(s=0.95, omega=0.4, beta=0.6)
    z = np.array([0.8, 0.2])
    smax, smin = SP.jacobian_product_logs(blk, z, 20)
    P = np.eye(2)
    zz = z.copy()
    for n in range(20):
        P = blk.jacobian(zz) @ P
        zz = blk.step(zz)
        sv = np.linalg.svd(P, compute_uv=False)
        assert smax[n] == pytest.approx(np.log(sv[0]), abs=1e-10)
        assert smin[n] == pytest.approx(np.log(sv[-1]), abs=1e-10)


def test_jacobian_product_logs_survive_far_past_underflow():
    """A raw product would be ~0.7^800 ~ 1e-124; the log form must stay exact."""
    blk = S.LinearBlock(np.diag([0.7, 0.7]))
    smax, smin = SP.jacobian_product_logs(blk, np.array([1.0, 1.0]), 800)
    assert smax[-1] == pytest.approx(800 * np.log(0.7), rel=1e-12)
    assert np.isfinite(smin[-1])


# --------------------------------------------------------------------------
# Cocycle
# --------------------------------------------------------------------------


def test_cocycle_rate_matches_the_lyapunov_prediction():
    f_target = S.TwistBlock(s=0.95, omega=0.4, beta=0.6)
    f_source = S.TwistBlock(s=0.70, omega=1.1, beta=-0.5)
    predicted = np.log(0.70) - np.log(0.95)
    cb = CC.cocycle_bound(f_target, np.array([0.9, 0.1]), f_source, np.array([0.7, 0.3]),
                          n_max=300, predicted_rate=predicted)
    assert cb.rate == pytest.approx(predicted, abs=1e-6)
    assert cb.forces_M_zero


def test_cocycle_stalls_without_a_gap():
    f_target = S.TwistBlock(s=0.85, omega=0.4, beta=0.6)
    f_source = S.TwistBlock(s=0.85, omega=1.1, beta=-0.5)
    cb = CC.cocycle_bound(f_target, np.array([0.9, 0.1]), f_source, np.array([0.7, 0.3]), n_max=300)
    assert cb.rate == pytest.approx(0.0, abs=1e-6)
    assert not cb.forces_M_zero


def test_cocycle_direction_matters():
    """Swapping target and source flips the sign: the gap is directional."""
    fast = S.TwistBlock(s=0.95, omega=0.4, beta=0.6)
    slow = S.TwistBlock(s=0.70, omega=1.1, beta=-0.5)
    a = CC.cocycle_bound(fast, np.array([0.9, 0.1]), slow, np.array([0.7, 0.3]), n_max=200)
    b = CC.cocycle_bound(slow, np.array([0.7, 0.3]), fast, np.array([0.9, 0.1]), n_max=200)
    assert a.rate < 0 < b.rate
    assert a.rate == pytest.approx(-b.rate, abs=1e-6)


def test_propagate_M_decays_at_the_cocycle_rate():
    """log||M_n|| = rate * n + bounded oscillation.

    The oscillation is real, not estimator error: the two modules rotate at
    different angles (omega 0.4 vs 1.1), so the alignment of M_0 with the
    singular directions of the two cocycles is quasi-periodic in n.  It stays
    bounded, so the rate is exact in the limit, but a finite-window slope
    inherits an O(1/window) bias -- hence abs=1e-3 rather than 1e-6.
    """
    f_target = S.TwistBlock(s=0.95, omega=0.4, beta=0.6)
    f_source = S.TwistBlock(s=0.70, omega=1.1, beta=-0.5)
    exact = np.log(0.70) - np.log(0.95)
    logM = CC.propagate_M(f_target, np.array([0.9, 0.1]), f_source, np.array([0.7, 0.3]),
                          np.eye(2), n_max=600)
    n = np.arange(1, 601, dtype=float)

    slope = np.polyfit(n[300:], logM[300:], 1)[0]
    assert slope == pytest.approx(exact, abs=1e-3)

    residual = logM - exact * n
    assert np.ptp(residual[300:]) < 2.0, "residual is bounded, so the rate is exact"
    assert logM[-1] < -100, "M is driven to zero"


def test_propagate_M_does_not_decay_without_a_gap():
    f = S.TwistBlock(s=0.85, omega=0.4, beta=0.6)
    g = S.TwistBlock(s=0.85, omega=1.1, beta=-0.5)
    logM = CC.propagate_M(f, np.array([0.9, 0.1]), g, np.array([0.7, 0.3]), np.eye(2), n_max=300)
    assert logM[-1] > -5.0


def test_propagate_M_uses_the_correct_product_order():
    """Df^(n) = J(z_{n-1}) ... J(z_0); the reversed order is a different matrix."""
    f = S.TwistBlock(s=0.95, omega=0.4, beta=0.6)
    g = S.TwistBlock(s=0.70, omega=1.1, beta=-0.5)
    z_t, z_s, M0 = np.array([0.9, 0.1]), np.array([0.7, 0.3]), np.eye(2)
    logM = CC.propagate_M(f, z_t, g, z_s, M0, n_max=3)

    Pt, Ps, zt, zs = np.eye(2), np.eye(2), z_t.copy(), z_s.copy()
    for n in range(3):
        Pt = f.jacobian(zt) @ Pt
        Ps = g.jacobian(zs) @ Ps
        zt, zs = f.step(zt), g.step(zs)
        expected = np.log(np.linalg.norm(np.linalg.solve(Pt, M0 @ Ps)))
        assert logM[n] == pytest.approx(expected, abs=1e-9)


def test_forces_M_zero_ignores_a_numerically_zero_rate():
    """A rate of -1e-17 forces nothing; the sign of a rounding error must not decide.

    The no-gap case fits a rate that is zero up to float noise, and which side
    of zero it lands on is arbitrary.  Testing ``rate < 0`` would let that decide
    whether the block-separation step reports as closing.
    """
    cb = CC.CocycleBound(n=np.arange(1.0), log_bound=np.zeros(1), rate=-1e-17)
    assert not cb.forces_M_zero
    assert not CC.CocycleBound(n=np.arange(1.0), log_bound=np.zeros(1), rate=0.0).forces_M_zero
    assert CC.CocycleBound(n=np.arange(1.0), log_bound=np.zeros(1), rate=-0.3).forces_M_zero


# --------------------------------------------------------------------------
# Conditioning: sigma_min of an accumulated product is not measurable
# --------------------------------------------------------------------------


def test_resolvable_horizon_predicts_where_sigma_min_dies():
    """cond(Df^n) = exp(n*spread) outruns float64 at n ~ 36/spread."""
    blk = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    spread = float(np.ptp(blk.lyapunov_spectrum_exact()))
    horizon = SP.resolvable_horizon(spread)
    assert horizon == pytest.approx(36.04 / spread, rel=1e-3)
    assert 30.0 < horizon < 45.0

    # below the horizon sigma_min is signal, above it is not
    z = np.array([1.0, 0.0])
    smax, smin = SP.jacobian_product_logs(blk, z, 120)
    n = np.arange(1, 121, dtype=float)
    below = np.polyfit(n[5:20], smin[5:20], 1)[0]
    above = np.polyfit(n[60:120], smin[60:120], 1)[0]
    lmin = float(blk.lyapunov_spectrum_exact().min())
    assert below == pytest.approx(lmin, abs=1e-3)
    assert abs(above - lmin) > 0.5, "past the horizon the slope is noise, not lambda_min"


def test_resolvable_horizon_is_infinite_for_a_flat_spectrum():
    """Why every exp05 measurement is sound: a TwistBlock has spread exactly 0."""
    assert SP.resolvable_horizon(0.0) == float("inf")
    blk = S.TwistBlock(s=0.95, omega=0.4, beta=0.6)
    assert SP.resolvable_horizon(float(np.ptp(blk.lyapunov_spectrum_exact()))) == float("inf")
    smax, smin = SP.jacobian_product_logs(blk, np.array([0.9, 0.1]), 400)
    assert smax[-1] - smin[-1] < 10.0, "condition number stays O(1) at n=400"


def test_inverse_product_agrees_with_sigma_min_while_sigma_min_is_still_accurate():
    """Equal in exact arithmetic; in float64 they part company at rate cond*eps.

    The horizon is where sigma_min becomes pure noise, but accuracy decays
    smoothly long before that: the absolute error in ``log sigma_min`` tracks the
    relative error in ``sigma_min``, which is ``cond(Df^n) * eps = exp(n*spread)*eps``.
    So the two routes agree to ~1e-11 at n = 12 and to only ~1e-7 by n = 25.
    """
    blk = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    z = np.array([1.0, 0.0])
    spread = float(np.ptp(blk.lyapunov_spectrum_exact()))
    eps = float(np.finfo(float).eps)

    inv = SP.inverse_jacobian_product_logs(blk, z, 25)
    _, smin = SP.jacobian_product_logs(blk, z, 25)
    assert inv[:12] == pytest.approx(-smin[:12], abs=1e-9)

    err = np.abs(inv + smin)
    budget = np.exp(np.arange(1, 26) * spread) * eps
    assert np.all(err <= 100.0 * budget), "discrepancy is bounded by cond * eps"
    assert err[24] > 100.0 * err[11], "and it grows exponentially, as that predicts"


@pytest.mark.parametrize("beta", [0.0, 0.6, 1.5])
def test_inverse_product_recovers_lambda_min_on_a_limit_cycle(beta):
    """Sound at n = 400, where the naive route is off by O(1) -- for any beta."""
    blk = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=beta)
    lmin = float(blk.lyapunov_spectrum_exact().min())
    y = SP.inverse_jacobian_product_logs(blk, np.array([1.0, 0.0]), 400)
    n = np.arange(1, 401, dtype=float)
    assert np.polyfit(n[200:], y[200:], 1)[0] == pytest.approx(-lmin, abs=1e-9)


def test_beta_shifts_the_intercept_not_the_rate():
    """The shear is non-normal, but on the cycle it contributes a constant.

    In the polar frame J = [[1-2a, 0], [beta, 1]], so
    J^n = [[(1-2a)^n, 0], [beta(1-(1-2a)^n)/2a, 1]] and sigma_max(J^n) tends to
    sqrt((beta/2a)^2 + 1) -- a constant.  At rho = 1 the polar frame is
    orthogonal, so the Cartesian singular values agree with it exactly.
    """
    a = 0.3
    for beta in (0.0, 0.6, 1.5):
        blk = S.LimitCycleBlock(a=a, rho=1.0, omega=0.5, beta=beta)
        smax, _ = SP.jacobian_product_logs(blk, np.array([1.0, 0.0]), 60)
        plateau = float(np.log(np.sqrt((beta / (2 * a)) ** 2 + 1.0)))
        assert smax[-1] == pytest.approx(plateau, abs=1e-9)


def test_naive_sigma_min_route_is_noise_on_a_limit_cycle():
    """The sound rate is exact; the naive one reads the wrong exponent.

    Past the resolvable horizon the SVD returns its noise floor, whose slope is
    lambda_max, not lambda_min -- so the naive rate lands on
    log s - lambda_max(cycle) instead of log s - lambda_min(cycle), an error of
    exactly the cycle's spread (CLAUDE.md 3.9).

    We assert that *wrongness*, not the n_max-instability an earlier revision of
    this test used as its proxy.  Whether the noise floor wanders or sits still
    is a property of the BLAS, and BOTH behaviours have now been observed on the
    same code: on numpy 2.5 / scipy 1.18 the naive rate converges cleanly to
    ``wrong_limit``; on numpy 2.4.3 it wanders across
    (-1.96, +0.73, -1.00, -1.15) as n_max runs (100, 200, 400, 800), landing
    within 0.2*spread of the wrong limit only at the last two.  A revision that
    asserted the convergence therefore failed here for a reason that has nothing
    to do with the defect.

    So the regression pins the one clause that is true in both environments and
    is what §3.9 actually claims: **the naive rate misses the truth by more than
    half the block's spread, at every horizon.**  Neither the variance of the
    error nor the particular wrong value it converges to is a property of the
    defect.

    Note the n_max=200 entry above: +0.73 against a true -0.29.  The naive route
    does not merely mis-estimate the rate, it can return the wrong *sign* -- a
    false negative on Lemma C, the mirror of the false positive pinned by
    ``test_the_naive_route_can_invent_a_gap_that_is_not_there``.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    fast = S.TwistBlock(s=0.30, omega=1.1, beta=-0.5)
    z_c, z_f = np.array([1.0, 0.0]), np.array([0.7, 0.3])
    lyap = cyc.lyapunov_spectrum_exact()
    predicted = float(np.log(0.30)) - float(lyap.min())
    wrong_limit = float(np.log(0.30)) - float(lyap.max())
    spread = float(lyap.max() - lyap.min())

    sound, naive = [], []
    for n_max in (100, 200, 400, 800):
        cb = CC.cocycle_bound(cyc, z_c, fast, z_f, n_max=n_max, predicted_rate=predicted)
        sound.append(cb.rate)
        naive.append(cb.naive_rate)
        assert not cb.naive_route_valid, "the fit window is past the horizon here"

    assert np.ptp(sound) < 1e-9, "the sound rate does not depend on n_max"
    assert all(r == pytest.approx(predicted, abs=1e-9) for r in sound)

    # the naive route misses the truth by more than half the block's spread.
    # This is the environment-independent clause; see the docstring for why the
    # "converges to wrong_limit" clause is not asserted.
    assert all(abs(r - predicted) > 0.5 * spread for r in naive)
    assert min(abs(r - wrong_limit) for r in naive) < 0.2 * spread, (
        "the naive rate should reach the lambda_max limit at some horizon, "
        f"even where it does not stay there: {naive}"
    )


def test_the_naive_route_can_invent_a_gap_that_is_not_there():
    """The failure that matters: a false positive on Lemma C's conclusion.

    With s = 0.45 the source contracts *slower* than the cycle's slowest mode,
    so there is no oriented gap and M is not forced to zero.  The naive route
    reports a decisively negative rate anyway.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    slow = S.TwistBlock(s=0.45, omega=1.1, beta=-0.5)
    predicted = float(np.log(0.45)) - float(cyc.lyapunov_spectrum_exact().min())
    assert predicted > 0.0

    cb = CC.cocycle_bound(cyc, np.array([1.0, 0.0]), slow, np.array([0.7, 0.3]),
                          n_max=400, predicted_rate=predicted)
    assert cb.rate == pytest.approx(predicted, abs=1e-9)
    assert not cb.forces_M_zero
    assert cb.naive_rate < -0.5, "the discarded route claims a gap that does not exist"


# --------------------------------------------------------------------------
# Lemma C on a genuine attractor (CLAUDE.md task 22)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("s_fast", [0.20, 0.25, 0.30, 0.35])
def test_lemma_C_rate_holds_on_a_limit_cycle_attractor(s_fast):
    """The cocycle argument never used the fixed point, and indeed does not need it.

    A LimitCycleBlock is a genuine attractor with no fixed point on it.  The
    predicted rate lambda_max(source) - lambda_min(target) holds to ~1e-9.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    fast = S.TwistBlock(s=s_fast, omega=1.1, beta=-0.5)
    predicted = float(np.log(s_fast)) - float(cyc.lyapunov_spectrum_exact().min())
    cb = CC.cocycle_bound(cyc, np.array([1.0, 0.0]), fast, np.array([0.7, 0.3]),
                          n_max=400, predicted_rate=predicted)
    assert cb.rate == pytest.approx(predicted, abs=1e-9)
    assert cb.forces_M_zero


def test_limit_cycle_gap_threshold_sits_exactly_at_the_radial_multiplier():
    """M is forced to zero iff s < |1 - 2a|; the crossing is at the predicted point."""
    a = 0.3
    cyc = S.LimitCycleBlock(a=a, rho=1.0, omega=0.5, beta=0.6)
    threshold = abs(1.0 - 2.0 * a)
    for s_fast in (0.30, 0.38, 0.42, 0.50):
        cb = CC.cocycle_bound(cyc, np.array([1.0, 0.0]),
                              S.TwistBlock(s=s_fast, omega=1.1, beta=-0.5),
                              np.array([0.7, 0.3]), n_max=300)
        assert cb.forces_M_zero == (s_fast < threshold)


def test_a_limit_cycle_can_only_ever_be_the_dominant_module():
    """Its neutral phase exponent is 0, and no contraction has lambda_min > 0.

    So the oriented gap lambda_max(cycle) < lambda_min(other) is unavailable for
    every contracting partner: an oscillatory module sits at the top of the
    filtration or is not separated at all.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    assert float(cyc.lyapunov_spectrum_exact().max()) == pytest.approx(0.0, abs=1e-12)
    for s_other in (0.20, 0.50, 0.90):
        other = S.TwistBlock(s=s_other, omega=1.1, beta=-0.5)
        # cycle as the *source* (dominated): rate = lambda_max(cycle) - lambda_min(other)
        cb = CC.cocycle_bound(other, np.array([0.7, 0.3]), cyc, np.array([1.0, 0.0]), n_max=300)
        assert not cb.forces_M_zero
        assert cb.rate == pytest.approx(-float(np.log(s_other)), abs=1e-9)


def test_limit_cycle_exponents_are_uniform_over_the_whole_basin():
    """Lemma C' concludes on all of Omega, and this is why.

    Oseledets alone gives the rate only mu-a.e., and for a limit cycle supp mu is
    the cycle -- which would leave the basin untouched, the same weakness that
    makes Route 2 of identifiability.md §5.1 useless.  A normally hyperbolic
    attracting circle has the *same* exponents at every basin point, so the
    cocycle bound decays at every z and M vanishes on the whole basin.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    fast = S.TwistBlock(s=0.30, omega=1.1, beta=-0.5)
    predicted = float(np.log(0.30)) - float(cyc.lyapunov_spectrum_exact().min())

    # basin is r < rho * sqrt((1+a)/a) = 2.0817; sweep 100x of radius inside it
    rates = [
        CC.cocycle_bound(cyc, np.array([r0, 0.0]), fast, np.array([0.7, 0.3]),
                         n_max=400, predicted_rate=predicted).rate
        for r0 in (0.02, 0.2, 0.9, 1.0, 1.5, 2.05)
    ]
    assert np.ptp(rates) < 1e-12, "the rate must not depend on where in the basin we start"
    assert all(r == pytest.approx(predicted, abs=1e-9) for r in rates)


def test_limit_cycle_basin_is_bounded():
    """Outside r = rho sqrt((1+a)/a) the radial map goes negative and escapes."""
    a, rho = 0.3, 1.0
    cyc = S.LimitCycleBlock(a=a, rho=rho, omega=0.5, beta=0.6)
    boundary = rho * np.sqrt((1.0 + a) / a)
    assert boundary == pytest.approx(2.0817, abs=1e-3)

    assert float(cyc._g(np.array(0.95 * boundary))) > 0.0
    assert float(cyc._g(np.array(1.05 * boundary))) < 0.0

    # and the guard fires rather than returning a plausible wrong number
    with pytest.raises(np.linalg.LinAlgError):
        SP.inverse_jacobian_product_logs(cyc, np.array([3.0, 0.0]), 400)


def test_propagate_M_is_sound_on_a_limit_cycle():
    """The sharper check must survive the attractor too.

    Solving against the accumulated target product (the obvious implementation)
    projects onto its dominant singular direction once it is numerically
    rank-deficient, and then measures lambda_max(source) - lambda_MAX(target).
    Here that would be -1.204 instead of -0.288.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    fast = S.TwistBlock(s=0.30, omega=1.1, beta=-0.5)
    exact = float(np.log(0.30)) - float(cyc.lyapunov_spectrum_exact().min())
    wrong = float(np.log(0.30)) - float(cyc.lyapunov_spectrum_exact().max())

    logM = CC.propagate_M(cyc, np.array([1.0, 0.0]), fast, np.array([0.7, 0.3]),
                          np.eye(2), n_max=400)
    n = np.arange(1, 401, dtype=float)
    slope = float(np.polyfit(n[200:], logM[200:], 1)[0])
    assert slope == pytest.approx(exact, abs=1e-3)
    assert abs(slope - wrong) > 0.5
    assert logM[-1] < -100, "M is driven to zero on the attractor"


def test_two_limit_cycles_can_never_be_separated():
    """Each contributes a 0 exponent, so the modules share one and (B4) fails."""
    c1 = S.LimitCycleBlock(a=0.30, rho=1.0, omega=0.50, beta=0.6)
    c2 = S.LimitCycleBlock(a=0.45, rho=1.0, omega=0.90, beta=-0.4)
    s1, s2 = c1.lyapunov_spectrum_exact(), c2.lyapunov_spectrum_exact()
    assert SP.spectral_gap([s1, s2]) == pytest.approx(0.0, abs=1e-15)

    z = np.array([1.0, 0.0])
    for target, source in ((c1, c2), (c2, c1)):
        predicted = float(source.lyapunov_spectrum_exact().max()) - float(
            target.lyapunov_spectrum_exact().min()
        )
        cb = CC.cocycle_bound(target, z, source, np.array([1.0, 0.2]), n_max=300,
                              predicted_rate=predicted)
        assert predicted > 0.0
        assert cb.rate == pytest.approx(predicted, abs=1e-9)
        assert not cb.forces_M_zero


# --------------------------------------------------------------------------
# Ordered separation -- hypothesis (F3) of Theorem F, identifiability.md §6.1
#
# spectral_gap is (B4): no two modules SHARE an exponent.  filtration_gap is
# (F3): the module spectra occupy disjoint ordered INTERVALS.  The gap between
# the two is not academic -- it is exactly the §3.1 regrouping counterexample.
# --------------------------------------------------------------------------


def test_filtration_gap_rejects_the_regrouping_that_spectral_gap_accepts():
    """(B4) passes the §3.1 counterexample at +0.18; (F3) rejects it at -0.22.

    This is the reason Theorem F is stated with ordered separation rather than
    disjointness: the regrouped representation keeps every exponent distinct
    while interleaving the two hulls, so no chain of oriented gaps exists and
    Lemma C has nothing to work with in either direction.
    """
    lg = np.log(np.array([0.90, 0.75, 0.60, 0.45]))
    true_grouping = [lg[[0, 1]], lg[[2, 3]]]
    regrouped = [lg[[0, 2]], lg[[1, 3]]]

    assert SP.spectral_gap(true_grouping) > 0.0
    assert SP.spectral_gap(regrouped) > 0.0, "(B4) cannot see the regrouping"
    assert SP.spectral_gap(regrouped) == pytest.approx(0.1823, abs=1e-3)

    assert SP.filtration_gap(true_grouping).ordered
    assert SP.filtration_gap(true_grouping).gap == pytest.approx(0.2231, abs=1e-3)
    assert not SP.filtration_gap(regrouped).ordered, "(F3) must see it"
    assert SP.filtration_gap(regrouped).gap == pytest.approx(-0.2231, abs=1e-3)


def test_filtration_gap_orders_modules_slowest_first():
    """Index 1 is the module with the largest exponents -- the top of the flag."""
    slow = np.array([-0.05, -0.05])
    fast = np.array([-0.90, -1.20])
    assert SP.filtration_gap([fast, slow]).order == [1, 0]
    assert SP.filtration_gap([slow, fast]).order == [0, 1]
    assert SP.filtration_gap([slow, fast]).gap == pytest.approx(0.85, abs=1e-12)


@pytest.mark.parametrize("s_fast", [0.20, 0.25, 0.30, 0.35, 0.38, 0.42, 0.50])
def test_filtration_gap_predicts_the_measured_cocycle_threshold(s_fast):
    """(F3) computed from spectra alone reproduces exp08's crossing, no free parameter.

    The cycle's hull is [log|1-2a|, 0], so a partner is ordered-separated from it
    iff log(s_fast) < log|1-2a| -- which is exactly where cocycle_bound flips.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)
    fast = S.TwistBlock(s=s_fast, omega=1.1, beta=-0.5)
    fo = SP.filtration_gap([cyc.lyapunov_spectrum_exact(), fast.lyapunov_spectrum_exact()])
    cb = CC.cocycle_bound(cyc, np.array([1.0, 0.0]), fast, np.array([0.7, 0.3]), n_max=300)
    assert fo.ordered == cb.forces_M_zero


def test_a_wide_hull_can_swallow_a_narrower_one_even_when_it_is_on_top():
    """Rider 1 does not make an oscillatory module safe (identifiability.md §6.5).

    A limit cycle spans [log|1-2a|, 0].  A contracting partner sitting *inside*
    that range is unseparable in both directions, even though the cycle has the
    larger lambda_max -- ordered separation is about intervals, not about which
    module is fastest.
    """
    cyc = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.6)   # [-0.9163, 0]
    mid = S.TwistBlock(s=0.50, omega=1.1, beta=-0.5)               # [-0.6931]*2
    specs = [cyc.lyapunov_spectrum_exact(), mid.lyapunov_spectrum_exact()]

    assert SP.spectral_gap(specs) > 0.0, "(B4) is satisfied -- and is not enough"
    assert not SP.filtration_gap(specs).ordered

    z_cyc, z_mid = np.array([1.0, 0.0]), np.array([0.7, 0.3])
    for target, z_t, source, z_s in ((cyc, z_cyc, mid, z_mid), (mid, z_mid, cyc, z_cyc)):
        assert not CC.cocycle_bound(target, z_t, source, z_s, n_max=300).forces_M_zero


def test_two_limit_cycles_fail_ordered_separation_as_well_as_disjointness():
    """Rider 2: identical hulls, so (F3) fails no matter how the two are ordered."""
    c1 = S.LimitCycleBlock(a=0.30, rho=1.0, omega=0.50, beta=0.6)
    c2 = S.LimitCycleBlock(a=0.45, rho=1.0, omega=0.90, beta=-0.4)
    specs = [c1.lyapunov_spectrum_exact(), c2.lyapunov_spectrum_exact()]
    assert not SP.filtration_gap(specs).ordered
    assert SP.filtration_gap(specs).gap < 0.0


# --------------------------------------------------------------------------
# Rotation number -- the invariant the Lyapunov spectrum cannot see (task 23)
# --------------------------------------------------------------------------


def _annulus(rng, n=6, lo=0.6, hi=1.1):
    th = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(lo, hi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)


@pytest.mark.parametrize("omega", [0.2, 0.4, 1.1])
def test_twist_block_rotation_number_is_omega_over_two_pi(omega):
    blk = S.TwistBlock(s=0.9, omega=omega, beta=0.6)
    r = SP.rotation_number_averaged(blk, _annulus(np.random.default_rng(0)), T=1200)
    assert abs(r.rho) == pytest.approx(omega / (2 * np.pi), abs=1e-9)
    assert r.coherence == pytest.approx(1.0, abs=1e-9)
    assert r.well_defined


@pytest.mark.parametrize("beta", [0.0, 0.4])
def test_limit_cycle_rotation_number_is_omega_and_does_not_depend_on_beta(beta):
    """On the cycle r = rho the shear term beta (r - rho) vanishes identically."""
    blk = S.LimitCycleBlock(a=0.3, omega=0.5, beta=beta)
    r = SP.rotation_number_averaged(blk, _annulus(np.random.default_rng(0)), T=1200)
    assert abs(r.rho) == pytest.approx(0.5 / (2 * np.pi), abs=1e-9)


def test_the_spectrum_is_blind_to_frequency_and_the_rotation_number_is_not():
    """Task 23 in one assertion: this is why the rotation number exists here.

    Two limit cycles with different omega have *identical* Lyapunov spectra, so
    ``spectral_gap`` is exactly 0 and Lemma C has nothing to work with.  Their
    rotation numbers differ by a wide margin.
    """
    rng = np.random.default_rng(0)
    z0 = _annulus(rng, lo=0.9, hi=1.1)
    a, b = S.LimitCycleBlock(a=0.3, omega=0.5), S.LimitCycleBlock(a=0.3, omega=1.3)
    sa = SP.lyapunov_spectrum_averaged(a, z0, T=800)
    sb = SP.lyapunov_spectrum_averaged(b, z0, T=800)
    assert np.abs(sa - sb).max() < 1e-9
    assert SP.spectral_gap([sa, sb]) == pytest.approx(0.0, abs=1e-9)

    ra = SP.rotation_number_averaged(a, z0, T=1200)
    rb = SP.rotation_number_averaged(b, z0, T=1200)
    assert abs(abs(ra.rho) - abs(rb.rho)) == pytest.approx(0.8 / (2 * np.pi), abs=1e-9)


def test_a_non_rotating_block_has_rotation_number_zero():
    blk = S.LinearBlock(np.diag([0.9, 0.5]))
    r = SP.rotation_number_averaged(blk, _annulus(np.random.default_rng(0)), T=1200)
    assert r.rho == pytest.approx(0.0, abs=1e-12)


def test_a_negative_eigenvalue_reads_as_half_a_turn():
    """A period-2 flip *is* a rotation number of 1/2, not an estimator failure."""
    blk = S.LinearBlock(np.diag([-0.8, 0.5]))
    r = SP.rotation_number_averaged(blk, _annulus(np.random.default_rng(0)), T=1200)
    assert abs(r.rho) == pytest.approx(0.5, abs=1e-12)


def test_rotation_number_survives_a_nonlinear_change_of_coordinates():
    """It is a conjugacy invariant, which is the whole reason it is usable.

    h(x, y) = (x + c y^3, y) is a within-module gauge change -- exactly the
    freedom §7 grants and never identifies.
    """

    class Sheared:
        def __init__(self, blk, c=0.7):
            self.blk, self.c, self.dim = blk, c, blk.dim

        def _h(self, z):
            z = np.asarray(z, float)
            return np.stack([z[..., 0] + self.c * z[..., 1] ** 3, z[..., 1]], axis=-1)

        def _hinv(self, w):
            w = np.asarray(w, float)
            return np.stack([w[..., 0] - self.c * w[..., 1] ** 3, w[..., 1]], axis=-1)

        def step(self, w):
            return self._h(self.blk.step(self._hinv(w)))

    z0 = _annulus(np.random.default_rng(0))
    for blk in (S.TwistBlock(s=0.9, omega=0.4, beta=0.6), S.LimitCycleBlock(a=0.3, omega=0.5)):
        raw = SP.rotation_number_averaged(blk, z0, T=1500, warmup=300)
        con = SP.rotation_number_averaged(Sheared(blk), z0, T=1500, warmup=300)
        # Exact for a rigid rotation; O(1/T) once h distorts the cycle, because
        # the window then covers a partial period.  Never worse than that.
        assert abs(raw.rho - con.rho) < 1e-3


@pytest.mark.parametrize("s,min_n", [(0.9, 2000), (0.5, 200), (0.2, 80)])
def test_a_contracting_orbit_reports_its_horizon_and_still_gets_the_answer(s, min_n):
    """§3.9 discipline: stop at the underflow, report n_used, do not average noise.

    The angle accumulates linearly in t, so a short clean window is worth more
    than a long dirty one -- unlike a Lyapunov average, truncation costs nothing
    here.  The answer is exact at every contraction rate; only ``n_used`` moves.
    """
    blk = S.TwistBlock(s=s, omega=0.7, beta=0.6)
    r = SP.rotation_number_averaged(blk, _annulus(np.random.default_rng(0)), T=4000, warmup=100)
    assert abs(r.rho) == pytest.approx(0.7 / (2 * np.pi), abs=1e-9)
    assert r.n_used >= min_n
    assert r.n_used < 4002


def test_a_one_dimensional_block_has_no_rotation_number():
    r = SP.rotation_number(S.LinearBlock([[0.7]]), np.array([1.0]), T=200)
    assert np.isnan(r.rho)
    assert not r.well_defined


def test_module_rotation_numbers_matches_the_per_block_answers():
    sysm = S.ModularSystem([S.TwistBlock(s=0.90, omega=0.40, beta=0.6),
                            S.LimitCycleBlock(a=0.3, omega=1.30)])
    rng = np.random.default_rng(0)
    z0 = np.concatenate([_annulus(rng), _annulus(rng, lo=0.9, hi=1.1)], axis=-1)
    rot = SP.module_rotation_numbers(sysm, z0, T=1200)
    assert [abs(r.rho) for r in rot] == pytest.approx(
        [0.40 / (2 * np.pi), 1.30 / (2 * np.pi)], abs=1e-9
    )


# ---------------------------------------------------------------------------
# Proposition N (identifiability.md 5.3): a single NEUTRAL exponent destroys
# cross-module non-resonance globally.  This is what makes Theorem B's
# restriction to fixed points a structural exclusion rather than a technical
# gap -- worth a test, because the natural reading ("limit cycles are Siegel,
# small divisors, hard") suggests a hypothesis that merely becomes difficult.
# ---------------------------------------------------------------------------

def test_one_neutral_exponent_kills_cross_module_nonresonance_everywhere():
    """nu = 0 + nu is an order-2 resonance against every other module."""
    from idyn import spectra as SPX

    cycle = np.array([0.0, np.log(0.4)])           # {0, log|1-2a|}
    spiral = np.array([np.log(0.55), np.log(0.55)])
    # two contracting spirals are fine -- so the failure below is the zero, not
    # some artefact of the checker
    assert SPX.is_cross_module_nonresonant(
        [np.array([np.log(0.92)] * 2), spiral], max_order=4)
    # one neutral direction is already enough
    assert not SPX.is_cross_module_nonresonant([cycle, spiral], max_order=4)
    res = SPX.cross_module_resonances([cycle, spiral], max_order=4)
    assert res, "expected explicit resonances"
    # and the mechanism is the appended zero, not a numerical coincidence
    assert any(abs(r.target - np.log(0.55)) < 1e-12 for r in res), [str(r) for r in res]


def test_two_oscillatory_modules_are_resonant_at_every_frequency():
    """So Theorem B can never cover the case the applied claim cares about."""
    from idyn import spectra as SPX

    for a1, a2 in ((0.3, 0.4), (0.3, 0.3), (0.2, 0.45)):
        s = [np.array([0.0, np.log(abs(1 - 2 * a))]) for a in (a1, a2)]
        assert not SPX.is_cross_module_nonresonant(s, max_order=4), (a1, a2)


# ---------------------------------------------------------------------------
# Theorem D (identifiability.md 15.12): Route D closes at a contracting fixed
# point.  The proof reduces to the cocycle A_s = f_1^{-n} A_{f_2^n s} f_1^n with
# A_{f_2^n s} -> I at rate rho_2, so the hypothesis is
#
#     log rho_2 + spread(f_1) < 0,
#
# comparing f_2's rate to f_1's OWN spread -- an internal property of one
# module, NOT a separation between modules as Lemma C needs.  For a conformal
# f_1 (any TwistBlock: spectrum {log s, log s}, spread 0) it is free.
# ---------------------------------------------------------------------------

def _theorem_d_residual(f1, rho2, n_lo=40, n_hi=70, seed=0):
    """sup over large n of ||f_1^{-n} (I + rho2^n D) f_1^n - I||, D fixed."""
    rng = np.random.default_rng(seed)
    D = 0.3 * rng.standard_normal((2, 2))
    worst = 0.0
    for n in range(n_lo, n_hi):
        F = np.linalg.matrix_power(f1, n)
        conj = np.linalg.solve(F, (np.eye(2) + rho2 ** n * D) @ F)
        worst = max(worst, float(np.abs(conj - np.eye(2)).max()))
    return worst


def _conformal(s, omega):
    return s * np.array([[np.cos(omega), -np.sin(omega)],
                         [np.sin(omega), np.cos(omega)]])


def test_theorem_D_is_free_for_a_conformal_module():
    """spread 0, so any contracting f_2 works -- the oscillatory case.

    This is the point of Theorem D: the module that (B4') excludes identically
    (Prop. N) and that (F3) cannot order is exactly where its hypothesis costs
    nothing.
    """
    f1 = _conformal(0.8, 0.7)
    assert _theorem_d_residual(f1, 0.9) < 1e-2
    assert _theorem_d_residual(f1, 0.5) < 1e-10


def test_theorem_D_condition_is_sharp_not_conservative():
    """Non-conformal f_1: converges iff rho_2 * exp(spread) < 1, and it is sharp."""
    f1 = np.diag([0.8, 0.2])
    spread = np.log(0.8) - np.log(0.2)
    ok, bad = 0.2, 0.9
    assert ok * np.exp(spread) < 1 < bad * np.exp(spread), "fixture must straddle"
    assert _theorem_d_residual(f1, ok) < 1e-3
    assert _theorem_d_residual(f1, bad) > 1e6, "must diverge on the wrong side"


def test_theorem_D_kills_the_cross_block_theorem_F_is_forced_to_permit():
    """The two results are complementary, and this is the configuration showing it.

    Take f_1 conformal at 0.8 and f_2 slower at 0.9.  Then:

    * (F3) HOLDS, but in the order [1, 0] -- f_2 leads.  So Theorem F gives
      h_2 = h_2(z_2) and explicitly PERMITS h_1 to depend on z_2.
    * Lemma C's oriented gap for killing that surviving block needs
      lambda_max(f_2) < lambda_min(f_1), i.e. log 0.9 < log 0.8, which is FALSE
      -- and by section 3.7 it can never hold in both directions anyway.
    * Theorem D's condition log rho_2 + spread(f_1) = log 0.9 + 0 < 0 holds, so
      it removes exactly the dependence Theorem F had to leave in.
    """
    from idyn import spectra as SPX

    f1 = _conformal(0.8, 0.7)                       # spectrum {log .8, log .8}
    rho2 = 0.9                                       # f_2 is the SLOWER module

    fg = SPX.filtration_gap([np.array([np.log(0.8)] * 2), np.array([np.log(rho2)] * 2)])
    assert fg.ordered and list(fg.order) == [1, 0], (fg.ordered, fg.order)

    # Lemma C cannot kill h_1's dependence on z_2 here
    assert np.log(rho2) > np.log(0.8), "the oriented gap Lemma C needs fails"

    # Theorem D can
    assert np.log(rho2) + 0.0 < 0, "Theorem D's condition holds (spread 0)"
    assert _theorem_d_residual(f1, rho2) < 1e-2


# ---------------------------------------------------------------------------
# Theorem D' (identifiability.md 15.13), linearisation step.  On a limit cycle
# the sketch's one remaining obligation is whether every u_s is linear.  The
# cocycle u_s = f_1^{-n} u_{f_2^n s} f_1^n acts on a degree-p Taylor term Q by
#
#     Q  |->  A_1^{-n} Q(A_1^n .),      ||.|| <= (rho_max^p)^n / rho_min^n,
#
# so the degree-2 term (the binding one) decays iff
#
#     log rho_1 + sigma_1 < 0,        sigma_1 = Lyapunov SPREAD of f_1.
#
# Free for a conformal f_1.  Note what this says: ANISOTROPY breaks it, not
# slowness -- diag(0.6, 0.3) contracts harder than a 0.95 spiral and still
# grows.  Same shape as 3.14's "width, not speed, is what disqualifies".
# ---------------------------------------------------------------------------

def _quadratic_jet_norm(A1, n, n_samp=4096):
    """||A_1^{-n} Q(A_1^n .)|| on the unit circle, for Q(y) = e_last * y_0^2.

    Deterministic angular grid, not a random sample: a conformal A_1 rotates
    the evaluation points by n*omega, so Monte-Carlo error does NOT cancel
    between two values of n and shows up as a ~1% bias in their ratio.
    """
    th = np.linspace(0.0, 2 * np.pi, n_samp, endpoint=False)
    X = np.stack([np.cos(th), np.sin(th)], axis=1)
    An = np.linalg.matrix_power(A1, n)
    Y = X @ An.T
    Q = np.zeros_like(Y)
    Q[:, -1] = Y[:, 0] ** 2
    Z = Q @ np.linalg.inv(An).T
    return float(np.sqrt((Z ** 2).sum(1).mean()))


def _spiral(s, omega):
    c, sn = np.cos(omega), np.sin(omega)
    return s * np.array([[c, -sn], [sn, c]])


def _log_rho_and_spread(A):
    sv = np.linalg.svd(A, compute_uv=False)
    return float(np.log(sv.max())), float(np.log(sv.max()) - np.log(sv.min()))


def test_theorem_d_prime_linearisation_boundary_is_log_rho_plus_spread():
    """The decay rate of the quadratic jet is exactly exp(log rho_1 + sigma_1).

    Checked against the closed form on both sides of the boundary, so the
    condition is certified sharp rather than merely sufficient.
    """
    cases = [
        _spiral(0.8, 0.7),
        _spiral(0.95, 0.4),
        np.diag([0.9, 0.3]),
        np.diag([0.6, 0.3]),
        np.diag([0.9, 0.75]),
    ]
    for A1 in cases:
        log_rho, sigma = _log_rho_and_spread(A1)
        emp = _quadratic_jet_norm(A1, 20) / _quadratic_jet_norm(A1, 10)
        pred = np.exp(10 * (log_rho + sigma))
        assert np.isclose(emp, pred, rtol=1e-3), (A1, emp, pred)


def test_theorem_d_prime_linearisation_is_free_for_a_conformal_module():
    """Conformal f_1 (spread 0) kills the nonlinear jet; anisotropy revives it.

    The comparison that matters: diag(0.6, 0.3) contracts *harder* than the
    0.95 spiral (log rho -0.51 vs -0.05) and its jet GROWS, because its spread
    is 0.69.  So the hypothesis is not "contract fast enough".
    """
    conformal = _spiral(0.95, 0.4)
    log_rho, sigma = _log_rho_and_spread(conformal)
    assert sigma < 1e-12, "a TwistBlock is conformal"
    assert log_rho + sigma < 0
    assert _quadratic_jet_norm(conformal, 40) < 0.2 * _quadratic_jet_norm(conformal, 0)

    anisotropic = np.diag([0.6, 0.3])
    log_rho_a, sigma_a = _log_rho_and_spread(anisotropic)
    assert log_rho_a < log_rho, "contracts harder than the spiral"
    assert log_rho_a + sigma_a > 0, "yet fails the condition"
    assert _quadratic_jet_norm(anisotropic, 20) > 5 * _quadratic_jet_norm(anisotropic, 10)
