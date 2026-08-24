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


# --------------------------------------------------------------------------
# 7. The rotation number does not pin the splitting (counterexamples.md 7)
# --------------------------------------------------------------------------


def _torus_annulus(rng, n=3000, lo=0.7, hi=1.3):
    r = rng.uniform(lo, hi, (n, 2))
    t = rng.uniform(-np.pi, np.pi, (n, 2))
    return np.stack([r[:, 0] * np.cos(t[:, 0]), r[:, 0] * np.sin(t[:, 0]),
                     r[:, 1] * np.cos(t[:, 1]), r[:, 1] * np.sin(t[:, 1])], axis=-1)


def test_torus_regrouping_is_an_exact_modular_conjugacy():
    w = S.torus_regrouping_counterexample()
    z = _torus_annulus(np.random.default_rng(0))
    assert np.abs(w["h"](w["F"](z)) - w["F_tilde"](w["h"](z))).max() < 1e-12
    assert np.abs(w["h_inv"](w["h"](z)) - z).max() < 1e-12
    assert w["system_tilde"].partition == [2, 2], "still modular, still 2-D blocks"


def test_torus_regrouping_moves_the_rotation_numbers():
    """(w1, w2) -> (w1 + w2, w2): a GL(2,Z) action, not a permutation."""
    w = S.torus_regrouping_counterexample(omega=(0.50, 1.30))
    assert w["rotation_true"][0] != pytest.approx(w["rotation_tilde"][0], abs=1e-3)
    assert w["rotation_true"][1] == pytest.approx(w["rotation_tilde"][1], abs=1e-12)
    assert w["rotation_tilde"][0] == pytest.approx(1.80 / (2 * np.pi), abs=1e-12)

    rot = [SP.rotation_number_averaged(b, _torus_annulus(np.random.default_rng(1), 16)[:, 2 * i:2 * i + 2],
                                       T=400, warmup=100).rho
           for i, b in enumerate(w["system_tilde"].blocks)]
    assert abs(abs(rot[0]) - w["rotation_tilde"][0]) < 1e-3, "measured, not just asserted"


def test_the_torus_counterexample_does_not_contradict_theorem_F():
    """(F3) fails outright for two cycles, so Theorem F never applied here."""
    w = S.torus_regrouping_counterexample()
    assert not SP.filtration_gap(w["spectra"]).ordered
    assert SP.filtration_gap(w["spectra"]).gap < -0.8


def test_the_torus_conjugacy_satisfies_F1_and_is_genuinely_coupled():
    w = S.torus_regrouping_counterexample()
    z = _torus_annulus(np.random.default_rng(2), n=500)

    def jac(fn, pts, eps=1e-6):
        out = np.zeros((pts.shape[0], 4, 4))
        for k in range(4):
            e = np.zeros(4)
            e[k] = eps
            out[:, :, k] = (fn(pts + e) - fn(pts - e)) / (2 * eps)
        return out

    J = jac(w["h"], z)
    assert np.linalg.norm(J, axis=(1, 2)).max() < 10.0, "bounded derivative: (F1)"
    assert np.linalg.norm(jac(w["h_inv"], w["h"](z)), axis=(1, 2)).max() < 10.0
    coupling = np.linalg.norm(J[:, :2, 2:], axis=(1, 2))
    assert coupling.min() > 0.3, "the cross-block is not marginal anywhere"


@pytest.mark.parametrize("beta_donor,breaks", [(0.0, False), (0.3, True), (0.8, True)])
def test_donor_shear_breaks_the_NAIVE_angle_construction(beta_donor, breaks):
    """The naive theta_2 increment depends on r_2, which module 1 cannot see."""
    w = S.torus_regrouping_counterexample(beta_donor=beta_donor, naive_phase=True)
    z = _torus_annulus(np.random.default_rng(3))
    resid = float(np.abs(w["h"](w["F"](z)) - w["F_tilde"](w["h"](z))).max())
    assert (resid > 1e-3) == breaks, f"beta_donor={beta_donor}, residual={resid}"


@pytest.mark.parametrize("beta_receiving", [0.0, 0.5])
@pytest.mark.parametrize("beta_donor", [0.0, 0.3, 0.8])
def test_asymptotic_phase_restores_it_at_every_shear(beta_receiving, beta_donor):
    """So shear is not an escape -- counterexamples.md 7.1.

    Theta = theta + beta * sum_k (g^k(r) - rho) advances rigidly for any beta,
    and rebuilding h with it is an exact conjugacy where the naive one fails.
    """
    w = S.torus_regrouping_counterexample(
        beta_receiving=beta_receiving, beta_donor=beta_donor)
    z = _torus_annulus(np.random.default_rng(4))
    assert np.abs(w["h"](w["F"](z)) - w["F_tilde"](w["h"](z))).max() < 1e-12
    assert np.abs(w["h_inv"](w["h"](z)) - z).max() < 1e-12


@pytest.mark.parametrize("beta", [0.0, 0.3, 0.8])
def test_asymptotic_phase_advances_rigidly(beta):
    """Theta(f z) = Theta(z) + omega, which the naive angle does not satisfy."""
    blk = S.LimitCycleBlock(a=0.3, rho=1.0, omega=1.3, beta=beta)
    z = _torus_annulus(np.random.default_rng(5))[:, 2:]
    d = S.asymptotic_phase(blk, blk.step(z)) - S.asymptotic_phase(blk, z) - blk.omega
    assert np.abs((d + np.pi) % (2 * np.pi) - np.pi).max() < 1e-12
    if beta != 0.0:
        naive = np.arctan2(z[:, 1], z[:, 0])
        assert np.abs(S.asymptotic_phase(blk, z) - naive).max() > 1e-3, "Theta != theta"


def test_shear_is_not_a_conjugacy_invariant_so_it_cannot_protect_one():
    """The regrouped module comes out shear-free even when the original is not."""
    w = S.torus_regrouping_counterexample(beta_receiving=0.5, beta_donor=0.8)
    assert w["system"].blocks[0].beta == 0.5
    assert w["system_tilde"].blocks[0].beta == 0.0


def test_lattice_margin_sees_through_the_regrouping_but_not_past_a_real_change():
    """Zero against the regrouping; still rejecting exp14's negative control."""
    w = S.torus_regrouping_counterexample(omega=(0.50, 1.30))
    same, _ = SP.rotation_lattice_margin(w["rotation_true"], w["rotation_tilde"])
    assert same == pytest.approx(0.0, abs=1e-12)

    control = (0.50 / (2 * np.pi), 0.90 / (2 * np.pi))
    naive = max(abs(a - b) for a, b in zip(w["rotation_true"], control))
    quotient, _ = SP.rotation_lattice_margin(w["rotation_true"], control)
    assert naive == pytest.approx(0.06366, abs=1e-4)
    assert quotient == pytest.approx(0.01592, abs=1e-4)
    assert 0.0 < quotient < naive, "the control still rejects, with less headroom"


# --------------------------------------------------------------------------
# What protects a contracting module from the lattice ambiguity: (F1), not (F3)
# (identifiability.md 11.6)
# --------------------------------------------------------------------------


def test_the_lattice_regrouping_also_conjugates_two_contracting_spirals():
    """The GL(2,Z) ambiguity is NOT special to limit cycles.

    Two contracting spirals with well-separated spectra -- (F3) holds
    comfortably -- still admit the exact regrouping h(z1,z2) = (z1 z2/|z2|, z2)
    carrying omega_1 -> omega_1 + omega_2.  So spectral separation does not
    protect the rotation numbers, and 7's counterexample is not about
    neutrality per se.

    NOTE `beta=0.0` is passed explicitly: TwistBlock's default beta is 0.6, and
    a sheared block is a different system.  That default silently broke an
    earlier version of this check.
    """
    w1, w2 = 0.35, 1.10
    A = S.ModularSystem([S.TwistBlock(s=0.92, omega=w1, beta=0.0),
                         S.TwistBlock(s=0.55, omega=w2, beta=0.0)])
    At = S.ModularSystem([S.TwistBlock(s=0.92, omega=w1 + w2, beta=0.0),
                          S.TwistBlock(s=0.55, omega=w2, beta=0.0)])

    def lattice(Z):
        z1 = Z[..., 0] + 1j * Z[..., 1]
        z2 = Z[..., 2] + 1j * Z[..., 3]
        w = z1 * z2 / np.maximum(np.abs(z2), 1e-300)
        return np.stack([w.real, w.imag, z2.real, z2.imag], -1)

    rng = np.random.default_rng(0)
    for donor_radius in (1.0, 1e-2, 1e-6):
        th = rng.uniform(-np.pi, np.pi, 500)
        r = rng.uniform(0.5, 1.2, 500)
        th2 = rng.uniform(-np.pi, np.pi, 500)
        Z = np.concatenate([
            np.stack([r * np.cos(th), r * np.sin(th)], -1),
            donor_radius * np.stack([np.cos(th2), np.sin(th2)], -1),
        ], -1)
        resid = np.abs(lattice(A.step(Z)) - At.step(lattice(Z))).max()
        assert resid < 1e-12, f"donor radius {donor_radius}: {resid}"

    # and (F3) holds for this system, so it is not what rules the map out
    from idyn import spectra as SP
    gap = SP.filtration_gap([np.array([np.log(0.92)] * 2), np.array([np.log(0.55)] * 2)])
    assert gap.ordered and gap.gap > 0.4


def test_F1_is_what_excludes_the_regrouping_for_a_contracting_module():
    """||Dh|| ~ 1/|z_donor|, so (F1) bites iff the donor decays to zero.

    This is the checkable diagnostic: bounded away from zero => the lattice
    ambiguity is live; decaying => (F1) excludes it.  It is a *different*
    quantity from `filtration_gap`.
    """
    rng = np.random.default_rng(1)

    def min_donor_radius(sysm, lo, hi, T=30):
        out = []
        for _ in range(2):
            th = rng.uniform(-np.pi, np.pi, 200)
            r = rng.uniform(lo, hi, 200)
            out.append(np.stack([r * np.cos(th), r * np.sin(th)], -1))
        Z = sysm.simulate(np.concatenate(out, -1), T)
        return float(np.hypot(Z[..., 2], Z[..., 3]).min())

    spirals = S.ModularSystem([S.TwistBlock(s=0.92, omega=0.35, beta=0.0),
                               S.TwistBlock(s=0.55, omega=1.10, beta=0.0)])
    cycles = S.torus_regrouping_counterexample()["system"]

    r_spiral = min_donor_radius(spirals, 0.5, 1.2)
    r_cycle = min_donor_radius(cycles, 0.8, 1.2)
    assert r_spiral < 1e-6, f"a contracting donor must reach ~0, got {r_spiral}"
    assert r_cycle > 0.5, f"a limit-cycle donor must stay away from 0, got {r_cycle}"
    # sup||Dh|| differs by seven orders of magnitude between the two
    assert (1.0 / r_spiral) / (1.0 / r_cycle) > 1e6


# ---------------------------------------------------------------------------
# Route D (identifiability.md section 15): independence of the module marginals.
# These pin what it rejects, what it must NOT reject, and the two escapes that
# bound it -- one of which turns out to be excluded by (A2) for free.
# ---------------------------------------------------------------------------

def _dep(Z, k=2):
    from idyn.metrics import distance_correlation as dc
    return float(dc(Z[:, :k], Z[:, k:]))


def _dep_base(n, k=2):
    from idyn.metrics import distance_correlation_baseline as dcb
    return float(dcb(n, k, k, seed=0))


def test_route_D_rejects_the_triangular_conjugacy():
    """The object that makes block-diagonality FALSE under (B1)-(B4).

    It is polynomial, hence C-infinity, so no regularity hypothesis removes it --
    but it makes the modules dependent, so independence does.
    """
    rng = np.random.default_rng(0)
    n = 1500
    Z = rng.standard_normal((n, 4))
    h = Z.copy()
    h[:, 0] = Z[:, 0] + 0.8 * np.sign(Z[:, 2]) * np.abs(Z[:, 2]) ** 2
    assert _dep(h) > 4 * _dep_base(n)


def test_route_D_rejects_the_lattice_regrouping_when_phases_are_concentrated():
    rng = np.random.default_rng(1)
    n = 1500
    t1, t2 = rng.vonmises(0.0, 4.0, n), rng.vonmises(0.3, 4.0, n)
    r1 = 1 + 0.2 * np.abs(rng.standard_normal(n))
    r2 = 1 + 0.2 * np.abs(rng.standard_normal(n))
    lat = np.stack([r1 * np.cos(t1), r1 * np.sin(t1),
                    r2 * np.cos(t2 + t1), r2 * np.sin(t2 + t1)], 1)
    assert _dep(lat) > 4 * _dep_base(n)


def test_route_D_is_correctly_blind_to_the_regrouping_counterexample():
    """Section 3.1 is (B2)'s job.  A criterion rejecting all three rejects too much."""
    rng = np.random.default_rng(2)
    n = 1500
    Z = rng.standard_normal((n, 4))
    assert _dep(Z[:, [0, 2, 1, 3]]) < 2 * _dep_base(n)


def test_route_D_gaussian_degeneracy_is_real_but_excluded_by_A2():
    """The classical ICA escape, and why it costs nothing here.

    iid Gaussian modules are rotation-invariant, so a rotation preserves
    independence and Route D is blind.  But that rotation is a *modular
    conjugacy* only when the two modules have the same map -- and (A2), which
    Theorem A already assumes, excludes equal spectra.  So Route D composes with
    (A2) without a Gaussian hole.
    """
    rng = np.random.default_rng(3)
    n = 4000
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    g = rng.standard_normal((n, 2)) @ R.T
    assert _dep(g, k=1) < 2 * _dep_base(n, 1), "Gaussian rotation hides -- the escape"
    u = rng.uniform(-1.7, 1.7, (n, 2)) @ R.T
    assert _dep(u, k=1) > 4 * _dep_base(n, 1), "non-Gaussian rotation does not"

    # and the escape is only a modular conjugacy at equal eigenvalues
    for (m1, m2), modular in (((0.8, 0.8), True), ((0.8, 0.5), False)):
        C = R @ np.diag([m1, m2]) @ np.linalg.inv(R)
        off = abs(C[0, 1]) + abs(C[1, 0])
        assert bool(off < 1e-12) == modular, (m1, m2, off)


def test_route_D_escape_is_stabiliser_valued_sign_flip():
    """A counterexample to the NAIVE Route D conjecture, and its shape.

    ``h_1 = z_1 sgn(z_2)`` is an exact modular conjugacy (f_1 odd, f_2
    orientation-preserving), preserves independence because for each fixed z_2 it
    is a p_1-preserving map, and is not block-diagonal.  It escapes because
    ``{+-1} = Stab(p_1)`` is nontrivial for symmetric p_1 -- the same object
    Proposition S identifies as Route B's residue (identifiability.md 15.11).

    It is discontinuous at ``z_2 = 0``, which is why the d_B = 1 case needs no
    extra hypothesis: the stabiliser is discrete, so no continuous family exists.
    """
    rng = np.random.default_rng(0)
    n = 4000
    mu1, mu2 = 0.6, 0.8
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    h1 = z1 * np.sign(z2)

    # exact conjugacy: h_1(F z) == f~_1(h_1(z)) with f~_1 = mu1 .
    np.testing.assert_allclose((mu1 * z1) * np.sign(mu2 * z2), mu1 * h1, atol=1e-12)
    # independence-preserving
    assert _dep(np.stack([h1, np.zeros(n), z2, np.zeros(n)], 1)) < 2 * _dep_base(n)
    # but genuinely not block-diagonal
    assert np.mean(h1 != z1) > 0.4


def test_route_D_escape_is_stabiliser_valued_rotation_and_symmetry_breaking_closes_it():
    """The SMOOTH escape, and the condition that removes it.

    ``h_1 = R(theta(z_2)) z_1`` with p_1 rotationally symmetric is
    independence-preserving because Stab(p_1) contains SO(2).  Concentrating
    p_1's phase makes Stab trivial and the escape becomes visible -- which is the
    sharpened conjecture's hypothesis, measured.
    """
    rng = np.random.default_rng(1)
    n = 4000
    th2 = rng.uniform(-np.pi, np.pi, n)
    r1 = 1 + 0.2 * np.abs(rng.standard_normal(n))
    donor = np.stack([np.cos(th2), np.sin(th2)], 1)

    def rotated(phase):
        return np.stack([r1 * np.cos(phase + th2), r1 * np.sin(phase + th2)], 1)

    from idyn.metrics import distance_correlation as dc
    sym = dc(rotated(rng.uniform(-np.pi, np.pi, n)), donor)      # Stab = SO(2)
    conc = dc(rotated(rng.vonmises(0.0, 4.0, n)), donor)         # Stab trivial
    assert sym < 2 * _dep_base(n), "symmetric p_1 hides the coupling"
    assert conc > 8 * sym, f"breaking the symmetry must expose it: {sym} -> {conc}"


# ---------------------------------------------------------------------------
# Lemma T (identifiability.md 15.13b): two time points collapse the circle
# stabiliser to its ROTATIONAL part.
#
# On a limit cycle the Route D escape lives in the phase, and
# Stab_Homeo(mu) = F^-1 o SO(2) o F is an entire circle group of non-rotation
# maps -- far bigger than O(2), which is why (D-d) looks insufficient there.
# But independence holds at every t and p^(t) is mu rotated by t*omega, so the
# escape must preserve TWO rotated copies.  That cuts it down to Stab_O2(mu).
# ---------------------------------------------------------------------------

_TAU = 2.0 * np.pi


def _circle_law(kappa, mu0=0.0, skew=0.0, n=100_000):
    th = np.linspace(0.0, _TAU, n, endpoint=False)
    p = np.exp(kappa * np.cos(th - mu0) + skew * np.cos(2.0 * (th - mu0)))
    return th, p / p.sum()


def _stab_element(th, p, alpha):
    """F^-1 o R_alpha o F -- the general orientation-preserving p-preserving homeo."""
    F = np.concatenate([[0.0], np.cumsum(p)[:-1]])

    def U(x):
        u = np.interp(np.mod(x, _TAU), th, F, period=_TAU)
        return np.interp(np.mod(u + alpha, 1.0), F, th)

    return U


def _tv(U, th, p, q, seed=1, n_samp=300_000, bins=120):
    rng = np.random.default_rng(seed)
    s = rng.choice(th, size=n_samp, p=p)
    h, _ = np.histogram(np.mod(U(s), _TAU), bins=bins, range=(0.0, _TAU), density=True)
    r, _ = np.histogram(th, bins=bins, range=(0.0, _TAU), weights=q * th.size,
                        density=True)
    return 0.5 * float(np.abs(h - r).sum()) * (_TAU / bins)


def _surviving_element(kappa, skew, omega=0.7):
    """The alpha != 0 best preserving BOTH mu and R_omega # mu, and where it sends 0."""
    th, p = _circle_law(kappa, skew=skew)
    _, prot = _circle_law(kappa, mu0=omega, skew=skew)
    alphas = np.arange(0.01, 1.0, 0.01)
    errs = np.array([_tv(_stab_element(th, p, a), th, prot, prot) for a in alphas])
    j = int(errs.argmin())
    U = _stab_element(th, p, alphas[j])
    return float(errs[j]), float(np.mod(U(np.array([0.0]))[0], _TAU))


def test_lemma_t_every_stab_element_preserves_mu_itself():
    """Sanity: the construction really does produce mu-preserving maps.

    Without this the collapse below would be vacuous -- a family that preserves
    nothing trivially fails to preserve two things.
    """
    th, p = _circle_law(2.0)
    for alpha in (0.05, 0.35, 0.8):
        assert _tv(_stab_element(th, p, alpha), th, p, p) < 0.02, alpha


def test_lemma_t_two_times_collapse_stabiliser_to_rotations():
    """A trivial Stab_O2 leaves only the identity; a Z_2 leaves exactly the half turn."""
    # trivial rotational symmetry -> only the identity survives
    err, shift = _surviving_element(kappa=2.0, skew=0.0)
    assert err > 0.03, f"a non-identity escape survived at TV {err}"
    assert min(shift, _TAU - shift) < 0.1, f"and the argmin is the identity, U(0)={shift}"

    # Z_2 rotational symmetry -> the half turn survives, and it is exactly pi
    err2, shift2 = _surviving_element(kappa=0.0, skew=0.8)
    assert err2 < 0.02, f"the half turn should survive, TV {err2}"
    assert abs(shift2 - np.pi) < 0.05, f"and it is the half turn, U(0)={shift2}"


def test_lemma_t_uniform_phase_is_the_degenerate_case():
    """Uniform mu: every element survives -- the Hyvarinen-Pajunen hole of 15.5."""
    th, p = _circle_law(0.0)
    _, prot = _circle_law(0.0, mu0=0.7)
    for alpha in (0.05, 0.35, 0.8):
        assert _tv(_stab_element(th, p, alpha), th, prot, prot) < 0.02, alpha


# ---------------------------------------------------------------------------
# Theorem D'' (identifiability.md 15.13c): the plumbing behind Lemma T.
#
# Independence at time t says the conditional law of zhat_1 given z_2 does not
# move with z_2.  For the task-23 lattice map h_1 = z_1 * z_2/|z_2| that
# conditional law is p_1's phase law ROTATED by arg(z_2) -- so it moves iff the
# phase law has a nontrivial rotational symmetry deficit, which is (D-d).
#
# This instantiates the theorem's hypothesis chain on the actual counterexample:
# concentrated phase -> hypothesis violated -> h rejected; uniform phase -> the
# 15.5 Hyvarinen-Pajunen hole, and h survives.
# ---------------------------------------------------------------------------

def _lattice_conditional_phase_shift(kappa, n=200_000, seed=3, n_harm=4):
    """How far the conditional law of arg(h_1) moves between two values of z_2.

    Returns the max over harmonics of |E e^{i k phi | s1} - E e^{i k phi | s2}|,
    which is 0 exactly when the conditional law is s-free.
    """
    rng = np.random.default_rng(seed)
    # phase law of module 1: von Mises(kappa); kappa = 0 is uniform
    if kappa == 0.0:
        phi = rng.uniform(0.0, _TAU, size=n)
    else:
        phi = rng.vonmises(0.0, kappa, size=n)
    out = 0.0
    for k in range(1, n_harm + 1):
        # h_1 = z_1 * z_2/|z_2| rotates module 1's phase by arg(z_2) = s
        c1 = np.exp(1j * k * (phi + 0.0)).mean()
        c2 = np.exp(1j * k * (phi + 1.1)).mean()
        out = max(out, abs(c1 - c2))
    return float(out)


def test_theorem_d2_lattice_moves_the_conditional_law_when_phase_is_concentrated():
    """(D-d) holds -> the lattice map violates independence -> it is rejected."""
    moved = _lattice_conditional_phase_shift(kappa=2.0)
    assert moved > 0.3, f"a concentrated phase law must move, got {moved}"


def test_theorem_d2_uniform_phase_is_the_hole_and_the_lattice_survives():
    """(D-d) fails -> the conditional law is s-free -> 15.5's escape is real."""
    moved = _lattice_conditional_phase_shift(kappa=0.0)
    assert moved < 0.02, f"a uniform phase law must NOT move, got {moved}"


def test_theorem_d2_the_boundary_is_monotone_in_concentration():
    """No sharp cliff: the violation grows with concentration, so the hypothesis
    is quantitative.  Matches 14.4's threshold reading rather than a dichotomy."""
    vals = [_lattice_conditional_phase_shift(kappa=k) for k in (0.0, 0.25, 1.0, 4.0)]
    assert all(b > a - 1e-3 for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] > 10 * max(vals[0], 1e-3), vals
