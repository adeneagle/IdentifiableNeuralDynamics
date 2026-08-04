"""Poincare-Dulac machinery and the Tier 2 witness (theory/approaches.md §A.2).

The claim under test is that Route A's Tier 2 is **non-empty**: there is a system
satisfying cross-module non-resonance that nonetheless does not linearise, so the
tier has genuine nonlinear content rather than collapsing to Theorem A.
"""

from __future__ import annotations

import numpy as np
import pytest

from idyn import linear as LIN
from idyn import normalform as NF
from idyn import spectra as SP
from idyn import systems as S


# --------------------------------------------------------------------------
# The homological operator
# --------------------------------------------------------------------------


def test_multi_indices_enumerate_the_right_monomials():
    assert set(NF.multi_indices(2, 2)) == {(2, 0), (1, 1), (0, 2)}
    assert set(NF.multi_indices(2, 3)) == {(3, 0), (2, 1), (1, 2), (0, 3)}
    assert len(NF.multi_indices(3, 2)) == 6  # C(3+2-1, 2)
    assert all(sum(m) == 4 for m in NF.multi_indices(3, 4))


def test_degree_one_is_rejected():
    with pytest.raises(ValueError):
        NF.homological_eigenvalues([0.5, 0.25], 1)


def test_homological_eigenvalue_is_lambda_i_minus_lambda_to_the_m():
    mu = 0.7
    eigs = [mu, mu**2]
    got = {(vm.component, vm.exponent): vm.value for vm in NF.homological_eigenvalues(eigs, 2)}
    # component 0 has lam = mu, component 1 has lam = mu^2
    assert got[(0, (2, 0))] == pytest.approx(mu - mu**2)
    assert got[(1, (2, 0))] == pytest.approx(mu**2 - mu**2)  # the resonance
    assert got[(0, (1, 1))] == pytest.approx(mu - mu * mu**2)
    assert got[(1, (0, 2))] == pytest.approx(mu**2 - (mu**2) ** 2)


def test_the_resonant_node_has_exactly_one_resonance_at_degree_two():
    mu = 0.7
    res = NF.resonances_at_degree([mu, mu**2], 2)
    assert len(res) == 1
    assert (res[0].component, res[0].exponent) == (1, (2, 0))
    assert abs(res[0].value) < 1e-15
    assert not NF.is_linearizable_at_degree([mu, mu**2], 2)


def test_a_nonresonant_spectrum_is_linearizable_at_every_low_degree():
    eigs = [0.7, 0.31]  # 0.31 is not 0.7^k nor a product hitting either
    for deg in (2, 3, 4):
        assert NF.is_linearizable_at_degree(eigs, deg)


def test_obstruction_is_exactly_the_resonant_coefficient():
    mu, c = 0.7, 0.9
    rep = NF.linearization_obstruction([mu, mu**2], {(1, (2, 0)): c}, degree=2)
    assert not rep.linearizable
    assert rep.obstruction[(1, (2, 0))] == pytest.approx(c)
    # a coefficient on a NON-resonant monomial is removable and does not appear
    assert (0, (2, 0)) not in rep.obstruction


def test_coefficient_tolerance_is_needed_for_estimated_jets():
    """A fitted jet is never exactly zero, so the exact test would always obstruct."""
    mu = 0.7
    noise = 1e-9  # a resonant coefficient that is zero up to fit error
    strict = NF.linearization_obstruction([mu, mu**2], {(1, (2, 0)): noise}, 2)
    assert not strict.linearizable, "exact test: any nonzero coefficient obstructs"

    tolerant = NF.linearization_obstruction([mu, mu**2], {(1, (2, 0)): noise}, 2, coeff_tol=1e-6)
    assert tolerant.linearizable, "with a fit-error budget it is correctly read as linear"
    assert tolerant.obstruction_norm == pytest.approx(noise)

    # and a genuine coefficient still obstructs at the same tolerance
    real = NF.linearization_obstruction([mu, mu**2], {(1, (2, 0)): 0.9}, 2, coeff_tol=1e-6)
    assert not real.linearizable
    assert real.obstruction_norm == pytest.approx(0.9)


def test_obstruction_vanishes_when_the_resonant_coefficient_does():
    """The control: with c = 0 the map IS its linear part."""
    mu = 0.7
    rep = NF.linearization_obstruction([mu, mu**2], {(1, (2, 0)): 0.0}, degree=2)
    assert rep.linearizable
    # and a purely non-resonant term never obstructs, whatever its size
    rep2 = NF.linearization_obstruction([mu, mu**2], {(0, (2, 0)): 1e6}, degree=2)
    assert rep2.linearizable


# --------------------------------------------------------------------------
# ResonantNodeBlock
# --------------------------------------------------------------------------


def test_resonant_node_jacobian_matches_finite_differences():
    blk = S.ResonantNodeBlock(mu=0.7, c=0.9)
    z = np.array([0.43, -0.21])
    J = blk.jacobian(z)
    eps = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        fd = (blk.step(z + e) - blk.step(z - e)) / (2 * eps)
        assert J[:, j] == pytest.approx(fd, abs=1e-8)


def test_resonant_node_closed_form_iterate_is_exact():
    blk = S.ResonantNodeBlock(mu=0.7, c=0.9)
    z0 = np.array([0.8, 0.3])
    z = z0.copy()
    for n in range(1, 12):
        z = blk.step(z)
        assert z == pytest.approx(blk.iterate_exact(z0, n), abs=1e-14)


def test_the_secular_term_grows_linearly_and_only_when_c_is_nonzero():
    """b_n / mu^(2n) = z_b + (n c / mu^2) z_a^2 -- the factor of n IS the resonance."""
    mu, c = 0.7, 0.9
    z0 = np.array([0.8, 0.3])

    blk = S.ResonantNodeBlock(mu=mu, c=c)
    ns = np.arange(1, 30)
    norm = np.array([blk.iterate_exact(z0, int(n))[1] / mu ** (2 * n) for n in ns])
    slope, intercept = np.polyfit(ns.astype(float), norm, 1)
    assert slope == pytest.approx(c * z0[0] ** 2 / mu**2, rel=1e-9)
    assert intercept == pytest.approx(z0[1], abs=1e-9)

    # the linear control: c = 0 gives a constant
    flat = S.ResonantNodeBlock(mu=mu, c=0.0)
    norm0 = np.array([flat.iterate_exact(z0, int(n))[1] / mu ** (2 * n) for n in ns])
    assert np.ptp(norm0) < 1e-12


def test_resonant_node_lyapunov_spectrum_is_log_mu_and_twice_it():
    blk = S.ResonantNodeBlock(mu=0.7, c=0.9)
    spec = SP.lyapunov_spectrum(blk, np.array([0.6, 0.2]), T=3000, warmup=300)
    assert np.sort(spec) == pytest.approx(np.sort(blk.lyapunov_spectrum_exact()), abs=1e-9)


def test_resonant_node_requires_a_contraction():
    with pytest.raises(ValueError):
        S.ResonantNodeBlock(mu=1.5)


# --------------------------------------------------------------------------
# The (B2) blind spot: linear part decomposable, map indecomposable
# --------------------------------------------------------------------------


def test_linear_part_reads_decomposable():
    """diag(mu, mu^2) has two distinct real eigenvalues, so it splits."""
    blk = S.ResonantNodeBlock(mu=0.7, c=0.9)
    L = blk.linear_part()
    assert L == pytest.approx(np.diag([0.7, 0.49]))
    assert LIN.n_indecomposable_summands(L) == 2
    assert not LIN.is_indecomposable(L)


def test_but_the_map_admits_no_invariant_curve_tangent_to_e_a():
    """So it is dynamically indecomposable, and the linearised test is a false negative.

    A complementary invariant factor would be a curve z_b = phi(z_a) with
    phi(0) = phi'(0) = 0, i.e. z_b/z_a^2 constant along orbits.  The closed form
    gives b_n/a_n^2 = z_b/z_a^2 + n c/mu^2, which is unbounded -- so no such curve
    exists for any c != 0.
    """
    mu, c = 0.7, 0.9
    blk = S.ResonantNodeBlock(mu=mu, c=c)
    z0 = np.array([0.8, 0.3])
    ratio = np.array(
        [
            (lambda w: w[1] / w[0] ** 2)(blk.iterate_exact(z0, int(n)))
            for n in range(1, 40)
        ]
    )
    d = np.diff(ratio)
    assert np.allclose(d, c / mu**2, rtol=1e-9), "ratio drifts by exactly c/mu^2 per step"
    assert ratio[-1] > 50.0, "unbounded, so no invariant curve z_b = k z_a^2"

    # the control: c = 0 pins the ratio, and then the map really does split
    flat = S.ResonantNodeBlock(mu=mu, c=0.0)
    ratio0 = np.array(
        [(lambda w: w[1] / w[0] ** 2)(flat.iterate_exact(z0, int(n))) for n in range(1, 40)]
    )
    assert np.ptp(ratio0) < 1e-12


# --------------------------------------------------------------------------
# Tier 2 is non-empty
# --------------------------------------------------------------------------


def test_tier2_witness_keeps_a_within_module_resonance():
    w = S.tier2_witness()
    mu = w["mu"]
    res = NF.resonances_at_degree([mu, mu**2], 2)
    assert (res[0].component, res[0].exponent) == w["resonant_monomial"]
    rep = NF.linearization_obstruction([mu, mu**2], {w["resonant_monomial"]: w["c"]}, 2)
    assert not rep.linearizable, "module 1 does not linearise -- that is the point"


def test_tier2_witness_satisfies_cross_module_nonresonance():
    """The resonance is WITHIN a module, so Tier 2's hypothesis survives it."""
    w = S.tier2_witness()
    assert SP.is_cross_module_nonresonant(w["spectra"], max_order=4)


@pytest.mark.parametrize("nu", [0.70**2, 0.70**3])
def test_a_resonant_partner_is_correctly_rejected(nu):
    """nu = mu^2 collides with an exponent; nu = mu^3 is a true cross resonance."""
    w = S.tier2_witness(nu=nu)
    assert not SP.is_cross_module_nonresonant(w["spectra"], max_order=4)


def test_coupling_resonances_pick_out_cross_group_terms():
    """z_a^2 e_b couples the two groups; z_a^2 e_a (were it resonant) would not."""
    mu = 0.7
    eigs = [mu, mu**2]
    groups = [[0], [1]]
    coup = NF.coupling_resonances(eigs, groups, degree=2)
    assert len(coup) == 1
    assert (coup[0].component, coup[0].exponent) == (1, (2, 0))

    # if the two coordinates are in the SAME group, nothing couples across
    assert NF.coupling_resonances(eigs, [[0, 1]], degree=2) == []


def test_coupling_resonances_reject_a_bad_grouping():
    with pytest.raises(ValueError):
        NF.coupling_resonances([0.7, 0.49], [[0]], degree=2)  # missing index 1


def test_coupling_components_single_edge_is_connected():
    """Two sub-blocks joined by one resonant coupling: one component."""
    mu = 0.7
    eigs = [mu, mu**2]
    coeffs = {(1, (2, 0)): 0.9}  # z_a^2 e_b, the witness edge
    comps = NF.resonance_coupling_components(eigs, [[0], [1]], 2, coeffs, coeff_tol=1e-2)
    assert comps == [[0, 1]]


def test_coupling_components_isolated_node_is_a_separate_factor():
    """3 sub-blocks, coupling only 0-1: node 2 splits off -> two components.

    This is the case 'any coupling exists' gets wrong: a coupling is present, yet
    the module is decomposable as {0,1} (+) {2}.
    """
    mu, nu = 0.7, 0.5  # nu non-resonant with mu, mu^2
    eigs = [mu, mu**2, nu]
    coeffs = {(1, (2, 0, 0)): 0.9}  # only the 0-1 edge (z_0^2 e_1)
    comps = NF.resonance_coupling_components(eigs, [[0], [1], [2]], 2, coeffs, coeff_tol=1e-2)
    assert comps == [[0, 1], [2]]


def test_coupling_components_connect_transitively_through_a_shared_node():
    """Edges 0-1 and 1-2 put all three in one component even with no 0-2 edge."""
    mu = 0.7
    eigs = [mu, mu**2, mu**4]
    coeffs = {(1, (2, 0, 0)): 0.9, (2, (0, 2, 0)): 0.6}  # z_0^2 e_1 and z_1^2 e_2
    comps = NF.resonance_coupling_components(eigs, [[0], [1], [2]], 2, coeffs, coeff_tol=1e-2)
    assert comps == [[0, 1, 2]]


def test_coupling_components_below_tolerance_are_not_edges():
    """A resonant monomial whose coefficient is ~0 does not connect anything."""
    mu, nu = 0.7, 0.5
    eigs = [mu, mu**2, nu]
    coeffs = {(1, (2, 0, 0)): 1e-9}  # present but negligible
    comps = NF.resonance_coupling_components(eigs, [[0], [1], [2]], 2, coeffs, coeff_tol=1e-2)
    assert comps == [[0], [1], [2]]  # fully disconnected -> decomposable into singletons


def test_quadratic_jet_coefficients_read_the_hessian():
    """f_b = mu^2 z_b + c z_a^2 => Hessian of component b is [[2c,0],[0,0]]."""
    c = 0.9
    H = np.zeros((2, 2, 2))
    H[1, 0, 0] = 2 * c  # d^2 f_b / d z_a^2
    coeffs = NF.quadratic_jet_coefficients(H)
    assert coeffs[(1, (2, 0))] == pytest.approx(c)  # ½ * 2c
    assert coeffs[(0, (2, 0))] == pytest.approx(0.0)
    assert coeffs[(1, (1, 1))] == pytest.approx(0.0)


def test_quadratic_jet_transform_is_consistent_under_change_of_basis():
    """Evaluating the transformed jet must match V^{-1} f(V w) directly."""
    rng = np.random.default_rng(0)
    c = 0.7
    H = np.zeros((2, 2, 2))
    H[1, 0, 0] = 2 * c

    def quad(z):  # the pure quadratic part in z-coords
        return np.array([0.0, c * z[0] ** 2])

    V = np.array([[1.0, 0.3], [-0.2, 1.0]])
    Vi = np.linalg.inv(V)
    coeffs = NF.quadratic_jet_coefficients(H, V=V)

    # reconstruct the quadratic form in w-coords from the coefficients and compare
    for _ in range(20):
        w = rng.normal(size=2)
        direct = Vi @ quad(V @ w)
        recon = np.zeros(2)
        for (i, m), a in coeffs.items():
            recon[i] += a * w[0] ** m[0] * w[1] ** m[1]
        assert recon == pytest.approx(direct, abs=1e-12)


def test_tier1_would_kill_the_witness():
    """Full-spectrum non-resonance excludes mu^2 = mu*mu, so Tier 1 linearises it.

    That is exactly why Tier 1 is billed as robustness of Theorem A: under its
    hypothesis there is no nonlinear invariant left to identify.
    """
    w = S.tier2_witness()
    mu = w["mu"]
    # pooled over the whole spectrum, mu^2 = mu*mu IS a resonance, and Tier 1
    # forbids exactly that -- so Tier 1 conjugates this module to its linear part
    assert NF.resonances_at_degree([mu, mu**2], 2), "the relation Tier 1 forbids"
    # and with it forbidden, the surviving system is the linear part
    assert LIN.n_indecomposable_summands(w["linear_part"]) == 2
