"""Partition-recovery metrics and the torch models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from idyn import metrics as MT
from idyn import systems as S
from idyn.models import LatentDynamicsModel, ModelConfig, ModularTransition, UnconstrainedTransition
from idyn.train import TrainConfig, fit, make_dataset


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_fit_linear_relation_recovers_a_known_map():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((400, 4))
    A = rng.standard_normal((4, 4))
    assert np.allclose(MT.fit_linear_relation(Z, Z @ A.T), A, atol=1e-9)


def test_recovery_report_flags_a_block_permutation_as_recovered():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((500, 4))
    B = np.zeros((4, 4))
    B[:2, 2:] = rng.standard_normal((2, 2))  # module 2 -> fitted module 1
    B[2:, :2] = rng.standard_normal((2, 2))
    rep = MT.recovery_report(Z, Z @ B.T, [2, 2], [2, 2])
    assert rep.recovered
    assert rep.assignment == (1, 0)
    assert rep.on_block_fraction == pytest.approx(1.0, abs=1e-6)


def test_recovery_report_rejects_a_mixing_map():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((500, 4))
    rep = MT.recovery_report(Z, Z @ rng.standard_normal((4, 4)).T, [2, 2], [2, 2])
    assert not rep.recovered
    assert rep.on_block_fraction < 0.95


def test_chance_level_is_reported_and_beaten_only_by_real_structure():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((2000, 4))
    fracs = [
        MT.recovery_report(Z, Z @ rng.standard_normal((4, 4)).T, [2, 2], [2, 2]).on_block_fraction
        for _ in range(30)
    ]
    assert np.mean(fracs) < 0.85
    assert MT.recovery_report(Z, Z, [2, 2], [2, 2]).chance_level == pytest.approx(0.5)


def test_mcc_is_one_for_a_permutation_and_scaling():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((500, 4))
    Zf = (Z @ np.eye(4)[[2, 0, 3, 1]].T) * np.array([2.0, -1.0, 0.5, 3.0])
    assert MT.mcc(Z, Zf) == pytest.approx(1.0, abs=1e-9)


def test_high_mcc_with_a_wrong_partition_is_not_recovery():
    """CLAUDE.md §7: coordinate recovery without the partition is a failure."""
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((800, 4))
    P = np.eye(4)[[0, 2, 1, 3]]  # the §3.1 swap: perfect MCC, wrong grouping
    rep = MT.recovery_report(Z, Z @ P.T, [2, 2], [2, 2])
    assert rep.mcc == pytest.approx(1.0, abs=1e-9)
    assert not rep.recovered
    assert rep.on_block_fraction == pytest.approx(0.5, abs=1e-6)


def test_filtration_report_block_diagonal():
    rng = np.random.default_rng(10)
    Z = rng.standard_normal((1500, 4))
    h = np.zeros((4, 4))
    h[:2, :2] = rng.standard_normal((2, 2))
    h[2:, 2:] = rng.standard_normal((2, 2))
    r = MT.filtration_report(Z, Z @ h.T, [2, 2], [2, 2], rate_order=[0, 1])
    assert r.is_block_diagonal
    assert r.is_triangular  # block-diagonal is a special case of triangular
    assert r.upper_mass == pytest.approx(0.0, abs=1e-9)


def test_filtration_report_skew_product_is_triangular_not_diagonal():
    """Lemma C conclusion: fast output may draw on slow input (allowed lower)."""
    rng = np.random.default_rng(11)
    Z = rng.standard_normal((1500, 4))
    h = np.zeros((4, 4))
    h[:2, :2] = rng.standard_normal((2, 2))
    h[2:, 2:] = rng.standard_normal((2, 2))
    h[2:, :2] = rng.standard_normal((2, 2)) * 0.8  # fast (module 1) depends on slow (module 0)
    r = MT.filtration_report(Z, Z @ h.T, [2, 2], [2, 2], rate_order=[0, 1])
    assert r.is_triangular
    assert not r.is_block_diagonal
    assert r.lower_mass > 0.05
    assert r.upper_mass == pytest.approx(0.0, abs=1e-9)


def test_filtration_report_forbidden_upper_coupling_is_flagged():
    """Slow output drawing on fast input is what Lemma C forbids."""
    rng = np.random.default_rng(12)
    Z = rng.standard_normal((1500, 4))
    h = np.zeros((4, 4))
    h[:2, :2] = rng.standard_normal((2, 2))
    h[2:, 2:] = rng.standard_normal((2, 2))
    h[:2, 2:] = rng.standard_normal((2, 2)) * 0.8  # slow depends on fast: forbidden
    r = MT.filtration_report(Z, Z @ h.T, [2, 2], [2, 2], rate_order=[0, 1])
    assert r.upper_mass > 0.1
    assert not r.is_triangular


def test_filtration_report_respects_rate_order():
    """Same h, reversed rate order: the allowed and forbidden triangles swap."""
    rng = np.random.default_rng(13)
    Z = rng.standard_normal((1500, 4))
    h = np.zeros((4, 4))
    h[:2, :2] = rng.standard_normal((2, 2))
    h[2:, 2:] = rng.standard_normal((2, 2))
    h[2:, :2] = rng.standard_normal((2, 2)) * 0.8
    r01 = MT.filtration_report(Z, Z @ h.T, [2, 2], [2, 2], rate_order=[0, 1])
    r10 = MT.filtration_report(Z, Z @ h.T, [2, 2], [2, 2], rate_order=[1, 0])
    assert r01.lower_mass == pytest.approx(r10.upper_mass, rel=1e-6)
    assert r01.upper_mass == pytest.approx(r10.lower_mass, rel=1e-6)


def test_filtration_report_validates_inputs():
    Z = np.zeros((10, 4))
    with pytest.raises(ValueError, match="equal block-size"):
        MT.filtration_report(Z, Z, [2, 2], [3, 1], rate_order=[0, 1])
    with pytest.raises(ValueError, match="permutation"):
        MT.filtration_report(Z, Z, [2, 2], [2, 2], rate_order=[0, 0])


def test_pairing_multiset_is_label_invariant():
    assert MT.pairing_multiset([0, 0, 1, 1]) == MT.pairing_multiset([1, 1, 0, 0])
    assert MT.pairing_multiset([0, 1, 1, 0]) != MT.pairing_multiset([0, 0, 1, 1])


def test_all_pairings_of_four_has_exactly_three_elements():
    assert len(MT.all_pairings(4)) == 3
    with pytest.raises(ValueError):
        MT.all_pairings(5)


def test_nonuniqueness_report_counts_distinct_near_optimal_groupings():
    rep = MT.nonuniqueness_report(
        [(0, 0, 1, 1), (0, 1, 1, 0), (0, 0, 1, 1), (1, 0, 0, 1)],
        [1.0, 1.05, 1.02, 50.0],  # the last one did not converge
        rel_tol=0.2,
    )
    assert rep.n_near_optimal == 3
    assert rep.n_distinct == 2
    assert rep.non_unique


def test_nonuniqueness_report_ignores_module_relabelling():
    rep = MT.nonuniqueness_report([(0, 0, 1, 1), (1, 1, 0, 0)], [1.0, 1.0])
    assert rep.n_distinct == 1
    assert not rep.non_unique


def test_nonuniqueness_report_handles_no_fits():
    assert MT.nonuniqueness_report([], []).n_distinct == 0


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def test_modular_transition_never_mixes_modules():
    """Block-diagonality is structural, so this must hold at initialisation."""
    torch.manual_seed(0)
    dyn = ModularTransition([2, 3])
    z = torch.randn(16, 5)
    z2 = z.clone()
    z2[:, 2:] = torch.randn(16, 3)  # perturb module 2 only
    assert torch.allclose(dyn(z)[:, :2], dyn(z2)[:, :2])
    assert not torch.allclose(dyn(z)[:, 2:], dyn(z2)[:, 2:])


def test_modular_transition_jacobian_is_block_diagonal():
    torch.manual_seed(1)
    dyn = ModularTransition([2, 2])
    J = torch.autograd.functional.jacobian(lambda z: dyn(z.unsqueeze(0)).squeeze(0), torch.randn(4))
    assert torch.allclose(J[:2, 2:], torch.zeros(2, 2), atol=1e-7)
    assert torch.allclose(J[2:, :2], torch.zeros(2, 2), atol=1e-7)


def test_unconstrained_transition_does_mix_modules():
    torch.manual_seed(2)
    dyn = UnconstrainedTransition(4)
    J = torch.autograd.functional.jacobian(lambda z: dyn(z.unsqueeze(0)).squeeze(0), torch.randn(4))
    assert J[:2, 2:].abs().max() > 1e-3


def test_model_config_validates_partition_and_names():
    with pytest.raises(ValueError, match="does not sum"):
        ModelConfig(n_obs=8, d=4, partition=[2, 3])
    with pytest.raises(ValueError, match="decoder"):
        ModelConfig(n_obs=8, d=4, decoder="quadratic")
    assert ModelConfig(n_obs=8, d=4, partition=[2, 2]).modular
    assert not ModelConfig(n_obs=8, d=4).modular


@pytest.mark.parametrize("decoder", ["linear", "mlp"])
def test_model_shapes_and_loss_terms(decoder):
    torch.manual_seed(3)
    cfg = ModelConfig(n_obs=7, d=4, partition=[2, 2], decoder=decoder)
    model = LatentDynamicsModel(cfg)
    x = torch.randn(5, 11, 7)
    out = model(x)
    assert out["z"].shape == (5, 11, 4)
    assert out["x_rec"].shape == (5, 11, 7)
    assert out["z_pred"].shape == (5, 10, 4)

    losses = model.losses(x)
    # "behavior" is the Route B u-dependence penalty; it is always reported and
    # is an exact zero unless u, an invariant slice and w_behavior are all given.
    assert set(losses) == {"total", "recon", "dyn", "white", "behavior", "fit_quality"}
    assert losses["behavior"].item() == 0.0
    assert losses["total"].requires_grad
    assert not losses["fit_quality"].requires_grad


def test_fit_quality_excludes_the_whitening_term():
    torch.manual_seed(4)
    model = LatentDynamicsModel(ModelConfig(n_obs=6, d=2, partition=[1, 1]))
    x = torch.randn(4, 9, 6)
    d = model.losses(x, w_recon=1.0, w_dyn=1.0, w_white=1.0)
    assert d["fit_quality"].item() == pytest.approx(d["recon"].item() + d["dyn"].item(), rel=1e-6)


def test_training_reduces_the_loss_and_is_reproducible():
    rng = np.random.default_rng(6)
    sys = S.two_oscillator_system(s=(0.95, 0.70))
    X, Z, _ = make_dataset(sys, n_obs=8, n_traj=48, T=12, rng=rng)
    cfg = ModelConfig(n_obs=8, d=4, partition=[2, 2])
    tcfg = TrainConfig(steps=150, seed=7)

    a = fit(X, cfg, tcfg)
    b = fit(X, cfg, tcfg)
    assert a.fit_quality == pytest.approx(b.fit_quality, rel=1e-9), "same seed => same fit"
    assert np.mean(a.history[-20:]) < np.mean(a.history[:20])


def test_make_dataset_shapes_and_exact_latent_dynamics():
    rng = np.random.default_rng(8)
    sys = S.two_oscillator_system()
    X, Z, dec = make_dataset(sys, n_obs=9, n_traj=12, T=15, rng=rng)
    assert X.shape == (12, 16, 9)
    assert Z.shape == (12, 16, 4)
    assert np.allclose(X, Z @ dec.W.T)
    assert np.allclose(Z[:, 1:], sys.step(Z[:, :-1]))


# --------------------------------------------------------------------------
# Nonlinear block structure: the linear probe's blind spot and its two fixes
# --------------------------------------------------------------------------

PART = [2, 2]
ORDER = [0, 1]
ASSIGN = [0, 1]


def _points(n=3000, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 4))


def _triangular(c, kind="quad", direction="lower"):
    """h with block B reading block A (lower) or A reading B (upper)."""
    def h(z):
        zA, zB = z[:, :2], z[:, 2:]
        src = zA if direction == "lower" else zB
        extra = c * (src * src) if kind == "quad" else c * src
        return np.hstack([zA, zB + extra]) if direction == "lower" else np.hstack([zA + extra, zB])
    return h


def test_jacobian_of_matches_an_analytic_jacobian():
    z = _points(200)

    def h(zz):
        return np.hstack([zz[:, :2] ** 2, np.sin(zz[:, 2:])])

    J = MT.jacobian_of(h, z)
    assert J.shape == (200, 4, 4)
    expected = np.zeros((200, 4, 4))
    expected[:, 0, 0] = 2 * z[:, 0]
    expected[:, 1, 1] = 2 * z[:, 1]
    expected[:, 2, 2] = np.cos(z[:, 2])
    expected[:, 3, 3] = np.cos(z[:, 3])
    assert np.abs(J - expected).max() < 1e-7


def test_the_linear_probe_is_blind_to_purely_quadratic_coupling():
    """The defect that motivates this whole section (CLAUDE.md §3.7).

    h = (z_A, z_B + 5 z_A^2) is overwhelmingly triangular -- the cross term has
    ~25x the variance of z_B's own contribution.  Cov(z_A, z_A^2) = 0 for
    symmetric z_A, so the linear relation cannot see it at all.
    """
    z = _points()
    h = _triangular(5.0)
    lin = MT.filtration_report(z, h(z), PART, PART, rate_order=ORDER)
    jac = MT.jacobian_block_report(h, z, PART, PART, rate_order=ORDER, assignment=ASSIGN)

    # thresholds are on the *standardised* scale, where this family saturates at
    # a diagonal share of 1/3 (fitted B becomes pure z_A, so E -> [[2,0],[4,0]])
    assert lin.on_block > 0.95, "the linear probe reports block-diagonal..."
    assert jac.on_block < 0.40, "...while the Jacobian sees a mostly off-block map"
    assert jac.lower_mass > 0.60
    assert jac.upper_mass == pytest.approx(0.0, abs=1e-12), "coupling is in the allowed direction"


def test_jacobian_report_is_monotone_in_the_coupling_strength():
    z = _points()
    prev = 1.1
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        r = MT.jacobian_block_report(_triangular(c), z, PART, PART,
                                     rate_order=ORDER, assignment=ASSIGN)
        assert r.on_block < prev, f"diagonal share must fall as c grows (c={c})"
        prev = r.on_block


def test_jacobian_report_is_exact_on_a_genuinely_block_diagonal_nonlinear_map():
    z = _points()

    def h(zz):
        return np.hstack([np.tanh(zz[:, :2]) + zz[:, :2] ** 3, 2.0 * np.sin(zz[:, 2:])])

    r = MT.jacobian_block_report(h, z, PART, PART, rate_order=ORDER, assignment=ASSIGN)
    assert r.on_block == pytest.approx(1.0, abs=1e-12)
    assert r.is_block_diagonal


def test_forbidden_upper_coupling_is_caught_by_jacobian_and_missed_by_linear():
    """The failure that matters: mass in the *forbidden* triangle (§4.2)."""
    z = _points()
    h = _triangular(2.0, direction="upper")
    lin = MT.filtration_report(z, h(z), PART, PART, rate_order=ORDER)
    jac = MT.jacobian_block_report(h, z, PART, PART, rate_order=ORDER, assignment=ASSIGN)
    dc = MT.distance_correlation_block_report(z, h(z), PART, PART,
                                              rate_order=ORDER, assignment=ASSIGN)
    assert lin.upper_mass < 0.01, "the linear probe barely registers it"
    assert jac.upper_mass > 0.50, "the Jacobian reports it as dominant"
    assert dc.upper_mass > 0.15, "and the model-free measure agrees it is there"


def test_max_energy_assignment_inverts_when_coupling_dominates():
    """Why ``assignment`` exists: the default matching flips at strong coupling.

    At c = 5 the off-block coupling exceeds the on-block one, so the max-energy
    matching pairs fitted-B with true-A and the relabelling moves that coupling
    *onto* the diagonal -- overstating how block-diagonal the map is.  Pinning
    the known correspondence fixes it.

    Standardisation (on by default) softens this a lot -- unstandardised, the
    same map reports 0.98 -- but does not remove it, because the inversion is a
    property of the matching, not of the scale.  Pin the correspondence whenever
    it is known; use auto-matching only when the fit may permute blocks, and
    then read the raw ``coupling`` matrix to check the pairing is sane.
    """
    z = _points()
    h = _triangular(5.0)
    auto = MT.jacobian_block_report(h, z, PART, PART, rate_order=ORDER)
    pinned = MT.jacobian_block_report(h, z, PART, PART, rate_order=ORDER, assignment=ASSIGN)
    assert auto.on_block > pinned.on_block + 0.20, "the automatic matching is fooled"
    assert pinned.lower_mass > 0.60, "the pinned correspondence sees the coupling"


def test_distance_correlation_basic_properties():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 2))
    assert MT.distance_correlation(x, x) == pytest.approx(1.0, abs=1e-10)
    assert MT.distance_correlation(x, 3.0 * x) == pytest.approx(1.0, abs=1e-10)
    # The point of dCor: quadratic dependence carries zero *population* linear
    # correlation for symmetric x.  Needs a big enough sample that the finite-
    # sample corrcoef (sd ~ 1/sqrt(n)) is actually small -- at n = 400 it lands
    # at 0.19 and the contrast this test is making would not be visible.
    big = rng.normal(size=(6000, 1))
    y = big ** 2
    assert abs(np.corrcoef(big[:, 0], y[:, 0])[0, 1]) < 0.05
    assert MT.distance_correlation(big, y) > 0.35


def test_distance_correlation_baseline_is_the_independence_floor():
    """dCor does not reach 0 for independent blocks; 0.03-0.07 is *no* coupling."""
    base = MT.distance_correlation_baseline(400, 2, 2, seed=3, reps=4)
    assert 0.0 < base < 0.25
    rng = np.random.default_rng(4)
    indep = MT.distance_correlation(rng.normal(size=(400, 2)), rng.normal(size=(400, 2)))
    assert indep < 4.0 * base


def test_jacobian_report_is_invariant_to_rescaling_a_block():
    """§7 grants within-module reparameterisation, so the metric must ignore it.

    Without standardisation the raw energy of a block scales with the square of
    its gain, and two equally-well-recovered blocks whose scales differ by 30x
    report energies differing by ~10^3 -- which is what made exp11's first
    Jacobian readout unreadable.
    """
    z = _points()
    h = _triangular(1.0)

    def h_scaled(zz):  # blow up fitted block A, shrink fitted block B
        out = h(zz)
        return np.hstack([30.0 * out[:, :2], 0.05 * out[:, 2:]])

    plain = MT.jacobian_block_report(h, z, PART, PART, rate_order=ORDER, assignment=ASSIGN)
    scaled = MT.jacobian_block_report(h_scaled, z, PART, PART, rate_order=ORDER, assignment=ASSIGN)
    assert scaled.on_block == pytest.approx(plain.on_block, abs=1e-9)
    assert scaled.lower_mass == pytest.approx(plain.lower_mass, abs=1e-9)

    raw = MT.jacobian_block_report(h_scaled, z, PART, PART, rate_order=ORDER,
                                   assignment=ASSIGN, standardize=False)
    assert abs(raw.on_block - plain.on_block) > 0.3, "unstandardised, the gauge dominates"


def test_block_reports_reject_a_bad_assignment():
    z = _points(300)
    h = _triangular(1.0)
    with pytest.raises(ValueError, match="assignment must be a permutation"):
        MT.jacobian_block_report(h, z, PART, PART, rate_order=ORDER, assignment=[0, 0])


# --------------------------------------------------------------------------
# Hypothesis diagnostics for a fitted h (CLAUDE.md task 32)
# --------------------------------------------------------------------------


def _lin(A):
    return lambda z: np.asarray(z, float) @ np.asarray(A, float).T


def test_conjugacy_residual_is_zero_for_a_genuine_conjugacy():
    """h A h^{-1} really conjugates A, so the residual must be at the noise floor."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 4))
    A = np.diag([0.9, 0.8, 0.5, 0.4])
    H = rng.standard_normal((4, 4)) + 3.0 * np.eye(4)
    rep = MT.conjugacy_residual(_lin(H), _lin(A), _lin(H @ A @ np.linalg.inv(H)), z)
    assert rep.rel_step < 1e-12
    assert rep.rel_state < 1e-12


def test_conjugacy_residual_rel_step_is_one_for_the_do_nothing_model():
    """The point of normalising by the increment: F~ = id must score 1, not ~0.

    Under a contraction both h(Fz) and h(z) shrink, so a state-relative residual
    flatters a model that has learned nothing.  rel_step does not.
    """
    rng = np.random.default_rng(1)
    z = rng.standard_normal((500, 4))
    A = np.diag([0.97, 0.97, 0.95, 0.95])  # a slow contraction: small increments
    rep = MT.conjugacy_residual(lambda x: x, _lin(A), lambda x: x, z)
    assert rep.rel_step == pytest.approx(1.0, abs=1e-12), "F~ = id reproduces none of the motion"
    assert rep.rel_state < 0.05, "yet the state-relative reading calls it a 4% error"


def test_conjugacy_residual_grows_with_the_mismatch():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((500, 4))
    A = np.diag([0.9, 0.8, 0.5, 0.4])
    prev = -1.0
    for eps in (0.0, 0.05, 0.2, 0.5):
        rep = MT.conjugacy_residual(lambda x: x, _lin(A), _lin(A + eps * np.eye(4)), z)
        assert rep.rel_step > prev
        prev = rep.rel_step


def test_hessian_of_matches_an_analytic_hessian():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((200, 3))
    Q = rng.standard_normal((3, 3))
    Q = Q + Q.T  # symmetric, so the analytic Hessian of z -> z Q z is 2Q

    def h(zz):
        return np.stack([np.einsum("ni,ij,nj->n", zz, Q, zz), zz[:, 0]], axis=1)

    H = MT.hessian_of(h, z, eps=1e-3)
    assert np.abs(H[:, 0] - 2.0 * Q).max() < 1e-6
    assert np.abs(H[:, 1]).max() < 1e-6, "a linear output has no curvature"


def test_additivity_defect_is_zero_for_an_additive_map_and_positive_otherwise():
    """Lemma D assumes h_B = z_B + psi(z_A); that is exactly a vanishing mixed partial."""
    rng = np.random.default_rng(4)
    z = rng.standard_normal((400, 4))
    part, out = [2, 2], slice(2, 4)

    additive = lambda zz: np.hstack([zz[:, :2], zz[:, 2:] + 1.5 * zz[:, :2] ** 2])
    assert MT.additivity_defect(additive, z, part, out) < 1e-6

    # z_A * z_B is the canonical non-additive coupling: pure mixed curvature
    product = lambda zz: np.hstack([zz[:, :2], zz[:, 2:] + 1.5 * zz[:, :2] * zz[:, 2:]])
    assert MT.additivity_defect(product, z, part, out) > 0.9


def test_additivity_defect_is_invariant_to_rescaling_a_block():
    """Same gauge argument as jacobian_block_report(standardize=True), §3.10 trap 1."""
    rng = np.random.default_rng(5)
    z = rng.standard_normal((400, 4))
    part, out = [2, 2], slice(2, 4)
    h = lambda zz: np.hstack([zz[:, :2], zz[:, 2:] + 0.8 * zz[:, :2] ** 2
                              + 0.6 * zz[:, :2] * zz[:, 2:]])
    plain = MT.additivity_defect(h, z, part, out)

    scaled = z.copy()
    scaled[:, :2] *= 20.0
    h_s = lambda zz: h(np.hstack([zz[:, :2] / 20.0, zz[:, 2:]]))
    assert MT.additivity_defect(h_s, scaled, part, out) == pytest.approx(plain, rel=1e-6)
