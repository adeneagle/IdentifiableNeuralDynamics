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


def test_fit_linear_relation_is_translation_invariant():
    """h is defined up to translation, and a learned latent's mean is unpinned.

    Without a fitted intercept an off-centre z_fit makes the solve misspecified,
    and it fails *silently* -- the returned matrix still splits into block
    energies. Measured at R^2 = -1.38 (below the mean baseline) on an MLP-decoder
    fit before this was fixed.
    """
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((400, 4)) + 3.0        # off-centre inputs
    A = rng.standard_normal((4, 4))
    offset = np.array([5.0, -2.0, 0.5, 1.0])       # and off-centre outputs
    assert np.allclose(MT.fit_linear_relation(Z, Z @ A.T + offset), A, atol=1e-9)
    assert MT.linear_relation_r2(Z, Z @ A.T + offset) == pytest.approx(1.0, abs=1e-12)


def test_linear_relation_r2_gates_a_readout_that_is_not_a_fit():
    """The validity gate: near zero means the block readouts are not measuring h."""
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((600, 3))
    assert MT.linear_relation_r2(Z, Z @ rng.standard_normal((3, 3)).T) == pytest.approx(1.0, abs=1e-12)
    # unrelated targets: an affine fit explains essentially nothing
    assert abs(MT.linear_relation_r2(Z, rng.standard_normal((600, 3)))) < 0.05


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


def _coupled(psi):
    """h(z) = (z_A, z_B + psi(z_A)) on R^4, blocks of 2."""
    return lambda z: np.hstack([z[:, :2], z[:, 2:] + psi(z[:, :2])])


@pytest.mark.parametrize(
    "name, psi, expected",
    [
        ("degree 2", lambda a: 0.7 * (a ** 2), 2.0),
        ("degree 1", lambda a: a @ np.array([[0.4, -0.9], [1.1, 0.3]]).T, 1.0),
        ("degree 3", lambda a: 0.5 * (a ** 3), 3.0),
        # the escape: direction only, so psi(sigma a) = psi(a) exactly
        ("degree 0", lambda a: 0.8 * a / np.linalg.norm(a, axis=1, keepdims=True), 0.0),
    ],
)
def test_coupling_homogeneity_degree_recovers_known_degrees(name, psi, expected):
    """Lemma D turns on this exponent, so it had better be measurable."""
    p, resid = MT.coupling_homogeneity_degree(
        _coupled(psi), slice(2, 4), slice(0, 2), np.array([0.3, -0.2])
    )
    assert p == pytest.approx(expected, abs=0.02), name
    assert resid < 1e-6, "a genuinely homogeneous psi is an exact power law"


def test_coupling_degree_flags_a_non_homogeneous_coupling_via_the_residual():
    """The escape is p = 0, but a *saturating* psi also defeats variance modulation.

    Its fitted slope is some small positive number that describes nothing, which
    is why the residual is returned alongside and must be read with it.
    """
    sat = lambda a: 0.8 * np.tanh(6.0 * a)  # linear at 0, flat by sigma ~ 1
    p, resid = MT.coupling_homogeneity_degree(
        _coupled(sat), slice(2, 4), slice(0, 2), np.array([0.3, -0.2])
    )
    assert 0.0 < p < 1.0, "saturation drags the apparent degree below the true small-z one"
    assert resid > 0.05, "and it is not a power law at all -- the residual says so"


def test_coupling_degree_works_in_either_block_orientation():
    """z_other holds the complementary coordinates, whichever side in_slice is."""
    h = lambda z: np.hstack([z[:, :2] + 0.6 * z[:, 2:] ** 2, z[:, 2:]])
    p, resid = MT.coupling_homogeneity_degree(h, slice(0, 2), slice(2, 4), np.array([0.1, 0.4]))
    assert p == pytest.approx(2.0, abs=0.02)
    assert resid < 1e-6


def test_coupling_degree_rejects_degenerate_sigmas():
    h = _coupled(lambda a: a)
    with pytest.raises(ValueError, match="positive scale factors"):
        MT.coupling_homogeneity_degree(h, slice(2, 4), slice(0, 2), np.zeros(2), sigmas=[1.0, -1.0])


# --------------------------------------------------------------------------
# Fit-to-fit invariant agreement (task 40) -- no ground truth anywhere
# --------------------------------------------------------------------------


class _Sheared:
    """h(x, y) = (x + c y^3, y) on one module: a §7 gauge change, nothing more."""

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

    def jacobian(self, w):
        w = np.asarray(w, float).reshape(2)
        z = self._hinv(w)
        Dh = np.array([[1.0, 3 * self.c * float(self.blk.step(z)[1]) ** 2], [0.0, 1.0]])
        Dhi = np.array([[1.0, -3 * self.c * float(w[1]) ** 2], [0.0, 1.0]])
        return Dh @ self.blk.jacobian(z) @ Dhi


def _ann(rng, n=6, lo=0.6, hi=1.1, k=2):
    out = []
    for _ in range(k):
        th, r = rng.uniform(-np.pi, np.pi, n), rng.uniform(lo, hi, n)
        out.append(np.stack([r * np.cos(th), r * np.sin(th)], axis=-1))
    return np.concatenate(out, axis=-1)


def _twin_twists():
    return S.ModularSystem([S.TwistBlock(s=0.90, omega=0.40, beta=0.6),
                            S.TwistBlock(s=0.55, omega=1.10, beta=0.3)])


def _fp(system, z0, T=800):
    return MT.dynamical_fingerprint(system, z0, T=T, warmup=200, T_rotation=1200)


def test_fingerprint_is_blind_to_a_within_module_change_of_coordinates():
    sysm = _twin_twists()
    z0 = _ann(np.random.default_rng(0))
    gauged = S.ModularSystem([_Sheared(b) for b in sysm.blocks])
    r = MT.invariant_agreement(_fp(sysm, z0), _fp(gauged, z0))
    assert r.agree
    assert r.spectrum_error < 1e-9
    assert r.rotation_error < 1e-9


def test_fingerprint_is_blind_to_module_relabelling():
    sysm = _twin_twists()
    z0 = _ann(np.random.default_rng(0))
    swapped = S.ModularSystem([sysm.blocks[1], sysm.blocks[0]])
    r = MT.invariant_agreement(_fp(sysm, z0), _fp(swapped, z0[:, [2, 3, 0, 1]]))
    assert r.agree
    assert r.order_agrees


def test_the_regrouping_counterexample_is_caught():
    """§3.1: three decompositions fit the same observations to 2.2e-16.

    Fit quality cannot separate them -- the tie is exact -- so this is precisely
    the case the invariants have to catch, and the negative control for the whole
    method.
    """
    rg = S.regrouping_counterexample()
    z0 = np.random.default_rng(0).uniform(0.5, 1.0, (6, 4))
    r = MT.invariant_agreement(_fp(rg["system"], z0), _fp(rg["system_tilde"], z0))
    assert not r.agree
    assert r.spectrum_error > 0.2


def test_rotation_catches_what_the_spectrum_provably_cannot():
    """Task 23: identical spectra, different frequencies."""
    a = S.ModularSystem([S.LimitCycleBlock(a=0.3, omega=0.5),
                         S.LimitCycleBlock(a=0.3, omega=1.3)])
    b = S.ModularSystem([S.LimitCycleBlock(a=0.3, omega=0.5),
                         S.LimitCycleBlock(a=0.3, omega=0.9)])
    z0 = _ann(np.random.default_rng(0), lo=0.9, hi=1.1)
    r = MT.invariant_agreement(_fp(a, z0), _fp(b, z0))
    assert not r.agree
    assert r.spectrum_error < 1e-6      # a spectrum-only test would say AGREE
    assert r.rotation_error == pytest.approx(0.4 / (2 * np.pi), abs=1e-6)


def test_two_oscillatory_modules_report_a_zero_order_margin():
    """Both lead with a neutral exponent, so the spectrum cannot order them.

    The margin is what makes that visible instead of letting an arbitrary
    tie-break read as a filtration.
    """
    sysm = S.ModularSystem([S.LimitCycleBlock(a=0.3, omega=0.5),
                            S.LimitCycleBlock(a=0.3, omega=1.3)])
    fp = _fp(sysm, _ann(np.random.default_rng(0), lo=0.9, hi=1.1))
    assert fp.order_margin < 1e-6
    assert _fp(_twin_twists(), _ann(np.random.default_rng(0))).order_margin > 0.4


def test_a_different_module_count_disagrees_without_crashing():
    z0 = _ann(np.random.default_rng(0))
    two = _fp(_twin_twists(), z0)
    one = MT.DynamicalFingerprint(partition=[4], spectra=[np.zeros(4)],
                                  rotations=[0.0], coherences=[1.0])
    r = MT.invariant_agreement(two, one)
    assert not r.agree and not r.same_K
    assert any("module count" in n for n in r.notes)


def test_unmeasurable_rotation_is_a_disagreement_not_a_silent_match():
    """NaN means two different things and conflating them hides a null result.

    A 1-D block *cannot* rotate, so two of them agree.  A 2-D module whose
    rotation could not be measured has an *unknown* one, and scoring that as a
    match would let the metric report agreement where it has no information --
    the §3.10 failure mode wearing a different hat.
    """
    flat = MT.DynamicalFingerprint(partition=[1, 1], spectra=[np.array([-0.3]),
                                                             np.array([-0.7])],
                                   rotations=[float("nan")] * 2, coherences=[0.0] * 2)
    assert MT.invariant_agreement(flat, flat).agree          # structural: fine

    known = MT.DynamicalFingerprint(partition=[2], spectra=[np.array([-0.1, -0.1])],
                                    rotations=[0.0636], coherences=[1.0])
    unknown = MT.DynamicalFingerprint(partition=[2], spectra=[np.array([-0.1, -0.1])],
                                      rotations=[float("nan")], coherences=[0.0])
    r = MT.invariant_agreement(known, unknown)
    assert not r.agree
    assert np.isinf(r.rotation_error)
    assert any("no measurable rotation" in n for n in r.notes)


def test_modules_are_paired_by_rotation_when_the_spectra_are_degenerate():
    """Regression: matching on spectra alone fails exactly where rho is needed.

    Two fitted limit cycles have near-identical spectra, so a spectrum-only cost
    matrix is flat and the pairing is decided by nothing.  `exp14` part 4a hit
    this in 5 of 16 comparisons, each returning a rotation error of 0.1274 --
    which is |rho_1 - rho_2|, the signature of a swap, not of a bad fit.

    Here module 0 of `a` (rho 0.207) must pair with module 1 of `b` (rho 0.207),
    against a spectral difference that mildly favours the wrong pairing.
    """
    a = MT.DynamicalFingerprint(
        partition=[2, 2],
        spectra=[np.array([0.0000, -0.60]), np.array([0.0000, -0.62])],
        rotations=[0.20690, 0.07958], coherences=[1.0, 1.0])
    b = MT.DynamicalFingerprint(
        partition=[2, 2],
        spectra=[np.array([0.0005, -0.62]), np.array([-0.0005, -0.60])],
        rotations=[0.07958, 0.20690], coherences=[1.0, 1.0])
    r = MT.invariant_agreement(a, b, spec_tol=0.05, rot_tol=0.01)
    assert r.rotation_error < 1e-4, "modules were paired by spectrum and swapped"
    assert r.agree


def test_an_undetermined_filtration_order_is_not_scored_as_a_disagreement():
    """When the spectrum cannot order the modules, there is no order to agree on.

    Two limit cycles both lead with a neutral exponent, so `order_margin` is ~0.
    Demanding that two fits list them in the same sequence scores an
    *undetermined* quantity as a disagreement -- `exp14` part 4a matched on
    rotation to 5e-4 in all 16 comparisons while `order_agrees` held in only 10.
    `agree` therefore drops the requirement when the margin is below tolerance,
    and says so in the notes; `order_agrees` is still reported for the caller.
    """
    a = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0005, -0.6]), np.array([-0.0005, -0.6])],
        rotations=[0.07958, 0.20690], coherences=[1.0, 1.0])
    b = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([-0.0004, -0.6]), np.array([0.0004, -0.6])],
        rotations=[0.07958, 0.20690], coherences=[1.0, 1.0])
    r = MT.invariant_agreement(a, b, spec_tol=0.05, rot_tol=0.01)
    assert r.order_margin < 0.05
    assert r.agree
    assert any("does not\n" not in n and "determine the ordering" in n for n in r.notes)

    # ... but a REAL disagreement is still caught, margin or no margin.
    c = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0005, -0.6]), np.array([-0.0005, -0.6])],
        rotations=[0.07958, 0.14324], coherences=[1.0, 1.0])
    assert not MT.invariant_agreement(a, c, spec_tol=0.05, rot_tol=0.01).agree


def test_near_tied_spectra_do_not_let_noise_decide_the_filtration_order():
    """Leading exponents 5e-4 apart are a tie; |rho| breaks it, not the noise."""
    fp = MT.DynamicalFingerprint(
        partition=[2, 2],
        spectra=[np.array([-0.0005, -0.60]), np.array([0.0005, -0.62])],
        rotations=[0.20690, 0.07958], coherences=[1.0, 1.0])
    assert fp.order == [0, 1]          # by |rho|, despite module 1 leading on lambda
    assert fp.order_margin == pytest.approx(1e-3, abs=1e-9)


def test_mode_collapse_is_detectable_without_ground_truth():
    """Two modules on one factor -- the failure neither coherence nor fit finds.

    Measured at 32 neurons/side: 2 of 12 restarts collapsed both modules onto the
    slower cycle.  `coherence` correlated with recovery error at only -0.48 (a
    collapsed fit scored 0.961, above several good ones) and `fit_quality` at
    +0.24 -- no information, wrong sign.  Duplicate invariants are a property of
    the fitted model alone, so they are checkable on real data.
    """
    collapsed = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0, -0.9]), np.array([0.0, -0.9])],
        rotations=[0.07958, 0.07960], coherences=[0.96, 0.95])
    healthy = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0, -0.9]), np.array([0.0, -0.9])],
        rotations=[0.07958, 0.20690], coherences=[0.96, 0.95])
    assert collapsed.duplicate_modules() == [(0, 1)]
    assert healthy.duplicate_modules() == []

    r = MT.invariant_agreement(healthy, collapsed, spec_tol=0.05, rot_tol=0.01)
    assert not r.agree
    assert any("mode collapse" in n for n in r.notes)
    # ...and a genuinely duplicated system is flagged, not rejected: the flag is
    # a thing to check, not a verdict.
    assert MT.invariant_agreement(collapsed, collapsed, spec_tol=0.05, rot_tol=0.01).agree


def test_canonical_order_puts_the_autonomous_module_first():
    """Lemma C's direction: the module with the LARGER exponents is the driver."""
    fp = _fp(_twin_twists(), _ann(np.random.default_rng(0))).reordered()
    assert fp.spectra[0][0] > fp.spectra[1][0]
    assert fp.spectra[0][0] == pytest.approx(np.log(0.90), abs=1e-6)
    assert fp.spectra[1][0] == pytest.approx(np.log(0.55), abs=1e-6)
