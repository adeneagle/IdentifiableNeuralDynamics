"""Partition-lattice model selection and fitted-model certification (exp06)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from idyn import selection as SEL


# --------------------------------------------------------------------------
# integer_partitions
# --------------------------------------------------------------------------


def test_integer_partitions_of_four():
    parts = SEL.integer_partitions(4)
    assert parts == [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]


@pytest.mark.parametrize("d, count", [(1, 1), (2, 2), (3, 3), (4, 5), (5, 7), (6, 11)])
def test_partition_counts_match_the_partition_function(d, count):
    parts = SEL.integer_partitions(d)
    assert len(parts) == count
    for p in parts:
        assert sum(p) == d
        assert list(p) == sorted(p, reverse=True), "blocks descending"


def test_integer_partitions_rejects_nonpositive():
    with pytest.raises(ValueError):
        SEL.integer_partitions(0)


# --------------------------------------------------------------------------
# select_finest_partition
# --------------------------------------------------------------------------


def test_selects_true_finest_when_finer_fits_worse():
    """Oscillator-like: finer partitions split an indecomposable block and pay."""
    scores = {(4,): 3.0e-4, (3, 1): 8e-3, (2, 2): 3.5e-4, (2, 1, 1): 1e-2, (1, 1, 1, 1): 2e-2}
    winner, rows = SEL.select_finest_partition(scores, rel_tol=3.0)
    assert winner == (2, 2)
    assert next(r for r in rows if r.partition == (2, 2)).selected
    assert not next(r for r in rows if r.partition == (1, 1, 1, 1)).acceptable


def test_selects_all_singletons_when_they_fit():
    """Regrouping-like: the finest partition fits, so it wins."""
    scores = {(4,): 3e-4, (3, 1): 3.2e-4, (2, 2): 3.1e-4, (2, 1, 1): 3.3e-4, (1, 1, 1, 1): 3.4e-4}
    winner, _ = SEL.select_finest_partition(scores, rel_tol=3.0)
    assert winner == (1, 1, 1, 1)


def test_uniqueness_gate_rejects_a_nonunique_partition():
    """A non-unique fit is the §3.1 signature and must be rejected even if it fits."""
    scores = {(4,): 3e-4, (2, 2): 3.1e-4, (1, 1, 1, 1): 9e-4}
    uniq = {(4,): True, (2, 2): False, (1, 1, 1, 1): True}
    winner, rows = SEL.select_finest_partition(scores, rel_tol=3.0, uniqueness=uniq)
    assert winner == (1, 1, 1, 1)
    assert not next(r for r in rows if r.partition == (2, 2)).acceptable


def test_selection_requires_scores():
    with pytest.raises(ValueError):
        SEL.select_finest_partition({})


# --------------------------------------------------------------------------
# fixed point + linearisation (stubbed, fast)
# --------------------------------------------------------------------------


class _Stub:
    """Minimal stand-in for a LatentDynamicsModel: just .cfg.d and .dyn."""

    def __init__(self, dyn, d):
        self.dyn = dyn
        self.cfg = type("C", (), {"d": d, "modular": True, "partition": None})()


def test_fixed_point_of_a_linear_contraction():
    stub = _Stub(lambda z: 0.5 * z, 2)
    z_star = SEL.fitted_transition_fixed_point(stub)
    assert np.allclose(z_star, [0.0, 0.0], atol=1e-6)


def test_fixed_point_of_an_affine_contraction():
    stub = _Stub(lambda z: 0.5 * z + 0.1, 2)  # fixed point at 0.1/(1-0.5) = 0.2
    z_star = SEL.fitted_transition_fixed_point(stub)
    assert np.allclose(z_star, [0.2, 0.2], atol=1e-6)


def test_linearisation_of_a_modular_transition_is_block_diagonal():
    from idyn.models import ModularTransition

    torch.manual_seed(0)
    dyn = ModularTransition([2, 2])
    stub = _Stub(dyn, 4)
    J = SEL.linearize_modular_transition(stub, z_star=np.zeros(4))
    assert np.allclose(J[:2, 2:], 0.0, atol=1e-6)
    assert np.allclose(J[2:, :2], 0.0, atol=1e-6)


# --------------------------------------------------------------------------
# certify_fitted_model
# --------------------------------------------------------------------------


def test_certification_runs_and_reports_per_block():
    from idyn.models import LatentDynamicsModel, ModelConfig

    torch.manual_seed(1)
    model = LatentDynamicsModel(ModelConfig(n_obs=8, d=4, partition=[2, 2]))
    cert = SEL.certify_fitted_model(model)
    assert cert.partition == [2, 2]
    assert len(cert.block_summands) == 2
    assert len(cert.indecomposable) == 2
    assert isinstance(cert.all_indecomposable, bool)


def test_certification_requires_a_modular_model():
    from idyn.models import LatentDynamicsModel, ModelConfig

    model = LatentDynamicsModel(ModelConfig(n_obs=8, d=4, partition=None))
    with pytest.raises(ValueError, match="modular"):
        SEL.certify_fitted_model(model)


def test_certification_detects_a_decomposable_block():
    """A learned block that is actually two 1-D directions must be flagged.

    Build a modular model and force one module's transition to a diagonal map
    with distinct eigenvalues (decomposable) and the other to a rotation
    (indecomposable), by overriding the linearisation input directly.
    """
    from idyn.models import LatentDynamicsModel, ModelConfig

    torch.manual_seed(2)
    model = LatentDynamicsModel(ModelConfig(n_obs=8, d=4, partition=[2, 2]))

    # Patch linearize to return a known block-diagonal Jacobian:
    # block 0 = distinct-eigenvalue diagonal (decomposable),
    # block 1 = scaled rotation (indecomposable).
    rot = 0.8 * np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    J = np.zeros((4, 4))
    J[:2, :2] = np.diag([0.9, 0.6])
    J[2:, 2:] = rot
    import idyn.selection as sel

    orig = sel.linearize_modular_transition
    sel.linearize_modular_transition = lambda m, z_star=None: J
    try:
        cert = SEL.certify_fitted_model(model)
    finally:
        sel.linearize_modular_transition = orig

    assert cert.block_summands == [2, 1]
    assert cert.indecomposable == [False, True]
    assert not cert.all_indecomposable


# --------------------------------------------------------------------------
# The nonlinear (B2) check: quadratic jet, not just the linear part
# --------------------------------------------------------------------------


def test_block_nonlinear_certificate_flags_the_resonant_node():
    """The linear part diag(mu, mu^2) splits, but the resonant map does not."""
    from idyn import systems as S

    node = S.ResonantNodeBlock(mu=0.7, c=0.9)
    chk = SEL.block_nonlinear_certificate(node.step, np.zeros(2), node.linear_part())
    assert chk.checked
    assert chk.indecomposable
    assert chk.max_coupling == pytest.approx(0.9, abs=1e-3)


def test_block_nonlinear_certificate_clears_the_linear_control():
    """c = 0 is genuinely its linear part, so it is decomposable."""
    from idyn import systems as S

    flat = S.ResonantNodeBlock(mu=0.7, c=0.0)
    chk = SEL.block_nonlinear_certificate(flat.step, np.zeros(2), flat.linear_part())
    assert chk.checked
    assert not chk.indecomposable
    assert chk.max_coupling < 1e-6


def test_a_nonresonant_cross_term_is_not_an_obstruction():
    """The subtle correctness point: only *resonant* coupling obstructs.

    f = (0.7 z_a, 0.6 z_b + 0.9 z_a^2) has a literal cross term, but
    lam_b - lam_a^2 = 0.6 - 0.49 != 0, so it is removable by a near-identity change
    of coordinates and the map is decomposable.
    """
    mu, nu, c = 0.7, 0.6, 0.9

    def step(z):
        z = np.asarray(z, float)
        return np.stack([mu * z[..., 0], nu * z[..., 1] + c * z[..., 0] ** 2], axis=-1)

    chk = SEL.block_nonlinear_certificate(step, np.zeros(2), np.diag([mu, nu]))
    assert chk.checked
    assert not chk.indecomposable


def test_nonlinear_check_is_coordinate_invariant():
    """The same map in sheared coordinates must give the same verdict."""
    from idyn import systems as S

    node = S.ResonantNodeBlock(mu=0.7, c=0.9)
    V = np.array([[1.0, 0.4], [0.0, 1.0]])
    Vi = np.linalg.inv(V)

    def step(z):
        return V @ node.step(Vi @ np.asarray(z, float))

    chk = SEL.block_nonlinear_certificate(step, np.zeros(2), V @ node.linear_part() @ Vi)
    assert chk.checked
    assert chk.indecomposable, "indecomposability is a conjugacy invariant"


def test_certificate_uses_graph_connectedness_not_mere_coupling():
    """A 3-block module coupled only on {0,1} is DECOMPOSABLE (node 2 splits off).

    'Any resonant coupling exists' would call this indecomposable; graph
    connectedness correctly calls it decomposable. This is the over-report the
    fit cannot catch, since the {0,1}(+){2} split fits the dynamics exactly.
    """
    mu, c, nu = 0.7, 0.9, 0.5  # nu non-resonant with mu, mu^2

    def step(z):
        z = np.asarray(z, float)
        return np.stack(
            [mu * z[..., 0], mu**2 * z[..., 1] + c * z[..., 0] ** 2, nu * z[..., 2]], axis=-1
        )

    chk = SEL.block_nonlinear_certificate(step, np.zeros(3), np.diag([mu, mu**2, nu]))
    assert chk.checked
    assert not chk.indecomposable, "node 2 is uncoupled, so the module splits"
    assert chk.n_components == 2
    assert chk.max_coupling == pytest.approx(c, abs=1e-2), "the 0-1 edge is still detected"


def test_certificate_connected_three_block_is_indecomposable():
    """nu = mu^3 makes z0*z1 -> z2 resonant, connecting node 2: one component."""
    mu, c = 0.7, 0.9

    def step(z):
        z = np.asarray(z, float)
        return np.stack(
            [
                mu * z[..., 0],
                mu**2 * z[..., 1] + c * z[..., 0] ** 2,
                mu**3 * z[..., 2] + 0.7 * z[..., 0] * z[..., 1],
            ],
            axis=-1,
        )

    chk = SEL.block_nonlinear_certificate(step, np.zeros(3), np.diag([mu, mu**2, mu**3]))
    assert chk.checked and chk.indecomposable
    assert chk.n_components == 1


def test_certificate_decomposable_three_block_split_is_exhibited():
    """The split the graph predicts is a genuine invariant product, not a claim.

    For the {0,1}(+){2} module, coordinate 2 evolves by z2 -> nu z2 with no
    dependence on z0, z1 and no z2-dependence anywhere else -- so {z2=0} and
    {z0=z1=0} are both invariant and the map is their product. Check that
    directly, so the certificate's 'decomposable' verdict is grounded.
    """
    mu, c, nu = 0.7, 0.9, 0.5

    def step(z):
        z = np.asarray(z, float)
        return np.stack(
            [mu * z[..., 0], mu**2 * z[..., 1] + c * z[..., 0] ** 2, nu * z[..., 2]], axis=-1
        )

    rng = np.random.default_rng(0)
    for _ in range(200):
        z = rng.normal(size=3)
        # {z2 = 0} invariant, and the (z0,z1) image is independent of z2
        z_masked = np.array([z[0], z[1], 0.0])
        assert step(z)[2] == pytest.approx(nu * z[2], abs=1e-12)          # z2 factor is autonomous
        assert step(z)[:2] == pytest.approx(step(z_masked)[:2], abs=1e-12)  # {0,1} ignores z2


def test_nonlinear_check_declines_when_it_does_not_apply():
    """A linearly indecomposable block, and a repeated-eigenvalue block, aren't checked."""
    # rotation: one real 2x2 block, already indecomposable
    rot = 0.8 * np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    chk = SEL.block_nonlinear_certificate(lambda z: rot @ np.asarray(z, float), np.zeros(2), rot)
    assert not chk.checked and chk.indecomposable

    # repeated eigenvalue: diagonalisable but the eigenbasis is ambiguous
    scal = np.diag([0.6, 0.6])
    chk2 = SEL.block_nonlinear_certificate(lambda z: scal @ np.asarray(z, float), np.zeros(2), scal)
    assert not chk2.checked


def test_certify_fitted_model_nonlinear_flips_the_witness_block():
    """End to end on a torch model whose first block IS the resonant node.

    The linear verdict calls block 0 decomposable; the nonlinear verdict, seeing
    the quadratic jet, corrects it.  This is the false negative of §A.2.2, closed.
    """
    import torch.nn as nn
    from idyn.models import LatentDynamicsModel, ModelConfig

    mu, c, nu = 0.7, 0.9, 0.5

    class WitnessResidual(nn.Module):
        def forward(self, z):
            za, zb = z[..., 0], z[..., 1]
            f = torch.stack([mu * za, mu**2 * zb + c * za**2], dim=-1)
            return f - z

    class LinearResidual(nn.Module):
        def forward(self, z):
            return (nu - 1.0) * z

    model = LatentDynamicsModel(ModelConfig(n_obs=6, d=3, partition=[2, 1]))
    model.dyn.nets[0] = WitnessResidual()
    model.dyn.nets[1] = LinearResidual()

    cert = SEL.certify_fitted_model(model, nonlinear=True)
    # linear verdict: block 0 splits (diag mu, mu^2), block 1 trivially indecomposable
    assert cert.block_summands == [2, 1]
    assert cert.indecomposable == [False, True]
    assert not cert.all_indecomposable
    # nonlinear verdict: block 0 is corrected to indecomposable
    assert cert.nonlinear_checked[0] and not cert.nonlinear_checked[1]
    assert cert.nonlinear_indecomposable == [True, True]
    assert cert.all_nonlinear_indecomposable
    assert cert.max_coupling[0] == pytest.approx(c, abs=1e-2)


def test_certify_fitted_model_linear_path_is_unchanged_by_default():
    """nonlinear=False leaves the certificate exactly as before (no new fields set)."""
    from idyn.models import LatentDynamicsModel, ModelConfig

    torch.manual_seed(3)
    model = LatentDynamicsModel(ModelConfig(n_obs=8, d=4, partition=[2, 2]))
    cert = SEL.certify_fitted_model(model)
    assert cert.nonlinear_indecomposable == []
    assert cert.nonlinear_checked == []
    # the fallback property uses the linear verdict when nonlinear wasn't run
    assert cert.all_nonlinear_indecomposable == cert.all_indecomposable
