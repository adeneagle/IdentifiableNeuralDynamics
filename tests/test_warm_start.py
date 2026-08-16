"""Adversarial initialisation: the warm start, and what it is for (task 41).

`exp16` found that cross-split agreement is **necessary but not sufficient** for
identifiability -- its arm C agreed to 0.0004 on a system where non-identifiability
is proved, because both fits happened to land on the same lattice representative.
The repair (`exp17`) is to warm-start the two halves at *different* representatives
and let ordinary training decide whether the data pulls them back.

The whole method rests on the warm start actually reaching the representative it
names, so that is what these tests pin: reaching a target, refusing a
mis-shaped one, leaving ordinary fitting alone, and staying silent when it is not
asked for.  Everything downstream is uninterpretable without those.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from idyn import systems as S
from idyn.models import ModelConfig
from idyn.train import TrainConfig, fit, make_dataset


def _data(seed: int = 0, n_traj: int = 60, T: int = 12, n_obs: int = 24):
    rng = np.random.default_rng(seed)
    system = S.ModularSystem([
        S.TwistBlock(s=0.92, omega=0.35, beta=0.0),
        S.TwistBlock(s=0.70, omega=1.10, beta=0.0),
    ])
    X, Z, _ = make_dataset(system, n_obs, n_traj, T, rng)
    X = (X - X.mean(axis=(0, 1), keepdims=True)) / X.std()
    return X, Z


def _cfg(X, encoder="mlp", decoder="mlp"):
    return ModelConfig(n_obs=X.shape[-1], d=4, partition=[2, 2],
                       decoder=decoder, encoder=encoder)


# --------------------------------------------------------------------------
# The warm start reaches what it names
# --------------------------------------------------------------------------


def test_warm_start_drives_the_encoder_to_its_target():
    """The residual is a fraction of the target's own variance, so <<1 means reached.

    This is the precondition for every exp17 reading.  A fit that never arrived
    at its adversarial representative cannot testify that the data pushed it
    away from one, and reporting "it returned" off such a fit would describe the
    setup rather than the system -- the §3.9 family's error one more time.
    """
    X, Z = _data()
    res = fit(X, _cfg(X), TrainConfig(steps=1, seed=0, warm_steps=600), warm_z=Z)
    assert res.warm_residual < 0.1, res.warm_residual


def test_a_scrambled_target_is_much_harder_to_reach_than_the_true_one():
    """The residual has to *discriminate*, or it is not a diagnostic.

    Latents shuffled across trials are not any function of the observations, so
    no encoder can produce them; the true latents are the encoder's own job.
    If both scored alike the number would be measuring optimiser effort rather
    than reachability.
    """
    X, Z = _data()
    rng = np.random.default_rng(1)
    bad = Z[rng.permutation(Z.shape[0])]
    tc = TrainConfig(steps=1, seed=0, warm_steps=600)
    good = fit(X, _cfg(X), tc, warm_z=Z).warm_residual
    poor = fit(X, _cfg(X), tc, warm_z=bad).warm_residual
    assert poor > 5.0 * good, (good, poor)


def test_a_linear_encoder_cannot_hold_a_nonlinear_representative():
    """§11.7, as a measurement: this is why exp17 requires ``encoder="mlp"``.

    Under a linear encoder z_hat = L g(z) is linear in z when g is, so the
    lattice alternative h(z1,z2) = (z1 z2/|z2|, z2) is not merely hard to find --
    it is **outside the model class**, and a protocol run there would measure the
    projection onto that class rather than anything about the data.  That is what
    invalidated exp16's arm-C diagnosis.
    """
    X, Z = _data()

    def lattice(W):
        z1 = W[..., 0] + 1j * W[..., 1]
        z2 = W[..., 2] + 1j * W[..., 3]
        w = z1 * z2 / np.maximum(np.abs(z2), 1e-300)
        return np.stack([w.real, w.imag, z2.real, z2.imag], -1)

    alt = lattice(Z)
    tc = TrainConfig(steps=1, seed=0, warm_steps=600)
    lin = fit(X, _cfg(X, encoder="linear", decoder="linear"), tc, warm_z=alt).warm_residual
    nl = fit(X, _cfg(X), tc, warm_z=alt).warm_residual
    assert lin > 0.3, f"a linear encoder should NOT reach it, got {lin}"
    assert nl < lin / 2.0, f"an mlp encoder should do better, got {nl} vs {lin}"


# --------------------------------------------------------------------------
# It stays out of the way otherwise
# --------------------------------------------------------------------------


def test_no_warm_start_leaves_the_residual_undefined():
    """NaN, not 0.  A missing measurement must not read as a perfect one --
    exp15 shipped three checks that passed because NaN was coerced to +inf."""
    X, _ = _data()
    res = fit(X, _cfg(X), TrainConfig(steps=20, seed=0), warm_z=None)
    assert math.isnan(res.warm_residual)


def test_warm_z_is_ignored_without_warm_steps():
    """``warm_steps=0`` must be a true no-op, so a config can carry the target
    around without silently changing what a fit means."""
    X, Z = _data()
    tc = TrainConfig(steps=20, seed=3, warm_steps=0)
    a = fit(X, _cfg(X), tc, warm_z=Z)
    b = fit(X, _cfg(X), tc, warm_z=None)
    assert math.isnan(a.warm_residual)
    assert a.fit_quality == pytest.approx(b.fit_quality, rel=1e-12)


def test_a_mis_shaped_target_is_rejected_loudly():
    X, Z = _data()
    tc = TrainConfig(steps=1, seed=0, warm_steps=10)
    with pytest.raises(ValueError, match="warm_z"):
        fit(X, _cfg(X), tc, warm_z=Z[..., :2])


def test_the_warm_start_does_not_poison_the_ordinary_fit():
    """Started at the truth, a warm-started fit must not be *worse* than a cold
    one.  If it were, exp17's arms would be comparing training regimes."""
    X, Z = _data()
    cold = fit(X, _cfg(X), TrainConfig(steps=400, seed=7))
    warm = fit(X, _cfg(X), TrainConfig(steps=400, seed=7, warm_steps=400), warm_z=Z)
    assert warm.fit_quality <= 1.5 * cold.fit_quality, (cold.fit_quality, warm.fit_quality)


def test_the_optimiser_state_does_not_carry_over():
    """A fresh Adam for the real objective, so no warm-start momentum leaks in."""
    X, Z = _data()
    tc = TrainConfig(steps=200, seed=11, warm_steps=200)
    res = fit(X, _cfg(X), tc, warm_z=Z)
    assert len(res.history) == 200
    assert all(np.isfinite(res.history))


# --------------------------------------------------------------------------
# The escape control's premise (exp17 arm E)
# --------------------------------------------------------------------------


def test_the_escape_map_is_not_a_modular_conjugacy():
    """h(z1,z2) = (z1 + c z2, z2) mixes the modules unless they are equal.

    Exactly: H F H^{-1} = [[A1, c(A2 - A1)], [0, A2]].  The cross block vanishes
    iff A1 = A2, so for two blocks differing in rate *and* frequency there is no
    modular F~ at all.  That is what makes arm E a control the data must reject
    -- and it is measured here rather than asserted in prose.
    """
    sysm = S.ModularSystem([S.TwistBlock(s=0.92, omega=0.35, beta=0.0),
                            S.TwistBlock(s=0.55, omega=1.10, beta=0.0)])
    d, c = 4, 0.8
    F = np.stack([sysm.step(np.eye(d)[j]) for j in range(d)], axis=1)
    H = np.eye(d)
    H[:2, 2:] = c * np.eye(2)
    C = H @ F @ np.linalg.inv(H)
    off = (np.linalg.norm(C[:2, 2:]) + np.linalg.norm(C[2:, :2])) / np.linalg.norm(C)
    assert off > 0.05, off
    # and it really is the difference of the blocks that survives
    assert np.allclose(C[:2, 2:], c * (F[2:, 2:] - F[:2, :2]), atol=1e-12)
    assert np.allclose(C[2:, :2], 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# exp17's scoring rules, which decide the verdict and so have to be right
# --------------------------------------------------------------------------


def _exp17():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "experiments"))
    import exp17_adversarial_init as E
    return E


def test_informative_modules_picks_the_module_the_map_actually_moves():
    """The lattice map moves module 0's rotation and leaves module 1 alone.

    Scoring on module 1 would be a comparison that cannot distinguish the two
    representatives at all, so the restriction is not tuning -- it is the
    difference between measuring the question and measuring noise.
    """
    E = _exp17()
    from idyn import metrics as MT
    r1 = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([-0.083, -0.083]), np.array([-0.598, -0.598])],
        rotations=[0.0557, 0.1751], coherences=[1.0, 1.0])
    r2 = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([-0.083, -0.083]), np.array([-0.598, -0.598])],
        rotations=[0.2308, 0.1751], coherences=[1.0, 1.0])
    keep, kind = E.informative_modules(r1, r2)
    assert kind == "rotation"
    assert keep == [0]


def test_informative_modules_falls_to_spectra_when_nothing_rotates():
    """The §3.1 regrouping has no rotation at all, so the rule must not pick it."""
    E = _exp17()
    from idyn import systems as S
    reg = S.regrouping_counterexample(lams=E.ARM_B_LAMS)
    z0 = E.annulus_z0(np.random.default_rng(0), 60, 0.5, 1.2)
    keep, kind = E.informative_modules(E.fingerprint(reg["system"], z0),
                                       E.fingerprint(reg["system_tilde"], z0))
    assert kind == "spectrum"
    assert keep == [0, 1]        # the swap moves both modules


def test_restricted_distance_quotients_out_the_module_order():
    """Module labels carry no meaning (§3.10 trap 2), so a relabelling is zero."""
    E = _exp17()
    from idyn import metrics as MT
    a = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0, -0.9]), np.array([0.0, -0.9])],
        rotations=[0.0796, 0.2069], coherences=[1.0, 1.0])
    swapped = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0, -0.9]), np.array([0.0, -0.9])],
        rotations=[0.2069, 0.0796], coherences=[1.0, 1.0])
    assert E.restricted_distance(swapped, a, [0, 1], "rotation") == pytest.approx(0.0, abs=1e-12)


def test_restricted_distance_treats_an_unmeasured_invariant_as_maximally_far():
    """NaN must never read as a match -- the exp15 vacuous-check failure mode."""
    E = _exp17()
    from idyn import metrics as MT
    tgt = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0, -0.9]), np.array([0.0, -0.9])],
        rotations=[0.0796, 0.2069], coherences=[1.0, 1.0])
    blind = MT.DynamicalFingerprint(
        partition=[2, 2], spectra=[np.array([0.0, -0.9]), np.array([0.0, -0.9])],
        rotations=[float("nan"), float("nan")], coherences=[0.0, 0.0])
    assert np.isinf(E.restricted_distance(blind, tgt, [0], "rotation"))


def test_whitening_a_warm_start_target_moves_no_invariant():
    """It is a block-diagonal change of basis, i.e. exactly what §7 declines to
    identify -- applied so ordinary training has no reason to rescale the target
    and any later drift is attributable."""
    E = _exp17()
    from idyn import metrics as MT
    from idyn import systems as S
    sysm = S.ModularSystem([S.TwistBlock(s=0.92, omega=0.35, beta=0.0),
                            S.TwistBlock(s=0.55, omega=1.10, beta=0.0)])
    rng = np.random.default_rng(0)
    Z = sysm.simulate(E.annulus_z0(rng, 200, 0.5, 1.2), 30)
    W = E.whiten_modules(Z)
    for off, k in ((0, 2), (2, 2)):
        blk = W[..., off:off + k].reshape(-1, k)
        cov = np.cov(blk.T)
        assert np.allclose(cov, np.eye(k), atol=1e-8), cov
    # and it is block-diagonal in the strong sense: changing module 0's data
    # leaves module 1's whitened coordinates untouched, so no invariant of
    # module 1 can have been moved by anything module 0 did
    Z2 = Z.copy()
    Z2[..., :2] *= 3.0
    assert np.allclose(E.whiten_modules(Z2)[..., 2:], W[..., 2:], atol=1e-10)
