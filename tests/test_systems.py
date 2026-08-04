"""Systems: analytic Jacobians, exact invariants, and the decoder contract."""

from __future__ import annotations

import numpy as np
import pytest

from idyn import systems as S

BLOCKS = [
    S.LinearBlock(0.8 * S.rotation(0.3)),
    S.TwistBlock(s=0.9, omega=0.4, beta=0.6),
    S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5, beta=0.2),
    S.ScalarBlock(s=0.9, gain=1.3),
]


def finite_difference_jacobian(blk, z, h=1e-6):
    J = np.zeros((blk.dim, blk.dim))
    for k in range(blk.dim):
        e = np.zeros(blk.dim)
        e[k] = h
        J[:, k] = (blk.step(z + e) - blk.step(z - e)) / (2 * h)
    return J


@pytest.mark.parametrize("blk", BLOCKS, ids=lambda b: type(b).__name__)
def test_analytic_jacobian_matches_finite_differences(blk):
    rng = np.random.default_rng(0)
    for _ in range(10):
        z = rng.standard_normal(blk.dim) * 0.5 + 1.0
        assert np.allclose(blk.jacobian(z), finite_difference_jacobian(blk, z), atol=1e-7)


def test_twist_jacobian_determinant_is_exactly_s_squared():
    """r' = s r, theta' = theta + w + b r^2 has Jacobian determinant s^2 everywhere."""
    blk = S.TwistBlock(s=0.87, omega=0.4, beta=1.3)
    rng = np.random.default_rng(1)
    for z in rng.standard_normal((20, 2)) * 2.0:
        assert np.linalg.det(blk.jacobian(z)) == pytest.approx(0.87**2, rel=1e-10)


def test_twist_is_invertible_by_construction():
    blk = S.TwistBlock(s=0.9, omega=0.4, beta=0.6)
    rng = np.random.default_rng(2)
    z = rng.standard_normal((50, 2)) * 1.5
    w = blk.step(z)
    # invert analytically: r = |w|/s, theta = angle(w) - omega - beta r^2
    r = np.linalg.norm(w, axis=1) / blk.s
    th = np.arctan2(w[:, 1], w[:, 0]) - blk.omega - blk.beta * r**2
    z_back = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    assert np.allclose(z_back, z, atol=1e-10)


def test_limit_cycle_attracts_to_rho():
    blk = S.LimitCycleBlock(a=0.3, rho=1.0, omega=0.5)
    z = np.array([[0.2, 0.0], [2.5, 0.0], [0.0, -1.7]])
    for _ in range(400):
        z = blk.step(z)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-6)


def test_modular_system_jacobian_is_block_diagonal():
    sys = S.two_oscillator_system()
    J = sys.jacobian(np.array([0.5, -0.3, 0.9, 0.2]))
    assert np.allclose(J[:2, 2:], 0.0)
    assert np.allclose(J[2:, :2], 0.0)


def test_modules_evolve_independently():
    """Changing module 2's state must not affect module 1's trajectory."""
    sys = S.two_oscillator_system()
    z_a = np.array([[0.5, -0.3, 0.9, 0.2]])
    z_b = np.array([[0.5, -0.3, -1.4, 0.7]])
    Za = sys.simulate(z_a, 15)
    Zb = sys.simulate(z_b, 15)
    assert np.allclose(Za[:, :, :2], Zb[:, :, :2])
    assert not np.allclose(Za[:, :, 2:], Zb[:, :, 2:])


def test_simulate_agrees_with_repeated_step():
    sys = S.two_oscillator_system()
    z0 = np.array([[0.4, 0.1, -0.6, 0.3]])
    Z = sys.simulate(z0, 7)
    z = z0
    for t in range(7):
        z = sys.step(z)
        assert np.allclose(Z[:, t + 1], z)


def test_coupled_system_reduces_to_base_at_eps_zero():
    rng = np.random.default_rng(3)
    base = S.two_oscillator_system()
    C = S.off_block_coupling(base.partition, rng)
    cs = S.CoupledSystem(base, C, eps=0.0)
    z = rng.standard_normal((5, 4))
    assert np.allclose(cs.step(z), base.step(z))


def test_off_block_coupling_is_strictly_off_block():
    rng = np.random.default_rng(4)
    C = S.off_block_coupling([2, 3], rng)
    assert np.allclose(C[:2, :2], 0.0)
    assert np.allclose(C[2:, 2:], 0.0)
    assert np.linalg.norm(C, 2) == pytest.approx(1.0)


def test_coupling_breaks_block_diagonality_proportionally_to_eps():
    rng = np.random.default_rng(5)
    base = S.two_oscillator_system()
    C = S.off_block_coupling(base.partition, rng)
    z = np.array([0.5, -0.3, 0.9, 0.2])
    for eps in (1e-3, 1e-2):
        J = S.CoupledSystem(base, C, eps=eps).jacobian(z)
        assert np.linalg.norm(J[:2, 2:]) == pytest.approx(eps * np.linalg.norm(C[:2, 2:]), rel=1e-9)


def test_decoder_rejects_rank_deficient_W():
    W = np.ones((6, 3))  # rank 1
    with pytest.raises(ValueError, match="full column rank"):
        S.LinearDecoder(W)


def test_decoder_rejects_wide_W():
    with pytest.raises(ValueError, match="tall"):
        S.LinearDecoder(np.eye(3, 5))


def test_decoder_requires_rng_when_noisy():
    dec = S.LinearDecoder.random(6, 3, np.random.default_rng(6), noise_std=0.1)
    with pytest.raises(ValueError, match="rng required"):
        dec(np.zeros((4, 3)))


def test_exact_lyapunov_spectra_are_documented_correctly():
    assert S.TwistBlock(s=0.8).lyapunov_spectrum_exact() == pytest.approx([np.log(0.8)] * 2)
    lc = S.LimitCycleBlock(a=0.3)
    assert lc.lyapunov_spectrum_exact() == pytest.approx([np.log(abs(1 - 0.6)), 0.0])


# --------------------------------------------------------------------------
# MLPDecoder: the Theorem B observation map (CLAUDE.md §3.5)
# --------------------------------------------------------------------------


def _invert_flow(dec, x):
    """Invert the coupling flow analytically -- the construction's whole point."""
    h = np.asarray(x, dtype=float)
    for layer in reversed(dec.layers):
        h = h @ layer.Q  # Q orthogonal, so Q^{-1} = Q^T and (h @ Q.T) @ Q = h
        j, k = h[:, : layer.split], h[:, layer.split:]
        s = layer.scale * np.tanh(layer._mlp(j, layer.s_w, layer.s_b))
        t = layer._mlp(j, layer.t_w, layer.t_b)
        h = np.hstack([j, (k - t) * np.exp(-s)])
    return h


def test_mlp_decoder_flow_is_exactly_invertible():
    """Not injective-by-estimate: the inverse is in closed form, so check it."""
    rng = np.random.default_rng(0)
    dec = S.MLPDecoder.random(10, 4, rng, strength=1.5)
    z = rng.normal(size=(300, 4)) * 2.0
    assert np.abs(_invert_flow(dec, dec.flow(z)) - z).max() < 1e-10


def test_mlp_decoder_is_injective_and_an_immersion():
    """Both halves of "injective immersion", checked rather than assumed."""
    from idyn import metrics as MT

    rng = np.random.default_rng(1)
    dec = S.MLPDecoder.random(9, 4, rng, strength=1.0)
    z = rng.normal(size=(300, 4)) * 2.0
    x = dec(z)
    assert x.shape == (300, 9)

    # injective: no two distinct latents collide in observation space
    dz = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    dx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    off = ~np.eye(len(z), dtype=bool)
    assert dx[off].min() > 1e-8
    assert (dx[off] / dz[off]).min() > 1e-3, "and not collapsing toward a collision"

    # immersion: the Jacobian has full column rank everywhere we look.  Each
    # coupling Jacobian is triangular with diagonal e^s >= e^-strength, and Q, W
    # are isometries, so smin is bounded below by e^{-n_layers * strength}.
    J = MT.jacobian_of(dec, z[:50])
    smin = np.linalg.svd(J, compute_uv=False)[:, -1]
    assert smin.min() > np.exp(-len(dec.layers) * 1.0) * 1e-2


def test_mlp_decoder_at_zero_strength_is_linear():
    """The control arm of any nonlinearity ablation must be a linear map."""
    rng = np.random.default_rng(2)
    dec = S.MLPDecoder.random(8, 4, rng, strength=0.0)
    z = np.random.default_rng(3).normal(size=(50, 4))
    A, *_ = np.linalg.lstsq(z, dec(z), rcond=None)
    assert np.abs(dec(z) - z @ A).max() < 1e-10


def test_mlp_decoder_is_strongly_nonlinear():
    """The reason this class exists: it must not be a linear map in disguise.

    The contractive construction this replaced managed only 3% here even at its
    limit, which is not a Theorem B test (see the class docstring).
    """
    rng = np.random.default_rng(4)
    dec = S.MLPDecoder.random(8, 4, rng, strength=1.0)
    z = rng.normal(size=(600, 4)) * 1.5
    x = dec(z)
    aug = np.hstack([z, np.ones((len(z), 1))])
    A, *_ = np.linalg.lstsq(aug, x, rcond=None)
    resid = np.linalg.norm(x - aug @ A) / np.linalg.norm(x)
    assert resid > 0.20, f"only {resid:.3f} of the map is nonlinear"


def test_mlp_decoder_nonlinearity_grows_with_strength():
    rng = np.random.default_rng(5)
    prev = -1.0
    for strength in (0.0, 0.25, 0.5, 1.0, 1.5):
        r = np.random.default_rng(6)
        dec = S.MLPDecoder.random(8, 4, r, strength=strength)
        z = r.normal(size=(500, 4)) * 1.5
        x = dec(z)
        aug = np.hstack([z, np.ones((len(z), 1))])
        A, *_ = np.linalg.lstsq(aug, x, rcond=None)
        resid = np.linalg.norm(x - aug @ A) / np.linalg.norm(x)
        assert resid > prev, f"nonlinearity must increase with strength (at {strength})"
        prev = resid


def test_mlp_decoder_handles_trajectory_shaped_input():
    """(n_traj, T+1, d) must give the same answer as flattening to (n, d).

    Regression: the coupling split indexed ``z[:, :k]``, which on a 3-D batch
    slices the *time* axis instead of the feature axis.  It still typechecks and
    still returns an array, so nothing raises until d and T happen to differ.
    """
    rng = np.random.default_rng(8)
    dec = S.MLPDecoder.random(11, 4, rng, strength=1.0)
    Z = rng.normal(size=(7, 13, 4))
    assert dec(Z).shape == (7, 13, 11)
    assert np.abs(dec(Z).reshape(-1, 11) - dec(Z.reshape(-1, 4))).max() < 1e-12


def test_mlp_decoder_rejects_bad_parameters():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="strength must be"):
        S.MLPDecoder.random(8, 4, rng, strength=-0.1)
    with pytest.raises(ValueError, match="coupling needs"):
        S.MLPDecoder.random(8, 1, rng)
