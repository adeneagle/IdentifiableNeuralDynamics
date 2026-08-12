"""Modular dynamical systems F = f_1 (+) ... (+) f_K and their perturbations.

Conventions (CLAUDE.md §8): a partition is a ``list[int]`` of block dimensions
summing to ``d``; everything numerical is float64 and takes an explicit rng.

Block maps implement three things:

* ``dim``            -- the block dimension d_i
* ``step(z)``        -- z of shape (..., d_i) -> (..., d_i), vectorised
* ``jacobian(z)``    -- z of shape (d_i,) -> (d_i, d_i)

Analytic Jacobians are used everywhere (tests/test_systems.py validates them
against finite differences) because the Lyapunov and cocycle code iterates
them thousands of times and finite differences lose too much accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "BlockMap",
    "LinearBlock",
    "TwistBlock",
    "LimitCycleBlock",
    "ScalarBlock",
    "ResonantNodeBlock",
    "ModularSystem",
    "CoupledSystem",
    "LinearDecoder",
    "rotation",
    "regrouping_counterexample",
    "nonlinear_regrouping_counterexample",
    "triangular_conjugacy_counterexample",
    "multiindex_resonance_counterexample",
    "repeated_exponent_resonance_counterexample",
    "nonadditive_behavioural_escape",
    "lemma_d_witness",
    "gapless_resonant_coupling",
    "sylvester_kernel_dim",
    "two_oscillator_system",
    "tier2_witness",
    "sample_initial_conditions",
]

_EPS = 1e-12


def rotation(theta: float) -> np.ndarray:
    """2x2 rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


# --------------------------------------------------------------------------
# Block maps
# --------------------------------------------------------------------------


class BlockMap:
    """Base class for a single module's transition f_i: R^{d_i} -> R^{d_i}."""

    dim: int

    def step(self, z: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def jacobian(self, z: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"


@dataclass
class LinearBlock(BlockMap):
    """z -> A z."""

    A: np.ndarray

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got {self.A.shape}")

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self.A.shape[0]

    def step(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) @ self.A.T

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        return self.A.copy()


@dataclass
class TwistBlock(BlockMap):
    """Nonlinear rotation-contraction on R^2: r' = s r, theta' = theta + w + b r^2.

    In Cartesian form ``z -> s R(w + b|z|^2) z``.  Genuinely nonlinear whenever
    ``b != 0``, a diffeomorphism of R^2 for any ``s > 0``, and its Lyapunov
    spectrum is exactly ``{log s, log s}`` -- the twist is unipotent in the
    orthonormal polar frame, so it contributes nothing exponentially.

    That exactness is why this is the default nonlinear block: the separation
    hypothesis of CLAUDE.md §3.4 can be checked against a known answer.
    """

    s: float = 0.9
    omega: float = 0.4
    beta: float = 0.6
    dim: int = field(default=2, init=False)

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        x, y = z[..., 0], z[..., 1]
        phi = self.omega + self.beta * (x * x + y * y)
        c, sn = np.cos(phi), np.sin(phi)
        return self.s * np.stack([x * c - y * sn, x * sn + y * c], axis=-1)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(2)
        x, y = z
        phi = self.omega + self.beta * (x * x + y * y)
        c, sn = np.cos(phi), np.sin(phi)
        xp = self.s * (x * c - y * sn)
        yp = self.s * (x * sn + y * c)
        return self.s * np.array([[c, -sn], [sn, c]]) + 2.0 * self.beta * np.outer(
            np.array([-yp, xp]), np.array([x, y])
        )

    def lyapunov_spectrum_exact(self) -> np.ndarray:
        return np.array([np.log(self.s), np.log(self.s)])


@dataclass
class LimitCycleBlock(BlockMap):
    """Discrete Stuart-Landau-like oscillator on R^2 with an attracting cycle.

    ``r' = r + a r (1 - r^2/rho^2)``, ``theta' = theta + omega + beta (r - rho)``.

    On the cycle r = rho the radial multiplier is ``1 - 2a`` and the tangential
    one is 1, so the Lyapunov spectrum is ``{log|1 - 2a|, 0}``.  The zero
    exponent is why two limit-cycle modules never satisfy the §3.4 separation
    hypothesis -- pair one of these with a TwistBlock instead.

    **The basin is bounded**, unlike the contracting blocks here: the radial map
    is ``g(r) = r(1 + a - a r^2/rho^2)``, so ``g(r) > 0`` only for
    ``r < rho sqrt((1+a)/a)`` and an initial radius past that escapes to
    infinity immediately (at the defaults, the boundary is ``2.0817`` and
    ``r0 = 3`` gives ``r1 = -4.2``).  Pick initial conditions inside it.  Within
    the basin the Lyapunov exponents are **uniform** -- every orbit converges to
    the cycle and realises the Floquet exponents above -- which is the hypothesis
    Lemma C' of identifiability.md §4.4 runs on, and is what lets that lemma
    conclude on the whole basin rather than only on the attractor.
    """

    a: float = 0.3
    rho: float = 1.0
    omega: float = 0.5
    beta: float = 0.0
    dim: int = field(default=2, init=False)

    def _g(self, r: np.ndarray) -> np.ndarray:
        return r + self.a * r * (1.0 - r * r / (self.rho**2))

    def _dg(self, r: np.ndarray) -> np.ndarray:
        return 1.0 + self.a - 3.0 * self.a * r * r / (self.rho**2)

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        x, y = z[..., 0], z[..., 1]
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        rp = self._g(r)
        thp = th + self.omega + self.beta * (r - self.rho)
        return np.stack([rp * np.cos(thp), rp * np.sin(thp)], axis=-1)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(2)
        x, y = z
        r = float(np.hypot(x, y))
        if r < _EPS:
            raise ValueError("LimitCycleBlock Jacobian is singular at the origin")
        th = float(np.arctan2(y, x))
        g, dg = float(self._g(r)), float(self._dg(r))
        thp = th + self.omega + self.beta * (r - self.rho)
        c, sn = np.cos(thp), np.sin(thp)
        # d(x', y') / d(r, theta)
        A = np.array(
            [[dg * c - g * sn * self.beta, -g * sn], [dg * sn + g * c * self.beta, g * c]]
        )
        # d(r, theta) / d(x, y)
        ct, st = np.cos(th), np.sin(th)
        B = np.array([[ct, st], [-st / r, ct / r]])
        return A @ B

    def lyapunov_spectrum_exact(self) -> np.ndarray:
        return np.array([np.log(abs(1.0 - 2.0 * self.a)), 0.0])


@dataclass
class ScalarBlock(BlockMap):
    """1-D block ``z -> s * tanh(gain * z) / gain``.

    Used to build the nonlinear version of the §3.1 regrouping counterexample:
    four *independent* 1-D nonlinear maps regrouped into two 2-D modules fail
    exactly the same way the linear ones do.  Derivative at 0 is ``s``.
    """

    s: float = 0.9
    gain: float = 1.0
    dim: int = field(default=1, init=False)

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return self.s * np.tanh(self.gain * z) / self.gain

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(1)
        return np.array([[self.s / np.cosh(self.gain * z[0]) ** 2]])

    def lyapunov_spectrum_exact(self) -> np.ndarray:
        # The origin is the global attractor, so the exponent is log|f'(0)|.
        return np.array([np.log(abs(self.s))])


@dataclass
class ResonantNodeBlock(BlockMap):
    """``(z_a, z_b) -> (mu z_a,  mu^2 z_b + c z_a^2)`` -- a live within-module resonance.

    The witness that Route A's **Tier 2 is non-empty** (`theory/approaches.md`
    §A.2).  Its linear part is ``diag(mu, mu^2)``, and the monomial ``z_a^2`` in
    the ``z_b`` slot has homological eigenvalue ``lam_b - lam_a^2 = mu^2 - mu^2 = 0``.
    Resonant, so ``c`` cannot be conjugated away: with ``c != 0`` this map is **not**
    linearisable, and ``c`` is a normal-form invariant.  (Diagonal rescaling sends
    ``c -> c beta/alpha^2``, so ``c`` normalises to 1; what is invariant is
    ``c != 0``, i.e. linearisable or not.)

    Three properties make it the right witness:

    * **Tier 1 would kill it.** Full-spectrum non-resonance excludes ``mu^2 = mu*mu``,
      so under Tier 1 this map linearises and there is nothing nonlinear to
      identify.  Tier 2 keeps exactly this.
    * **The resonance is within-module**, so it does not trip
      ``spectra.is_cross_module_nonresonant`` -- Tier 2's hypotheses stay
      satisfiable alongside it (``mu = 0.7`` pairs fine with ``nu = 0.5``; it does
      *not* pair with ``nu = mu^2`` or ``mu^3``).
    * **Its linear part is decomposable but the map is not.** ``diag(mu, mu^2)``
      has two distinct real eigenvalues, so a linearised (B2) test splits it into
      two 1-D summands -- yet no invariant curve tangent to ``e_a`` exists when
      ``c != 0`` (``phi(mu z_a) = mu^2 phi(z_a) + c z_a^2`` has no solution), so
      the map admits no direct-product splitting.  ``selection.certify_fitted_model``
      linearises, so it reports this module DECOMPOSABLE when it is not.  That
      false negative sits precisely in Tier 2.

    Note the spectrum ``{log mu, 2 log mu}`` is **spread**, so the sigma_min trap
    of CLAUDE.md §3.9 applies to any cocycle measurement on this block.
    """

    mu: float = 0.70
    c: float = 0.90
    dim: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.mu < 1.0:
            raise ValueError(f"need 0 < mu < 1 for a contraction, got {self.mu}")

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        za, zb = z[..., 0], z[..., 1]
        return np.stack([self.mu * za, self.mu**2 * zb + self.c * za * za], axis=-1)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        za = float(np.asarray(z, dtype=float).reshape(2)[0])
        return np.array([[self.mu, 0.0], [2.0 * self.c * za, self.mu**2]])

    def linear_part(self) -> np.ndarray:
        """``Df(0) = diag(mu, mu^2)`` -- the part a linearised test can see."""
        return np.diag([self.mu, self.mu**2])

    def iterate_exact(self, z: np.ndarray, n: int) -> np.ndarray:
        """Closed form ``f^n``, which exposes the resonance as a *secular* term.

        ``b_n = mu^{2n} z_b + n c mu^{2(n-1)} z_a^2`` -- the factor of ``n`` is
        the signature.  Dividing out ``mu^{2n}`` leaves ``z_b + (n c/mu^2) z_a^2``,
        which grows linearly in n, whereas the linear map ``diag(mu, mu^2)`` gives
        a constant.  Unbounded normalised iterate <=> ``c != 0``.
        """
        z = np.asarray(z, dtype=float)
        za, zb = z[..., 0], z[..., 1]
        an = self.mu**n * za
        bn = self.mu ** (2 * n) * zb + n * self.c * self.mu ** (2 * (n - 1)) * za * za
        return np.stack([an, bn], axis=-1)

    def lyapunov_spectrum_exact(self) -> np.ndarray:
        return np.array([np.log(self.mu), 2.0 * np.log(self.mu)])


# --------------------------------------------------------------------------
# Modular system
# --------------------------------------------------------------------------


@dataclass
class ModularSystem:
    """F = f_1 (+) ... (+) f_K acting on R^d, d = sum of block dims."""

    blocks: Sequence[BlockMap]

    def __post_init__(self) -> None:
        self.blocks = list(self.blocks)
        if not self.blocks:
            raise ValueError("need at least one block")

    @property
    def partition(self) -> list[int]:
        return [b.dim for b in self.blocks]

    @property
    def dim(self) -> int:
        return int(sum(self.partition))

    @property
    def K(self) -> int:
        return len(self.blocks)

    @property
    def slices(self) -> list[slice]:
        out, off = [], 0
        for d in self.partition:
            out.append(slice(off, off + d))
            off += d
        return out

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.concatenate(
            [b.step(z[..., sl]) for b, sl in zip(self.blocks, self.slices)], axis=-1
        )

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(self.dim)
        J = np.zeros((self.dim, self.dim))
        for b, sl in zip(self.blocks, self.slices):
            J[sl, sl] = b.jacobian(z[sl])
        return J

    def simulate(self, z0: np.ndarray, T: int) -> np.ndarray:
        """Roll out T steps.  z0 is (d,) or (n_traj, d); returns (n_traj, T+1, d)."""
        z = np.atleast_2d(np.asarray(z0, dtype=float))
        out = np.empty((z.shape[0], T + 1, self.dim))
        out[:, 0] = z
        for t in range(T):
            z = self.step(z)
            out[:, t + 1] = z
        return out

    def matrix(self) -> np.ndarray:
        """The block-diagonal matrix, for all-linear systems only."""
        if not all(isinstance(b, LinearBlock) for b in self.blocks):
            raise TypeError("matrix() requires every block to be a LinearBlock")
        F = np.zeros((self.dim, self.dim))
        for b, sl in zip(self.blocks, self.slices):
            F[sl, sl] = b.A  # type: ignore[attr-defined]
        return F


@dataclass
class CoupledSystem:
    """``F_eps(z) = F(z) + eps * C z`` -- the §4-step-7 perturbation.

    Breaks modularity by exactly ``eps``, which is what lets us ask how the
    recovered partition degrades as a function of eps and the spectral gap.
    """

    base: ModularSystem
    C: np.ndarray
    eps: float

    def __post_init__(self) -> None:
        self.C = np.asarray(self.C, dtype=float)
        if self.C.shape != (self.base.dim, self.base.dim):
            raise ValueError(f"C must be {(self.base.dim,) * 2}, got {self.C.shape}")

    @property
    def dim(self) -> int:
        return self.base.dim

    @property
    def partition(self) -> list[int]:
        return self.base.partition

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return self.base.step(z) + self.eps * (z @ self.C.T)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        return self.base.jacobian(z) + self.eps * self.C

    def simulate(self, z0: np.ndarray, T: int) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z0, dtype=float))
        out = np.empty((z.shape[0], T + 1, self.dim))
        out[:, 0] = z
        for t in range(T):
            z = self.step(z)
            out[:, t + 1] = z
        return out


def off_block_coupling(partition: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Random matrix supported strictly off the diagonal blocks, unit spectral norm."""
    d = int(sum(partition))
    C = rng.standard_normal((d, d))
    off = 0
    for dk in partition:
        C[off : off + dk, off : off + dk] = 0.0
        off += dk
    nrm = np.linalg.norm(C, 2)
    return C / nrm if nrm > _EPS else C


# --------------------------------------------------------------------------
# Observation model
# --------------------------------------------------------------------------


@dataclass
class LinearDecoder:
    """x = W z + noise, with W of full column rank (CLAUDE.md §3.5)."""

    W: np.ndarray
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=float)
        if self.W.shape[0] < self.W.shape[1]:
            raise ValueError("W must be tall (n_obs >= d) to have full column rank")
        if np.linalg.matrix_rank(self.W) < self.W.shape[1]:
            raise ValueError("W does not have full column rank")

    @property
    def d(self) -> int:
        return self.W.shape[1]

    def __call__(self, z: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        x = np.asarray(z, dtype=float) @ self.W.T
        if self.noise_std > 0:
            if rng is None:
                raise ValueError("rng required when noise_std > 0")
            x = x + self.noise_std * rng.standard_normal(x.shape)
        return x

    @staticmethod
    def random(n_obs: int, d: int, rng: np.random.Generator, noise_std: float = 0.0) -> "LinearDecoder":
        W = rng.standard_normal((n_obs, d))
        W, _ = np.linalg.qr(W)  # orthonormal columns: full column rank, well conditioned
        return LinearDecoder(W, noise_std=noise_std)


@dataclass
class _CouplingLayer:
    """One affine coupling block plus a rotation: (z_J, z_K) -> Q (z_J, z_K e^s + t)."""

    s_w: list[np.ndarray]
    s_b: list[np.ndarray]
    t_w: list[np.ndarray]
    t_b: list[np.ndarray]
    Q: np.ndarray
    split: int
    scale: float

    @staticmethod
    def _mlp(z, ws, bs):
        h = z
        for k, (w, b) in enumerate(zip(ws, bs)):
            h = h @ w.T + b
            if k < len(ws) - 1:
                h = np.tanh(h)
        return h

    def __call__(self, z: np.ndarray) -> np.ndarray:
        # index on the *last* axis: callers pass (n, d) for Jacobians but
        # (n_traj, T+1, d) for whole datasets, and slicing [:, :k] would cut the
        # time axis on the latter -- silently, since it still typechecks
        j, k = z[..., : self.split], z[..., self.split:]
        # s is squashed so e^s stays in [e^-scale, e^scale]: keeps the Jacobian
        # bounded away from singular in *both* directions, so the map is a
        # bi-Lipschitz bijection rather than merely an injection.
        s = self.scale * np.tanh(self._mlp(j, self.s_w, self.s_b))
        t = self._mlp(j, self.t_w, self.t_b)
        return np.concatenate([j, k * np.exp(s) + t], axis=-1) @ self.Q.T


@dataclass
class MLPDecoder:
    """x = W . (coupling flow)(z) -- the Theorem B observation map.

    CLAUDE.md §3.5 splits the theory in two on exactly this object: a *linear*
    full-column-rank decoder forces h in GL(d) before any dynamics are used
    (Theorem A), so the nonlinear reparameterisation ambiguity the project is
    about only exists once g is nonlinear (Theorem B).  `LinearDecoder` was the
    only data-generating decoder in the repo, which meant no experiment tested
    the Theorem B observation model end to end.  This is that decoder.

    **Invertible by construction, and strongly nonlinear.**  Each layer sends
    ``(z_J, z_K) -> (z_J, z_K e^{s(z_J)} + t(z_J))``, whose inverse is
    ``z_K = (x_K - t(x_J)) e^{-s(x_J)}`` -- exact for *arbitrary* s and t, with
    no smallness condition.  A random orthogonal ``Q`` after each layer stops any
    coordinate being permanently pass-through, so the composition preserves no
    block structure that could bias a block-structure measurement.

    *Why not the obvious construction.*  The first version here was a
    contractive perturbation ``z + eps m(z)`` with ``eps Lip(m) < 1`` -- injective
    by Banach, and clean.  Measured, it caps out at a **3% nonlinear residual**
    even at ``strength = 0.99``, and gets *worse* with depth (0.5% at two hidden
    layers) because ``Lip <= prod ||W_k||_2`` grows and shrinks eps.  That bound
    is a worst case over all directions and all points, while tanh is far gentler
    in fact, so the whole budget buys a bound that is not tight.  A decoder within
    3% of linear does not test Theorem B; coupling has no such ceiling.

    **Analytic**, since tanh and exp are: §3.7's live route needs
    h = g~^{-1} . g real-analytic, and that is *free* only if the decoders are.
    Do not swap the activation for a ReLU without reading §3.7 first.

    ``strength=0`` gives s = t = 0 identically, recovering ``LinearDecoder`` up
    to the fixed rotations -- the control arm of a nonlinearity ablation.
    """

    W: np.ndarray
    layers: list[_CouplingLayer]
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=float)
        if self.W.shape[0] < self.W.shape[1]:
            raise ValueError("W must be tall (n_obs >= d) to have full column rank")
        if np.linalg.matrix_rank(self.W) < self.W.shape[1]:
            raise ValueError("W does not have full column rank")

    @property
    def d(self) -> int:
        return self.W.shape[1]

    def flow(self, z: np.ndarray) -> np.ndarray:
        """The invertible R^d -> R^d part, before the lift to R^n_obs."""
        h = np.asarray(z, dtype=float)
        for layer in self.layers:
            h = layer(h)
        return h

    def __call__(self, z: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        x = self.flow(z) @ self.W.T
        if self.noise_std > 0:
            if rng is None:
                raise ValueError("rng required when noise_std > 0")
            x = x + self.noise_std * rng.standard_normal(x.shape)
        return x

    @staticmethod
    def random(
        n_obs: int,
        d: int,
        rng: np.random.Generator,
        strength: float = 1.0,
        hidden: Sequence[int] = (32,),
        n_layers: int = 3,
        noise_std: float = 0.0,
    ) -> "MLPDecoder":
        """``strength`` bounds |log| of the coupling scale; 0 gives a linear map."""
        if d < 2:
            raise ValueError("coupling needs d >= 2")
        if strength < 0.0:
            raise ValueError(f"strength must be >= 0, got {strength}")
        W = rng.standard_normal((n_obs, d))
        W, _ = np.linalg.qr(W)
        split = d // 2
        layers = []
        for _ in range(n_layers):
            dims = [split, *hidden, d - split]
            nets = []
            for _net in range(2):  # s and t
                ws = [rng.standard_normal((b, a)) / np.sqrt(a)
                      for a, b in zip(dims[:-1], dims[1:])]
                bs = [rng.standard_normal(b) * 0.1 for b in dims[1:]]
                nets.append((ws, bs))
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            layers.append(_CouplingLayer(
                s_w=nets[0][0], s_b=nets[0][1], t_w=nets[1][0], t_b=nets[1][1],
                Q=Q, split=split, scale=float(strength),
            ))
        if strength == 0.0:  # exact linear control: kill t as well as s
            for layer in layers:
                layer.t_w = [np.zeros_like(w) for w in layer.t_w]
                layer.t_b = [np.zeros_like(b) for b in layer.t_b]
        return MLPDecoder(W, layers, noise_std=noise_std)


# --------------------------------------------------------------------------
# Named constructions
# --------------------------------------------------------------------------


def regrouping_counterexample(
    lams: Sequence[float] = (0.90, 0.75, 0.60, 0.45),
    n_obs: int = 8,
    seed: int = 0,
) -> dict:
    """The CLAUDE.md §3.1 counterexample, built explicitly.

    F = diag(l1,l2,l3,l4) split as (l1,l2) (+) (l3,l4).  P swaps coordinates 2
    and 3.  Then P F P^{-1} = diag(l1,l3) (+) diag(l2,l4) is *also* modular with
    K = 2 and block dims (2,2), and W~ = W P^{-1} reproduces the observations
    exactly -- yet P is not h_1 (+) h_2 up to module permutation.

    The defect is that diag(l1,l2) is decomposable, so the partition into two
    2-D modules is not the finest one.  Returns everything a test or experiment
    needs to assert that.
    """
    lams = tuple(float(x) for x in lams)
    if len(lams) != 4:
        raise ValueError("this construction is the d=4 one; pass exactly 4 eigenvalues")
    if len(set(lams)) != 4:
        raise ValueError("eigenvalues must be distinct")
    rng = np.random.default_rng(seed)

    F = np.diag(lams)
    P = np.eye(4)[[0, 2, 1, 3]]  # swaps coordinates 2 and 3 (0-indexed 1 and 2)
    F_tilde = P @ F @ P.T

    sys = ModularSystem([LinearBlock(np.diag(lams[:2])), LinearBlock(np.diag(lams[2:]))])
    sys_tilde = ModularSystem(
        [LinearBlock(np.diag([lams[0], lams[2]])), LinearBlock(np.diag([lams[1], lams[3]]))]
    )
    dec = LinearDecoder.random(n_obs, 4, rng)
    dec_tilde = LinearDecoder(dec.W @ P.T)  # W~ = W P^{-1}, P orthogonal so P^{-1} = P^T

    return {
        "lams": lams,
        "F": F,
        "F_tilde": F_tilde,
        "P": P,
        "system": sys,
        "system_tilde": sys_tilde,
        "decoder": dec,
        "decoder_tilde": dec_tilde,
        "partition": [2, 2],
    }


def nonlinear_regrouping_counterexample(
    scales: Sequence[float] = (0.90, 0.75, 0.60, 0.45),
    gains: Sequence[float] = (1.0, 1.3, 0.8, 1.6),
    n_obs: int = 8,
    seed: int = 0,
) -> dict:
    """§3.1 again, with four independent *nonlinear* 1-D maps.

    Same failure: the coordinate swap regroups them into a different pair of
    2-D modules with identical observations.  Included because CLAUDE.md is
    explicit that this is not a linear artifact.
    """
    if len(scales) != 4 or len(gains) != 4:
        raise ValueError("need exactly 4 scales and 4 gains")
    rng = np.random.default_rng(seed)
    b = [ScalarBlock(s=float(s), gain=float(g)) for s, g in zip(scales, gains)]
    perm = [0, 2, 1, 3]
    P = np.eye(4)[perm]

    # Modules are formed by concatenating 1-D blocks; a "module" here is the
    # product system on the pair, which we realise as a ModularSystem of 1-D
    # blocks (the pairing only matters through the partition we impose later).
    sys = ModularSystem(b)
    sys_tilde = ModularSystem([b[i] for i in perm])
    dec = LinearDecoder.random(n_obs, 4, rng)
    dec_tilde = LinearDecoder(dec.W @ P.T)

    return {
        "blocks": b,
        "P": P,
        "perm": perm,
        "system": sys,
        "system_tilde": sys_tilde,
        "decoder": dec,
        "decoder_tilde": dec_tilde,
        "grouped_partition": [2, 2],
        "finest_partition": [1, 1, 1, 1],
    }


def triangular_conjugacy_counterexample(
    mu1: float = 0.30, mu2: float = 0.50, c: float = 0.7, resonant_m: int | None = None
) -> dict:
    """Theorem B's target conclusion is FALSE under (B1)-(B4) as stated.

    Take f_i(z_i) = mu_i z_i with 0 < mu1 < mu2 < 1, and

        h(z1, z2) = (z1 + c sgn(z2)|z2|^p,  z2),   p = log(mu1)/log(mu2).

    Then h . F = F . h exactly, because the cross term picks up mu2^p = mu1,
    matching the mu1 that F applies to the first coordinate.  And:

    * (B1) holds -- dh1/dz2 = c p |z2|^(p-1) with p > 1, so h is C^1 with
      bounded derivative on compacts; Dh is unit lower-triangular, det = 1;
    * (B2) holds -- 1-D blocks are trivially indecomposable;
    * (B3) holds -- f~_i = f_i, sigma = identity;
    * (B4) holds **as written in identifiability.md**: the Lyapunov spectra
      {log mu1} and {log mu2} are disjoint.

    Yet h is triangular and *not* block-diagonal.  So the triangular conclusion
    of identifiability.md §4.2 is sharp: block-diagonality is not merely
    unreachable by the cocycle bound (CLAUDE.md §3.7), it is **false**.

    Note this does not contradict Lemma C.  Lemma C needs the *oriented* gap
    lambda_max(f2) < lambda_min(f~1), which fails here (log mu2 > log mu1); the
    orientation that does hold kills M21, and indeed M21 = dh2/dz1 = 0.  (B4) as
    stated asks only for disjoint spectra, which is strictly weaker.

    With ``resonant_m`` set, mu1 = mu2**m makes p = m an integer and h a
    polynomial, hence C^infinity -- so raising the regularity of h does not by
    itself rescue the theorem.  Cross-module non-resonance is necessary.
    """
    if resonant_m is not None:
        if resonant_m < 2:
            raise ValueError("resonant_m must be at least 2")
        mu1 = mu2**resonant_m
        p = float(resonant_m)
    else:
        if not 0.0 < mu1 < mu2 < 1.0:
            raise ValueError("need 0 < mu1 < mu2 < 1")
        p = float(np.log(mu1) / np.log(mu2))

    system = ModularSystem([LinearBlock([[mu1]]), LinearBlock([[mu2]])])

    def h(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z1, z2 = z[..., 0], z[..., 1]
        return np.stack([z1 + c * np.sign(z2) * np.abs(z2) ** p, z2], axis=-1)

    def h_inv(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        w1, w2 = w[..., 0], w[..., 1]
        return np.stack([w1 - c * np.sign(w2) * np.abs(w2) ** p, w2], axis=-1)

    def cross_derivative(z2: np.ndarray) -> np.ndarray:
        """M12 = dh1/dz2, the block that block-diagonality would need to vanish."""
        return c * p * np.abs(np.asarray(z2, dtype=float)) ** (p - 1.0)

    return {
        "mu1": float(mu1),
        "mu2": float(mu2),
        "c": float(c),
        "p": p,
        "resonant": resonant_m is not None,
        "smooth": resonant_m is not None,  # polynomial h in the resonant case
        "system": system,
        "h": h,
        "h_inv": h_inv,
        "cross_derivative": cross_derivative,
        "lyapunov": (float(np.log(mu1)), float(np.log(mu2))),
    }


def multiindex_resonance_counterexample(
    mu2: float = 0.50, mu3: float = 0.30, c: float = 0.8
) -> dict:
    """Pairwise non-resonance is NOT sufficient for Route A.

    Three 1-D modules with ``mu1 = mu2 * mu3``.  Then

        h(z) = (z1 + c z2 z3,  z2,  z3)

    is an exact conjugacy -- the cross term picks up ``mu2 mu3 = mu1`` -- and it
    is a *polynomial*, hence C^infinity.  Yet every **pairwise** log-ratio
    ``log mu_i / log mu_j`` sits far from an integer, so a pairwise-stated
    non-resonance hypothesis is satisfied while the conclusion fails.

    The offending relation is the multi-index one ``log mu1 = log mu2 + log mu3``.
    Non-resonance therefore has to be checked over multi-indices across *all*
    modules -- see ``spectra.cross_module_resonances``.
    """
    mu1 = mu2 * mu3
    system = ModularSystem(
        [LinearBlock([[mu1]]), LinearBlock([[mu2]]), LinearBlock([[mu3]])]
    )

    def h(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z1, z2, z3 = z[..., 0], z[..., 1], z[..., 2]
        return np.stack([z1 + c * z2 * z3, z2, z3], axis=-1)

    lyap = [np.array([np.log(mu1)]), np.array([np.log(mu2)]), np.array([np.log(mu3)])]
    return {
        "mu": (float(mu1), float(mu2), float(mu3)),
        "c": float(c),
        "system": system,
        "h": h,
        "lyapunov": lyap,
        "cross_derivative": lambda z2, z3: c * np.asarray(z3, dtype=float),
    }


def repeated_exponent_resonance_counterexample(
    rho: float = 0.80, theta: float = 0.4, c: float = 0.6
) -> dict:
    """Rotation angles give no protection: phases cancel in ``x^2 + y^2``.

    Module 1 is a 2-D scaled rotation (spectrum ``{log rho, log rho}`` -- a
    *repeated* exponent, exactly what every ``TwistBlock`` has); module 2 is 1-D
    at rate ``rho^2``.  Then

        h(x, y, z2) = (x, y,  z2 + c (x^2 + y^2))

    is an exact polynomial conjugacy **for every** ``theta``, because
    ``x^2 + y^2 = |w|^2`` is rotation invariant.

    The moral for our own test systems: non-resonance must be checked on the
    radial rates *with multiplicity*.  A module with a repeated exponent
    ``log rho`` resonates with any module at rate ``2 log rho``, and moduli like
    ``(0.95, 0.9025)`` look innocuous but satisfy ``0.9025 = 0.95**2``.
    """
    R = rho * rotation(theta)
    system = ModularSystem([LinearBlock(R), LinearBlock([[rho**2]])])

    def h(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        x, y, w = z[..., 0], z[..., 1], z[..., 2]
        return np.stack([x, y, w + c * (x * x + y * y)], axis=-1)

    lyap = [np.array([np.log(rho), np.log(rho)]), np.array([2.0 * np.log(rho)])]
    return {
        "rho": float(rho),
        "theta": float(theta),
        "c": float(c),
        "system": system,
        "h": h,
        "lyapunov": lyap,
    }


def nonadditive_behavioural_escape(
    gamma: float = 1.0, s_a: float = 0.90, alpha: float = 0.40, omega_b: float = 1.10
) -> dict:
    """Lemma D open item (a): (D4) alone cannot carry the non-additive case.

    Lemma D (``identifiability.md`` §4.5) proves ``psi = 0`` for **additive**
    ``h_B(z_A,z_B) = z_B + psi(z_A)``.  Open item (a) notes the graded reduction
    survives for non-additive ``h_B = sum_m z_A^m c_m(z_B)`` -- Steps 1-3 go
    through by evaluating at ``z_B = 0`` -- but that Step 4's characteristic-
    function factorisation needs the independence that ``c_m(z_B)`` destroys.

    This is the witness showing that the missing piece **cannot** be recovered by
    strengthening the behavioural hypothesis.  Take ``p(z_B | u) = N(0, I_2)`` and

        h(z_A, z_B) = ( z_A,  R(gamma * z_A_1) z_B ),   R = rotation.

    This is non-additive (``c_m`` depends on ``z_B``, linearly), and for **every**
    ``u`` and every fixed ``z_A`` the rotated ``z_B`` is again ``N(0, I)`` and
    independent of ``z_A``.  So the law of ``h_B`` is exactly ``u``-invariant:
    **(D4) holds, with (D2) and (D3) untouched, while ``M_BA != 0``.**  Step 4's
    entire behavioural input is therefore satisfied by a map it must exclude, so
    no sharpening of (D4) closes item (a) -- the work has to come from Steps 1-3.

    Consistency with Lemma D's *conclusion*: this ``h`` is **not** a conjugacy
    between modular systems.  With ``f_B`` a scaled rotation (which commutes with
    ``R``) the ``B``-component of ``h . F = F~ . h`` needs
    ``theta . f_A - theta`` constant, and at the fixed point of a contracting
    ``f_A`` that constant is ``0``, forcing ``theta`` -- hence ``gamma`` -- to
    vanish.  ``dynamics_defect`` measures this and is exactly ``0`` iff
    ``gamma = 0``.  So Step 1 is doing the work that Step 4 does in the additive
    case, which is precisely the shape any proof of item (a) must have.

    Orientation follows §4.5: block ``A`` is the ``u``-varying, spectrally
    dominant one (``rho = s_a``) and block ``B`` the invariant, dominated one
    (``rho = s_a^2 < s_a``), so **(D1) holds** and this is not an alignment
    artefact.  ``d_B = 2`` is necessary: at ``d_B = 1`` the ``p_B``-preserving
    transports are the two isolated points ``+-id``, so a family continuous in
    ``z_A`` is constant and no such escape exists.
    """
    s_b = s_a**2  # (D1): rho(f_B) = s_a^2 < s_a = rho_min(f_A)
    f_A = TwistBlock(s=s_a, omega=alpha, beta=0.0)
    f_B = TwistBlock(s=s_b, omega=omega_b, beta=0.0)  # commutes with R

    def _rot(b: np.ndarray, theta: np.ndarray) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        out = np.array(b, dtype=float, copy=True)
        out[..., 0] = c * b[..., 0] - s * b[..., 1]
        out[..., 1] = s * b[..., 0] + c * b[..., 1]
        return out

    def theta(za: np.ndarray) -> np.ndarray:
        return gamma * np.asarray(za, dtype=float)[..., 0]

    def h(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        za, zb = z[..., :2], z[..., 2:]
        return np.concatenate([za, _rot(zb, theta(za))], axis=-1)

    def h_inv(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        wa, wb = w[..., :2], w[..., 2:]
        return np.concatenate([wa, _rot(wb, -theta(wa))], axis=-1)

    def cross_derivative(z: np.ndarray) -> np.ndarray:
        """``|M_BA| = |d h_B / d z_A|`` per sample; a rotation derivative is an isometry."""
        z = np.asarray(z, dtype=float)
        return abs(gamma) * np.linalg.norm(z[..., 2:], axis=-1)

    def dynamics_defect(za: np.ndarray, w_b: Sequence[float] = (1.0, 0.0)) -> float:
        """Obstruction to ``h`` conjugating modular dynamics to modular dynamics.

        Fix one value ``w_b`` of the reparameterised invariant block.  Its
        pre-image depends on ``z_A``, so states sharing that ``wtilde_B`` now reach
        different successors; an autonomous ``ftilde_B`` would need them to agree.
        Exactly ``0`` iff ``gamma = 0``.  No binning, no estimation.
        """
        za = np.atleast_2d(np.asarray(za, dtype=float))[:, :2]
        w = np.broadcast_to(np.asarray(w_b, dtype=float), za.shape[:-1] + (2,))
        zb = _rot(w, -theta(za))
        succ = _rot(f_B.step(zb), theta(f_A.step(za)))
        return float(np.linalg.norm(succ - succ.mean(axis=0), axis=-1).mean())

    return {
        "gamma": float(gamma),
        "system": ModularSystem([f_A, f_B]),
        "f_A": f_A,
        "f_B": f_B,
        "rho_a": float(s_a),
        "rho_b": float(s_b),
        "one_sided_gap": float(s_a - s_b),  # (D1) margin, > 0
        "h": h,
        "h_inv": h_inv,
        "cross_derivative": cross_derivative,
        "dynamics_defect": dynamics_defect,
        "additive": False,
        "satisfies_D4": True,
    }


def two_oscillator_system(
    s: Sequence[float] = (0.95, 0.70),
    omega: Sequence[float] = (0.40, 1.10),
    beta: Sequence[float] = (0.60, -0.50),
) -> ModularSystem:
    """Two 2-D nonlinear oscillators with well-separated Lyapunov exponents.

    The §4-step-6 positive control.  Module i has spectrum {log s_i, log s_i}
    exactly, so ``s[0] != s[1]`` gives the §3.4 separation by construction and
    the spectral gap is ``|log s_0 - log s_1|``.
    """
    if not (len(s) == len(omega) == len(beta)):
        raise ValueError("s, omega, beta must have equal length")
    return ModularSystem(
        [TwistBlock(s=float(a), omega=float(w), beta=float(b)) for a, w, b in zip(s, omega, beta)]
    )


def lemma_d_witness(s: float = 0.90, alpha: float = 0.40, c: float = 0.70) -> dict:
    """Lemma D's witness: the coupling the spectral gap provably cannot kill.

    In complex coordinates ``f_A(z) = s e^{i a} z`` and ``psi(z) = z^2`` satisfy
    ``psi(f_A z) = s^2 e^{2ia} z^2 = f_B(psi(z))`` with ``f_B(w) = s^2 e^{2ia} w``.
    So ``h(z_A, z_B) = (z_A, z_B + c z_A^2)`` is an **exact** conjugacy of the
    modular ``F = f_A (+) f_B`` to itself, and is *not* block-diagonal.

    What makes it a witness rather than a curiosity: ``rho(f_B) = s^2 < s``, so
    the **one-sided gap holds**.  That is the orientation Lemma C can close, and
    it kills ``M_AB``; the surviving coupling is ``M_BA``, which CLAUDE.md 3.7
    proves the gap can *never* reach.  The dynamics are therefore exhausted here
    and ``h`` stays triangular -- this is exactly the gap Lemma D fills.

    Behaviour resolves it because ``psi`` is homogeneous of degree 2 (the gap
    forces degree >= 2, see identifiability.md 4.5 Step 3), so under variance
    modulation ``z_A ~ sigma_u * N(0,I)`` the law of ``h_B`` scales as
    ``sigma_u^2``: ``var(h_B) = 1 + 4 c^2 sigma^4`` per component.

    Returns the maps and the diagnostic quantities; see ``theory/
    identifiability.md`` 4.5.
    """
    lam_a = s * np.exp(1j * alpha)
    lam_b = lam_a ** 2  # the resonance, by construction

    def _to_c(z):
        z = np.asarray(z, dtype=float)
        return z[..., 0] + 1j * z[..., 1], z[..., 2] + 1j * z[..., 3]

    def _from_c(a, b):
        return np.stack([a.real, a.imag, b.real, b.imag], axis=-1)

    def F(z):
        a, b = _to_c(z)
        return _from_c(lam_a * a, lam_b * b)

    def h(z):
        a, b = _to_c(z)
        return _from_c(a, b + c * a ** 2)

    return {
        "F": F,
        "h": h,
        "lam_a": lam_a,
        "lam_b": lam_b,
        "s_a": s,
        "s_b": s ** 2,
        "c": c,
        "resonance_residual": float(abs(lam_a ** 2 - lam_b)),
        # the gap Lemma C needs, in the orientation it can close
        "gap_holds": bool(np.log(s ** 2) < np.log(s)),
        "psi_degree": 2,
        "var_h_b": lambda sigma: 1.0 + 4.0 * c ** 2 * sigma ** 4,
    }


def gapless_resonant_coupling(
    s: float = 0.85, omega: float = 0.70, c: float = 0.70, omega_b: float | None = None
) -> dict:
    """Lemma D' -- the same coupling with **no spectral gap at all**.

    ``lemma_d_witness`` above satisfies (D1), the one-sided gap, and is degree 2
    because identifiability.md 4.5 Step 3 says the gap forces that.  This one
    drops the gap entirely: both modules are ``s * Rot(omega)``, so their
    Lyapunov spectra are *identical* -- ``spectral_gap`` is exactly 0 and
    ``filtration_gap`` is not ordered.  Lemma C has nothing to work with,
    Theorem F does not apply, and (D1) fails ("rho(f~_B) < rho_min(f_A)" reads
    ``s < s``).

    The point: behaviour does not care.  The surviving coupling here is
    **degree 1** -- a linear ``psi = c I``, resonant because ``f_A`` and
    ``f~_B`` share their spectrum -- and Step 4's scaling iteration works at
    every degree ``>= 1``, not only at ``>= 2``.  So the conclusion survives
    under the far weaker

        (D1')  1 not in spec(f~_B),

    which is all that is needed to exclude the degree-0 (scale-invariant)
    escape that 4.5 already identifies as the unique one.

    Why it matters: two modules with identical spectra is the *linear* form of
    task 23's two-oscillator case -- exactly where the spectral route is dead.
    ``omega_b`` defaults to ``omega`` (resonant).  Set it to anything other than
    ``+/- omega`` and ``sylvester_kernel_dim`` drops to 0: no linear ``psi``
    exists at all, and behaviour is not even needed.
    """
    omega_b = omega if omega_b is None else float(omega_b)

    def _rot(w: float) -> np.ndarray:
        return s * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])

    R_a, R_b = _rot(omega), _rot(omega_b)

    # Step 1 of Lemma D forces f_B = f~_B, so F and F~ coincide here.
    def F(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.concatenate([z[..., :2] @ R_a.T, z[..., 2:] @ R_b.T], axis=-1)

    def h(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.concatenate([z[..., :2], z[..., 2:] + c * z[..., :2]], axis=-1)

    lam_a, lam_b = np.linalg.eigvals(R_a), np.linalg.eigvals(R_b)
    spec = np.array([np.log(s), np.log(s)])

    return {
        "F": F,
        "h": h,
        "R_a": R_a,
        "R_b": R_b,
        "c": c,
        "s": s,
        "omega": omega,
        "omega_b": omega_b,
        "spectra": [spec, spec.copy()],
        "psi_degree": 1,
        "resonance_residual": float(np.abs(lam_a[:, None] - lam_b[None, :]).min()),
        "sylvester_kernel_dim": sylvester_kernel_dim(R_a, R_b),
        # (D1) needs rho(f~_B) < rho_min(f_A); both are s, so it fails
        "gap_holds": False,
        # (D1') needs 1 not in spec(f~_B)
        "unit_eigenvalue_distance": float(np.abs(lam_b - 1.0).min()),
        "cross_derivative": float(abs(c) * np.sqrt(2.0)),
        "var_h_b": lambda sigma, tau=1.0: tau ** 2 + c ** 2 * sigma ** 2,
    }


def sylvester_kernel_dim(A: np.ndarray, B: np.ndarray, tol: float = 1e-10) -> int:
    """Dimension of ``{P : P A = B P}`` -- the linear (degree-1) couplings.

    Nonzero exactly when ``A`` and ``B`` share an eigenvalue, which for two
    rotation-scalings at the same rate means ``omega_A = +/- omega_B``.  This is
    the degree-1 instance of Lemma D Step 2's resonance condition, and it is
    what makes "equal rotation numbers" the resonance analogue for oscillatory
    modules (CLAUDE.md task 35).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    d = A.shape[0]
    M = np.kron(A.T, np.eye(d)) - np.kron(np.eye(d), B)
    sv = np.linalg.svd(M, compute_uv=False)
    return int((sv < tol * max(1.0, float(sv.max()))).sum())


def tier2_witness(mu: float = 0.70, c: float = 0.90, nu: float = 0.50) -> dict:
    """Route A's **Tier 2 is non-empty**: a modular system that does not linearise.

    Module 1 is a ``ResonantNodeBlock`` -- linear part ``diag(mu, mu^2)``, with the
    within-module resonance ``mu^2 = mu * mu`` keeping a quadratic term that no
    change of coordinates can remove.  Module 2 is a 1-D contraction at rate
    ``nu``, chosen so that **cross-module** non-resonance holds.

    Why this matters (`theory/approaches.md` §A.2).  Tier 1 assumes non-resonance
    on the full spectrum and Poincare then linearises ``F`` outright, so the object
    identified is linear and the tier is only robustness of Theorem A.  Tier 2
    keeps within-module resonances.  If no system could satisfy Tier 2's
    hypotheses *while* carrying a live resonance, Tier 2 would be empty and Route
    A's nonlinear content would be vacuous.  This construction shows it is not.

    ``nu`` must avoid the module-1 exponents and their integer combinations:
    ``nu = mu**2`` collides with the second exponent, ``nu = mu**3`` is a genuine
    cross-module resonance, and both are rejected by
    ``spectra.is_cross_module_nonresonant``.  The default ``(0.70, 0.50)`` passes.
    """
    node = ResonantNodeBlock(mu=float(mu), c=float(c))
    partner = ScalarBlock(s=float(nu), gain=1.0)
    system = ModularSystem([node, partner])
    spectra = [node.lyapunov_spectrum_exact(), partner.lyapunov_spectrum_exact()]
    return {
        "mu": float(mu),
        "c": float(c),
        "nu": float(nu),
        "system": system,
        "node": node,
        "partner": partner,
        "partition": [2, 1],
        "spectra": spectra,
        # the resonant vector monomial: z_a^2 in the z_b slot of module 1
        "resonant_monomial": (1, (2, 0)),
        "linear_part": node.linear_part(),
    }


def sample_initial_conditions(
    d: int, n: int, rng: np.random.Generator, radius: float = 1.0, annulus: float = 0.35
) -> np.ndarray:
    """Initial conditions on an annulus of the given radius.

    Contracting systems collapse to a point, so the visited region is the union
    over many short trajectories rather than a single long one.  This is the
    CLAUDE.md §3.6 support caveat made operational: conclusions only hold on the
    closure of what we actually visit.

    Spreading the starts over an annulus (rather than one orbit) is also what
    makes the visited region have **nonempty interior** -- hypothesis (B1) of
    identifiability.md §4.  On a thin support the normal-form route's conclusion
    is false (theory/route_a_assessment.md §3.5), so this is load-bearing, not
    cosmetic.
    """
    v = rng.standard_normal((n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = radius * (1.0 + annulus * (2.0 * rng.random((n, 1)) - 1.0))
    return v * r
