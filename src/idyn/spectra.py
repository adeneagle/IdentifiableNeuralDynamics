"""Lyapunov / dichotomy spectra -- the replacement for the unusable Assumption 4.

CLAUDE.md §3.4: "Jacobian spectra generically distinct" cannot be an assumption,
because pointwise ``eig(Df(z))`` varies with z and will cross somewhere.  The
separation hypothesis has to be stated on objects that are constant along an
orbit.  Everything downstream (cocycle.py, the experiments) uses this module,
never a pointwise Jacobian spectrum.

The estimator is the standard Benettin/QR method: propagate an orthonormal
frame through the Jacobian cocycle, re-orthonormalising each step, and average
the logs of the diagonal of R.  By Oseledets this converges for a.e. initial
condition to the Lyapunov spectrum of the ergodic component containing it --
which is also the caveat: on a system with several attractors the answer
depends on where you start, so callers should average over initial conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Protocol, Sequence

import numpy as np

from idyn.linear import slices_of

__all__ = [
    "HasJacobian",
    "lyapunov_spectrum",
    "lyapunov_spectrum_averaged",
    "module_lyapunov_spectra",
    "ModuleSpectra",
    "spectral_gap",
    "jacobian_product_logs",
    "inverse_jacobian_product_logs",
    "resolvable_horizon",
    "Resonance",
    "cross_module_resonances",
    "is_cross_module_nonresonant",
]


class HasJacobian(Protocol):
    dim: int

    def step(self, z: np.ndarray) -> np.ndarray: ...

    def jacobian(self, z: np.ndarray) -> np.ndarray: ...


def lyapunov_spectrum(
    system: HasJacobian,
    z0: np.ndarray,
    T: int = 2000,
    warmup: int = 200,
    k: int | None = None,
) -> np.ndarray:
    """Lyapunov exponents along the orbit of ``z0``, largest first.

    ``k`` limits the number of exponents (default: all).  ``warmup`` steps are
    run before accumulation so the frame aligns with the Oseledets filtration.
    """
    d = int(np.asarray(z0).size)
    k = d if k is None else int(k)
    z = np.asarray(z0, dtype=float).reshape(d).copy()
    Q = np.linalg.qr(np.eye(d)[:, :k])[0]

    for _ in range(warmup):
        Q, _ = _qr_pos(system.jacobian(z) @ Q)
        z = system.step(z)

    total = np.zeros(k)
    for _ in range(T):
        Q, R = _qr_pos(system.jacobian(z) @ Q)
        total += np.log(np.abs(np.diag(R)) + 1e-300)
        z = system.step(z)
    return total / T


def _qr_pos(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR with a positive diagonal on R, so the exponents are sign-unambiguous."""
    Q, R = np.linalg.qr(M)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s, R * s[:, None]


def lyapunov_spectrum_averaged(
    system: HasJacobian,
    z0s: np.ndarray,
    T: int = 2000,
    warmup: int = 200,
) -> np.ndarray:
    """Mean spectrum over several initial conditions.

    Use this rather than a single orbit: it is the honest estimator when the
    system is not known to be uniquely ergodic on the visited region.
    """
    specs = [lyapunov_spectrum(system, z, T=T, warmup=warmup) for z in np.atleast_2d(z0s)]
    return np.mean(np.stack(specs), axis=0)


@dataclass
class ModuleSpectra:
    """Per-module Lyapunov spectra plus the separation between them."""

    partition: list[int]
    spectra: list[np.ndarray] = field(default_factory=list)
    gap: float = 0.0

    @property
    def separated(self) -> bool:
        return self.gap > 0.0

    def summary(self) -> str:
        parts = ", ".join(
            f"f{i + 1}={np.array2string(s, precision=4)}" for i, s in enumerate(self.spectra)
        )
        return f"gap={self.gap:+.4f}  [{parts}]"


def module_lyapunov_spectra(
    system, z0s: np.ndarray, T: int = 2000, warmup: int = 200
) -> ModuleSpectra:
    """Lyapunov spectrum of each module of a ModularSystem, and the gap.

    Only meaningful for a genuinely modular system: each block is iterated on
    its own coordinates, which is exactly what block-diagonality of the
    Jacobian cocycle lets us do.
    """
    partition = list(system.partition)
    sls = slices_of(partition)
    z0s = np.atleast_2d(np.asarray(z0s, dtype=float))
    specs = []
    for blk, sl in zip(system.blocks, sls):
        specs.append(lyapunov_spectrum_averaged(blk, z0s[:, sl], T=T, warmup=warmup))
    return ModuleSpectra(partition=partition, spectra=specs, gap=spectral_gap(specs))


def spectral_gap(spectra: Sequence[np.ndarray]) -> float:
    """Minimum distance between exponents belonging to *different* modules.

    This is the quantity the §3.3 cocycle argument needs to be positive; it is
    the dichotomy-spectrum analogue of "disjoint eigenvalues".  Returns 0.0 if
    any two modules share an exponent (to numerical precision).
    """
    best = np.inf
    for i in range(len(spectra)):
        for j in range(i + 1, len(spectra)):
            a = np.asarray(spectra[i]).ravel()
            b = np.asarray(spectra[j]).ravel()
            best = min(best, float(np.abs(a[:, None] - b[None, :]).min()))
    return float(best) if np.isfinite(best) else 0.0


@dataclass(frozen=True)
class Resonance:
    """A cross-module resonance ``lambda_i = sum_k m_k nu_k`` with ``|m| >= 2``.

    ``target_module`` / ``target`` identify the exponent on the left; ``terms``
    lists the (module, exponent) pairs on the right, with repeats.
    """

    target_module: int
    target: float
    terms: tuple[tuple[int, float], ...]

    @property
    def order(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        rhs = " + ".join(f"L{m + 1}({v:.4f})" for m, v in self.terms)
        return f"Resonance(L{self.target_module + 1}({self.target:.4f}) = {rhs})"


def cross_module_resonances(
    spectra: Sequence[np.ndarray], max_order: int = 4, tol: float = 1e-9
) -> list[Resonance]:
    """Find resonances between one module's exponent and a sum of others'.

    A cross-module resonance is ``lambda = sum_k m_k nu_k`` with total order
    ``|m| >= 2``, where ``lambda`` belongs to module i and at least one ``nu_k``
    comes from a module other than i.  These are exactly the relations that let a
    polynomial change of coordinates mix modules while remaining an exact
    conjugacy, so they must be excluded -- see theory/counterexamples.md.

    **Pairwise non-resonance is not enough.**  With ``mu_1 = mu_2 mu_3`` every
    pairwise log-ratio can sit far from an integer while
    ``h_1 = z_1 + c z_2 z_3`` is still an exact polynomial conjugacy.  The
    condition has to be checked on multi-indices, which is what this does.

    Exponents are taken **with multiplicity**: a module whose spectrum is
    ``{log s, log s}`` (any ``TwistBlock``) resonates with any module at rate
    ``2 log s``, regardless of rotation angles -- the phases cancel.
    """
    if max_order < 2:
        raise ValueError("resonances have order >= 2")

    flat: list[tuple[int, float]] = []
    for i, s in enumerate(spectra):
        flat.extend((i, float(v)) for v in np.asarray(s).ravel())

    out: list[Resonance] = []
    for order in range(2, max_order + 1):
        for combo in combinations_with_replacement(range(len(flat)), order):
            terms = tuple(flat[k] for k in combo)
            total = sum(v for _, v in terms)
            involved = {m for m, _ in terms}
            for i, s in enumerate(spectra):
                # cross-module: the right-hand side must touch some module != i
                if involved == {i}:
                    continue
                for lam in np.asarray(s).ravel():
                    if abs(float(lam) - total) <= tol:
                        out.append(Resonance(i, float(lam), terms))
    return out


def is_cross_module_nonresonant(
    spectra: Sequence[np.ndarray], max_order: int = 4, tol: float = 1e-9
) -> bool:
    """True if no cross-module resonance of order <= ``max_order`` exists.

    Use this to guard test systems.  Innocuous-looking parameters fail it:
    ``TwistBlock`` moduli ``(0.95, 0.9025)`` are resonant because
    ``0.9025 = 0.95**2``.
    """
    return not cross_module_resonances(spectra, max_order=max_order, tol=tol)


def jacobian_product_logs(
    system: HasJacobian, z0: np.ndarray, n_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Log of the largest and smallest singular value of Df^(n) along an orbit.

    Returns ``(log_smax, log_smin)`` of shape (n_max,), indexed by n = 1..n_max.
    A scalar is factored out of the running product at every step, so this
    stays exact well past where the raw product would underflow -- necessary
    because the cocycle bound in §3.3 is evaluated at n in the hundreds.

    **``log_smin`` is only trustworthy for n below** ``resolvable_horizon``.
    Factoring out a scalar cures underflow but not *conditioning*: the ratio
    sigma_min/sigma_max of the normalised product decays like
    ``exp(-n (lmax - lmin))``, and once that falls under machine epsilon the
    smallest singular value is numerical noise, not signal.  Past that point
    ``log_smin`` tracks ``log_smax`` plus a noise floor, so its *slope* reads
    lambda_MAX rather than lambda_min -- silently, and with the wrong sign as
    often as not.  Use ``inverse_jacobian_product_logs`` whenever you actually
    want ``1 / sigma_min``; it is stable at every n.  ``log_smax`` is fine
    throughout (sigma_max is the well-determined end of a product).
    """
    d = int(np.asarray(z0).size)
    z = np.asarray(z0, dtype=float).reshape(d).copy()
    P = np.eye(d)
    log_scale = 0.0
    smax = np.empty(n_max)
    smin = np.empty(n_max)
    for n in range(n_max):
        P = system.jacobian(z) @ P
        c = float(np.linalg.norm(P))
        if c <= 0.0 or not np.isfinite(c):
            raise FloatingPointError(f"degenerate Jacobian product at step {n}")
        P /= c
        log_scale += np.log(c)
        sv = np.linalg.svd(P, compute_uv=False)
        smax[n] = log_scale + np.log(sv[0])
        smin[n] = log_scale + np.log(max(sv[-1], 1e-300))
        z = system.step(z)
    return smax, smin


def resolvable_horizon(spread: float, eps: float = float(np.finfo(float).eps)) -> float:
    """Largest n at which ``sigma_min(Df^(n))`` is still signal in float64.

    ``spread`` is ``lambda_max - lambda_min`` of the system, so
    ``cond(Df^(n)) ~ exp(n * spread)``; the smallest singular value drops below
    the noise floor of the SVD once that exceeds ``1 / eps``.  Returns ``inf``
    for a flat spectrum.

    This is not a rounding nicety, it is the difference between a result and an
    artifact.  A ``TwistBlock`` has spectrum ``{log s, log s}`` -- spread 0,
    horizon infinite -- which is why every fixed-point measurement in ``exp05``
    is sound at n = 400.  A ``LimitCycleBlock`` has ``{0, log|1-2a|}``, a spread
    of 0.92 at the default ``a = 0.3``, so the horizon is n ~ 39: a cocycle rate
    fitted over n in [200, 400) there is fitting pure noise, and it wanders over
    several units (and both signs) with ``n_max`` and with the initial
    condition.  See ``exp08`` and ``theory/identifiability.md`` §4.4.
    """
    spread = float(spread)
    if not np.isfinite(spread) or spread <= 0.0:
        return float("inf")
    return float(np.log(1.0 / eps) / spread)


def inverse_jacobian_product_logs(
    system: HasJacobian, z0: np.ndarray, n_max: int
) -> np.ndarray:
    """``log ||[Df^(n)]^{-1}||_2`` along the orbit of ``z0``, for n = 1..n_max.

    Mathematically this is ``-log sigma_min(Df^(n))``, but computing it that way
    is wrong past ``resolvable_horizon`` (see ``jacobian_product_logs``).  Here
    the *inverse* cocycle is propagated directly,

        [Df^(n)]^{-1} = [Df^(n-1)]^{-1} @ J(z_{n-1})^{-1}   (accumulates RIGHT),

    and its **largest** singular value is read off -- the numerically
    well-determined end of a matrix product, and exactly the quantity wanted,
    since ``sigma_max([Df^(n)]^{-1}) = 1 / sigma_min(Df^(n))``.  Stable to ~1e-14
    in the fitted rate at n = 400 on a limit cycle, where the naive route is off
    by O(1).

    Requires each Jacobian along the orbit to be invertible, which holds for the
    diffeomorphisms in ``systems.py``.
    """
    d = int(np.asarray(z0).size)
    z = np.asarray(z0, dtype=float).reshape(d).copy()
    Q = np.eye(d)
    log_scale = 0.0
    out = np.empty(n_max)
    for n in range(n_max):
        J = system.jacobian(z)
        try:
            Q = Q @ np.linalg.inv(J)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise np.linalg.LinAlgError(
                f"Jacobian is singular at step {n}; the inverse cocycle is undefined"
            ) from exc
        c = float(np.linalg.norm(Q))
        if c <= 0.0 or not np.isfinite(c):
            raise FloatingPointError(f"degenerate inverse Jacobian product at step {n}")
        Q /= c
        log_scale += np.log(c)
        out[n] = log_scale + np.log(np.linalg.svd(Q, compute_uv=False)[0])
        z = system.step(z)
    return out
