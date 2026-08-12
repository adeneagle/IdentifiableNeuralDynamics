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
    "FiltrationOrder",
    "filtration_gap",
    "rotation_lattice_margin",
    "RotationNumber",
    "rotation_number",
    "rotation_number_averaged",
    "module_rotation_numbers",
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
class FiltrationOrder:
    """The ordered-separation data Theorem F needs, which ``spectral_gap`` is not.

    ``order`` lists module indices from slowest to fastest (descending
    ``lambda_max``); ``gap`` is the smallest step down that chain,

        min_i [ lambda_min(module order[i]) - lambda_max(module order[i+1]) ].

    Positive iff the modules occupy **pairwise disjoint intervals** of the real
    line, in that order.
    """

    order: list[int]
    gap: float
    hulls: list[tuple[float, float]]

    @property
    def ordered(self) -> bool:
        return self.gap > 0.0

    def summary(self) -> str:
        chain = " > ".join(
            f"f{i + 1}[{lo:+.4f},{hi:+.4f}]" for i, (lo, hi) in zip(self.order, self.hulls)
        )
        return f"chain-gap={self.gap:+.4f}  {chain}"


def filtration_gap(spectra: Sequence[np.ndarray]) -> FiltrationOrder:
    """Ordered separation of the module spectra -- hypothesis (F3) of Theorem F.

    ``spectral_gap`` above asks only that no two modules **share** an exponent.
    That is hypothesis (B4), and identifiability.md §4.3 records that it is
    strictly too weak: Lemma C needs the *oriented* gap
    ``lambda_max(f_j) < lambda_min(f_i)``, and disjointness does not supply it.
    The difference is not academic -- it is exactly what separates the CLAUDE.md
    §3.1 regrouping counterexample from a genuine filtration.  There the
    alternative grouping interleaves two modules whose exponent sets are still
    pairwise distinct, so ``spectral_gap`` reports a comfortable +0.18 while the
    convex hulls overlap and no chain of oriented gaps exists.

    Returns the descending order and the weakest link in the chain.  Negative
    (or zero) means the modules cannot be arranged as a filtration at all, and
    Theorem F does not apply -- whatever ``spectral_gap`` says.
    """
    hulls = [
        (float(np.asarray(s).ravel().min()), float(np.asarray(s).ravel().max()))
        for s in spectra
    ]
    if not hulls:
        return FiltrationOrder(order=[], gap=0.0, hulls=[])
    order = sorted(range(len(hulls)), key=lambda i: -hulls[i][1])
    if len(order) == 1:
        return FiltrationOrder(order=order, gap=float("inf"), hulls=[hulls[order[0]]])
    gap = min(
        hulls[order[k]][0] - hulls[order[k + 1]][1] for k in range(len(order) - 1)
    )
    return FiltrationOrder(order=order, gap=float(gap), hulls=[hulls[i] for i in order])


def rotation_lattice_margin(
    rho_a: Sequence[float], rho_b: Sequence[float], max_coeff: int = 3
) -> tuple[float, np.ndarray | None]:
    """How far apart two rotation vectors are **after quotienting by GL(2,Z)**.

    Two limit cycles carry an invariant torus, and a conjugacy acts on
    ``H_1(T^2) = Z^2``.  So the rotation *vector* is identified only up to an
    integer unimodular change of basis -- see
    ``systems.torus_regrouping_counterexample``, which realises
    ``(w1, w2) -> (w1 + w2, w2)`` as an exact modular conjugacy.  Comparing
    rotation numbers coordinatewise therefore over-states how much the data
    pins down.

    Returns ``(margin, A)``: the smallest ``max |A rho_a - rho_b|`` over integer
    ``A`` with ``|det A| = 1`` and entries bounded by ``max_coeff``, and the
    minimiser.  A margin near zero means the two fits may describe the *same*
    dynamics in different lattice bases; a large one means they genuinely
    differ.  Two modules only -- raises otherwise.
    """
    a = np.asarray(rho_a, dtype=float).ravel()
    b = np.asarray(rho_b, dtype=float).ravel()
    if a.size != 2 or b.size != 2:
        raise ValueError("the lattice quotient is only implemented for K = 2")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf"), None
    best: tuple[float, np.ndarray | None] = (float("inf"), None)
    rng = range(-max_coeff, max_coeff + 1)
    for p in rng:
        for q in rng:
            for r in rng:
                for s in rng:
                    if abs(p * s - q * r) != 1:
                        continue
                    A = np.array([[p, q], [r, s]], dtype=float)
                    err = float(np.abs(A @ a - b).max())
                    if err < best[0]:
                        best = (err, A.astype(int))
    return best


# ---------------------------------------------------------------------------
# Rotation number -- the conjugacy invariant the Lyapunov spectrum cannot see
# ---------------------------------------------------------------------------


@dataclass
class RotationNumber:
    """Average turns per step on the attractor, with how well-defined it is.

    CLAUDE.md task 23: two oscillatory modules are never separated by their
    Lyapunov spectra.  A limit cycle carries a neutral exponent 0, so
    ``spectral_gap`` returns exactly 0.0 for any pair of them, and
    ``LimitCycleBlock(a=0.3)`` has spectrum ``{0, log|1-2a|}`` for *every*
    ``omega``.  The rotation number is a genuine conjugacy invariant that the
    spectrum discards, which is what makes it the missing coordinate of the
    task-37 fingerprint.

    ``rho`` is signed, in **turns per step** -- multiply by the sampling rate for
    Hz.  It is invariant under orientation-*preserving* conjugacy and flips sign
    under an orientation-reversing one, so cross-fit comparisons should use
    ``abs``.

    ``coherence`` is the resultant length of the per-step angle increments: 1.0
    for a rigid rotation, near 0 when the module is not rotating in any plane.
    **A rho with low coherence describes nothing** -- read the pair together, the
    same way §3.9 insists a fitted rate be read with its horizon.
    """

    rho: float
    coherence: float
    n_used: int
    plane: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def well_defined(self) -> bool:
        return bool(self.n_used >= 32 and self.coherence >= 0.95)

    def summary(self) -> str:
        return (
            f"rho={self.rho:+.6f} turns/step  coherence={self.coherence:.4f}  n={self.n_used}"
        )


def rotation_number(
    system: HasJacobian,
    z0: np.ndarray,
    T: int = 2000,
    warmup: int = 200,
    min_norm: float = 1e-150,
    center: bool = False,
    plane: np.ndarray | None = None,
) -> RotationNumber:
    """Rotation number of ``system`` along the orbit of ``z0``.

    Method: iterate past the transient, normalise each state onto the unit
    sphere -- which removes the radial decay of a contraction without touching
    the angle -- take the dominant 2-D plane of the resulting direction curve,
    and average the per-step angle increment in it.

    **Numerical horizon, cf. §3.9.** A contracting module underflows: at
    ``s = 0.5`` the state passes ``1e-300`` after about a thousand steps and the
    direction becomes 0/0.  The loop stops once ``||z||`` drops below
    ``min_norm`` and reports ``n_used`` rather than averaging noise.  Truncation
    is cheap here in a way it is not for a Lyapunov average: the angle
    accumulates linearly in t, so a short clean window is worth more than a long
    dirty one.

    **Nyquist.** Increments beyond pi per step alias, so a rotation faster than
    half a turn per step reads as a slower one in the other direction.  That is
    a property of the sampling, not of the estimator, and nothing here can
    detect it.

    ``center=True`` subtracts the orbit mean before normalising, for an
    attractor not centred at the origin.  Off by default: for a contraction *to*
    the origin the mean is dominated by the largest early points, and centring
    would move the origin off the fixed point.

    Pass ``plane`` (a ``(d, 2)`` orthonormal basis) to measure several orbits in
    one fixed frame -- for ``d > 2`` the plane's orientation, hence the sign of
    ``rho``, is otherwise arbitrary per orbit.
    """
    d = int(np.asarray(z0).size)
    z = np.asarray(z0, dtype=float).reshape(d).copy()

    def alive(v: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(v)) and np.linalg.norm(v) > min_norm)

    for _ in range(int(warmup)):
        if not alive(z):
            break
        z = np.asarray(system.step(z), dtype=float).reshape(d)

    pts: list[np.ndarray] = []
    for _ in range(int(T) + 1):
        if not alive(z):
            break
        pts.append(z.copy())
        z = np.asarray(system.step(z), dtype=float).reshape(d)

    empty = np.zeros((d, 2))
    if d < 2 or len(pts) < 3:
        return RotationNumber(float("nan"), 0.0, len(pts), empty)

    P = np.stack(pts)
    if center:
        P = P - P.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(P, axis=1)
    P = P[nrm > min_norm]
    if P.shape[0] < 3:
        return RotationNumber(float("nan"), 0.0, int(P.shape[0]), empty)
    U = P / np.linalg.norm(P, axis=1)[:, None]

    if plane is not None:
        V = np.asarray(plane, dtype=float).reshape(d, 2)
    elif d == 2:
        V = np.eye(2)  # keep the caller's own orientation, so the sign is meaningful
    else:
        V = np.linalg.svd(U - U.mean(axis=0, keepdims=True), full_matrices=False)[2][:2].T

    proj = U @ V
    th = np.arctan2(proj[:, 1], proj[:, 0])
    resultant = np.mean(np.exp(1j * np.diff(th)))  # wraps the increments for free
    return RotationNumber(
        rho=float(np.angle(resultant) / (2.0 * np.pi)),
        coherence=float(np.abs(resultant)),
        n_used=int(P.shape[0]),
        plane=V,
    )


def rotation_number_averaged(
    system: HasJacobian, z0s: np.ndarray, T: int = 2000, warmup: int = 200, **kw
) -> RotationNumber:
    """Median rotation number over several initial conditions, in one fixed plane.

    Median rather than mean, and a shared plane taken from the first orbit:
    without both, a single orbit whose SVD basis came out reflected would flip
    the sign of its ``rho`` and drag the average toward zero.
    """
    z0s = np.atleast_2d(np.asarray(z0s, dtype=float))
    first = rotation_number(system, z0s[0], T=T, warmup=warmup, **kw)
    if z0s.shape[0] == 1 or not np.isfinite(first.rho):
        return first
    rest = [
        rotation_number(system, z, T=T, warmup=warmup, plane=first.plane, **kw)
        for z in z0s[1:]
    ]
    fin = [r for r in [first, *rest] if np.isfinite(r.rho)]
    if not fin:
        return RotationNumber(float("nan"), 0.0, 0, first.plane)
    return RotationNumber(
        rho=float(np.median([r.rho for r in fin])),
        coherence=float(np.median([r.coherence for r in fin])),
        n_used=int(min(r.n_used for r in fin)),
        plane=first.plane,
    )


def module_rotation_numbers(
    system, z0s: np.ndarray, T: int = 2000, warmup: int = 200
) -> list[RotationNumber]:
    """Rotation number of each module of a ModularSystem.

    Same shape as ``module_lyapunov_spectra``, and meaningful for the same
    reason: a modular system's blocks iterate on their own coordinates.
    """
    sls = slices_of(list(system.partition))
    z0s = np.atleast_2d(np.asarray(z0s, dtype=float))
    return [
        rotation_number_averaged(blk, z0s[:, sl], T=T, warmup=warmup)
        for blk, sl in zip(system.blocks, sls)
    ]


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
