"""Partition-recovery metrics.

CLAUDE.md §7 is the constraint that shapes this module: what the theory claims
is identified is the **partition** plus each module's conjugacy class, never the
coordinates.  So the headline metric is block structure, and MCC is reported
second as a diagnostic.  *A high MCC with the wrong partition is a failure*, and
``recovery_report`` is built so that it reads as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from idyn.linear import block_energy_matrix, block_permutation_report, slices_of

__all__ = [
    "fit_linear_relation",
    "mcc",
    "RecoveryReport",
    "recovery_report",
    "coordinate_pairing",
    "pairing_multiset",
    "nonuniqueness_report",
    "NonUniquenessReport",
    "all_pairings",
    "FiltrationReport",
    "filtration_report",
    "BlockStructureReport",
    "jacobian_of",
    "jacobian_block_report",
    "distance_correlation",
    "distance_correlation_baseline",
    "distance_correlation_block_report",
    "ConjugacyReport",
    "conjugacy_residual",
    "hessian_of",
    "additivity_defect",
    "coupling_homogeneity_degree",
]


def fit_linear_relation(z_true: np.ndarray, z_fit: np.ndarray) -> np.ndarray:
    """Least-squares A with ``z_fit ~= z_true @ A.T``; returns A of shape (d_fit, d_true).

    In the linear-decoder setting this A *is* the reparameterisation h of the
    theory (§3.5 forces h to be linear), so asking whether A is a block
    permutation is exactly asking whether the partition was recovered.
    """
    Zt = np.asarray(z_true, dtype=float).reshape(-1, np.asarray(z_true).shape[-1])
    Zf = np.asarray(z_fit, dtype=float).reshape(-1, np.asarray(z_fit).shape[-1])
    if Zt.shape[0] != Zf.shape[0]:
        raise ValueError("z_true and z_fit must have the same number of samples")
    A, *_ = np.linalg.lstsq(Zt, Zf, rcond=None)
    return A.T


def mcc(z_true: np.ndarray, z_fit: np.ndarray) -> float:
    """Mean absolute correlation after optimal coordinate matching.

    The standard nonlinear-ICA identifiability score (cf. the sibling
    IdentifiableCommunication project).  Reported here only as a diagnostic:
    it measures coordinate recovery, which this project does *not* claim.
    """
    Zt = np.asarray(z_true, dtype=float).reshape(-1, np.asarray(z_true).shape[-1])
    Zf = np.asarray(z_fit, dtype=float).reshape(-1, np.asarray(z_fit).shape[-1])
    d = min(Zt.shape[1], Zf.shape[1])
    C = np.corrcoef(Zt.T, Zf.T)[: Zt.shape[1], Zt.shape[1] :]
    C = np.nan_to_num(np.abs(C))
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / d)


@dataclass
class RecoveryReport:
    """Everything needed to decide whether a fit recovered the partition."""

    on_block_fraction: float = 0.0
    chance_level: float = 0.0
    assignment: tuple[int, ...] = ()
    is_block_permutation: bool = False
    invertible: bool = False
    mcc: float = 0.0
    energy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)

    @property
    def recovered(self) -> bool:
        """Partition recovered, in the sense that matters for §7."""
        return self.invertible and self.on_block_fraction > 0.95

    def summary(self) -> str:
        verdict = "RECOVERED" if self.recovered else "not recovered"
        return (
            f"{verdict}: on-block {self.on_block_fraction:.4f} "
            f"(chance {self.chance_level:.4f}), sigma={self.assignment}, mcc={self.mcc:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "on_block_fraction": self.on_block_fraction,
            "chance_level": self.chance_level,
            "assignment": list(self.assignment),
            "is_block_permutation": self.is_block_permutation,
            "invertible": self.invertible,
            "mcc": self.mcc,
            "recovered": self.recovered,
        }


def _chance_on_block(part_row: Sequence[int], part_col: Sequence[int]) -> float:
    """On-block fraction expected from an isotropic random A.

    Energy spreads uniformly over entries, so the best matching collects the
    largest sum of d_i * d_sigma(i) over permutations, divided by d^2.
    """
    E = np.array([[r * c for c in part_col] for r in part_row], dtype=float)
    row, col = linear_sum_assignment(-E)
    return float(E[row, col].sum() / (sum(part_row) * sum(part_col)))


def recovery_report(
    z_true: np.ndarray,
    z_fit: np.ndarray,
    part_true: Sequence[int],
    part_fit: Sequence[int],
    tol: float = 1e-6,
) -> RecoveryReport:
    """Did the fit recover the true module partition?"""
    A = fit_linear_relation(z_true, z_fit)
    rep = block_permutation_report(A, part_fit, part_true, tol=tol)
    return RecoveryReport(
        on_block_fraction=rep.on_block_fraction,
        chance_level=_chance_on_block(part_fit, part_true),
        assignment=rep.assignment,
        is_block_permutation=rep.is_block_permutation,
        invertible=rep.invertible,
        mcc=mcc(z_true, z_fit),
        energy=rep.energy,
    )


# --------------------------------------------------------------------------
# Non-uniqueness diagnostics (the §3.1 negative control)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Filtration recovery -- the metric for Route C (identifiability.md §4.2)
# --------------------------------------------------------------------------


@dataclass
class FiltrationReport:
    """How close the recovered h is to *triangular* in Lyapunov (rate) order.

    Route C claims h is triangular, not block-diagonal: in rate order (slowest
    module first), the slow output depends only on the slow input, the fast
    output on both. So the block-energy of h should be **lower-triangular**.

    Three quantities partition the mass of ``h``:

    * ``diagonal`` -- within matched modules (what block-diagonality would need);
    * ``lower`` -- a faster module's output drawing on a slower module's input,
      which is *allowed* by the filtration (the skew-product coupling);
    * ``upper`` -- a slower output drawing on a faster input, which is
      *forbidden* and is what Lemma C drives to zero.

    ``triangular_mass = diagonal + lower`` is the Route-C success quantity;
    ``on_block = diagonal`` is the stricter Route-A one. Their gap is exactly the
    §5 finding made measurable -- the mass Lemma C cannot remove.
    """

    triangular_mass: float = 0.0
    on_block: float = 0.0
    lower_mass: float = 0.0
    upper_mass: float = 0.0
    rate_order: tuple[int, ...] = ()
    assignment: tuple[int, ...] = ()
    energy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)

    @property
    def is_triangular(self) -> bool:
        return self.upper_mass < 1e-6 * max(self.triangular_mass + self.upper_mass, 1e-30) or \
            self.triangular_mass > 1.0 - 1e-6

    @property
    def is_block_diagonal(self) -> bool:
        return self.on_block > 1.0 - 1e-6

    def summary(self) -> str:
        return (
            f"triangular_mass {self.triangular_mass:.4f} "
            f"(diag {self.on_block:.4f} + lower {self.lower_mass:.4f}), "
            f"forbidden upper {self.upper_mass:.4f}; "
            f"{'BLOCK-DIAGONAL' if self.is_block_diagonal else ('TRIANGULAR' if self.is_triangular else 'neither')}"
        )

    def to_dict(self) -> dict:
        return {
            "triangular_mass": self.triangular_mass,
            "on_block": self.on_block,
            "lower_mass": self.lower_mass,
            "upper_mass": self.upper_mass,
            "rate_order": list(self.rate_order),
            "assignment": list(self.assignment),
            "is_triangular": self.is_triangular,
            "is_block_diagonal": self.is_block_diagonal,
        }


def filtration_report(
    z_true: np.ndarray,
    z_fit: np.ndarray,
    part_true: Sequence[int],
    part_fit: Sequence[int],
    rate_order: Sequence[int],
) -> FiltrationReport:
    """Score recovery of the *filtration* (Route C), not the symmetric partition.

    ``rate_order`` lists the true module indices slowest-first (largest Lyapunov
    exponent first -- least contracting), e.g. from ``spectra.module_lyapunov_spectra``.
    The recovered ``h = fit_linear_relation(z_true, z_fit)`` is matched to the true
    modules by block energy, reordered into rate order, and its mass split into
    diagonal / lower / upper triangles.

    A perfect block-diagonal recovery gives ``triangular_mass = on_block = 1``.
    A genuine skew-product (the §4.2 conclusion, sharp per §5) gives
    ``triangular_mass = 1`` with ``on_block < 1``. A failure leaves mass in the
    forbidden upper triangle.

    Scope: this is a **linear** measure (it reads the linear relation between the
    two latent sets), so it detects linear triangular structure -- exactly the
    Theorem A / linear-decoder regime. In the nonlinear-decoder regime it is a
    first-order proxy: it sees the linear part of a nonlinear triangular ``h``,
    not its higher-order coupling.
    """
    part_true, part_fit = list(part_true), list(part_fit)
    if sorted(part_true) != sorted(part_fit):
        raise ValueError("filtration recovery compares equal block-size multisets")
    if sorted(rate_order) != list(range(len(part_true))):
        raise ValueError(f"rate_order must be a permutation of 0..{len(part_true) - 1}")

    A = fit_linear_relation(z_true, z_fit)  # (d_fit, d_true)
    E = block_energy_matrix(A, part_fit, part_true)  # rows fitted, cols true
    split = _split_block_matrix(E, rate_order)
    if split is None:
        return FiltrationReport(rate_order=tuple(rate_order))
    return FiltrationReport(
        triangular_mass=split["triangular"],
        on_block=split["diag"],
        lower_mass=split["lower"],
        upper_mass=split["upper"],
        rate_order=tuple(split["order"]),
        assignment=tuple(split["assignment"]),
        energy=E,
    )


def _split_block_matrix(
    E: np.ndarray, rate_order: Sequence[int], assignment: Sequence[int] | None = None
) -> dict | None:
    """Match fitted->true blocks, reorder slowest-first, split diag/lower/upper.

    Shared by every block-structure metric in this module so that the linear,
    Jacobian and distance-correlation readouts cannot drift apart in how they
    define "upper".  ``E`` has rows indexed by fitted block, columns by true
    block; any nonnegative coupling measure works.  Returns ``None`` when the
    total is zero (nothing to normalise against).

    ``assignment[r]`` fixes which true block fitted block ``r`` corresponds to.
    **Pass it whenever the correspondence is known** -- the default max-energy
    matching inverts precisely when the metric matters most.  For
    h = (z_A, z_B + c z_A^2) at c = 5 the off-diagonal Jacobian energy (~100)
    dwarfs the diagonal (~2), so the assignment pairs fitted-B with true-A and
    the relabelling moves that coupling *onto* the diagonal: the report swings
    back to 0.98 block-diagonal for the most triangular map in the family.  The
    matching is only safe when each block's own energy dominates its coupling,
    which is the regime the answer is already known in.
    """
    K = E.shape[0]
    if sorted(rate_order) != list(range(K)):
        raise ValueError(f"rate_order must be a permutation of 0..{K - 1}")
    total = float(E.sum())
    if total <= 0:
        return None

    if assignment is None:
        row, col = linear_sum_assignment(-E)
        fit_to_true = {int(r): int(c) for r, c in zip(row, col)}
    else:
        if sorted(assignment) != list(range(K)):
            raise ValueError(f"assignment must be a permutation of 0..{K - 1}")
        fit_to_true = {int(r): int(c) for r, c in enumerate(assignment)}
    # relabel: put fitted block at the position of the true module it matches,
    # so both axes are indexed by true module, then reorder both by rate.
    M = np.zeros((K, K))  # M[true-of-output, true-of-input]
    for r in range(K):
        M[fit_to_true[r], :] = E[r, :]
    order = list(rate_order)
    M = M[np.ix_(order, order)]  # slowest-first on both axes

    diag = float(np.trace(M))
    lower = float(np.tril(M, -1).sum())  # output faster than input: allowed
    upper = float(np.triu(M, 1).sum())   # output slower than input: forbidden
    return {
        "diag": diag / total,
        "lower": lower / total,
        "upper": upper / total,
        "triangular": (diag + lower) / total,
        "order": order,
        "assignment": [fit_to_true[r] for r in range(K)],
    }


def coordinate_pairing(A: np.ndarray, part_fit: Sequence[int]) -> tuple[int, ...]:
    """Which fitted module each *true coordinate* is assigned to.

    ``A`` maps true coordinates to fitted ones (shape (d_fit, d_true)); a true
    coordinate belongs to the fitted module carrying most of its energy.  For
    the §3.1 construction -- four independent 1-D systems grouped into two 2-D
    modules -- this tuple is the recovered grouping, and the whole point of the
    counterexample is that it is not unique.
    """
    A = np.asarray(A, dtype=float)
    sls = slices_of(part_fit)
    energy = np.stack([np.sum(A[sl, :] ** 2, axis=0) for sl in sls])  # (K, d_true)
    return tuple(int(i) for i in np.argmax(energy, axis=0))


def pairing_multiset(assign: Sequence[int]) -> frozenset[frozenset[int]]:
    """Canonical, label-invariant form of a coordinate grouping.

    ``(0,0,1,1)`` and ``(1,1,0,0)`` are the same grouping; comparing raw tuples
    would count a module relabelling as a different solution and inflate the
    apparent non-uniqueness.
    """
    groups: dict[int, set[int]] = {}
    for coord, mod in enumerate(assign):
        groups.setdefault(int(mod), set()).add(coord)
    return frozenset(frozenset(g) for g in groups.values())


@dataclass
class NonUniquenessReport:
    """Distinct groupings found among near-optimal fits."""

    groupings: list[frozenset[frozenset[int]]] = field(default_factory=list, repr=False)
    counts: dict[str, int] = field(default_factory=dict)
    n_distinct: int = 0
    n_near_optimal: int = 0
    best_loss: float = float("nan")
    loss_spread: float = 0.0

    @property
    def non_unique(self) -> bool:
        """More than one grouping fits essentially as well."""
        return self.n_distinct > 1

    def summary(self) -> str:
        # Deliberately neutral: non-uniqueness is the *expected* outcome in the
        # §3.1 negative control and a *failure* in the positive control, so the
        # verdict belongs to the caller, not to this dataclass.
        verdict = "NON-UNIQUE" if self.non_unique else "UNIQUE"
        return (
            f"{verdict}: {self.n_distinct} distinct grouping(s) among "
            f"{self.n_near_optimal} near-optimal fits; best loss {self.best_loss:.3e}, "
            f"spread {self.loss_spread:.3e}  {self.counts}"
        )


def _fmt(g: frozenset[frozenset[int]]) -> str:
    return "|".join("".join(str(c) for c in sorted(grp)) for grp in sorted(g, key=sorted))


def nonuniqueness_report(
    assignments: Sequence[Sequence[int]],
    losses: Sequence[float],
    rel_tol: float = 0.05,
) -> NonUniquenessReport:
    """Collect groupings from restarts whose loss is within ``rel_tol`` of the best.

    Restarts that simply failed to converge must not be counted as evidence of
    non-uniqueness, which is what the loss filter is for.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return NonUniquenessReport()
    best = float(np.nanmin(losses))
    keep = losses <= best * (1.0 + rel_tol) if best > 0 else losses <= best + rel_tol

    groupings = [pairing_multiset(a) for a, k in zip(assignments, keep) if k]
    counts: dict[str, int] = {}
    for g in groupings:
        counts[_fmt(g)] = counts.get(_fmt(g), 0) + 1

    return NonUniquenessReport(
        groupings=groupings,
        counts=counts,
        n_distinct=len(counts),
        n_near_optimal=int(keep.sum()),
        best_loss=best,
        loss_spread=float(losses[keep].max() - best) if keep.any() else 0.0,
    )


def all_pairings(n: int = 4) -> list[frozenset[frozenset[int]]]:
    """All ways to split ``n`` coordinates into two equal groups (n even).

    For n = 4 there are exactly 3, and the §3.1 claim is that all 3 give
    identical observations.
    """
    if n % 2:
        raise ValueError("n must be even")
    half = n // 2
    seen = set()
    for c in combinations(range(n), half):
        seen.add(frozenset({frozenset(c), frozenset(set(range(n)) - set(c))}))
    return sorted(seen, key=_fmt)


# --------------------------------------------------------------------------
# Nonlinear block structure (CLAUDE.md §3.3, §3.7)
#
# ``filtration_report`` reads the *linear* relation between the two latent sets
# and so, by its own docstring, is a first-order proxy once the decoder is
# nonlinear.  The blind spot is not academic: for h(z) = (z_A, z_B + c z_A^2)
# with z_A symmetric about 0, Cov(z_A, z_A^2) = 0 exactly, so the linear probe
# reports on_block = 0.997 for a map whose B-block is almost entirely A.  A
# metric that cannot distinguish block-diagonal from triangular is useless for
# the one question this project asks.
#
# Two replacements, deliberately of different kinds:
#   * ``jacobian_block_report``  -- M_ij = dh_i/dz_j itself, the object Lemma C
#     forces to zero.  Theory-aligned: what it reports is what the proof bounds.
#   * ``distance_correlation_block_report`` -- model-free dependence, no
#     derivatives and no fitting.  Catches coupling the Jacobian would miss if
#     h were nondifferentiable or the finite-difference step were mis-scaled.
# They answer the same question by different routes; disagreement is a signal.
# --------------------------------------------------------------------------


@dataclass
class BlockStructureReport:
    """Block structure of a nonnegative coupling matrix between two latent sets."""

    on_block: float = 1.0
    lower_mass: float = 0.0
    upper_mass: float = 0.0
    triangular_mass: float = 1.0
    rate_order: tuple[int, ...] = ()
    assignment: tuple[int, ...] = ()
    coupling: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    kind: str = ""

    @property
    def is_triangular(self) -> bool:
        return self.upper_mass < 1e-6

    @property
    def is_block_diagonal(self) -> bool:
        return self.on_block > 1.0 - 1e-6

    def __repr__(self) -> str:
        shape = (
            "BLOCK-DIAGONAL" if self.is_block_diagonal
            else ("TRIANGULAR" if self.is_triangular else "neither")
        )
        return (
            f"BlockStructureReport({self.kind}: diag {self.on_block:.4f} + "
            f"lower {self.lower_mass:.4f}, forbidden upper {self.upper_mass:.4f}; {shape})"
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "on_block": self.on_block,
            "lower_mass": self.lower_mass,
            "upper_mass": self.upper_mass,
            "triangular_mass": self.triangular_mass,
            "rate_order": list(self.rate_order),
            "assignment": list(self.assignment),
            "coupling": self.coupling.tolist(),
        }


def jacobian_of(h, z_points: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Pointwise Jacobian dh/dz by central differences; shape (N, d_fit, d_true).

    ``h`` maps an ``(N, d_true)`` array to ``(N, d_fit)``.  Central differences
    (not forward) because the error is then O(eps^2): with eps = 1e-5 in float64
    that is ~1e-10, far below any coupling this project cares about, while the
    subtractive cancellation floor stays around 1e-11.  2*d_true evaluations.
    """
    Z = np.asarray(z_points, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"z_points must be (N, d), got {Z.shape}")
    n, d = Z.shape
    base = np.asarray(h(Z), dtype=float)
    if base.ndim != 2 or base.shape[0] != n:
        raise ValueError(f"h must map (N, d) -> (N, d_fit), got {base.shape}")
    J = np.empty((n, base.shape[1], d))
    for j in range(d):
        step = np.zeros(d)
        step[j] = eps
        plus = np.asarray(h(Z + step), dtype=float)
        minus = np.asarray(h(Z - step), dtype=float)
        J[:, :, j] = (plus - minus) / (2.0 * eps)
    return J


def jacobian_block_report(
    h,
    z_points: np.ndarray,
    part_true: Sequence[int],
    part_fit: Sequence[int],
    rate_order: Sequence[int],
    eps: float = 1e-5,
    assignment: Sequence[int] | None = None,
    standardize: bool = True,
) -> BlockStructureReport:
    """Block structure of the mean squared Jacobian -- the Lemma C object itself.

    Entry (i, j) is ``mean_z ||dh_i/dz_j(z)||_F^2`` over ``z_points``.  Lemma C
    concludes ``M_ij = dh_i/dz_j == 0``, so this is the quantity the theorem
    bounds, evaluated rather than proxied.  Unlike the linear relation it is
    blind to nothing: any dependence of output block i on input block j, of any
    order, shows up in the derivative somewhere on the support.

    Averaging squares (not norms of averages) matters -- for h = z_B + c z_A^2
    the derivative 2c z_A averages to zero over a symmetric z_A while its square
    does not, which is precisely the case the linear probe misses.

    ``standardize`` (default on) rescales entry (i, j) by
    ``sigma_true[j] / sigma_fit[i]``, making it a dimensionless elasticity.
    **This is not cosmetic.**  Raw energy is not invariant under rescaling a
    block, but §7 grants exactly that freedom -- within-module h_i is an
    arbitrary diffeomorphism -- so a metric that moves when a block is rescaled
    is measuring the gauge, not the structure.  It bites hardest where this
    project lives: contracting modules have wildly different variances, a
    whitened encoder applies correspondingly different gains, and the raw
    energies then differ by ~10^3 between blocks that are equally well
    recovered (observed in exp11: 30.9 against 0.011, with dCor 0.96 and 0.99).
    """
    J = jacobian_of(h, z_points, eps=eps)
    sl_true, sl_fit = slices_of(list(part_true)), slices_of(list(part_fit))
    if standardize:
        Zt = np.asarray(z_points, dtype=float)
        Zf = np.asarray(h(Zt), dtype=float)
        s_true = np.array([max(float(Zt[:, sj].std()), 1e-12) for sj in sl_true])
        s_fit = np.array([max(float(Zf[:, si].std()), 1e-12) for si in sl_fit])
    else:
        s_true = np.ones(len(sl_true))
        s_fit = np.ones(len(sl_fit))
    E = np.zeros((len(sl_fit), len(sl_true)))
    for i, si in enumerate(sl_fit):
        for j, sj in enumerate(sl_true):
            scale = s_true[j] / s_fit[i]
            E[i, j] = float(np.mean(np.sum((scale * J[:, si, sj]) ** 2, axis=(1, 2))))
    split = _split_block_matrix(E, rate_order, assignment)
    if split is None:
        return BlockStructureReport(rate_order=tuple(rate_order), coupling=E, kind="jacobian")
    return BlockStructureReport(
        on_block=split["diag"], lower_mass=split["lower"], upper_mass=split["upper"],
        triangular_mass=split["triangular"], rate_order=tuple(split["order"]),
        assignment=tuple(split["assignment"]), coupling=E, kind="jacobian",
    )


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Szekely-Rizzo distance correlation in [0, 1]; 0 iff independent.

    Model-free: no derivatives, no fitting, no assumption that the dependence is
    linear or even monotone.  Ordinary correlation is 0 for Y = X^2 with
    symmetric X; dCor is not, which is the whole reason it is here.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X.reshape(len(X), -1)
    Y = Y.reshape(len(Y), -1)
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of samples")
    a = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    b = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()
    dcov2 = float((A * B).mean())
    dvarx, dvary = float((A * A).mean()), float((B * B).mean())
    denom = np.sqrt(dvarx * dvary)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denom))


def distance_correlation_baseline(
    n: int, p: int, q: int, seed: int = 0, reps: int = 8
) -> float:
    """Mean dCor between *independent* Gaussian samples of shape (n,p), (n,q).

    dCor is biased upward in finite samples -- it does not reach 0 for genuinely
    independent blocks, it settles around O(1/sqrt(n)) (~0.033 at n = 1500).  So
    an off-diagonal dCor of 0.03 is **not** weak coupling, it is *no* coupling,
    and reading it as mass would manufacture a leak that is not there.  Compare
    off-diagonal entries against this baseline, not against zero.
    """
    rng = np.random.default_rng(seed)
    vals = [
        distance_correlation(rng.normal(size=(n, p)), rng.normal(size=(n, q)))
        for _ in range(reps)
    ]
    return float(np.mean(vals))


def distance_correlation_block_report(
    z_true: np.ndarray,
    z_fit: np.ndarray,
    part_true: Sequence[int],
    part_fit: Sequence[int],
    rate_order: Sequence[int],
    max_points: int = 1500,
    seed: int = 0,
    assignment: Sequence[int] | None = None,
) -> BlockStructureReport:
    """Block structure of the dCor matrix between true and fitted blocks.

    The model-free cross-check on ``jacobian_block_report``: it needs neither a
    callable ``h`` nor differentiability, only paired samples.  dCor is O(n^2)
    in memory, so ``max_points`` subsamples (seeded -- CLAUDE.md §8).

    Read it as dependence, not as mass: entry (i, j) is a correlation in [0, 1],
    so the diag/lower/upper split is a *share of total dependence*, not of
    energy.  A block-diagonal h still leaves diagonal dCor near 1 and
    off-diagonal near 0, which is what the split reports.
    """
    Zt = np.asarray(z_true, dtype=float).reshape(-1, np.asarray(z_true).shape[-1])
    Zf = np.asarray(z_fit, dtype=float).reshape(-1, np.asarray(z_fit).shape[-1])
    if len(Zt) != len(Zf):
        raise ValueError("z_true and z_fit must have the same number of samples")
    if len(Zt) > max_points:
        idx = np.random.default_rng(seed).choice(len(Zt), max_points, replace=False)
        Zt, Zf = Zt[idx], Zf[idx]

    sl_true, sl_fit = slices_of(list(part_true)), slices_of(list(part_fit))
    E = np.zeros((len(sl_fit), len(sl_true)))
    for i, si in enumerate(sl_fit):
        for j, sj in enumerate(sl_true):
            E[i, j] = distance_correlation(Zt[:, sj], Zf[:, si])
    split = _split_block_matrix(E, rate_order, assignment)
    if split is None:
        return BlockStructureReport(rate_order=tuple(rate_order), coupling=E, kind="dcor")
    return BlockStructureReport(
        on_block=split["diag"], lower_mass=split["lower"], upper_mass=split["upper"],
        triangular_mass=split["triangular"], rate_order=tuple(split["order"]),
        assignment=tuple(split["assignment"]), coupling=E, kind="dcor",
    )


# --------------------------------------------------------------------------
# Hypothesis diagnostics for a *fitted* h (CLAUDE.md task 32)
#
# Lemma C and Lemma D constrain an **exact** conjugacy h o F = F~ o h.  A fitted
# model supplies only an approximate one, so a numerical result about a learned
# h says nothing about identifiability until the hypotheses are checked on the
# object actually measured.  These two functions check the two that a fit can
# silently break: exactness of the conjugacy, and additivity of h_B.
# --------------------------------------------------------------------------


@dataclass
class ConjugacyReport:
    """How far a candidate ``h`` is from satisfying ``h o F = F~ o h``.

    ``rel_step`` is the number to read.  Normalising by the *state* scale
    flatters any contracting system -- both sides shrink toward the fixed point,
    so a model that has learned nothing but "everything decays" scores well.
    Normalising by the **increment** ``h(Fz) - h(z)`` does not: it is the share
    of the actual motion that ``F~`` fails to reproduce, and the do-nothing model
    ``F~ = id`` scores exactly 1.0 rather than something small.
    """

    residual: float = 0.0     # RMS ||h(Fz) - F~(h z)||
    state_scale: float = 0.0  # RMS ||h(Fz)||
    step_scale: float = 0.0   # RMS ||h(Fz) - h(z)||
    rel_state: float = 0.0
    rel_step: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ConjugacyReport(rel_step {self.rel_step:.4f}, "
            f"rel_state {self.rel_state:.4f}, abs {self.residual:.3e})"
        )

    def to_dict(self) -> dict:
        return {
            "residual": self.residual,
            "state_scale": self.state_scale,
            "step_scale": self.step_scale,
            "rel_state": self.rel_state,
            "rel_step": self.rel_step,
        }


def conjugacy_residual(h, step_true, step_fit, z_points: np.ndarray) -> ConjugacyReport:
    """Measure ``||h(F z) - F~(h z)||`` on ``z_points``.

    ``step_true`` is the true transition ``F`` acting on true latents;
    ``step_fit`` is the learned transition ``F~`` acting on fitted latents; both
    must accept and return ``(N, d)``.  ``h`` maps true latents to fitted ones.

    Note what this is *not*: on trajectory data ``h(F z_t) = h(z_{t+1})`` is the
    encoder's own next-step latent, so the residual there coincides with the
    prediction loss the optimiser already minimises.  That makes it a fair
    readout of the hypothesis but not an independent one -- if you want the
    conjugacy tested somewhere training did not push on it, pass ``z_points``
    off the sampled trajectories.
    """
    Z = np.asarray(z_points, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"z_points must be (N, d), got {Z.shape}")
    hz = np.asarray(h(Z), dtype=float)
    h_of_Fz = np.asarray(h(np.asarray(step_true(Z), dtype=float)), dtype=float)
    F_of_hz = np.asarray(step_fit(hz), dtype=float)
    if h_of_Fz.shape != F_of_hz.shape:
        raise ValueError(f"h(F z) is {h_of_Fz.shape} but F~(h z) is {F_of_hz.shape}")

    rms = lambda A: float(np.sqrt(np.mean(np.sum(A ** 2, axis=1))))
    resid, state, step = rms(h_of_Fz - F_of_hz), rms(h_of_Fz), rms(h_of_Fz - hz)
    return ConjugacyReport(
        residual=resid,
        state_scale=state,
        step_scale=step,
        rel_state=resid / max(state, 1e-30),
        rel_step=resid / max(step, 1e-30),
    )


def hessian_of(h, z_points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Pointwise second derivatives of ``h`` by central differences; (N, d_out, d, d).

    The symmetric four-point stencil
    ``[h(z+e_p+e_q) - h(z+e_p-e_q) - h(z-e_p+e_q) + h(z-e_p-e_q)] / 4 eps^2``
    handles ``p == q`` correctly (it degenerates to the usual second difference
    at step ``2 eps``), so one loop covers the whole Hessian.

    ``eps`` is much larger than the ``1e-5`` used for first derivatives, and has
    to be: roundoff in a second difference scales as ``delta / eps^2`` where
    ``delta`` is the precision of ``h``.  **Cast a torch model to float64 before
    calling this** -- at float32 (``delta ~ 1e-7``) the roundoff floor is ~0.1 at
    ``eps = 1e-3`` and the result is noise.
    """
    Z = np.asarray(z_points, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"z_points must be (N, d), got {Z.shape}")
    n, d = Z.shape
    d_out = np.asarray(h(Z), dtype=float).shape[1]
    H = np.empty((n, d_out, d, d))
    e = np.eye(d) * eps
    for p in range(d):
        for q in range(p, d):
            v = (
                np.asarray(h(Z + e[p] + e[q]), dtype=float)
                - np.asarray(h(Z + e[p] - e[q]), dtype=float)
                - np.asarray(h(Z - e[p] + e[q]), dtype=float)
                + np.asarray(h(Z - e[p] - e[q]), dtype=float)
            ) / (4.0 * eps * eps)
            H[:, :, p, q] = v
            H[:, :, q, p] = v
    return H


def additivity_defect(
    h,
    z_points: np.ndarray,
    part_true: Sequence[int],
    out_slice: slice,
    eps: float = 1e-3,
) -> float:
    """Share of ``h_out``'s curvature that *mixes* input blocks; 0 iff additive.

    Lemma D (identifiability.md §4.5) assumes ``h_B(z_A, z_B) = z_B + psi(z_A)``.
    Additive separability in that sense is exactly the vanishing of the mixed
    second derivative ``d^2 h_B / dz_A dz_B``, so this returns

        sum of off-block Hessian energy / total Hessian energy

    for the output coordinates ``out_slice``, with each entry ``(p, q)`` weighted
    by ``sigma_{block(p)} sigma_{block(q)}``.  That weighting is what makes it a
    gauge quantity rather than a units quantity: rescaling any input block leaves
    the ratio fixed (the sigmas absorb it), and rescaling the output scales every
    term alike.  Compare ``jacobian_block_report``'s ``standardize``, which
    exists for the same reason (§3.10 trap 1).

    A *linear* ``h`` returns 0 trivially -- there is no second derivative at all.
    So this only carries information once the map is genuinely curved, and it
    should be read alongside the total curvature, not on its own.
    """
    Z = np.asarray(z_points, dtype=float)
    H = hessian_of(h, Z, eps=eps)[:, out_slice, :, :]
    sls = slices_of(list(part_true))
    sigma = np.ones(Z.shape[1])
    blk = np.zeros(Z.shape[1], dtype=int)
    for i, sl in enumerate(sls):
        sigma[sl] = max(float(Z[:, sl].std()), 1e-12)
        blk[sl] = i

    W = sigma[:, None] * sigma[None, :]
    E = np.mean(np.sum(H ** 2, axis=1), axis=0) * W ** 2  # (d, d), energy per input pair
    same = blk[:, None] == blk[None, :]
    total = float(E.sum())
    if total <= 0:
        return 0.0
    return float(E[~same].sum() / total)


def coupling_homogeneity_degree(
    h,
    out_slice: slice,
    in_slice: slice,
    z_other: np.ndarray,
    sigmas: Sequence[float] | None = None,
    n_dirs: int = 3000,
    seed: int = 0,
) -> tuple[float, float]:
    """Homogeneity degree ``p`` of the coupling ``psi(z_A) = h_out(z_A, z_other) - h_out(0, z_other)``.

    ``z_other`` holds only the *complementary* coordinates (everything outside
    ``in_slice``), held fixed while ``in_slice`` is swept radially.  Returns
    ``(p, max_residual)``: ``p`` is the log-log slope against ``sigma`` of the
    **spread across directions** of ``h_out(sigma d, z_other)``, and
    ``max_residual`` is the largest deviation from that straight line.

    Spread across directions, rather than displacement from ``h_out(0, .)``:
    for ``psi`` homogeneous of degree ``p`` the two agree up to a constant, since
    ``psi(sigma d) = sigma^p psi(d)`` makes the directional spread exactly
    ``sigma^p`` times its value at ``sigma = 1``.  But the reference form needs
    ``psi(0)`` to exist, and **the one case this function is here to detect --
    degree 0 -- is precisely the case where it does not** (``z_A/||z_A||`` has no
    limit at the origin).  Constant offsets drop out of a spread anyway.

    **This is Lemma D's Step-3/Step-4 quantity, made measurable on any ``h``.**
    The proof (identifiability.md §4.5) turns entirely on ``p``: a homogeneous
    coupling of degree ``p`` scales as ``sigma^p`` under variance modulation, so
    behaviour detects it for every ``p >= 1``; the one escape is ``p = 0``, a
    scale-invariant (direction-only) coupling, which (D1) forbids by requiring
    ``rho(f~_B) < 1``.  A fitted model need not respect (D1) -- so before reading
    any behavioural result off a fit, check whether it found the escape.

    ``max_residual`` is not decoration.  ``p`` is only meaningful if ``psi``
    really is close to homogeneous; a coupling that saturates, or that is large
    at small ``sigma`` and flattens, produces a fitted slope that describes
    nothing.  Read the pair, never the slope alone.

    Scope (§3.8): ``sigmas`` must stay inside the visited region.  For the
    contracting systems here an annulus of initial conditions sweeps a wide range
    of radii as it decays, so a decade around the data scale is legitimate --
    but off the support ``h`` is unconstrained and ``p`` is meaningless.
    """
    sig = np.asarray(list(sigmas) if sigmas is not None else np.logspace(-0.8, 0.15, 9), float)
    if sig.ndim != 1 or sig.size < 3 or np.any(sig <= 0):
        raise ValueError("sigmas must be at least 3 positive scale factors")
    rng = np.random.default_rng(seed)
    d_in = in_slice.stop - in_slice.start
    dirs = rng.normal(size=(n_dirs, d_in))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-300)

    other = np.asarray(z_other, dtype=float).ravel()
    d_total = len(other) + d_in

    def spread_at(scale: float) -> float:
        Z = np.tile(other, (n_dirs, 1))
        Z = np.insert(Z, [in_slice.start] * d_in, 0.0, axis=1)[:, :d_total]
        Z[:, in_slice] = scale * dirs
        out = np.asarray(h(Z), dtype=float)[:, out_slice]
        return float(np.sqrt(np.mean(np.sum((out - out.mean(0)) ** 2, axis=1))))

    mags = np.array([spread_at(s) for s in sig])
    if np.any(mags <= 0):
        return 0.0, float("inf")
    slope, icpt = np.polyfit(np.log(sig), np.log(mags), 1)
    resid = np.log(mags) - (slope * np.log(sig) + icpt)
    return float(slope), float(np.abs(resid).max())
