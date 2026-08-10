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
from idyn.spectra import module_lyapunov_spectra, module_rotation_numbers

__all__ = [
    "fit_linear_relation",
    "linear_relation_r2",
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
    "DynamicalFingerprint",
    "dynamical_fingerprint",
    "AgreementReport",
    "invariant_agreement",
]


def fit_linear_relation(z_true: np.ndarray, z_fit: np.ndarray) -> np.ndarray:
    """Least-squares A with ``z_fit - mean ~= (z_true - mean) @ A.T``.

    Returns A of shape (d_fit, d_true).  In the linear-decoder setting this A *is*
    the reparameterisation h of the theory (§3.5 forces h to be linear), so asking
    whether A is a block permutation is exactly asking whether the partition was
    recovered.

    Both sides are **centred**, i.e. an intercept is fitted and discarded.  It has
    to be: $h$ is only ever defined up to translation, and nothing pins the mean of
    a learned latent (the whitening penalty constrains the covariance, not the
    mean; an MLP encoder carries biases).  Without the intercept the solve is
    misspecified whenever the fitted latents are off-centre, and it fails silently
    — measured at $R^2 = -1.38$, *below* the mean baseline, on an MLP-decoder fit,
    while still returning a matrix that the block-energy readouts happily split.
    Centring takes the same fit to $0.879$.  Use :func:`linear_relation_r2` as the
    gate before believing any readout built on this.

    Centring is a no-op for the zero-mean linear-decoder experiments (exp02/03/06),
    which is why the defect stayed invisible there.
    """
    Zt = np.asarray(z_true, dtype=float).reshape(-1, np.asarray(z_true).shape[-1])
    Zf = np.asarray(z_fit, dtype=float).reshape(-1, np.asarray(z_fit).shape[-1])
    if Zt.shape[0] != Zf.shape[0]:
        raise ValueError("z_true and z_fit must have the same number of samples")
    A, *_ = np.linalg.lstsq(Zt - Zt.mean(0), Zf - Zf.mean(0), rcond=None)
    return A.T


def linear_relation_r2(z_true: np.ndarray, z_fit: np.ndarray) -> float:
    """Fraction of ``z_fit``'s variance the affine relation to ``z_true`` explains.

    The validity gate for anything built on :func:`fit_linear_relation`.  §3.10
    already establishes that a linear probe is *blind* to nonlinear block
    structure; this is the complementary check, that the probe is even a fit.
    Near zero (or negative) means the readout is not measuring $h$ at all and must
    not be reported as if it were.
    """
    Zt = np.asarray(z_true, dtype=float).reshape(-1, np.asarray(z_true).shape[-1])
    Zf = np.asarray(z_fit, dtype=float).reshape(-1, np.asarray(z_fit).shape[-1])
    A = fit_linear_relation(Zt, Zf)
    Zfc = Zf - Zf.mean(0)
    resid = Zfc - (Zt - Zt.mean(0)) @ A.T
    denom = float((Zfc**2).sum())
    return float(1.0 - (resid**2).sum() / denom) if denom > 0 else 0.0


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


# ---------------------------------------------------------------------------
# Fit-to-fit invariant agreement -- CLAUDE.md task 40
#
# Everything above this line compares a fit to the GROUND TRUTH, which is
# exactly what real data does not have.  These two objects compare two fits to
# EACH OTHER.  If the dynamics are identifiable, independent fits recover
# different coordinates and the same invariants; agreement confirms, and
# disagreement falsifies.  Nothing here ever refers to a true latent.
# ---------------------------------------------------------------------------


#: Two modules whose exponents differ by less than this per step are not ordered
#: by the spectrum at any usable horizon, so the sort must fall through to the
#: next invariant rather than let estimator noise decide.  See ``_module_sort_key``.
ORDER_TOL = 1e-2


def _module_sort_key(spectrum: np.ndarray, dim: int, rho: float, tol: float = ORDER_TOL) -> tuple:
    """Canonical order: slowest module first, i.e. descending leading exponent.

    That direction is forced by Lemma C, not chosen.  ``M_ij = d h_i / d z_j``
    vanishes iff ``lambda_max(f_j) < lambda_min(f_i)``, so the module with the
    *larger* exponents is the one that does not see the other -- the autonomous
    factor at the top of the filtration.  Sorting descending therefore lists the
    driver before the driven.

    **The spectral keys are quantised, and that is load-bearing.**  Two fitted
    limit cycles both lead with an exponent near 0; the fitted values differ by
    estimator noise of order 1e-3, which is enough for an exact comparison to
    order them -- by noise.  Quantising lets the ``|rho|`` tie-break actually
    fire, which is the whole reason the rotation number is in the fingerprint.
    Read ``order_margin`` alongside: it reports how far the leading exponents
    actually are apart, so a quantised tie is visible rather than implied.
    """
    s = np.sort(np.asarray(spectrum, dtype=float).ravel())[::-1]
    r = abs(float(rho))
    return (
        -round(float(s[0]) / tol),
        -round(float(s[-1]) / tol),
        int(dim),
        0.0 if np.isnan(r) else -r,
    )


@dataclass
class DynamicalFingerprint:
    """What §1.2 Tier 2 claims is identified, read off one fitted model.

    Deliberately *not* coordinates: module count, module dimensions, per-module
    Lyapunov spectra, per-module rotation numbers.  §7 says these are what
    survives the reparameterisation, so these are what two fits must agree on.

    Modules are stored in the caller's order; ``order`` gives the canonical
    filtration order and every comparison applies it first, because module
    labels carry no meaning (§3.10 trap 2 is the same point for a different
    metric).
    """

    partition: list[int]
    spectra: list[np.ndarray]
    rotations: list[float]
    coherences: list[float]

    def __post_init__(self) -> None:
        self.partition = [int(d) for d in self.partition]
        self.spectra = [
            np.sort(np.asarray(s, dtype=float).ravel())[::-1] for s in self.spectra
        ]
        self.rotations = [float(r) for r in self.rotations]
        self.coherences = [float(c) for c in self.coherences]
        n = len(self.partition)
        if not (len(self.spectra) == len(self.rotations) == len(self.coherences) == n):
            raise ValueError("partition, spectra, rotations and coherences must agree in length")

    @property
    def K(self) -> int:
        return len(self.partition)

    @property
    def order(self) -> list[int]:
        """Indices of the modules in canonical (slowest-first) filtration order."""
        return sorted(
            range(self.K),
            key=lambda i: _module_sort_key(self.spectra[i], self.partition[i], self.rotations[i]),
        )

    @property
    def order_margin(self) -> float:
        """Smallest separation between adjacent modules' leading exponents.

        How robustly the filtration *order* is determined by the spectrum.  Zero
        means two modules lead at the same rate and the ordering is not decided
        by the spectrum at all -- which is precisely task 23's two-oscillator
        case, where both modules carry a neutral exponent 0.  Read the order
        with this number next to it.
        """
        if self.K < 2:
            return float("inf")
        lead = sorted((float(self.spectra[i][0]) for i in range(self.K)), reverse=True)
        return float(min(lead[i] - lead[i + 1] for i in range(len(lead) - 1)))

    def duplicate_modules(
        self, spec_tol: float = 0.05, rot_tol: float = 0.01
    ) -> list[tuple[int, int]]:
        """Module pairs whose invariants coincide -- the mode-collapse signature.

        A fit can satisfy a modular constraint by putting **two modules on the
        same factor**, duplicating one and missing another entirely.  Measured at
        32 neurons/side: 2 of 12 restarts did exactly this, each erring against
        the truth by $|rho_1 - rho_2|$ to three digits.

        It matters that this is checkable **without ground truth**, because
        nothing else here is: coherence correlates with recovery error at only
        $-0.48$ (one collapsed fit scored $0.961$, above several good ones) and
        fit quality at $+0.24$ -- no information and the wrong sign, which is
        §3.11's result arriving in a new regime.  Duplicate invariants are a
        property of the fitted model alone.

        Not proof of failure: a system genuinely *can* carry two identical
        factors, and then the duplication is the right answer.  It is a flag to
        check, not a verdict -- but an unflagged fit is one fewer thing to worry
        about, and a flagged one explains a disagreement that would otherwise be
        unattributable.

        **Keyed on the well-determined invariants only** -- the rotation number
        and the *leading* exponent, never the full spectrum.  That is not a
        convenience: 3.13(b) established that the transverse exponent is the one
        quantity the data does not constrain, and an earlier version of this
        detector compared whole spectra and therefore **missed a textbook
        collapse** -- a fit reporting rotation numbers 0.0793 and 0.0804 (the same
        cycle twice) went unflagged because its two badly-estimated transverse
        exponents happened to differ by more than the tolerance.  A detector must
        not depend on a number that was never measured.
        """
        out = []
        for i in range(self.K):
            for j in range(i + 1, self.K):
                if self.partition[i] != self.partition[j]:
                    continue
                ri, rj = abs(self.rotations[i]), abs(self.rotations[j])
                rot_same = (np.isnan(ri) and np.isnan(rj)) or (
                    not np.isnan(ri) and not np.isnan(rj) and abs(ri - rj) <= rot_tol
                )
                lead_same = abs(float(self.spectra[i][0] - self.spectra[j][0])) <= spec_tol
                if lead_same and rot_same:
                    out.append((i, j))
        return out

    def reordered(self) -> "DynamicalFingerprint":
        o = self.order
        return DynamicalFingerprint(
            partition=[self.partition[i] for i in o],
            spectra=[self.spectra[i] for i in o],
            rotations=[self.rotations[i] for i in o],
            coherences=[self.coherences[i] for i in o],
        )

    def summary(self) -> str:
        f = self.reordered()
        parts = [
            f"d={f.partition[i]} lam={np.array2string(f.spectra[i], precision=4)} "
            f"rho={f.rotations[i]:+.5f}(coh {f.coherences[i]:.2f})"
            for i in range(f.K)
        ]
        return f"K={f.K} margin={f.order_margin:+.4f} | " + " || ".join(parts)


def dynamical_fingerprint(
    system, z0s: np.ndarray, T: int = 2000, warmup: int = 200, T_rotation: int | None = None
) -> DynamicalFingerprint:
    """Fingerprint of any modular system exposing ``partition`` and ``blocks``.

    Works equally on a ground-truth ``ModularSystem`` and on a fitted model
    wrapped to the ``spectra.HasJacobian`` protocol -- which is the point, since
    task 40 compares two *fits*.
    """
    ms = module_lyapunov_spectra(system, z0s, T=T, warmup=warmup)
    rot = module_rotation_numbers(system, z0s, T=T_rotation or T, warmup=warmup)
    return DynamicalFingerprint(
        partition=list(ms.partition),
        spectra=list(ms.spectra),
        rotations=[r.rho for r in rot],
        coherences=[r.coherence for r in rot],
    )


@dataclass
class AgreementReport:
    """Do two independent fits describe the same dynamics?

    ``order_agrees`` and the errors answer different questions on purpose.  Two
    fits can recover the same *set* of modules and disagree about which drives
    which; §3.7 says the ordering is the part the theory actually delivers, so
    it is reported separately rather than folded into a single score.
    """

    same_K: bool
    same_dims: bool
    order_agrees: bool
    order_margin: float
    spectrum_error: float
    rotation_error: float
    min_coherence: float
    matching: list[tuple[int, int]]
    agree: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "AGREE" if self.agree else "DISAGREE"
        return (
            f"{head}  K={self.same_K} dims={self.same_dims} order={self.order_agrees} "
            f"(margin {self.order_margin:+.4f})  spec_err={self.spectrum_error:.4g} "
            f"rot_err={self.rotation_error:.4g}  min_coh={self.min_coherence:.3f}"
        )


def invariant_agreement(
    a: DynamicalFingerprint,
    b: DynamicalFingerprint,
    spec_tol: float = 0.05,
    rot_tol: float = 0.01,
    coherence_floor: float = 0.5,
) -> AgreementReport:
    """Compare two fingerprints, with module labels treated as meaningless.

    Both are put in canonical filtration order, then modules are paired by
    Hungarian matching on spectral distance -- *not* by position.  The
    difference between those two things is the content of ``order_agrees``: if
    the nearest-spectrum pairing is the identity, the two fits ranked their
    modules the same way; if it is a permutation, they found the same factors
    and disagreed about the hierarchy.

    ``rot_tol`` is in turns per step and compares ``abs(rho)``: rotation number
    flips sign under an orientation-reversing conjugacy, which is a gauge
    freedom §7 grants, so the sign carries no cross-fit meaning.
    """
    notes: list[str] = []
    fa, fb = a.reordered(), b.reordered()
    margin = min(fa.order_margin, fb.order_margin)
    # Only over modules that actually have a rotation number: a 1-D block scores
    # coherence 0 because it cannot rotate, which is not a measurement problem.
    rotating = [
        c
        for f in (fa, fb)
        for c, r in zip(f.coherences, f.rotations)
        if not np.isnan(r)
    ]
    min_coh = float(min(rotating)) if rotating else float("nan")

    same_K = fa.K == fb.K
    if not same_K:
        notes.append(f"module count differs: {fa.K} vs {fb.K}")
        return AgreementReport(
            False, False, False, margin, float("inf"), float("inf"), min_coh, [], False, notes
        )

    # Pair modules on the FULL invariant vector, not on spectra alone.  Matching
    # by spectrum is degenerate in exactly the case the rotation number exists to
    # handle: two limit cycles have identical spectra, so the cost matrix is flat
    # and the pairing is decided by nothing.  Measured before this was fixed --
    # `exp14` part 4a paired the wrong modules in 5 of 16 comparisons, each time
    # producing a rotation error of 0.1274, which is exactly |rho_1 - rho_2|
    # rather than any recovery failure.  The cost is only used to PAIR; the
    # reported spectrum_error and rotation_error stay separate.
    BIG = 1e6
    cost = np.full((fa.K, fb.K), BIG)
    for i in range(fa.K):
        for j in range(fb.K):
            if fa.partition[i] != fb.partition[j]:
                continue
            c = float(np.abs(fa.spectra[i] - fb.spectra[j]).max())
            ra, rb = abs(float(fa.rotations[i])), abs(float(fb.rotations[j]))
            if not (np.isnan(ra) or np.isnan(rb)):
                c += abs(ra - rb)
            cost[i, j] = c
    rows, cols = linear_sum_assignment(cost)
    matching = [(int(i), int(j)) for i, j in zip(rows, cols)]

    same_dims = all(cost[i, j] < BIG for i, j in matching)
    if not same_dims:
        notes.append(
            f"no dimension-compatible pairing: {fa.partition} vs {fb.partition}"
        )
        return AgreementReport(
            True, False, False, margin, float("inf"), float("inf"), min_coh, matching, False, notes
        )

    order_agrees = all(i == j for i, j in matching)
    # Recomputed from the spectra, NOT read off `cost` -- the cost now carries a
    # rotation term and would silently inflate the reported spectral error.
    spec_err = float(
        max(np.abs(fa.spectra[i] - fb.spectra[j]).max() for i, j in matching)
    )

    def rot_diff(x: float, y: float, dx: int, dy: int) -> float:
        """|rho| difference, distinguishing "cannot rotate" from "could not measure".

        Both report NaN, and conflating them is a silent-agreement bug: a 1-D
        block has no rotation number *structurally*, so two of them agree; a
        module whose orbit underflowed has an *unknown* one, and calling that
        agreement would let the metric report a match where it has no
        information at all -- the §3.10 failure mode in a new place.
        """
        ax, ay = abs(float(x)), abs(float(y))
        if dx < 2 and dy < 2:
            return 0.0
        if np.isnan(ax) or np.isnan(ay):
            return float("inf")
        return abs(ax - ay)

    rot_err = float(
        max(
            rot_diff(fa.rotations[i], fb.rotations[j], fa.partition[i], fb.partition[j])
            for i, j in matching
        )
    )
    if np.isinf(rot_err):
        notes.append(
            "a module of dimension >= 2 has no measurable rotation number "
            "(orbit underflowed, or fewer than 3 usable points); scored as "
            "disagreement rather than as a match"
        )

    # The filtration order is only a claim when the spectrum actually determines
    # it.  Two limit cycles both lead with a neutral exponent, so `order_margin`
    # is ~0 and there is no ordering to agree about -- demanding one would score
    # an UNDETERMINED quantity as a disagreement.  Measured: `exp14` part 4a has
    # every one of 16 comparisons matching on rotation to 5e-4 while
    # `order_agrees` holds in only 10 of them, at a median margin of 0.0011.
    order_determined = margin > spec_tol
    order_ok = order_agrees or not order_determined
    if not order_agrees:
        notes.append(f"same modules, different filtration order: pairing {matching}")
    if not order_determined:
        notes.append(
            f"order margin {margin:.4g} <= spec_tol: the spectrum does not "
            "determine the ordering here (cf. task 23), so `agree` does not "
            "require the orders to match -- read `order_agrees` yourself if the "
            "hierarchy is the claim"
        )
    for tag, f in (("A", fa), ("B", fb)):
        dup = f.duplicate_modules(spec_tol=spec_tol, rot_tol=rot_tol)
        if dup:
            notes.append(
                f"fingerprint {tag} has modules with identical invariants {dup}: "
                "possible mode collapse (two modules on one factor). Check before "
                "attributing any disagreement to the data"
            )
    if not np.isnan(min_coh) and min_coh < coherence_floor:
        notes.append(
            f"min rotation coherence {min_coh:.3f} < {coherence_floor}: "
            "rotation_error is not meaningful for at least one module"
        )

    agree = bool(same_dims and order_ok and spec_err <= spec_tol and rot_err <= rot_tol)
    return AgreementReport(
        same_K=True,
        same_dims=True,
        order_agrees=order_agrees,
        order_margin=margin,
        spectrum_error=spec_err,
        rotation_error=rot_err,
        min_coherence=min_coh,
        matching=matching,
        agree=agree,
        notes=notes,
    )
