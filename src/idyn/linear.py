"""The linear base case (CLAUDE.md §4 step 2).

Because a full-column-rank linear decoder already forces h in GL(d) (§3.5),
the linear setting is not a warm-up for the nonlinear one -- it is the whole
question for the model as originally specified.  This module settles it.

The result proved in ``theory/linear_case.md`` and certified here:

    Let F be invertible with R^d = U_1 (+) ... (+) U_K, U_i F-invariant.
    (A1) each F|U_i is indecomposable, and
    (A2) the minimal polynomials of the F|U_i are pairwise coprime
         (equivalently: the spectra are pairwise disjoint over C).
    Then {U_i} is the primary decomposition of F, hence canonical; and any
    S in GL(d) with S F S^{-1} modular *with indecomposable blocks* maps each
    U_i onto one block of the new decomposition.  So S is a block permutation
    composed with a block-diagonal map.

Note what (A1) + (A2) buy: the matching multiset of block dimensions does not
have to be assumed, it follows.  Note also what they cost: §3.1 fails (A1) and
``theory/counterexamples.md`` §2 gives a system satisfying (A1) but not (A2)
where the conclusion still fails -- so neither hypothesis is removable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment

__all__ = [
    "EigClass",
    "eigenvalue_classes",
    "n_indecomposable_summands",
    "is_indecomposable",
    "spectral_separation",
    "DecompositionCertificate",
    "certify_finest_decomposition",
    "primary_decomposition",
    "invariant_subspace",
    "intertwiner_space",
    "block_energy_matrix",
    "BlockPermutationReport",
    "block_permutation_report",
    "subspace_angle",
    "blocks_of",
    "slices_of",
]

_RANK_RTOL = 1e-8


def slices_of(partition: Sequence[int]) -> list[slice]:
    out, off = [], 0
    for d in partition:
        out.append(slice(off, off + d))
        off += d
    return out


def blocks_of(F: np.ndarray, partition: Sequence[int]) -> list[np.ndarray]:
    """The K diagonal blocks of F under the given partition."""
    return [F[sl, sl] for sl in slices_of(partition)]


# --------------------------------------------------------------------------
# Spectra and indecomposability
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EigClass:
    """One conjugation-closed cluster of eigenvalues of a real matrix.

    ``rep`` has non-negative imaginary part; ``size`` is the algebraic
    multiplicity counted over C (so a simple complex pair has size 2).
    """

    rep: complex
    size: int
    geometric: int

    @property
    def is_real(self) -> bool:
        return abs(self.rep.imag) < 1e-12


def eigenvalue_classes(A: np.ndarray, tol: float = 1e-7) -> list[EigClass]:
    """Cluster the eigenvalues of a real square matrix into conjugate classes.

    Two eigenvalues join the same class when they agree to within ``tol`` after
    conjugating to the upper half plane.  Geometric multiplicity is measured as
    ``n - rank(A - rep*I)`` over C, which is the number of Jordan blocks for
    that class -- and therefore the number of *real* indecomposable summands it
    contributes, since a complex pair and its conjugate share one real block.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    ev = np.linalg.eigvals(A)
    keys = np.array([complex(v.real, abs(v.imag)) for v in ev])

    reps: list[complex] = []
    counts: list[int] = []
    for k in keys:
        for i, r in enumerate(reps):
            if abs(k - r) <= tol:
                # running mean keeps the representative centred in the cluster
                counts[i] += 1
                reps[i] = r + (k - r) / counts[i]
                break
        else:
            reps.append(complex(k))
            counts.append(1)

    out = []
    for r, c in zip(reps, counts):
        M = A.astype(complex) - r * np.eye(n)
        s = np.linalg.svd(M, compute_uv=False)
        rank = int(np.sum(s > max(s[0], 1.0) * _RANK_RTOL)) if s.size else 0
        out.append(EigClass(rep=r, size=c, geometric=n - rank))
    return sorted(out, key=lambda e: (-abs(e.rep), e.rep.real, e.rep.imag))


def n_indecomposable_summands(A: np.ndarray, tol: float = 1e-7) -> int:
    """Number of indecomposable summands of A as a real R[t]-module.

    Equals the number of real Jordan blocks: sum of geometric multiplicities
    over eigenvalue classes.
    """
    return int(sum(c.geometric for c in eigenvalue_classes(A, tol=tol)))


def is_indecomposable(A: np.ndarray, tol: float = 1e-7) -> bool:
    """A is indecomposable iff it is a single real Jordan block.

    ``diag(l1, l2)`` with distinct eigenvalues is *not* -- which is precisely
    the defect behind the CLAUDE.md §3.1 counterexample.  A 2x2 rotation *is*
    (its two complex eigenvalues form one real block).
    """
    return n_indecomposable_summands(A, tol=tol) == 1


def spectral_separation(As: Sequence[np.ndarray]) -> float:
    """Minimum distance in C between eigenvalues of *different* blocks.

    Zero (to numerical precision) means condition (A2) fails.
    """
    specs = [np.linalg.eigvals(np.asarray(A, dtype=float)) for A in As]
    best = np.inf
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            d = np.abs(specs[i][:, None] - specs[j][None, :]).min()
            best = min(best, float(d))
    return best if np.isfinite(best) else 0.0


# --------------------------------------------------------------------------
# Certificate for the finest modular decomposition
# --------------------------------------------------------------------------


@dataclass
class DecompositionCertificate:
    partition: list[int]
    n_summands: list[int]
    indecomposable: list[bool]
    spectra: list[np.ndarray] = field(repr=False)
    separation: float = 0.0
    spectra_disjoint: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def A1(self) -> bool:
        """Every block is indecomposable."""
        return all(self.indecomposable)

    @property
    def A2(self) -> bool:
        """Block spectra are pairwise disjoint."""
        return self.spectra_disjoint

    @property
    def canonical(self) -> bool:
        """(A1) and (A2): the decomposition is the primary one, hence unique."""
        return self.A1 and self.A2

    def summary(self) -> str:
        head = "CANONICAL (finest, unique)" if self.canonical else "NOT canonical"
        return f"{head}: partition={self.partition}, " + "; ".join(self.reasons)


def certify_finest_decomposition(
    F: np.ndarray, partition: Sequence[int], tol: float = 1e-7, sep_tol: float = 1e-7
) -> DecompositionCertificate:
    """Check (A1) and (A2) for F under the given partition.

    This is the computable form of "prove uniqueness of the finest modular
    decomposition" (CLAUDE.md §3.1 fix).  It does not check that F is actually
    block diagonal -- call ``block_permutation_report`` for that -- it checks
    that the decomposition, if present, is the canonical one.
    """
    F = np.asarray(F, dtype=float)
    partition = list(partition)
    if sum(partition) != F.shape[0]:
        raise ValueError(f"partition {partition} does not sum to d={F.shape[0]}")

    blks = blocks_of(F, partition)
    n_sum = [n_indecomposable_summands(B, tol=tol) for B in blks]
    indec = [n == 1 for n in n_sum]
    specs = [np.linalg.eigvals(B) for B in blks]
    sep = spectral_separation(blks) if len(blks) > 1 else np.inf
    disjoint = bool(sep > sep_tol)

    reasons = []
    for i, (n, ok) in enumerate(zip(n_sum, indec)):
        if not ok:
            reasons.append(f"(A1) fails: block {i} splits into {n} summands")
    if not disjoint:
        reasons.append(f"(A2) fails: block spectra separated by only {sep:.3e}")
    if not reasons:
        reasons.append(f"(A1) ok; (A2) ok with separation {sep:.3e}")

    return DecompositionCertificate(
        partition=partition,
        n_summands=n_sum,
        indecomposable=indec,
        spectra=specs,
        separation=float(sep if np.isfinite(sep) else 0.0),
        spectra_disjoint=disjoint,
        reasons=reasons,
    )


# --------------------------------------------------------------------------
# Invariant subspaces
# --------------------------------------------------------------------------


def invariant_subspace(F: np.ndarray, selected: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Real orthonormal basis of the F-invariant subspace for ``selected`` eigenvalues.

    ``selected`` is a set of target eigenvalues (as complex numbers); the
    subspace is spanned by the generalised eigenvectors whose eigenvalues are
    nearest them.  Uses an ordered complex Schur form, which stays well behaved
    when F is defective (an eigenvector basis does not).
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    sel = np.atleast_1d(np.asarray(selected, dtype=complex))

    ev = np.linalg.eigvals(F)
    # Greedily assign each target to its nearest unclaimed eigenvalue, then
    # select by membership so that repeated eigenvalues are handled correctly.
    claimed = np.zeros(n, dtype=bool)
    for s in sel:
        d = np.abs(ev - s)
        d[claimed] = np.inf
        claimed[int(np.argmin(d))] = True
    keep = ev[claimed]

    remaining = list(keep)

    def pick(x: complex) -> bool:
        for i, r in enumerate(remaining):
            if abs(x - r) <= max(tol, 1e-10 * (1.0 + abs(r))):
                remaining.pop(i)
                return True
        return False

    T, Z, sdim = sla.schur(F.astype(complex), output="complex", sort=pick)
    if sdim != len(keep):
        raise RuntimeError(f"Schur reordering selected {sdim} of {len(keep)} eigenvalues")

    V = Z[:, :sdim]
    real_span = np.hstack([V.real, V.imag])
    Q = sla.orth(real_span, rcond=_RANK_RTOL)
    if Q.shape[1] != sdim:
        raise RuntimeError(
            f"real span has dimension {Q.shape[1]}, expected {sdim}; "
            "the selected eigenvalue set is probably not closed under conjugation"
        )
    return Q


def primary_decomposition(F: np.ndarray, tol: float = 1e-7) -> list[tuple[EigClass, np.ndarray]]:
    """The canonical decomposition of R^d into F-primary components.

    Returns ``[(eigclass, Q)]`` with Q an orthonormal real basis.  These are the
    subspaces the linear theorem says are forced; every other modular
    decomposition into indecomposable blocks is a permutation of them.
    """
    F = np.asarray(F, dtype=float)
    ev = np.linalg.eigvals(F)
    out = []
    for cls in eigenvalue_classes(F, tol=tol):
        near = ev[np.abs(np.array([complex(v.real, abs(v.imag)) for v in ev]) - cls.rep) <= tol]
        out.append((cls, invariant_subspace(F, near, tol=tol)))
    return out


def subspace_angle(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """Largest principal angle (radians) between two subspaces given by bases."""
    ang = sla.subspace_angles(np.asarray(Q1, dtype=float), np.asarray(Q2, dtype=float))
    return float(np.max(ang)) if ang.size else 0.0


# --------------------------------------------------------------------------
# Intertwiners:  { S : S F = F~ S }
# --------------------------------------------------------------------------


def intertwiner_space(F: np.ndarray, F_tilde: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Basis for Hom(F, F~) = { S : S F = F~ S }, shape (n_basis, d~, d).

    This is the exact solution set of the identifiability question in the
    linear case: the admissible reparameterisations are precisely the
    *invertible* elements of this space.  Its dimension is the sharp numerical
    certificate -- under (A1)+(A2) it equals sum_i dim End(f_i), and every
    invertible element is a block permutation.  In the §3.1 counterexample it
    is strictly larger and contains invertible non-block-permutations.
    """
    F = np.asarray(F, dtype=float)
    Ft = np.asarray(F_tilde, dtype=float)
    d, dt = F.shape[0], Ft.shape[0]
    # column-major vec:  vec(S F) = (F^T (x) I) vec(S),  vec(F~ S) = (I (x) F~) vec(S)
    L = np.kron(F.T, np.eye(dt)) - np.kron(np.eye(d), Ft)
    _, s, Vh = np.linalg.svd(L)
    cutoff = max(s[0], 1.0) * rtol if s.size else rtol
    null = Vh[np.sum(s > cutoff) :].conj()
    return np.stack([v.reshape((dt, d), order="F") for v in null]) if null.size else np.zeros((0, dt, d))


# --------------------------------------------------------------------------
# Block-permutation structure
# --------------------------------------------------------------------------


def block_energy_matrix(
    S: np.ndarray, part_row: Sequence[int], part_col: Sequence[int]
) -> np.ndarray:
    """E[i, j] = squared Frobenius mass of S in block-row i, block-column j."""
    S = np.asarray(S, dtype=float)
    rs, cs = slices_of(part_row), slices_of(part_col)
    return np.array([[float(np.sum(S[r, c] ** 2)) for c in cs] for r in rs])


@dataclass
class BlockPermutationReport:
    energy: np.ndarray = field(repr=False)
    assignment: tuple[int, ...] = ()
    on_block_fraction: float = 0.0
    is_block_permutation: bool = False
    invertible: bool = False

    def __repr__(self) -> str:
        return (
            f"BlockPermutationReport(on_block={self.on_block_fraction:.4f}, "
            f"sigma={self.assignment}, block_perm={self.is_block_permutation}, "
            f"invertible={self.invertible})"
        )


def block_permutation_report(
    S: np.ndarray,
    part_row: Sequence[int],
    part_col: Sequence[int],
    tol: float = 1e-8,
) -> BlockPermutationReport:
    """Is S a block permutation from ``part_col`` blocks to ``part_row`` blocks?

    ``on_block_fraction`` is the share of ||S||_F^2 sitting in the best matched
    blocks: 1.0 for an exact block permutation, about 1/K for a fully mixed
    map.  This is the headline number for partition recovery (see metrics.py) --
    CLAUDE.md §7 is explicit that the partition, not the coordinates, is what
    the theory claims.
    """
    S = np.asarray(S, dtype=float)
    E = block_energy_matrix(S, part_row, part_col)
    total = float(E.sum())
    if E.shape[0] != E.shape[1]:
        raise ValueError("block permutation requires equally many row and column blocks")

    row, col = linear_sum_assignment(-E)
    sigma = tuple(int(c) for c in col[np.argsort(row)])
    on = float(E[row, col].sum())
    frac = on / total if total > 0 else 0.0

    # exact block permutation: matched blocks square, all other mass zero
    dims_match = all(part_row[i] == part_col[sigma[i]] for i in range(len(sigma)))
    off = total - on
    exact = bool(dims_match and off <= tol * max(total, 1.0))

    invertible = False
    if S.shape[0] == S.shape[1]:
        sv = np.linalg.svd(S, compute_uv=False)
        invertible = bool(sv[-1] > max(sv[0], 1.0) * 1e-12)

    return BlockPermutationReport(
        energy=E,
        assignment=sigma,
        on_block_fraction=frac,
        is_block_permutation=exact,
        invertible=invertible,
    )
