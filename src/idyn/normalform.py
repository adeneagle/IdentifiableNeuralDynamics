"""Poincare-Dulac machinery: which nonlinear terms survive linearisation.

This is Route A's Tier 2 apparatus (`theory/approaches.md` §A.2).  Tier 1 assumes
non-resonance on the *full* spectrum, and there Poincare linearises F outright --
so the object identified is a linear map and the tier is honestly billed as
robustness of Theorem A.  Tier 2 keeps *within-module* resonances, and those
survive normal-form reduction as finitely many nonlinear terms whose coefficients
are conjugacy invariants.  That is where the nonlinear content lives, so it
matters whether Tier 2 is non-empty at all.  It is: see ``exp09``.

The mechanism.  Write ``f(z) = L z + P(z) + O(|z|^{deg+1})`` with ``P`` homogeneous
of degree ``deg``, and look for a near-identity ``h(z) = z + Q(z)`` killing ``P``.
Matching terms of degree ``deg`` in ``h . f = L . h`` gives the **homological
equation**

    L Q(z) - Q(L z) = P(z).

For diagonal ``L = diag(lam_1, ..., lam_d)`` the operator on the left is diagonal
in the basis of vector monomials ``z^m e_i``, with eigenvalues

    lam_i - lam^m,        lam^m := prod_k lam_k^{m_k},  |m| = deg.

A **zero** eigenvalue is a resonance: that monomial cannot be removed, its
coefficient is a normal-form invariant, and if ``P`` has a component along it the
map is not linearisable.  Everything here is stated multiplicatively in the
eigenvalues; ``spectra.cross_module_resonances`` states the same condition
additively in the Lyapunov exponents (``log lam``), and the two agree.

Only diagonalisable (here diagonal) linear parts are handled -- the Poincare-Dulac
setting.  A nontrivial Jordan block makes the operator merely triangular and the
eigenvalues are unchanged, but the eigen*vectors* are not, so the projection used
by ``linearization_obstruction`` would need revisiting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

import numpy as np

__all__ = [
    "multi_indices",
    "VectorMonomial",
    "homological_eigenvalues",
    "resonances_at_degree",
    "is_linearizable_at_degree",
    "linearization_obstruction",
    "LinearizationReport",
    "coupling_resonances",
    "quadratic_jet_coefficients",
    "resonance_coupling_components",
]


def multi_indices(d: int, degree: int) -> list[tuple[int, ...]]:
    """All exponent multi-indices ``m`` on ``d`` variables with ``|m| = degree``."""
    if d < 1 or degree < 0:
        raise ValueError("need d >= 1 and degree >= 0")
    out = [m for m in product(range(degree + 1), repeat=d) if sum(m) == degree]
    return sorted(out, reverse=True)


@dataclass(frozen=True)
class VectorMonomial:
    """The basis element ``z^m e_i`` and its homological eigenvalue.

    ``value = lam_i - lam^m``.  Zero means resonant: this term is not removable.
    """

    component: int
    exponent: tuple[int, ...]
    value: complex

    @property
    def degree(self) -> int:
        return sum(self.exponent)

    def resonant(self, tol: float = 1e-12) -> bool:
        return abs(self.value) <= tol

    def __repr__(self) -> str:
        mono = " ".join(
            f"z{k + 1}^{p}" if p > 1 else f"z{k + 1}"
            for k, p in enumerate(self.exponent)
            if p
        )
        return f"VectorMonomial({mono} e{self.component + 1}, lam_i - lam^m = {self.value:.6g})"


def homological_eigenvalues(
    eigs: Sequence[complex], degree: int
) -> list[VectorMonomial]:
    """Eigenvalues ``lam_i - lam^m`` of the homological operator at ``degree``.

    ``eigs`` are the eigenvalues of the linear part, in coordinate order.
    """
    if degree < 2:
        raise ValueError("normal-form degrees start at 2; degree 1 is the linear part")
    lam = np.asarray(eigs, dtype=complex)
    d = lam.size
    out: list[VectorMonomial] = []
    for m in multi_indices(d, degree):
        lam_m = complex(np.prod(lam ** np.asarray(m, dtype=float)))
        for i in range(d):
            out.append(VectorMonomial(i, tuple(m), complex(lam[i]) - lam_m))
    return out


def resonances_at_degree(
    eigs: Sequence[complex], degree: int, tol: float = 1e-12
) -> list[VectorMonomial]:
    """The vector monomials of this degree that cannot be removed."""
    return [vm for vm in homological_eigenvalues(eigs, degree) if vm.resonant(tol)]


def is_linearizable_at_degree(
    eigs: Sequence[complex], degree: int, tol: float = 1e-12
) -> bool:
    """True if every degree-``degree`` term is removable (no resonance)."""
    return not resonances_at_degree(eigs, degree, tol=tol)


def coupling_resonances(
    eigs: Sequence[complex],
    groups: Sequence[Sequence[int]],
    degree: int,
    tol: float = 1e-12,
) -> list[VectorMonomial]:
    """Resonant monomials that couple *across* a grouping of the coordinates.

    ``groups`` partitions ``range(d)`` into the linear indecomposable subspaces.
    A resonant vector monomial ``z^m e_i`` obstructs a **direct-product**
    splitting into those groups exactly when its support ``supp(m) union {i}``
    is not contained in a single group: a non-resonant coupling term is removed
    by the normalising change of coordinates, but a resonant one cannot be, so it
    genuinely ties two factors together.

    This is the nonlinear analogue of ``linear.is_indecomposable``: the linear
    part may split (distinct eigenvalues) while the map does not, precisely when
    such a coupling resonance carries a nonzero coefficient.  The witness is
    ``systems.ResonantNodeBlock`` -- ``z_a^2 e_b`` with ``lam_b = lam_a^2`` (see
    ``theory/approaches.md`` §A.2.2).
    """
    group_of: dict[int, int] = {}
    for g, idxs in enumerate(groups):
        for i in idxs:
            group_of[i] = g
    d = len(eigs)
    if set(group_of) != set(range(d)):
        raise ValueError("groups must partition range(d) exactly")

    out: list[VectorMonomial] = []
    for vm in resonances_at_degree(eigs, degree, tol=tol):
        involved = {group_of[k] for k in range(d) if vm.exponent[k] > 0}
        involved.add(group_of[vm.component])
        if len(involved) > 1:
            out.append(vm)
    return out


def resonance_coupling_components(
    eigs: Sequence[complex],
    groups: Sequence[Sequence[int]],
    degree: int,
    coefficients: dict[tuple[int, tuple[int, ...]], complex],
    coeff_tol: float = 0.0,
    res_tol: float = 1e-12,
) -> list[list[int]]:
    """Connected components of the resonance-coupling graph over ``groups``.

    Nodes are the group indices (the linear indecomposable sub-blocks). Two
    groups are joined by an edge when some resonant monomial ``z^m e_i`` with
    ``|coefficient| > coeff_tol`` touches both — i.e. ``supp(m) union {i}`` meets
    each. Returns the components as sorted lists of group indices.

    **A module is indecomposable iff this graph is connected** (a single
    component); a graph with $r$ components exhibits $f$ as a direct product of
    $r$ factors, one per component. This is the criterion of
    ``theory/route_a_assessment.md`` §4.1, proved there at degree 2 for distinct
    eigenvalues. Note "some resonant coupling exists" (a nonempty edge set) is
    *not* the same as connectedness once there are three or more sub-blocks: a
    coupling between two of them still leaves a third as its own factor.

    Only monomials of the given ``degree`` are considered. When the normal form
    has resonances of higher degree, the graph computed here can miss edges, so
    it is a *sub*-graph of the true coupling graph — a disconnected result at
    degree 2 is then only suggestive, whereas a connected one is conclusive.
    """
    group_of: dict[int, int] = {}
    for g, idxs in enumerate(groups):
        for i in idxs:
            group_of[i] = g
    n = len(list(groups))
    d = len(eigs)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for vm in resonances_at_degree(eigs, degree, tol=res_tol):
        touched = sorted(
            {group_of[k] for k in range(d) if vm.exponent[k] > 0} | {group_of[vm.component]}
        )
        if len(touched) < 2:
            continue  # a within-group resonance couples nothing across the partition
        if abs(complex(coefficients.get((vm.component, vm.exponent), 0.0))) <= coeff_tol:
            continue  # a resonance that is present but carries no coefficient is no edge
        for t in touched[1:]:
            union(touched[0], t)

    comps: dict[int, list[int]] = {}
    for g in range(n):
        comps.setdefault(find(g), []).append(g)
    return sorted(comps.values())


def quadratic_jet_coefficients(
    hessians: np.ndarray, V: np.ndarray | None = None
) -> dict[tuple[int, tuple[int, ...]], float]:
    """Coefficients of the degree-2 vector monomials from per-component Hessians.

    ``hessians[i]`` is the Hessian of output component ``i`` at the fixed point,
    so that component's quadratic part is ``½ z^T hessians[i] z``.  Returns a map
    ``(i, m) -> coefficient of z^m e_i``, with ``m`` a degree-2 multi-index.

    If ``V`` is given (real, e.g. the eigenvector matrix of the linear part with
    **distinct real** eigenvalues), the jet is first expressed in the coordinates
    ``w`` with ``z = V w`` -- i.e. for ``f~(w) = V^{-1} f(V w)`` the component
    Hessians become ``H~_i = sum_j (V^{-1})_{ij} V^T H_j V``.  That is the basis
    in which resonances are diagonal, so it is the basis ``coupling_resonances``
    must be read against.  Complex ``V`` (a rotation block) is out of scope here;
    the caller restricts to the real diagonalisable case.

    **Degree 2 only**, and that limit is load-bearing, not incidental: at degree 2
    the raw jet coefficient of a resonant monomial *is* its normal-form invariant,
    because there are no lower-degree nonlinear terms to feed into it under the
    normalising change of coordinates.  A degree-3 analogue could not read the raw
    third-derivative tensor -- it would first have to normalise the quadratic part
    (see ``theory/approaches.md`` §A.2.2(i)).
    """
    H = np.asarray(hessians, dtype=float)
    if H.ndim != 3 or H.shape[1] != H.shape[2] or H.shape[0] != H.shape[1]:
        raise ValueError(f"hessians must be (d, d, d), got {H.shape}")
    d = H.shape[0]

    if V is not None:
        V = np.asarray(V, dtype=float)
        Vinv = np.linalg.inv(V)
        VtHV = np.stack([V.T @ H[j] @ V for j in range(d)])  # congruence per component
        H = np.einsum("ij,jpq->ipq", Vinv, VtHV)  # H~_i = sum_j Vinv[i,j] VtHV[j]

    coeffs: dict[tuple[int, tuple[int, ...]], float] = {}
    for i in range(d):
        for p in range(d):
            for q in range(p, d):
                m = [0] * d
                m[p] += 1
                m[q] += 1
                # ½ z^T H z gives coeff H[p,p]/2 on z_p^2 and H[p,q] on z_p z_q
                coeffs[(i, tuple(m))] = (
                    0.5 * H[i, p, p] if p == q else float(H[i, p, q])
                )
    return coeffs


@dataclass
class LinearizationReport:
    """Whether a given homogeneous term can be conjugated away."""

    degree: int
    resonant: list[VectorMonomial] = field(default_factory=list)
    obstruction: dict[tuple[int, tuple[int, ...]], complex] = field(default_factory=dict)
    min_abs_eigenvalue: float = 0.0
    coeff_tol: float = 0.0

    @property
    def obstruction_norm(self) -> float:
        """Size of the surviving resonant part -- the normal-form invariant's scale."""
        vals = [abs(v) for v in self.obstruction.values()]
        return float(max(vals)) if vals else 0.0

    @property
    def linearizable(self) -> bool:
        """False iff the resonant part exceeds ``coeff_tol``.

        The default ``coeff_tol = 0`` is the exact statement, right for coefficients
        given in closed form.  Raise it when the coefficients are *estimated* -- a
        jet read off a fitted model carries fit error, and every resonant slot will
        then be nonzero by noise alone.  That is the setting a nonlinear (B2) test
        needs (`approaches.md` §A.2.2), so the knob exists for it.
        """
        return self.obstruction_norm <= self.coeff_tol

    def summary(self) -> str:
        if not self.resonant:
            return f"degree {self.degree}: no resonances, term is removable"
        tag = "REMOVABLE" if self.linearizable else "OBSTRUCTED"
        terms = ", ".join(
            f"{vm!r} coeff={self.obstruction.get((vm.component, vm.exponent), 0j):.6g}"
            for vm in self.resonant
        )
        return f"degree {self.degree}: {tag}; resonant terms: {terms}"


def linearization_obstruction(
    eigs: Sequence[complex],
    coefficients: dict[tuple[int, tuple[int, ...]], complex],
    degree: int,
    tol: float = 1e-12,
    coeff_tol: float = 0.0,
) -> LinearizationReport:
    """Can the degree-``degree`` part ``P`` be removed by a near-identity change?

    ``coefficients`` maps ``(component i, exponent m)`` to the coefficient of
    ``z^m e_i`` in ``P``.  Since the homological operator is diagonal in this
    basis, ``L Q - Q(L .) = P`` is solvable **iff** ``P`` has no component along a
    resonant monomial -- solve coefficientwise, dividing by ``lam_i - lam^m``, and
    the division is only impossible exactly where that vanishes.

    So the obstruction is just the restriction of ``P`` to the resonant monomials,
    and those surviving coefficients are the normal-form invariants.
    """
    # enumerate the vector monomials once, then split into resonant / not
    all_vm = homological_eigenvalues(eigs, degree)
    res = [vm for vm in all_vm if vm.resonant(tol)]
    obstruction = {
        (vm.component, vm.exponent): complex(coefficients.get((vm.component, vm.exponent), 0.0))
        for vm in res
    }
    vals = [abs(vm.value) for vm in all_vm if not vm.resonant(tol)]
    return LinearizationReport(
        degree=degree,
        resonant=res,
        obstruction=obstruction,
        min_abs_eigenvalue=float(min(vals)) if vals else 0.0,
        coeff_tol=float(coeff_tol),
    )
