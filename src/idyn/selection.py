"""Learning an indecomposable model: partition-lattice model selection.

CLAUDE.md §6 (cross-cutting) and `theory/approaches.md`: nothing in training
constrains or checks indecomposability, so a `[2,2]` fit can silently converge to
two modules that are each secretly two independent 1-D systems -- the §3.1
non-identifiable regime.  Every indecomposability check elsewhere in the repo
runs on a *ground-truth* object; this module is the first that operates on the
*fitted* model.

Two tools:

1. **Certify a fitted model.**  Linearise the learned `ModularTransition` at its
   fixed point and test each block for indecomposability with `linear.py`.  This
   is the local (linearised) handle the theory admits -- `route_a_assessment.md`
   §4 is explicit that off the fixed point we have no test.

2. **Search the partition lattice.**  Indecomposability is not enforced
   pointwise; you obtain it by fitting the *finest* partition that still explains
   the data.  This is the identifiability claim of `linear_case.md` operationalised
   as model selection: the theory says the finest decomposition is canonical, and
   lattice search is how you find it without knowing the answer in advance.

The validation signature is `exp02`: at a *non-finest* partition the fit must
come back non-unique.  So a partition that both fits well and is unique across
restarts is evidence it is at-or-above the finest; the finest such is the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from idyn.linear import eigenvalue_classes, n_indecomposable_summands, slices_of
from idyn.normalform import (
    coupling_resonances,
    quadratic_jet_coefficients,
    resonance_coupling_components,
)

__all__ = [
    "integer_partitions",
    "fitted_transition_fixed_point",
    "linearize_modular_transition",
    "FittedCertificate",
    "certify_fitted_model",
    "numerical_hessians",
    "BlockNonlinearCheck",
    "block_nonlinear_certificate",
    "PartitionScore",
    "select_finest_partition",
]


def integer_partitions(d: int) -> list[tuple[int, ...]]:
    """All integer partitions of ``d`` as descending-sorted tuples of block sizes.

    These are the candidate module structures: only the multiset of block *sizes*
    matters, because the encoder is free to permute coordinates into blocks.  For
    d = 4 this is [(4,), (3,1), (2,2), (2,1,1), (1,1,1,1)].
    """
    if d < 1:
        raise ValueError("d must be >= 1")

    def _parts(n: int, cap: int):
        if n == 0:
            yield ()
            return
        for first in range(min(n, cap), 0, -1):
            for rest in _parts(n - first, first):
                yield (first, *rest)

    return list(_parts(d, d))


@torch.no_grad()
def fitted_transition_fixed_point(model, iters: int = 2000, tol: float = 1e-10) -> np.ndarray:
    """Fixed point of the fitted transition, found by iteration from the origin.

    The learned dynamics contract (they are fit to contracting systems), so
    iterating the transition converges to its unique fixed point in the visited
    basin.  Returns it in the model's latent coordinates.
    """
    d = model.cfg.d
    z = torch.zeros(1, d)
    for _ in range(iters):
        z_next = model.dyn(z)
        if float(torch.max(torch.abs(z_next - z))) < tol:
            z = z_next
            break
        z = z_next
    return z.squeeze(0).cpu().numpy().astype(np.float64)


def linearize_modular_transition(model, z_star: np.ndarray | None = None) -> np.ndarray:
    """Jacobian of the learned `ModularTransition` at its fixed point, float64.

    Block-diagonal by construction (each module ignores the others), so the
    per-block diagonal pieces are what get certified.
    """
    if z_star is None:
        z_star = fitted_transition_fixed_point(model)
    z = torch.tensor(np.asarray(z_star, dtype=np.float64), dtype=torch.float32)
    J = torch.autograd.functional.jacobian(
        lambda x: model.dyn(x.unsqueeze(0)).squeeze(0), z
    )
    return J.detach().cpu().numpy().astype(np.float64)


def numerical_hessians(step_fn, z_star: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Component Hessians ``H[i] = d^2 f_i / dz^2`` at ``z_star`` by central differences.

    ``step_fn`` is any ``R^k -> R^k`` map.  ``H[i, p, q]`` is the second partial of
    component ``i``, so component ``i``'s quadratic Taylor part is ``½ z^T H[i] z``
    -- exactly the input ``normalform.quadratic_jet_coefficients`` expects.  The
    fixed point is the right expansion point because that is where the resonance
    condition on the linear part is defined.

    This is the **float64 fallback**, for exact ``step_fn`` (the ground-truth
    ``systems`` blocks).  Do not point it at a float32 torch net: the ``1/eps^2``
    in a second difference amplifies the net's ~1e-7 roundoff into a coefficient
    error that can swamp ``coeff_tol``.  For fitted models take the Hessian by
    autodiff instead -- ``certify_fitted_model`` does, via ``model.dyn``.
    """
    z0 = np.asarray(z_star, dtype=float).ravel()
    d = z0.size
    H = np.zeros((d, d, d))
    e = np.eye(d)
    f0 = step_fn(z0)  # base point, independent of the perturbation
    for p in range(d):
        fpp = step_fn(z0 + eps * e[p])
        fpm = step_fn(z0 - eps * e[p])
        H[:, p, p] = (fpp - 2.0 * f0 + fpm) / (eps * eps)
        for q in range(p + 1, d):
            a = step_fn(z0 + eps * e[p] + eps * e[q])
            b = step_fn(z0 + eps * e[p] - eps * e[q])
            c = step_fn(z0 - eps * e[p] + eps * e[q])
            g = step_fn(z0 - eps * e[p] - eps * e[q])
            mixed = (a - b - c + g) / (4.0 * eps * eps)
            H[:, p, q] = mixed
            H[:, q, p] = mixed
    return H


@dataclass
class BlockNonlinearCheck:
    """Whether a fitted block is indecomposable once its quadratic jet is seen.

    ``checked`` is False when the test does not apply -- a linearly indecomposable
    block (nothing to obstruct) or one whose linear part is not real-diagonalisable
    with distinct eigenvalues, where the degree-2 diagonal argument is not valid.
    In that case the caller keeps the linear verdict.

    ``n_components`` is the number of connected components of the resonance-coupling
    graph over the eigendirections; the block is indecomposable iff that is 1.
    ``max_coupling`` is the largest surviving resonant-coupling coefficient (an
    edge weight), kept as a magnitude diagnostic.
    """

    checked: bool
    indecomposable: bool
    max_coupling: float = 0.0
    couplings: list = field(default_factory=list)
    n_components: int = 1


def block_nonlinear_certificate(
    step_fn,
    z_star: np.ndarray,
    L: np.ndarray,
    hessians: np.ndarray | None = None,
    lin_tol: float = 1e-7,
    coeff_tol: float = 1e-2,
    res_tol: float = 1e-2,
    eps: float = 1e-3,
) -> BlockNonlinearCheck:
    """Is a block indecomposable once the quadratic jet is taken into account?

    The linear part ``L`` can split (distinct real eigenvalues) while the map does
    not: a *resonant* monomial coupling two eigendirections cannot be conjugated
    away, so it ties the factors together.  This is the false negative in the
    purely linear ``certify_fitted_model`` (``theory/approaches.md`` §A.2.2), and
    the witness is ``systems.ResonantNodeBlock`` -- ``diag(mu, mu^2)`` linear part,
    ``z_a^2 e_b`` resonant coupling.

    The verdict is graph **connectedness**, not merely "a coupling exists": the
    module is indecomposable iff the resonance-coupling graph over its
    eigendirections is connected (``normalform.resonance_coupling_components``).
    With three or more eigendirections these differ -- a coupling between two of
    them leaves a third splitting off -- so "any coupling" would over-report
    indecomposability, and over-reporting is the direction the fit cannot catch
    (a decomposable module fits a split perfectly).  Proved at degree 2 for
    distinct eigenvalues in ``route_a_assessment.md`` §4.1.

    Applies only when ``L`` is real-diagonalisable with **distinct** eigenvalues:
    that is the Poincare-domain case where resonances are diagonal in the
    eigenbasis and the degree-2 coefficient *is* the normal-form invariant (no
    lower-degree terms feed in).  Otherwise ``checked=False`` and the linear
    verdict stands.

    The quadratic jet is supplied as ``hessians`` (component Hessians, the shape
    ``numerical_hessians`` returns) when the caller has an exact one -- e.g.
    ``certify_fitted_model`` takes it from ``model.dyn`` by autodiff.  If omitted,
    it is estimated from ``step_fn`` by ``numerical_hessians`` (float64 fallback).
    The jet is only consulted when a resonance actually exists, so a decomposable
    non-resonant block never pays for it.

    ``res_tol`` is how close ``lam_i - lam^m`` must be to 0 to read as resonant, and
    ``coeff_tol`` how large the resonant coefficient must be to obstruct.  Both
    matter for *fitted* models, whose eigenvalues and jet carry fit error -- an
    exact resonance like ``mu^2 = mu*mu`` only holds approximately once learned.
    A near-resonance is genuinely ambiguous from finite data; the defaults are
    deliberately loose and the knobs are exposed.
    """
    L = np.asarray(L, dtype=float)
    classes = eigenvalue_classes(L, tol=lin_tol)
    if sum(c.geometric for c in classes) == 1:  # linearly indecomposable already
        return BlockNonlinearCheck(checked=False, indecomposable=True)
    if not (len(classes) == L.shape[0] and all(c.is_real for c in classes)):
        # not real-diagonalisable with distinct eigenvalues: the eigenbasis, and
        # hence the resonance reading, would be basis-dependent -- abstain.
        return BlockNonlinearCheck(checked=False, indecomposable=False)

    eigs, V = np.linalg.eig(L)
    eigs, V = eigs.real, V.real
    groups = [[i] for i in range(len(eigs))]  # distinct real => each its own factor
    res = coupling_resonances(eigs, groups, degree=2, tol=res_tol)
    if not res:  # no resonance to obstruct: decomposable, and no need for the jet
        return BlockNonlinearCheck(checked=True, indecomposable=False)

    if hessians is None:
        hessians = numerical_hessians(step_fn, z_star, eps=eps)
    coeffs = quadratic_jet_coefficients(hessians, V=V)
    comps = resonance_coupling_components(
        eigs, groups, 2, coeffs, coeff_tol=coeff_tol, res_tol=res_tol
    )
    found = [
        (vm, abs(coeffs.get((vm.component, vm.exponent), 0.0)))
        for vm in res
        if abs(coeffs.get((vm.component, vm.exponent), 0.0)) > coeff_tol
    ]
    return BlockNonlinearCheck(
        checked=True,
        indecomposable=len(comps) == 1,
        max_coupling=max((c for _, c in found), default=0.0),
        couplings=found,
        n_components=len(comps),
    )


@dataclass
class FittedCertificate:
    """Per-block indecomposability of a fitted modular model.

    ``block_summands`` and ``indecomposable`` describe the **linear part** at the
    fixed point.  When ``certify_fitted_model`` is called with ``nonlinear=True``,
    the quadratic jet is additionally inspected: ``nonlinear_indecomposable`` holds
    the corrected per-block verdict (linear verdict OR a resonant coupling in the
    jet), ``nonlinear_checked`` says whether that inspection actually ran for the
    block, and ``max_coupling`` records the largest obstructing coefficient found.
    With ``nonlinear=False`` (the default) the nonlinear fields are left empty and
    the certificate is exactly the linear one.
    """

    partition: list[int]
    block_summands: list[int]
    indecomposable: list[bool]
    block_spectra: list[np.ndarray] = field(repr=False, default_factory=list)
    nonlinear_indecomposable: list[bool] = field(default_factory=list)
    nonlinear_checked: list[bool] = field(default_factory=list)
    max_coupling: list[float] = field(default_factory=list)

    @property
    def all_indecomposable(self) -> bool:
        return all(self.indecomposable)

    @property
    def all_nonlinear_indecomposable(self) -> bool:
        """Combined verdict; falls back to the linear one where nonlinear wasn't run."""
        if not self.nonlinear_indecomposable:
            return self.all_indecomposable
        return all(self.nonlinear_indecomposable)

    def summary(self) -> str:
        tag = "all blocks indecomposable" if self.all_indecomposable else "SOME blocks split"
        out = f"{tag}: partition={self.partition}, summands per block={self.block_summands}"
        if self.nonlinear_indecomposable:
            flipped = [
                i for i, (lin, nl) in enumerate(zip(self.indecomposable, self.nonlinear_indecomposable))
                if nl and not lin
            ]
            if flipped:
                out += f"; blocks {flipped} are indecomposable only nonlinearly (resonant coupling)"
        return out


def certify_fitted_model(
    model,
    tol: float = 1e-6,
    nonlinear: bool = False,
    coeff_tol: float = 1e-2,
    res_tol: float = 1e-2,
) -> FittedCertificate:
    """Linearise a fitted modular model and test each learned block.

    A block that splits into >1 summand means the learned module is *locally
    decomposable* -- the partition used was not the finest, and this fit sits in
    the non-identifiable regime.  ``tol`` is the eigenvalue-clustering tolerance;
    a learned near-rotation has a genuine complex pair (indecomposable) while a
    learned pair of 1-D maps has two real eigenvalues (decomposable).

    **The linear verdict has a false negative** (``theory/approaches.md`` §A.2.2):
    a block whose linear part is ``diag(mu, mu^2)`` reads as decomposable, yet if a
    resonant quadratic term ``z_a^2 e_b`` is present the map is indecomposable and
    does not split -- and that regime is exactly Route A's Tier 2, so the false
    negative is not a corner case.  Pass ``nonlinear=True`` to inspect the quadratic
    jet at the fixed point and correct the verdict; the result is in
    ``nonlinear_indecomposable`` / ``all_nonlinear_indecomposable``.  The linear
    fields are unchanged, so callers that only read them are unaffected.
    """
    if not model.cfg.modular:
        raise ValueError("certification requires a modular model")
    partition = list(model.cfg.partition)
    z_star = fitted_transition_fixed_point(model)
    J = linearize_modular_transition(model, z_star=z_star)
    H = _model_component_hessians(model, z_star) if nonlinear else None

    summ, indec, specs = [], [], []
    nl_indec, nl_checked, nl_coupling = [], [], []
    for sl in slices_of(partition):
        block = J[sl, sl]
        s = n_indecomposable_summands(block, tol=tol)
        lin_indec = s == 1
        summ.append(s)
        indec.append(lin_indec)
        specs.append(np.linalg.eigvals(block))

        if nonlinear:
            chk = block_nonlinear_certificate(
                None, z_star[sl], block, hessians=H[sl, sl, sl],
                lin_tol=tol, coeff_tol=coeff_tol, res_tol=res_tol,
            )
            nl_checked.append(chk.checked)
            nl_indec.append(lin_indec or chk.indecomposable)
            nl_coupling.append(chk.max_coupling)

    return FittedCertificate(
        partition=partition, block_summands=summ, indecomposable=indec,
        block_spectra=specs, nonlinear_indecomposable=nl_indec,
        nonlinear_checked=nl_checked, max_coupling=nl_coupling,
    )


def _model_component_hessians(model, z_star: np.ndarray) -> np.ndarray:
    """Component Hessians of the learned transition at ``z_star``, by autodiff.

    ``H[i]`` is the Hessian of output component ``i`` of the public ``model.dyn``,
    so the block-diagonal structure need not be reached into by hand -- each
    block's jet is the corresponding diagonal sub-tensor ``H[sl, sl, sl]``.
    Exact to float32 precision (no ``1/eps^2`` finite-difference amplification),
    which matters for the resonant-coefficient threshold on a genuinely fitted net.
    """
    z = torch.tensor(np.asarray(z_star, dtype=np.float64), dtype=torch.float32)

    def component(i: int):
        return lambda x: model.dyn(x.unsqueeze(0)).squeeze(0)[i]

    rows = [
        torch.autograd.functional.hessian(component(i), z).detach().cpu().numpy()
        for i in range(model.cfg.d)
    ]
    return np.stack(rows).astype(np.float64)


@dataclass
class PartitionScore:
    partition: tuple[int, ...]
    fit_quality: float
    n_blocks: int
    unique: bool | None = None
    acceptable: bool = False
    selected: bool = False


def select_finest_partition(
    scores: dict[tuple[int, ...], float],
    rel_tol: float = 3.0,
    uniqueness: dict[tuple[int, ...], bool] | None = None,
) -> tuple[tuple[int, ...], list[PartitionScore]]:
    """Pick the finest partition whose fit is within ``rel_tol`` of the best.

    "Best" is the smallest fit_quality over all candidates -- attained by the
    coarsest partition, which is the least constrained.  "Finest" = the most
    blocks (largest K).  A partition that fits acceptably but has *more* blocks
    than needed would over-split an indecomposable module and pay for it in the
    fit, so it will not be acceptable; the finest acceptable partition is
    therefore the finest decomposition the data supports.

    If ``uniqueness`` is supplied (from restart spread, cf. `exp02`), a partition
    must also be unique to be acceptable -- a non-unique fit is the §3.1
    signature of a non-finest partition and must be rejected even if it fits.
    """
    if not scores:
        raise ValueError("no partitions scored")
    best = min(scores.values())
    thresh = best * (1.0 + rel_tol) if best > 0 else best + rel_tol

    rows: list[PartitionScore] = []
    for part, q in scores.items():
        uniq = None if uniqueness is None else uniqueness.get(part)
        ok = q <= thresh and (uniq is not False)
        rows.append(PartitionScore(part, q, len(part), uniq, ok))

    acceptable = [r for r in rows if r.acceptable]
    # finest = max blocks; tie-break toward the smaller fit_quality
    winner = max(acceptable, key=lambda r: (r.n_blocks, -r.fit_quality))
    for r in rows:
        r.selected = r.partition == winner.partition
    rows.sort(key=lambda r: (-r.n_blocks, r.fit_quality))
    return winner.partition, rows
