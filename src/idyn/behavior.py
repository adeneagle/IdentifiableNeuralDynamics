"""Route B: the behavioural auxiliary variable, and the canonical invariant subspace.

Route B (``theory/approaches.md`` §B) adds an observed behaviour label ``u`` and
uses it to kill **one** latent cross-derivative that the cocycle argument (Route C)
provably cannot supply on its own.  The mechanism is distributional, not dynamical:

    partition the latents into z^A (its conditional law p(z^A | u) moves with u)
    and z^B (its conditional law does not).  The u-INVARIANT subspace is
    *canonical* -- a direction lies in it iff its conditional law is fixed in u --
    so any observation-equivalent reparameterisation h must map it into itself.
    Hence h_B cannot depend on z^A: M_BA = d h_B / d z^A = 0.

No spectral, regularity or resonance hypothesis enters -- only that some directions
carry u-variation and others do not.  Its complement is *not* canonical (adding an
invariant direction to a varying one keeps it varying), so behaviour alone gives a
**triangular** h; the cocycle closes the other cross-derivative under a one-sided
gap (``theory/approaches.md`` §B, the B∘C composition).

This module provides the two pieces the mechanism is measured with: a u-conditioned
initial-condition sampler, and a detector that scores how much a latent block's
conditional law moves with u.  The detector is what "behaviour sees the leak"
means operationally: a block that has picked up z^A becomes u-dependent, and the
score rises from ~0 with the leak size.

One caveat is load-bearing and is exposed as the ``mode`` argument (Khemakhem et
al. 2020, Prop. 1, verified in ``route_a_assessment.md`` §6.1): behaviour that
modulates only the **mean** of z^A caps identifiability at a linear indeterminacy
-- it cannot resolve rotations.  To reach permutation-level structure the
modulation must move **variances** (or use >= 2 sufficient statistics).  The
``"mean"`` and ``"variance"`` modes let the experiments show both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConditionedSample",
    "conditioned_initial_conditions",
    "UDependence",
    "block_u_dependence",
]


@dataclass
class ConditionedSample:
    """Latents drawn with p(z^A | u) varying and p(z^B | u) fixed.

    ``Z`` is ``(n, d_a + d_b)`` with the varying block first; ``U`` is ``(n,)``
    integer u-labels; ``slice_a`` / ``slice_b`` index the two blocks.
    """

    Z: np.ndarray
    U: np.ndarray
    d_a: int
    d_b: int

    @property
    def slice_a(self) -> slice:
        return slice(0, self.d_a)

    @property
    def slice_b(self) -> slice:
        return slice(self.d_a, self.d_a + self.d_b)


def conditioned_initial_conditions(
    d_a: int,
    d_b: int,
    u_values: np.ndarray,
    n_per_u: int,
    rng: np.random.Generator,
    mode: str = "variance",
    base_scale: float = 1.0,
    modulation: float = 0.6,
) -> ConditionedSample:
    """Sample ``z = (z^A, z^B)`` with the A-block's law conditioned on ``u``.

    ``z^B ~ N(0, base_scale^2 I)`` for every ``u`` -- the invariant block.  The
    A-block is conditioned per ``mode``:

    * ``"variance"``: ``z^A ~ N(0, s(u)^2 I)`` with ``s(u)`` spread by
      ``modulation`` around ``base_scale`` -- moves the second moment, so it can
      drive permutation-level identifiability (Prop. 1).
    * ``"mean"``: ``z^A ~ N(m(u), base_scale^2 I)`` with ``m(u)`` spread by
      ``modulation`` -- moves only the first moment, the Prop. 1 cap.

    ``u_values`` are the discrete behaviour levels (any real labels; used to index
    the conditioning). Returns a :class:`ConditionedSample`.
    """
    if mode not in {"variance", "mean"}:
        raise ValueError("mode must be 'variance' or 'mean'")
    u_values = np.asarray(u_values, dtype=float).ravel()
    if u_values.size < 2:
        raise ValueError("need at least two behaviour levels for variability")

    # spread the conditioning factor evenly over the u levels, centred so the
    # marginal is symmetric and no single u is special
    span = np.linspace(-1.0, 1.0, u_values.size)

    Z_parts, U_parts = [], []
    for k, s in enumerate(span):
        za = rng.standard_normal((n_per_u, d_a))
        zb = base_scale * rng.standard_normal((n_per_u, d_b))
        if mode == "variance":
            scale = base_scale * (1.0 + modulation * s)
            za = scale * za
        else:  # mean
            za = base_scale * za + modulation * s * base_scale
        Z_parts.append(np.concatenate([za, zb], axis=1))
        U_parts.append(np.full(n_per_u, k, dtype=int))

    return ConditionedSample(
        Z=np.concatenate(Z_parts, axis=0),
        U=np.concatenate(U_parts, axis=0),
        d_a=d_a,
        d_b=d_b,
    )


@dataclass
class UDependence:
    """How much a block's conditional law ``p(w | u)`` moves with ``u``.

    ``mean_variation`` is the spread of the conditional means across ``u``;
    ``cov_variation`` the spread of the conditional covariances.  ``total`` sums
    them.  A **mean-only** detector (Prop. 1) would read ``mean_variation`` alone
    and miss a pure variance leak -- that is the cap, made measurable.
    """

    mean_variation: float
    cov_variation: float

    @property
    def total(self) -> float:
        return self.mean_variation + self.cov_variation

    @property
    def mean_only(self) -> float:
        return self.mean_variation


def block_u_dependence(
    w: np.ndarray, U: np.ndarray, normalize: bool = False
) -> UDependence:
    """Score how strongly ``p(w | u)`` varies with ``u`` (0 iff u-invariant).

    ``w`` is ``(n, d)`` samples of a latent block, ``U`` the ``(n,)`` u-labels.
    The score is the between-``u`` spread of the first two conditional moments:
    the standard deviation across ``u`` of the conditional means (per coordinate,
    Euclidean-aggregated) and of the conditional covariance entries (Frobenius).
    Both are zero exactly when the block's law does not move with ``u``, and grow
    with a leak of a u-varying direction into ``w``.

    ### ``normalize`` -- mandatory when comparing blocks of different scale

    The raw score has the units of ``w``: under ``w -> eps w`` the mean term
    scales like ``eps`` and the covariance term like ``eps^2``.  That is harmless
    for the sampler in this module, whose blocks are built at comparable scale,
    and **actively misleading for fitted latents**, where a block can be small
    for reasons that have nothing to do with ``u``.  A rescaled copy of a
    u-varying latent scores near zero while carrying every bit of the
    u-variation -- see CLAUDE.md §3.12, where exactly that happened.

    ``normalize=True`` whitens ``w`` by its own pooled covariance first, making
    the score invariant under any invertible linear map of the block -- the
    freedom §7 grants within a module.  Use it for anything read off a fit.  The
    default stays ``False`` so the raw quantity, and the tests written against
    it, keep their meaning.
    """
    w = np.atleast_2d(np.asarray(w, dtype=float))
    U = np.asarray(U).ravel()
    labels = np.unique(U)
    if labels.size < 2:
        raise ValueError("need at least two u levels to measure dependence")

    if normalize:
        c = w - w.mean(axis=0, keepdims=True)
        cov = np.cov(c, rowvar=False).reshape(w.shape[1], w.shape[1])
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(w.shape[1]))
        w = np.linalg.solve(L, c.T).T

    means = np.stack([w[U == u].mean(axis=0) for u in labels])          # (n_u, d)
    covs = np.stack([np.cov(w[U == u], rowvar=False).reshape(w.shape[1], w.shape[1])
                     for u in labels])                                    # (n_u, d, d)

    # spread across u: std over the u axis, then aggregate over coordinates
    mean_variation = float(np.linalg.norm(means.std(axis=0)))
    cov_variation = float(np.linalg.norm(covs.std(axis=0)))
    return UDependence(mean_variation=mean_variation, cov_variation=cov_variation)
