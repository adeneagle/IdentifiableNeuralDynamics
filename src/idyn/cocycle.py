"""The §3.3 cocycle argument, made numerical.

The draft differentiated ``h_1(f_1(z_1), f_2(z_2)) = f~_1(h_1(z_1, z_2))`` with
respect to z_2 and wrote

    M(z) Df_2(z_2) = Df~_1(h_1(z)) M(z),        M := d h_1 / d z_2

which is a *pointwise Sylvester equation* and is only correct at fixed points of
F.  The true relation carries an argument shift:

    M(F z) Df_2(z_2) = Df~_1(h_1(z)) M(z)

which is a cocycle relation.  Iterating it n times,

    M(F^n z) Df_2^{(n)}(z_2) = Df~_1^{(n)}(h_1(z)) M(z)

and solving for M(z),

    M(z) = [Df~_1^{(n)}]^{-1} M(F^n z) Df_2^{(n)}
    ||M(z)|| <= ||[Df~_1^{(n)}]^{-1}|| . sup||M|| . ||Df_2^{(n)}||

so if ``sup ||M|| < infinity`` (compact invariant set, C^1 conjugacy) and the
bound ``B_n`` decays, then M == 0 identically.  The decay rate of B_n is

    (1/n) log B_n  ->  lambda_max(f_2) - lambda_min(f~_1)

so the hypothesis that actually does the work is the *Lyapunov* gap
``lambda_max(f_2) < lambda_min(f~_1)`` -- not eigenvalue disjointness, and not
anything evaluated pointwise.

This module measures B_n directly, so the argument can be checked rather than
believed, and so the no-gap case can be shown to genuinely stall.

Nothing in the derivation above mentions a fixed point -- it is a statement
about Lyapunov exponents, so it should hold on any compact invariant set.  It
does, but checking it on a genuine attractor needs care: see the note on
``sigma_min`` in ``spectra.jacobian_product_logs`` and §4.4 of
``theory/identifiability.md``.  ``B_n`` is assembled here from
``inverse_jacobian_product_logs``, which is stable at every n, rather than from
``-log sigma_min``, which is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from idyn.spectra import (
    HasJacobian,
    inverse_jacobian_product_logs,
    jacobian_product_logs,
)

__all__ = ["CocycleBound", "cocycle_bound", "propagate_M"]

# cond(Df^(n)) beyond which sigma_min of the accumulated product is noise.
_LOG_COND_LIMIT = float(np.log(1.0 / np.finfo(float).eps))


@dataclass
class CocycleBound:
    """The sequence B_n and its empirical exponential rate.

    ``rate`` is computed from the inverse cocycle and is reliable at every n.
    ``naive_rate`` is what the discarded ``-log sigma_min`` route reports on the
    same data, and ``n_resolvable`` is the n past which that route is measuring
    the SVD noise floor.  Both are kept so the failure can be *shown* rather
    than asserted (``exp08``); when ``n_resolvable`` exceeds the fit window the
    two rates agree to ~1e-12, which is why every ``exp05`` number was correct.
    """

    n: np.ndarray = field(repr=False)
    log_bound: np.ndarray = field(repr=False)
    rate: float = 0.0
    predicted_rate: float = float("nan")
    naive_rate: float = float("nan")
    n_resolvable: float = float("inf")
    fit_start: int = 0
    zero_tol: float = 1e-9

    @property
    def forces_M_zero(self) -> bool:
        """A *decisively* negative rate is what makes the §3.3 argument close.

        The comparison is against ``-zero_tol``, not against 0.  A fitted rate
        of -1e-17 is zero to numerical precision and drives ``B_n`` down by a
        factor ``exp(-3e-15)`` over the whole run -- it forces nothing.  Testing
        ``rate < 0`` would let the sign of a rounding error decide whether the
        no-gap case reports the block-separation step as closing, which is the
        one thing this measurement exists to rule out.
        """
        return self.rate < -abs(self.zero_tol)

    @property
    def naive_route_valid(self) -> bool:
        """Whether ``-log sigma_min`` would have been trustworthy on this fit."""
        return self.fit_start < self.n_resolvable

    def __repr__(self) -> str:
        return (
            f"CocycleBound(rate={self.rate:+.4f}, predicted={self.predicted_rate:+.4f}, "
            f"forces_M_zero={self.forces_M_zero})"
        )


def cocycle_bound(
    f_target: HasJacobian,
    z0_target: np.ndarray,
    f_source: HasJacobian,
    z0_source: np.ndarray,
    n_max: int = 300,
    fit_from: float = 0.5,
    predicted_rate: float = float("nan"),
) -> CocycleBound:
    """Measure ``B_n = ||(Df_target^{(n)})^{-1}|| . ||Df_source^{(n)}||``.

    ``f_target`` plays the role of f~_1 (its derivative gets inverted) and
    ``f_source`` the role of f_2.  The rate is fitted on the last
    ``1 - fit_from`` fraction of n to avoid the transient.  Pass
    ``predicted_rate = lambda_max(f_source) - lambda_min(f_target)`` to record
    the theoretical value alongside the measured one.

    A negative rate means M is forced to zero and the block-separation step of
    the proof goes through on this orbit.  A rate >= 0 means it does not -- and
    that is the correct outcome when the modules share an exponent, which is
    the situation the §3.1 counterexample creates.

    ``||(Df_target^{(n)})^{-1}||`` is obtained from the inverse cocycle, not
    from ``-log sigma_min`` of the forward product.  The two are equal in exact
    arithmetic and wildly different in float64 once the target's spectrum has
    any spread -- see ``spectra.resolvable_horizon``.  The naive value is
    returned alongside for comparison, never used.
    """
    smax_t, smin_t = jacobian_product_logs(f_target, z0_target, n_max)
    smax_s, _ = jacobian_product_logs(f_source, z0_source, n_max)

    # log ||(Df_target^{(n)})^{-1}||_2 = -log sigma_min(Df_target^{(n)}), but
    # computed as sigma_max of the inverse cocycle so it survives conditioning.
    inv_t = inverse_jacobian_product_logs(f_target, z0_target, n_max)
    log_bound = inv_t + smax_s
    naive_bound = (-smin_t) + smax_s
    n = np.arange(1, n_max + 1, dtype=float)

    # first n where the forward product's condition number outruns float64,
    # i.e. where the naive route stops measuring anything real
    cond_log = smax_t - smin_t
    over = np.flatnonzero(cond_log > _LOG_COND_LIMIT)
    n_resolvable = float(n[over[0]]) if over.size else float("inf")

    start = max(1, int(fit_from * n_max))
    slope = float(np.polyfit(n[start:], log_bound[start:], 1)[0])
    naive_slope = float(np.polyfit(n[start:], naive_bound[start:], 1)[0])
    return CocycleBound(
        n=n,
        log_bound=log_bound,
        rate=slope,
        predicted_rate=float(predicted_rate),
        naive_rate=naive_slope,
        n_resolvable=n_resolvable,
        fit_start=int(start),
    )


def propagate_M(
    f_target: HasJacobian,
    z0_target: np.ndarray,
    f_source: HasJacobian,
    z0_source: np.ndarray,
    M0: np.ndarray,
    n_max: int = 300,
) -> np.ndarray:
    """Evaluate ``M_n := [Df_target^{(n)}]^{-1} M_0 Df_source^{(n)}``.

    Returns ``log ||M_n||_F`` for n = 1..n_max.  Same statement as
    ``cocycle_bound`` but without the operator-norm slack, so it is the sharper
    check that a specific candidate cross-derivative M_0 is killed.  Logs are
    returned because both cocycles contract and the raw norms underflow well
    before n_max.

    Order matters and is easy to get wrong: ``Df^{(n)} = J(z_{n-1}) ... J(z_0)``,
    so the forward product accumulates on the *left* and the inverse one on the
    *right*.

    The target factor is built by propagating the **inverse** cocycle rather than
    by solving against the accumulated forward product.  Solving is what the
    obvious implementation does and it is wrong off a fixed point: the target
    product becomes numerically rank-deficient once its condition number outruns
    float64 (``spectra.resolvable_horizon``), and the solve then projects onto
    the dominant singular direction, so the measured rate drifts to
    ``lambda_max(f_source) - lambda_max(f_target)`` instead of
    ``... - lambda_min(f_target)``.  On a limit cycle that is an error of ~0.9 in
    the rate, silently.  Each *single-step* Jacobian is well conditioned, so
    inverting one at a time is safe; the ill-conditioned product is only ever
    multiplied, never inverted.
    """
    M0 = np.asarray(M0, dtype=float)
    zt = np.asarray(z0_target, dtype=float).reshape(f_target.dim).copy()
    zs = np.asarray(z0_source, dtype=float).reshape(f_source.dim).copy()
    Qt = np.eye(f_target.dim)  # [Df_target^{(n)}]^{-1}
    Ps = np.eye(f_source.dim)  # Df_source^{(n)}
    log_t = log_s = 0.0
    out = np.empty(n_max)
    for i in range(n_max):
        try:
            Qt = Qt @ np.linalg.inv(f_target.jacobian(zt))
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"target Jacobian is singular at step {i}; f_target must be a "
                "local diffeomorphism along this orbit"
            ) from exc
        Ps = f_source.jacobian(zs) @ Ps
        ct, cs = float(np.linalg.norm(Qt)), float(np.linalg.norm(Ps))
        Qt /= ct
        Ps /= cs
        log_t += np.log(ct)
        log_s += np.log(cs)
        out[i] = float(np.log(np.linalg.norm(Qt @ M0 @ Ps))) + log_s + log_t
        zt = f_target.step(zt)
        zs = f_source.step(zs)
    return out
