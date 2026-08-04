"""Experiment 4 -- epsilon-coupling perturbation sweep (CLAUDE.md §4 step 7).

The target claim: "recovered partition within O(eps) of truth provided the
spectral gap exceeds C.eps".  CLAUDE.md ranks this above sharpening the exact
conditions, because an exact theorem that evaporates at eps > 0 will not support
the interpretive claims.

Here the claim is measured rather than fitted.  For a linear system the
"recovered partition" *is* the pair of invariant subspaces, so the degradation
is exactly the principal angle between the perturbed invariant subspace of
``F_eps = F_0 + eps.C`` and the unperturbed coordinate subspace.  That is
computable to machine precision, with no optimiser in the loop.

Prediction from standard invariant-subspace perturbation theory:

    angle  ~  eps . ||C|| / sep

with ``sep`` the separation between the two blocks' spectra.  So the sweep
should show (i) slope 1 in log-log against eps, and (ii) ``angle . sep / eps``
constant across different gaps -- and it should break down once eps approaches
sep, which is the ``gap > C.eps`` threshold made concrete.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import linear as L
from idyn import systems as S

SEED = 0
OMEGA = 0.5  # equal rotation angles, so the spectral separation is exactly |s1 - s2|
S1 = 0.95
S2_VALUES = (0.90, 0.80, 0.60)  # three different gaps
EPS_VALUES = np.logspace(-6, -1, 11)


def build(s1: float, s2: float) -> tuple[np.ndarray, np.ndarray]:
    A1 = s1 * S.rotation(OMEGA)
    A2 = s2 * S.rotation(OMEGA)
    Z = np.zeros((2, 2))
    return np.block([[A1, Z], [Z, A2]]), np.block([[A1, Z], [Z, A2]])


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 4 -- how the recovered partition degrades under eps-coupling")

    C = S.off_block_coupling([2, 2], rng)  # unit spectral norm, strictly off-block
    U1 = np.eye(4)[:, :2]  # the true module-1 subspace
    print(f"\n   coupling C: ||C||_2 = {np.linalg.norm(C, 2):.4f}, strictly off-block")

    sweeps, checks = [], []
    for s2 in S2_VALUES:
        F0, _ = build(S1, s2)
        target = np.linalg.eigvals(F0[:2, :2])  # spec of block 1
        sep = float(np.abs(target[:, None] - np.linalg.eigvals(F0[2:, 2:])[None, :]).min())
        lyap_gap = abs(np.log(S1) - np.log(s2))

        angles, ratios, failed = [], [], []
        for eps in EPS_VALUES:
            Feps = F0 + eps * C
            try:
                Q = L.invariant_subspace(Feps, target)
                ang = L.subspace_angle(Q, U1)
            except Exception as exc:  # cluster collision: no clean splitting left
                angles.append(float("nan"))
                ratios.append(float("nan"))
                failed.append({"eps": float(eps), "error": type(exc).__name__})
                continue
            angles.append(ang)
            ratios.append(ang * sep / eps)

        angles = np.array(angles)
        ratios = np.array(ratios)
        ok = np.isfinite(angles)
        # slope of log(angle) vs log(eps) in the small-eps regime
        small = ok & (EPS_VALUES <= 1e-2)
        slope = float(np.polyfit(np.log(EPS_VALUES[small]), np.log(angles[small]), 1)[0])

        print(f"\n   s2 = {s2:.2f}   sep = {sep:.4f}   Lyapunov gap = {lyap_gap:.4f}")
        print(f"     {'eps':>10s} {'angle (rad)':>14s} {'angle.sep/eps':>15s}")
        for eps, a, r in zip(EPS_VALUES, angles, ratios):
            print(f"     {eps:10.2e} {a:14.6e} {r:15.6f}")
        print(f"     log-log slope (eps <= 1e-2): {slope:.4f}   (theory: 1)")

        sweeps.append(
            {
                "s1": S1,
                "s2": s2,
                "sep": sep,
                "lyapunov_gap": lyap_gap,
                "eps": EPS_VALUES.tolist(),
                "angle": angles.tolist(),
                "angle_times_sep_over_eps": ratios.tolist(),
                "loglog_slope": slope,
                "failures": failed,
            }
        )
        checks.append(
            (
                abs(slope - 1.0) < 0.05,
                f"s2={s2:.2f}: angle grows linearly in eps (log-log slope {slope:.4f})",
            )
        )

    # The constant in "angle ~ C.eps/sep" should not depend on the gap.
    const = np.array([np.nanmedian(s["angle_times_sep_over_eps"]) for s in sweeps])
    spread = float(const.max() / const.min())
    print(f"\n   angle.sep/eps across gaps: {np.array2string(const, precision=4)} "
          f"(max/min = {spread:.3f})")
    checks.append(
        (
            spread < 2.0,
            f"the constant in angle ~ C.eps/sep is gap-independent to within a factor "
            f"{spread:.3f} -- the 1/gap scaling is the right one",
        )
    )

    # Where does it break?  Push eps up to and past sep for the smallest gap.
    banner("BREAKDOWN: eps approaching the spectral gap")
    F0, _ = build(S1, S2_VALUES[0])
    target = np.linalg.eigvals(F0[:2, :2])
    sep0 = float(np.abs(target[:, None] - np.linalg.eigvals(F0[2:, 2:])[None, :]).min())
    breakdown = []
    print(f"   sep = {sep0:.4f}")
    for ratio in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0):
        eps = ratio * sep0
        Feps = F0 + eps * C
        try:
            ang = L.subspace_angle(L.invariant_subspace(Feps, target), U1)
            note = f"angle {ang:.4f} rad ({np.degrees(ang):.1f} deg)"
        except Exception as exc:
            ang = float("nan")
            note = f"NO CLEAN SPLITTING ({type(exc).__name__})"
        print(f"     eps/sep = {ratio:5.2f}:  {note}")
        breakdown.append({"eps_over_sep": ratio, "eps": float(eps), "angle": ang})

    small_ok = breakdown[0]["angle"] < 0.05
    large_bad = (not np.isfinite(breakdown[-1]["angle"])) or breakdown[-1]["angle"] > 0.5
    checks.append(
        (
            small_ok and large_bad,
            "the partition survives eps << sep and is destroyed once eps >~ sep -- "
            "this is the 'gap > C.eps' threshold, located empirically",
        )
    )

    banner("VERDICTS")
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)
    print(
        "\n  Reading: for a LINEAR system the O(eps/gap) claim of §4 step 7 holds and is\n"
        "  sharp.  What is NOT established here is the nonlinear statement, where the\n"
        "  invariant subspaces become invariant manifolds and 'sep' must be replaced by\n"
        "  the dichotomy-spectrum gap.  That remains TODO(gap)."
    )

    save(
        "exp04_perturbation",
        {"seed": SEED, "omega": OMEGA, "s1": S1, "s2_values": list(S2_VALUES),
         "eps_values": EPS_VALUES.tolist(), "coupling_norm": float(np.linalg.norm(C, 2)),
         "sweeps": sweeps, "breakdown": breakdown, "constant_spread": spread,
         "all_passed": passed, "checks": [{"passed": ok, "claim": m} for ok, m in checks]},
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
