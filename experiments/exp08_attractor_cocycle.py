"""Experiment 8 -- does Lemma C survive off the fixed point?  (CLAUDE.md task 22)

Lemma C (identifiability.md §4.1) is a statement about Lyapunov exponents.  Its
derivation never mentions a fixed point, so it *should* hold on any compact
invariant set carrying an ergodic measure and a module Lyapunov gap.  Every
measurement certifying it so far (`exp05`) was nonetheless taken at an attracting
fixed point, and the block-diagonality route above it (Theorem B §5.3) genuinely
does need one -- Poincare-Dulac is a Poincare-domain phenomenon.  So the question
is open for the *filtration* claim specifically, which is the one route standing
on proved ground.

Part 1 is the measurement that has to come first, because the obvious way to run
this experiment is wrong.  `sigma_min` of an accumulated Jacobian product stops
being measurable once cond(Df^n) = exp(n*(lmax-lmin)) outruns float64, at
n ~ 36/spread.  A TwistBlock has spectrum {log s, log s} -- spread 0, no horizon
-- which is why every exp05 number is sound at n = 400.  A LimitCycleBlock has
{0, log|1-2a|}, spread 0.92, horizon n ~ 39: a rate fitted over n in [200, 400)
there is fitting the SVD noise floor.  It wanders by units, and by sign, with
n_max and with the initial condition, and it will happily report that the
block-separation step closes when the gap it needs does not exist.

Parts 2-4 then answer the question with the sound bound:

  2. On a limit cycle the predicted rate lambda_max(f_2) - lambda_min(f~_1) holds
     to ~1e-14, and the threshold sits exactly at the radial multiplier |1-2a|.
  3. A limit cycle carries a NEUTRAL exponent 0, so it can only ever be the
     dominant module -- an oscillatory factor sits at the top of the filtration
     or is not separated at all.
  4. Two limit cycles therefore can never be separated: each contributes a 0, so
     the modules share an exponent and (B4) fails outright.  This is the case
     that produced irreconcilable numbers before part 1 was understood; it is
     outside Lemma C's hypotheses, not a counterexample to it.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import cocycle as CC
from idyn import spectra as SP
from idyn import systems as S

SEED = 0
N_MAX = 400
A_CYCLE = 0.30          # radial multiplier |1 - 2a| = 0.40, so lmin = log 0.4
BETAS = (0.0, 0.6, 1.5)
S_FAST = (0.20, 0.25, 0.30, 0.35, 0.38, 0.42, 0.50)
Z_CYCLE = np.array([1.0, 0.0])
Z_FAST = np.array([0.7, 0.3])


def fitted_rate(y: np.ndarray, fit_from: float = 0.5) -> float:
    n = np.arange(1, len(y) + 1, dtype=float)
    start = max(1, int(fit_from * len(y)))
    return float(np.polyfit(n[start:], y[start:], 1)[0])


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 8 -- Lemma C on a genuine attractor (task 22)")

    cyc = S.LimitCycleBlock(a=A_CYCLE, rho=1.0, omega=0.5, beta=0.6)
    lam_cyc = cyc.lyapunov_spectrum_exact()
    lmin_cyc, lmax_cyc = float(lam_cyc.min()), float(lam_cyc.max())
    spread_cyc = lmax_cyc - lmin_cyc
    horizon = SP.resolvable_horizon(spread_cyc)

    # ---------------------------------------------------------------- part 1
    banner("PART 1 -- why the naive bound cannot be used off a fixed point")
    print(f"   LimitCycleBlock(a={A_CYCLE}) spectrum {np.round(lam_cyc, 5)}  "
          f"spread {spread_cyc:.4f}")
    print(f"   resolvable horizon for sigma_min: n ~ {horizon:.0f}\n")

    print(f"     {'n':>5s} {'sigma_min/sigma_max':>21s} {'resolvable?':>12s}")
    cond_rows = []
    smax_c, smin_c = SP.jacobian_product_logs(cyc, Z_CYCLE, 120)
    for n in (10, 20, 30, 40, 60, 100, 120):
        ratio = float(np.exp(smin_c[n - 1] - smax_c[n - 1]))
        ok = ratio > 1e-15
        print(f"     {n:5d} {ratio:21.4e} {str(ok):>12s}")
        cond_rows.append({"n": n, "sigma_ratio": ratio, "resolvable": ok})

    print(f"\n   Effect on the fitted rate (target = cycle, source = TwistBlock(s=0.30)):")
    fast = S.TwistBlock(s=0.30, omega=1.1, beta=-0.5)
    predicted_030 = float(np.log(0.30)) - lmin_cyc
    print(f"     true rate {predicted_030:+.5f}")
    print(f"     {'n_max':>7s} {'naive rate':>12s} {'sound rate':>12s}")
    stability = []
    for n_max in (100, 200, 400, 800):
        cb = CC.cocycle_bound(cyc, Z_CYCLE, fast, Z_FAST, n_max=n_max,
                              predicted_rate=predicted_030)
        print(f"     {n_max:7d} {cb.naive_rate:+12.5f} {cb.rate:+12.5f}")
        stability.append({"n_max": n_max, "naive_rate": cb.naive_rate, "sound_rate": cb.rate})

    naive_spread = float(np.ptp([r["naive_rate"] for r in stability]))
    sound_spread = float(np.ptp([r["sound_rate"] for r in stability]))
    # What condemns the naive route is that it reads the WRONG exponent, not that
    # it is jittery.  Past the horizon the SVD returns its noise floor, whose slope
    # is lambda_max, so the fit lands on log s - lambda_max instead of
    # log s - lambda_min -- off by exactly the block's spread.  Whether that floor
    # also wanders is a BLAS detail (it did on numpy 2.4, it does not on 2.5); the
    # convergent version is the more dangerous one, since a stable wrong number
    # reads as a measurement.  So we assert the error, not the variance.
    wrong_limit_030 = float(np.log(0.30)) - lam_cyc.max()
    naive_err = float(np.max([abs(r["naive_rate"] - predicted_030) for r in stability]))
    naive_to_wrong = float(np.max([abs(r["naive_rate"] - wrong_limit_030) for r in stability]))
    print(f"\n   naive rate varies by {naive_spread:.3f} across n_max; "
          f"sound rate by {sound_spread:.2e}.")
    print(f"   but the damning number is the bias: naive is off the true rate by "
          f"{naive_err:.3f}\n   (= the spread {spread_cyc:.3f}), sitting within "
          f"{naive_to_wrong:.3f} of log s - lambda_MAX = {wrong_limit_030:+.5f}.")

    print("\n   Same wrong exponent across initial conditions on the SAME cycle:")
    ic_rows = []
    for _ in range(5):
        th = rng.uniform(0.0, 2.0 * np.pi)
        z = np.array([np.cos(th), np.sin(th)])
        cb = CC.cocycle_bound(cyc, z, fast, Z_FAST, n_max=N_MAX)
        ic_rows.append({"theta": float(th), "naive_rate": cb.naive_rate, "sound_rate": cb.rate})
        print(f"     theta={th:5.2f}: naive {cb.naive_rate:+9.5f}   sound {cb.rate:+9.5f}")
    naive_ic_spread = float(np.ptp([r["naive_rate"] for r in ic_rows]))
    sound_ic_spread = float(np.ptp([r["sound_rate"] for r in ic_rows]))
    naive_ic_err = float(np.max([abs(r["naive_rate"] - predicted_030) for r in ic_rows]))

    print("\n   And beta (the radius->phase shear) is NOT the culprit -- it shifts the")
    print("   intercept, not the rate.  Analytically smax(J^n) -> sqrt((beta/2a)^2+1):")
    beta_rows = []
    for beta in BETAS:
        blk = S.LimitCycleBlock(a=A_CYCLE, rho=1.0, omega=0.5, beta=beta)
        sm, _ = SP.jacobian_product_logs(blk, Z_CYCLE, 60)
        plateau = float(np.log(np.sqrt((beta / (2 * A_CYCLE)) ** 2 + 1.0)))
        sound = fitted_rate(SP.inverse_jacobian_product_logs(blk, Z_CYCLE, N_MAX))
        print(f"     beta={beta:4.1f}: predicted plateau {plateau:+.5f}  measured {sm[-1]:+.5f}"
              f"   |  sound rate {sound:+.6f} vs -lmin {-lmin_cyc:+.6f}")
        beta_rows.append({"beta": beta, "plateau_predicted": plateau,
                          "plateau_measured": float(sm[-1]), "sound_rate": sound})

    # ---------------------------------------------------------------- part 2
    banner("PART 2 -- Lemma C's rate holds on the attractor, with the right threshold")
    threshold = abs(1.0 - 2.0 * A_CYCLE)
    print(f"   Target = the limit cycle (lambda_min = {lmin_cyc:+.5f}).  M is forced to")
    print(f"   zero iff lambda_max(source) < lambda_min(target), i.e. iff s < {threshold:.2f}.\n")
    print(f"     {'s_fast':>7s} {'lmax(src)':>10s} {'predicted':>11s} {'measured':>11s} "
          f"{'err':>10s} {'M->0?':>7s} {'expected':>9s}")
    sweep = []
    for s_fast in S_FAST:
        src = S.TwistBlock(s=s_fast, omega=1.1, beta=-0.5)
        lmax_src = float(np.log(s_fast))
        predicted = lmax_src - lmin_cyc
        cb = CC.cocycle_bound(cyc, Z_CYCLE, src, Z_FAST, n_max=N_MAX,
                              predicted_rate=predicted)
        expected = s_fast < threshold
        print(f"     {s_fast:7.2f} {lmax_src:10.4f} {predicted:+11.5f} {cb.rate:+11.5f} "
              f"{abs(cb.rate - predicted):10.2e} {str(cb.forces_M_zero):>7s} "
              f"{str(expected):>9s}")
        sweep.append({"s_fast": s_fast, "lambda_max_source": lmax_src,
                      "predicted": predicted, "measured": cb.rate,
                      "error": abs(cb.rate - predicted),
                      "forces_M_zero": cb.forces_M_zero, "expected": expected})

    max_err = max(r["error"] for r in sweep)
    threshold_ok = all(r["forces_M_zero"] == r["expected"] for r in sweep)

    # ---------------------------------------------------------------- part 2b
    banner("PART 2b -- the uniformity clause, which is what Lemma C' runs on")
    boundary = 1.0 * np.sqrt((1.0 + A_CYCLE) / A_CYCLE)
    print(f"""   Oseledets gives the rate only mu-a.e., and for a limit cycle supp mu is the
   CYCLE -- which would leave the basin untouched, exactly the omega-limit-set
   weakness that makes Route 2 of identifiability.md §5.1 useless.  A normally
   hyperbolic attracting circle instead has the SAME exponents at every basin
   point, so the bound decays at every z and M vanishes on all of Omega.

   The basin is bounded: g(r) = r(1+a - a r^2) is positive only for
   r < sqrt((1+a)/a) = {boundary:.4f}.  Sweep inside it.\n""")
    print(f"     {'r0':>7s} {'cocycle rate':>14s} {'err':>10s}")
    basin = []
    for r0 in (0.02, 0.05, 0.20, 0.50, 0.90, 1.00, 1.50, 1.90, 2.05):
        cb = CC.cocycle_bound(cyc, np.array([r0, 0.0]), fast, Z_FAST,
                              n_max=N_MAX, predicted_rate=predicted_030)
        print(f"     {r0:7.2f} {cb.rate:+14.7f} {abs(cb.rate - predicted_030):10.1e}")
        basin.append({"r0": r0, "rate": cb.rate, "error": abs(cb.rate - predicted_030)})
    basin_spread = float(np.ptp([r["rate"] for r in basin]))
    print(f"\n   rate varies by {basin_spread:.1e} over a 100x range of starting radius.")
    print(f"   Outside the basin (r0 = 3.0) the map escapes at once (r1 = -4.2) and the")
    print(f"   inverse cocycle is undefined -- the guard raises rather than returning a")
    print(f"   plausible wrong number.")
    outside_guarded = False
    try:
        SP.inverse_jacobian_product_logs(cyc, np.array([3.0, 0.0]), 50)
    except np.linalg.LinAlgError:
        outside_guarded = True

    # ---------------------------------------------------------------- part 3
    banner("PART 3 -- a neutral exponent forces the cycle to the top of the filtration")
    print(f"""   lambda_max(limit cycle) = {lmax_cyc:+.5f} exactly -- the phase direction is
   neutral.  For the cycle to be the DOMINATED module j we would need
   lambda_max(cycle) < lambda_min(f_i), i.e. 0 < lambda_min(f_i), which no
   contraction satisfies.  So it is never dominated:\n""")
    print(f"     {'partner s':>10s} {'rate (cycle as source)':>24s} {'M->0?':>7s}")
    dominated = []
    for s_other in (0.20, 0.50, 0.90):
        other = S.TwistBlock(s=s_other, omega=1.1, beta=-0.5)
        cb = CC.cocycle_bound(other, Z_FAST, cyc, Z_CYCLE, n_max=N_MAX)
        print(f"     {s_other:10.2f} {cb.rate:+24.5f} {str(cb.forces_M_zero):>7s}")
        dominated.append({"s_other": s_other, "rate": cb.rate,
                          "forces_M_zero": cb.forces_M_zero})
    never_dominated = not any(r["forces_M_zero"] for r in dominated)

    # ---------------------------------------------------------------- part 4
    banner("PART 4 -- two limit cycles can never be separated")
    c1 = S.LimitCycleBlock(a=0.30, rho=1.0, omega=0.50, beta=0.6)
    c2 = S.LimitCycleBlock(a=0.45, rho=1.0, omega=0.90, beta=-0.4)
    s1, s2 = c1.lyapunov_spectrum_exact(), c2.lyapunov_spectrum_exact()
    gap = SP.spectral_gap([s1, s2])
    print(f"   cycle 1 spectrum {np.round(s1, 5)}")
    print(f"   cycle 2 spectrum {np.round(s2, 5)}")
    print(f"   spectral_gap = {gap:.3e}  -> the modules SHARE the exponent 0, so (B4) fails\n")
    print(f"     {'direction':>22s} {'predicted':>11s} {'measured':>11s} {'err':>10s} {'M->0?':>7s}")
    pair = []
    for tag, tgt, src, zs in (("M_12  (cycle1 <- cycle2)", c1, c2, np.array([1.0, 0.2])),
                              ("M_21  (cycle2 <- cycle1)", c2, c1, np.array([1.0, 0.2]))):
        predicted = float(src.lyapunov_spectrum_exact().max()) - float(
            tgt.lyapunov_spectrum_exact().min()
        )
        cb = CC.cocycle_bound(tgt, Z_CYCLE, src, zs, n_max=N_MAX, predicted_rate=predicted)
        print(f"     {tag:>22s} {predicted:+11.5f} {cb.rate:+11.5f} "
              f"{abs(cb.rate - predicted):10.2e} {str(cb.forces_M_zero):>7s}")
        pair.append({"direction": tag, "predicted": predicted, "measured": cb.rate,
                     "error": abs(cb.rate - predicted), "forces_M_zero": cb.forces_M_zero})
    both_positive = all(r["predicted"] > 0 for r in pair)
    neither_forced = not any(r["forces_M_zero"] for r in pair)

    # ---------------------------------------------------------------- part 5
    banner("PART 5 -- exp05's fixed-point numbers are unaffected")
    f1 = S.TwistBlock(s=0.95, omega=0.40, beta=0.60)
    f2 = S.TwistBlock(s=0.70, omega=1.10, beta=-0.50)
    pred05 = np.log(0.70) - np.log(0.95)
    cb05 = CC.cocycle_bound(f1, np.array([0.9, 0.1]), f2, np.array([0.7, 0.3]),
                            n_max=N_MAX, predicted_rate=pred05)
    print(f"   TwistBlock spread = 0 exactly, so the horizon is infinite and the naive")
    print(f"   route was always valid there:")
    print(f"     predicted {pred05:+.7f}   sound {cb05.rate:+.7f}   naive {cb05.naive_rate:+.7f}")
    print(f"     n_resolvable = {cb05.n_resolvable}   naive_route_valid = {cb05.naive_route_valid}")
    exp05_intact = (
        abs(cb05.rate - pred05) < 1e-6
        and abs(cb05.naive_rate - pred05) < 1e-6
        and cb05.naive_route_valid
    )

    banner("VERDICTS")
    checks = [
        (
            naive_err > 0.5 * spread_cyc
            and naive_ic_err > 0.5 * spread_cyc
            and naive_to_wrong < 0.2 * spread_cyc
            and max(sound_spread, sound_ic_spread) < 1e-9,
            f"the naive sigma_min bound is not a measurement off a fixed point: it reads "
            f"lambda_MAX, missing the true rate by {naive_err:.2f} across n_max and "
            f"{naive_ic_err:.2f} across initial conditions on the same cycle (the block's "
            f"spread is {spread_cyc:.2f}), while the sound bound is stable to "
            f"{max(sound_spread, sound_ic_spread):.1e}",
        ),
        (
            all(abs(r["plateau_measured"] - r["plateau_predicted"]) < 1e-6 for r in beta_rows)
            and float(np.ptp([r["sound_rate"] for r in beta_rows])) < 1e-6,
            "the shear beta shifts the intercept by exactly sqrt((beta/2a)^2+1) and leaves "
            "the rate invariant -- non-normality was not the problem, conditioning was",
        ),
        (
            max_err < 1e-9,
            f"Lemma C's predicted rate lambda_max(f_j) - lambda_min(f~_i) holds on a genuine "
            f"limit-cycle attractor to {max_err:.1e} across the sweep -- the argument does "
            f"not need the fixed point",
        ),
        (
            threshold_ok,
            f"and the threshold sits exactly at the radial multiplier |1-2a| = {threshold:.2f}: "
            f"M is forced to zero for every s below it and for none above",
        ),
        (
            basin_spread < 1e-12 and outside_guarded,
            f"the cycle's exponents are uniform over its whole basin (rate constant to "
            f"{basin_spread:.1e} across a 100x range of starting radius), which is the "
            f"hypothesis Lemma C' needs to conclude on all of Omega and not merely on "
            f"supp mu -- and outside the basin the guard raises instead of guessing",
        ),
        (
            never_dominated and abs(lmax_cyc) < 1e-12,
            "a limit cycle's neutral exponent 0 makes it undominatable, so an oscillatory "
            "module sits at the TOP of the filtration or is not separated at all",
        ),
        (
            gap < 1e-15 and both_positive and neither_forced,
            "two limit cycles share the exponent 0, so (B4) fails and neither cross-derivative "
            "is forced to zero -- the case that produced irreconcilable numbers is outside "
            "Lemma C's hypotheses, not a counterexample to it",
        ),
        (
            exp05_intact,
            f"exp05's fixed-point measurements are untouched: TwistBlock spread is 0, the "
            f"horizon is infinite, and both routes give {cb05.rate:+.7f}",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    save(
        "exp08_attractor_cocycle",
        {
            "seed": SEED, "n_max": N_MAX, "a_cycle": A_CYCLE,
            "cycle_spectrum": lam_cyc.tolist(), "cycle_spread": spread_cyc,
            "resolvable_horizon": horizon,
            "conditioning": cond_rows,
            "rate_vs_n_max": stability,
            "rate_vs_initial_condition": ic_rows,
            "naive_rate_spread_n_max": naive_spread,
            "naive_rate_spread_ic": naive_ic_spread,
            "sound_rate_spread_n_max": sound_spread,
            "sound_rate_spread_ic": sound_ic_spread,
            # the bias is what condemns the naive route; the spreads above are
            # BLAS-dependent and are recorded for provenance, not asserted on
            "naive_rate_error_n_max": naive_err,
            "naive_rate_error_ic": naive_ic_err,
            "naive_wrong_limit": wrong_limit_030,
            "naive_dist_to_wrong_limit": naive_to_wrong,
            "beta_invariance": beta_rows,
            "threshold": threshold,
            "gap_sweep": sweep, "max_rate_error": max_err,
            "basin_boundary": float(boundary),
            "basin_uniformity": basin, "basin_rate_spread": basin_spread,
            "outside_basin_guarded": outside_guarded,
            "cycle_as_dominated": dominated,
            "two_cycles": {"spectral_gap": gap, "directions": pair},
            "exp05_recheck": {"predicted": float(pred05), "sound": cb05.rate,
                              "naive": cb05.naive_rate,
                              "n_resolvable": cb05.n_resolvable},
            "all_passed": passed,
            "checks": [{"passed": ok, "claim": m} for ok, m in checks],
        },
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
