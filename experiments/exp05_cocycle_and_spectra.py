"""Experiment 5 -- the two proof-level defects, §3.3 and §3.4.

Part 1 (§3.4): Assumption 4 of the draft -- "Jacobian spectra generically
distinct" -- is not a property of the system at all.  Sweeping how much of state
space the trajectories visit shows the pointwise Jacobian spectra of the two
modules are well separated on a small region and *cross* on a larger one, while
the Lyapunov spectra are unchanged throughout.  So the pointwise assumption
depends on the data-collection regime and the dichotomy-spectrum replacement
does not.  This also sharpens §3.6: the support caveat is not only about where
h is constrained, it decides whether the hypotheses hold.

Part 2 (§3.3): the draft turned a cocycle relation into a pointwise Sylvester
equation by dropping an argument shift.  Two consequences, both measured:

  (a) At a point where the pointwise Jacobian spectra cross, the pointwise
      Sylvester operator is singular -- it has nonzero solutions M, so the
      draft's step cannot conclude M = 0 there.  The cocycle version still can,
      because it only needs the *averaged* rates to differ.
  (b) The cocycle bound B_n decays at exactly the predicted rate
      lambda_max(f_2) - lambda_min(f~_1) when there is a gap, and stalls at rate
      ~0 when the two modules share an exponent.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import cocycle as CC
from idyn import spectra as SP
from idyn import systems as S

SEED = 0
RADII = (0.6, 1.0, 1.4, 1.8, 2.2)
N_MAX = 400


def pointwise_spectral_distance(f1, f2, Z1, Z2) -> tuple[float, float, tuple[int, int]]:
    """min, mean, and argmin over pairs of dist(spec Df_1(z1), spec Df_2(z2)).

    All pairs, not the trajectory diagonal.  Initial conditions are drawn
    independently per module, so the visited set really is the product
    ``visited(module 1) x visited(module 2)`` and every pair here is a state the
    system actually reaches.
    """
    s1 = np.array([np.linalg.eigvals(f1.jacobian(p)) for p in Z1])
    s2 = np.array([np.linalg.eigvals(f2.jacobian(p)) for p in Z2])
    D = np.abs(s1[:, None, :, None] - s2[None, :, None, :]).min(axis=(2, 3))
    i, j = np.unravel_index(int(D.argmin()), D.shape)
    return float(D.min()), float(D.mean()), (int(i), int(j))


def sylvester_smallest_singular_value(A: np.ndarray, B: np.ndarray) -> float:
    """sigma_min of  M -> A M - M B,  which is 0 iff spec(A) and spec(B) meet.

    This is the operator the draft's pointwise step needs to be injective.
    """
    n, m = A.shape[0], B.shape[0]
    L = np.kron(np.eye(m), A) - np.kron(B.T, np.eye(n))
    return float(np.linalg.svd(L, compute_uv=False)[-1])


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 5 -- §3.4 (spectra) and §3.3 (cocycle), measured")

    sys = S.two_oscillator_system(s=(0.95, 0.70), omega=(0.40, 1.10), beta=(0.60, -0.50))
    f1, f2 = sys.blocks
    lyap_gap_exact = abs(np.log(0.95) - np.log(0.70))

    # ---------------------------------------------------------------- part 1
    banner("PART 1 (§3.4) -- pointwise Jacobian spectra vs Lyapunov spectra")
    print("   The draft assumes the former are distinct.  Sweep the visited region:\n")
    print(f"     {'radius':>8s} {'max |z| visited':>16s} {'min pointwise dist':>20s} "
          f"{'Lyapunov gap':>14s}")
    rows = []
    for radius in RADII:
        z0 = S.sample_initial_conditions(4, 300, rng, radius=radius)
        Z = sys.simulate(z0, 25).reshape(-1, 4)
        idx = rng.choice(len(Z), 900, replace=False)
        dmin, dmean, _ = pointwise_spectral_distance(f1, f2, Z[idx, :2], Z[idx, 2:])
        ms = SP.module_lyapunov_spectra(sys, z0[:8], T=600, warmup=150)
        maxr = float(max(np.linalg.norm(Z[:, :2], axis=1).max(), np.linalg.norm(Z[:, 2:], axis=1).max()))
        print(f"     {radius:8.1f} {maxr:16.3f} {dmin:20.3e} {ms.gap:14.4f}")
        rows.append({"radius": radius, "max_visited_norm": maxr, "pointwise_min_distance": dmin,
                     "pointwise_mean_distance": dmean, "lyapunov_gap": ms.gap})

    pw = np.array([r["pointwise_min_distance"] for r in rows])
    lg = np.array([r["lyapunov_gap"] for r in rows])
    print(f"\n   pointwise minimum varies by a factor {pw.max() / max(pw.min(), 1e-30):.3g} "
          f"across the sweep;")
    print(f"   the Lyapunov gap varies by {lg.max() - lg.min():.2e} (exact value {lyap_gap_exact:.4f}).")
    print("   => Assumption 4 is a property of the data-collection regime, not the system.")

    # ---------------------------------------------------------------- part 2a
    banner("PART 2a (§3.3) -- the pointwise Sylvester step fails where spectra cross")
    z_small = S.sample_initial_conditions(4, 300, rng, radius=0.6)
    z_large = S.sample_initial_conditions(4, 300, rng, radius=2.2)
    sv_rows = []
    for label, z0 in (("small region (r=0.6)", z_small), ("large region (r=2.2)", z_large)):
        Z = sys.simulate(z0, 25).reshape(-1, 4)
        idx = rng.choice(len(Z), 900, replace=False)
        Z1, Z2 = Z[idx, :2], Z[idx, 2:]
        # locate the worst pair rather than hoping to sample it: the crossing
        # set has codimension 1, so random sampling would essentially never
        # land on it and would understate the failure.
        dmin, _, (i, j) = pointwise_spectral_distance(f1, f2, Z1, Z2)
        sv_worst = sylvester_smallest_singular_value(f1.jacobian(Z1[i]), f2.jacobian(Z2[j]))
        sv_typ = np.median(
            [
                sylvester_smallest_singular_value(f1.jacobian(Z1[k]), f2.jacobian(Z2[k]))
                for k in range(300)
            ]
        )
        print(f"   {label}: closest spectra {dmin:.3e} at z1={np.round(Z1[i], 3)}, "
              f"z2={np.round(Z2[j], 3)}")
        print(f"      sigma_min of the Sylvester operator there: {sv_worst:.3e} "
              f"(typical pair: {sv_typ:.3e})")
        sv_rows.append({"region": label, "closest_spectra": dmin, "sigma_min_worst": float(sv_worst),
                        "sigma_min_typical": float(sv_typ),
                        "z1": Z1[i].tolist(), "z2": Z2[j].tolist()})
    print("\n   A near-zero sigma_min means the pointwise equation M Df_2 = Df~_1 M admits")
    print("   nonzero M, so the draft's argument cannot conclude M = 0 at those points.")

    # ---------------------------------------------------------------- part 2b
    banner("PART 2b (§3.3) -- the cocycle bound, with and without a spectral gap")
    z1 = np.array([0.9, 0.1])
    z2 = np.array([0.7, 0.3])

    predicted = np.log(0.70) - np.log(0.95)
    gap_case = CC.cocycle_bound(f1, z1, f2, z2, n_max=N_MAX, predicted_rate=predicted)
    logM = CC.propagate_M(f1, z1, f2, z2, np.eye(2), n_max=N_MAX)
    print(f"   WITH gap    (s = 0.95 vs 0.70): {gap_case}")
    print(f"                measured rate {gap_case.rate:+.5f}  predicted {predicted:+.5f}  "
          f"error {abs(gap_case.rate - predicted):.2e}")
    print(f"                log||M_n||: n=1 {logM[0]:+.3f}, n={N_MAX} {logM[-1]:+.3f} "
          f"-> M is forced to zero")

    g1 = S.TwistBlock(s=0.85, omega=0.40, beta=0.60)
    g2 = S.TwistBlock(s=0.85, omega=1.10, beta=-0.50)
    no_gap = CC.cocycle_bound(g1, z1, g2, z2, n_max=N_MAX, predicted_rate=0.0)
    logM_ng = CC.propagate_M(g1, z1, g2, z2, np.eye(2), n_max=N_MAX)
    print(f"\n   WITHOUT gap (s = 0.85 vs 0.85): {no_gap}")
    print(f"                measured rate {no_gap.rate:+.5f}  predicted {0.0:+.5f}")
    print(f"                log||M_n||: n=1 {logM_ng[0]:+.3f}, n={N_MAX} {logM_ng[-1]:+.3f} "
          f"-> M is NOT forced to zero")

    # ---------------------------------------------------------------- part 2c
    banner("PART 2c (NEW) -- the cocycle argument cannot close in both directions")
    print("""   Block-diagonality of h needs BOTH cross-derivative blocks to vanish:
       M_12 = dh_1/dz_2  needs  lam_max(f_2) < lam_min(f~_1)
       M_21 = dh_2/dz_1  needs  lam_max(f_1) < lam_min(f~_2)
   Once the modules are correctly matched, f~_i is conjugate to f_i, so the two
   requirements read lam_max(f_2) < lam_min(f_1) and lam_max(f_1) < lam_min(f_2).
   Chaining them gives lam_max(f_2) < lam_min(f_1) <= lam_max(f_1) < lam_min(f_2)
   <= lam_max(f_2), a contradiction.  They can never both hold.\n""")

    print(f"     {'s_2':>6s} {'rate(M_12)':>12s} {'rate(M_21)':>12s} {'sum':>10s} {'both < 0?':>10s}")
    two_sided = []
    for s2 in (0.30, 0.50, 0.70, 0.90):
        g = S.TwistBlock(s=s2, omega=1.1, beta=-0.5)
        r12 = CC.cocycle_bound(f1, z1, g, z2, n_max=300).rate
        r21 = CC.cocycle_bound(g, z2, f1, z1, n_max=300).rate
        both = bool(r12 < 0 and r21 < 0)
        print(f"     {s2:6.2f} {r12:12.5f} {r21:12.5f} {r12 + r21:10.2e} {str(both):>10s}")
        two_sided.append({"s2": s2, "rate_M12": r12, "rate_M21": r21, "both_negative": both})

    print("""
   Consequence: the §3.3 fix, even executed correctly, yields a TRIANGULAR h
   (a skew product: h_1 depends on z_1 only, h_2 on both), not a block-diagonal
   one.  Recovering the second direction needs a genuinely different argument.
   This is a new blocking issue, not one of the six in CLAUDE.md §3.""")

    banner("VERDICTS")
    checks = [
        (
            pw.min() < 1e-2 < pw.max(),
            f"pointwise Jacobian spectra are separated on a small visited region "
            f"({pw.max():.2e}) and cross on a larger one ({pw.min():.2e}) -- "
            "Assumption 4 as written is unusable (§3.4)",
        ),
        (
            float(lg.max() - lg.min()) < 1e-3 and abs(lg.mean() - lyap_gap_exact) < 1e-3,
            f"the Lyapunov gap is invariant across the same sweep "
            f"({lg.mean():.4f} vs exact {lyap_gap_exact:.4f}) -- the dichotomy-spectrum "
            "replacement is well posed",
        ),
        (
            sv_rows[1]["sigma_min_worst"] < 1e-2 <= sv_rows[0]["sigma_min_worst"],
            f"the pointwise Sylvester operator is well conditioned on the small region "
            f"(sigma_min {sv_rows[0]['sigma_min_worst']:.2e}) and singular on the larger one "
            f"({sv_rows[1]['sigma_min_worst']:.2e}), so the draft's pointwise step "
            "genuinely fails there (§3.3)",
        ),
        (
            abs(gap_case.rate - predicted) < 1e-3 and gap_case.forces_M_zero,
            f"with a gap the cocycle bound decays at the predicted rate "
            f"({gap_case.rate:+.5f} vs {predicted:+.5f}), forcing M = 0",
        ),
        (
            abs(no_gap.rate) < 1e-3 and not no_gap.forces_M_zero,
            f"without a gap the bound stalls (rate {no_gap.rate:+.5f}) and M is not "
            "forced to zero -- the gap is doing the work, not the algebra",
        ),
        (
            not any(t["both_negative"] for t in two_sided)
            and all(abs(t["rate_M12"] + t["rate_M21"]) < 1e-6 for t in two_sided),
            "NEW BLOCKING ISSUE: the two cross-derivative rates are exact negatives, so "
            "the cocycle argument can never kill both -- it yields a triangular h, "
            "not a block-diagonal one",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    save(
        "exp05_cocycle_and_spectra",
        {"seed": SEED, "radii": list(RADII), "n_max": N_MAX,
         "spectra_sweep": rows, "sylvester": sv_rows,
         "lyapunov_gap_exact": float(lyap_gap_exact),
         "cocycle_gap": {"rate": gap_case.rate, "predicted": predicted,
                         "log_M_final": float(logM[-1])},
         "cocycle_no_gap": {"rate": no_gap.rate, "predicted": 0.0,
                            "log_M_final": float(logM_ng[-1])},
         "two_sided_obstruction": two_sided,
         "all_passed": passed, "checks": [{"passed": ok, "claim": m} for ok, m in checks]},
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
