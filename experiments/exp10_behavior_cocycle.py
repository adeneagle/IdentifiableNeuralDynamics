"""Experiment 10 -- the B∘C composition: behaviour + one-sided gap = block-diagonal.

Route C's cocycle argument supplies **one** cross-derivative and provably not both
(``counterexamples.md`` §3): a spectral gap is one-sided, and the two-sided version
is a contradiction, so dynamics alone give a *triangular* h.  Route B supplies the
*other* cross-derivative from behaviour, with no spectral or regularity hypothesis:
the u-invariant subspace is canonical, so the invariant block's reconstruction
cannot depend on the varying block (``M_BA = d h_B / d z^A = 0``).  Neither alone
gives block-diagonality; together they do -- **if** the two kills land on
*different* cross-derivatives, which is the alignment condition Part 4 pins down.

Setup: two 2-D modules.  Block A is behaviour-**varying** (its initial law is
conditioned on u) and spectrally **dominant** (slow, ``s = 0.9``).  Block B is
behaviour-**invariant** and **dominated** (fast, ``s = 0.5``).  Then behaviour must
kill ``M_BA`` and the cocycle can kill ``M_AB`` -- complementary.

  1. Behaviour detects a ``M_BA`` leak: h_B = z^B + eps z^A becomes u-dependent,
     net score rising from ~0 with eps.  (A valid representation forces eps = 0.)
  2. The cocycle kills ``M_AB`` under the one-sided gap (A dominates B), and
     cannot kill ``M_BA`` (reverse rate is positive) -- so the two tools are
     genuinely complementary, not redundant.
  3. Truth table: block-diagonal passes both tests; each one-sided leak is caught
     by exactly one test; neither test alone certifies block-diagonality.
  4. Alignment: if the varying block is the *dominated* one, both tools kill the
     same cross-derivative and ``M_AB`` survives -- only triangular results.  So
     block-diagonal needs varying = dominant.
  5. Prop. 1 (Khemakhem et al., verified in route_a_assessment §6.1): a mean-only
     behavioural signal is detectable at O(eps) but a variance leak hides from a
     mean-only detector -- variance modulation (or >=2 statistics) is needed to
     reach permutation-level structure.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import behavior as BEH
from idyn import cocycle as CC
from idyn import systems as S

SEED = 0
U_LEVELS = np.array([0, 1, 2, 3, 4])
N_PER_U = 12000
N_MAX = 300
EPS_SWEEP = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
Z_A = np.array([0.7, 0.2])
Z_B = np.array([0.5, 0.3])


def net_u_dependence(sample, w, use="total") -> float:
    """u-dependence of block ``w`` minus the invariant-block floor (pure noise)."""
    floor = BEH.block_u_dependence(sample.Z[:, sample.slice_b], sample.U)
    d = BEH.block_u_dependence(w, sample.U)
    return getattr(d, use) - getattr(floor, use)


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 10 -- B+C: behaviour kills one cross-derivative, the gap the other")

    # A: varying + dominant (slow s=0.9); B: invariant + dominated (fast s=0.5)
    f_a = S.TwistBlock(s=0.90, omega=0.40, beta=0.6)
    f_b = S.TwistBlock(s=0.50, omega=1.10, beta=-0.5)
    lam_a, lam_b = np.log(0.90), np.log(0.50)
    sample = BEH.conditioned_initial_conditions(2, 2, U_LEVELS, N_PER_U, rng, mode="variance")
    za, zb = sample.Z[:, sample.slice_a], sample.Z[:, sample.slice_b]

    # ---------------------------------------------------------------- part 1
    banner("PART 1 (behaviour) -- a M_BA leak makes the invariant block u-dependent")
    print("   h_B = z^B + eps * z^A.  Since p(z^A|u) moves with u and p(z^B|u) does not,")
    print("   any eps != 0 makes h_B u-dependent -- behaviour rejects it.\n")
    print(f"   {'eps':>6s} {'net u-dependence of h_B':>26s}")
    beh_rows = []
    for eps in EPS_SWEEP:
        nd = net_u_dependence(sample, zb + eps * za)
        print(f"   {eps:6.2f} {nd:26.4f}")
        beh_rows.append({"eps": eps, "net_u_dependence": nd})
    beh_zero = beh_rows[0]["net_u_dependence"]
    beh_leak = beh_rows[-1]["net_u_dependence"]
    beh_monotone = all(
        beh_rows[i]["net_u_dependence"] <= beh_rows[i + 1]["net_u_dependence"] + 1e-3
        for i in range(len(beh_rows) - 1)
    )
    print(f"\n   eps=0: {beh_zero:+.4f} (~0, invariant);  eps=0.5: {beh_leak:+.4f} (clearly u-dependent)")

    # ---------------------------------------------------------------- part 2
    banner("PART 2 (dynamics) -- the cocycle kills M_AB, and cannot kill M_BA")
    predicted_ab = lam_b - lam_a  # rate for M_AB = d h_A / d z_B (A dominates B)
    predicted_ba = lam_a - lam_b  # rate for M_BA (reverse orientation)
    cb_ab = CC.cocycle_bound(f_a, Z_A, f_b, Z_B, n_max=N_MAX, predicted_rate=predicted_ab)
    cb_ba = CC.cocycle_bound(f_b, Z_B, f_a, Z_A, n_max=N_MAX, predicted_rate=predicted_ba)
    print(f"   M_AB = d h_A/d z_B :  rate {cb_ab.rate:+.4f} (pred {predicted_ab:+.4f})  "
          f"forces_M_zero={cb_ab.forces_M_zero}")
    print(f"   M_BA = d h_B/d z_A :  rate {cb_ba.rate:+.4f} (pred {predicted_ba:+.4f})  "
          f"forces_M_zero={cb_ba.forces_M_zero}")
    print("""
   The gap kills M_AB (A dominant) and leaves M_BA alone -- exactly the one
   behaviour kills.  The two tools are complementary: dynamics cannot supply the
   behavioural direction (that is the two-sided obstruction, counterexamples.md sec 3).""")
    complementary = cb_ab.forces_M_zero and not cb_ba.forces_M_zero

    # ---------------------------------------------------------------- part 3
    banner("PART 3 -- truth table: only block-diagonal passes both tests")
    beh_thresh = 0.02   # net u-dependence above this = behaviour rejects (leak present)
    print(f"   behaviour rejects when net u-dependence(h_B) > {beh_thresh}; "
          f"dynamics rejects a nonzero M_AB (gap forces it to 0).\n")
    print(f"   {'candidate h':>22s} {'eps_BA':>7s} {'eps_AB':>7s} "
          f"{'behaviour':>11s} {'dynamics':>10s} {'valid?':>7s}")
    table = []
    for name, e_ba, e_ab in (
        ("block-diagonal", 0.0, 0.0),
        ("leak M_BA only", 0.3, 0.0),
        ("leak M_AB only", 0.0, 0.3),
        ("both (full mix)", 0.3, 0.3),
    ):
        beh_reject = net_u_dependence(sample, zb + e_ba * za) > beh_thresh
        dyn_reject = e_ab != 0.0 and cb_ab.forces_M_zero  # a nonzero M_AB is inconsistent under the gap
        valid = not beh_reject and not dyn_reject
        print(f"   {name:>22s} {e_ba:7.2f} {e_ab:7.2f} "
              f"{('reject' if beh_reject else 'ok'):>11s} "
              f"{('reject' if dyn_reject else 'ok'):>10s} {str(valid):>7s}")
        table.append({"candidate": name, "eps_BA": e_ba, "eps_AB": e_ab,
                      "behaviour_rejects": beh_reject, "dynamics_rejects": dyn_reject,
                      "valid": valid})
    only_bd_valid = (table[0]["valid"] and not any(r["valid"] for r in table[1:]))
    beh_covers_ba = table[1]["behaviour_rejects"] and not table[1]["dynamics_rejects"]
    dyn_covers_ab = table[2]["dynamics_rejects"] and not table[2]["behaviour_rejects"]

    # ---------------------------------------------------------------- part 4
    banner("PART 4 -- alignment: block-diagonal needs varying = dominant")
    print("""   Swap the roles: let the VARYING block be the spectrally DOMINATED (fast) one.
   Behaviour still kills M_BA; but now the gap ALSO kills M_BA (dominant block is
   the invariant one), redundantly, and M_AB survives -- only triangular.\n""")
    # now A' (first block) = varying + dominated (fast), B' = invariant + dominant (slow)
    f_a2 = S.TwistBlock(s=0.50, omega=0.40, beta=0.6)   # varying + fast (dominated)
    f_b2 = S.TwistBlock(s=0.90, omega=1.10, beta=-0.5)  # invariant + slow (dominant)
    # cocycle for M_AB (d h_A/d z_B): A' dominated so this should NOT vanish
    cb_ab2 = CC.cocycle_bound(f_a2, Z_A, f_b2, Z_B, n_max=N_MAX)
    # cocycle for M_BA (d h_B/d z_A): B' dominant so this DOES vanish -- same as behaviour
    cb_ba2 = CC.cocycle_bound(f_b2, Z_B, f_a2, Z_A, n_max=N_MAX)
    print(f"   M_AB rate {cb_ab2.rate:+.4f}  forces_M_zero={cb_ab2.forces_M_zero}  (survives)")
    print(f"   M_BA rate {cb_ba2.rate:+.4f}  forces_M_zero={cb_ba2.forces_M_zero}  "
          f"(killed by the gap -- but behaviour already kills it)")
    misaligned_triangular = (not cb_ab2.forces_M_zero) and cb_ba2.forces_M_zero
    print("""
   Both tools now target M_BA; M_AB is unconstrained by either.  So a leak in M_AB
   survives and the result is triangular, not block-diagonal.  Block-diagonality
   requires the behaviour-varying block to be the spectrally dominant one.""")

    # ---------------------------------------------------------------- part 5
    banner("PART 5 -- Prop. 1: mean-only modulation caps at a linear indeterminacy")
    mean_s = BEH.conditioned_initial_conditions(2, 2, U_LEVELS, N_PER_U, rng, mode="mean")
    za_m, zb_m = mean_s.Z[:, mean_s.slice_a], mean_s.Z[:, mean_s.slice_b]
    print("   mean modulation: a M_BA leak is detectable at O(eps) (linear, sensitive):")
    print(f"     {'eps':>6s} {'net mean-dependence':>20s}")
    mean_rows = []
    for eps in (0.0, 0.02, 0.05, 0.1):
        floor = BEH.block_u_dependence(zb_m, mean_s.U).mean_variation
        nd = BEH.block_u_dependence(zb_m + eps * za_m, mean_s.U).mean_variation - floor
        print(f"     {eps:6.2f} {nd:20.5f}")
        mean_rows.append({"eps": eps, "net_mean_dependence": nd})
    # but a mean-only detector misses a variance leak -> the cap
    var_leak = zb + 0.3 * za  # variance-modulated leak
    floor_v = BEH.block_u_dependence(zb, sample.U)
    total_catch = BEH.block_u_dependence(var_leak, sample.U).total - floor_v.total
    mean_only_miss = BEH.block_u_dependence(var_leak, sample.U).mean_variation - floor_v.mean_variation
    print(f"\n   a variance leak (eps=0.3): full detector sees {total_catch:+.4f}, "
          f"mean-only sees {mean_only_miss:+.4f} (misses it)")
    print("   => mean-only behaviour cannot resolve variance/rotation structure (Prop. 1 cap);")
    print("      variance modulation is what reaches permutation-level identifiability.")
    prop1 = (mean_rows[-1]["net_mean_dependence"] > 0.01) and (abs(mean_only_miss) < 0.3 * total_catch)

    banner("VERDICTS")
    checks = [
        (
            abs(beh_zero) < 0.01 and beh_leak > 0.05 and beh_monotone,
            f"behaviour detects a M_BA leak: net u-dependence of h_B is ~0 at eps=0 "
            f"({beh_zero:+.4f}) and rises monotonically to {beh_leak:+.4f} at eps=0.5 -- the "
            f"invariant subspace is canonical, so a valid h has M_BA = 0",
        ),
        (
            complementary,
            f"the cocycle kills M_AB (rate {cb_ab.rate:+.4f}) and cannot kill M_BA "
            f"(rate {cb_ba.rate:+.4f} > 0) -- dynamics and behaviour are complementary, each "
            f"supplying the cross-derivative the other cannot",
        ),
        (
            only_bd_valid and beh_covers_ba and dyn_covers_ab,
            "only the block-diagonal h passes both tests; the M_BA leak is caught by behaviour "
            "alone and the M_AB leak by dynamics alone -- neither ingredient suffices, both "
            "together certify block-diagonality",
        ),
        (
            misaligned_triangular,
            "alignment matters: when the varying block is the dominated one, both tools kill "
            "M_BA and M_AB survives -- only triangular. Block-diagonal needs varying = dominant",
        ),
        (
            prop1,
            "Prop. 1: mean modulation gives O(eps) detection but a mean-only detector misses a "
            "variance leak -- variance modulation (or >=2 statistics) is required for "
            "permutation-level structure",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    save(
        "exp10_behavior_cocycle",
        {
            "seed": SEED, "u_levels": U_LEVELS.tolist(), "n_per_u": N_PER_U, "n_max": N_MAX,
            "modules": {"A_varying_dominant_s": 0.90, "B_invariant_dominated_s": 0.50,
                        "lambda_A": lam_a, "lambda_B": lam_b},
            "behaviour_leak_sweep": beh_rows,
            "cocycle": {"rate_M_AB": cb_ab.rate, "predicted_M_AB": predicted_ab,
                        "rate_M_BA": cb_ba.rate, "predicted_M_BA": predicted_ba,
                        "complementary": complementary},
            "truth_table": table,
            "alignment": {"rate_M_AB_misaligned": cb_ab2.rate,
                          "M_AB_survives": not cb_ab2.forces_M_zero,
                          "rate_M_BA_misaligned": cb_ba2.rate,
                          "only_triangular": misaligned_triangular},
            "prop1": {"mean_sweep": mean_rows, "variance_leak_full": total_catch,
                      "variance_leak_mean_only": mean_only_miss},
            "all_passed": passed,
            "checks": [{"passed": ok, "claim": m} for ok, m in checks],
        },
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
