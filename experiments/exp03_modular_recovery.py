"""Experiment 3 -- modular recovery, POSITIVE CONTROL (CLAUDE.md §4 step 6).

Two 2-D nonlinear oscillators with well-separated Lyapunov exponents.  Each
module is indecomposable (its linearisation at the fixed point is a scaled
rotation, one real Jordan block) and their spectra are disjoint, so (A1) and
(A2) hold and the theory predicts the partition *is* recovered.

This is the companion to experiment 2.  Read them together: experiment 2 shows
the method reports non-uniqueness when the truth is non-unique, so a positive
result here is not just the metric being easy to satisfy.

Also fits an unconstrained model on the same data, to show what is lost without
the modular constraint -- the unconstrained fit has no partition to recover and
its h is an arbitrary element of GL(d).
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import linear as L
from idyn import metrics as MT
from idyn import spectra as SP
from idyn import systems as S
from idyn.models import ModelConfig
from idyn.train import TrainConfig, fit, fit_many, make_dataset

SEED = 0
N_RESTARTS = 8
S_VALUES = (0.95, 0.70)


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 3 -- modular recovery with separated exponents (POSITIVE CONTROL)")

    sys = S.two_oscillator_system(s=S_VALUES, omega=(0.40, 1.10), beta=(0.60, -0.50))
    partition = sys.partition

    # ---- the hypotheses, verified rather than assumed --------------------
    z0s = S.sample_initial_conditions(4, 16, rng, radius=1.0)
    ms = SP.module_lyapunov_spectra(sys, z0s, T=1000, warmup=200)
    exact_gap = abs(np.log(S_VALUES[0]) - np.log(S_VALUES[1]))
    print(f"\n   Lyapunov spectra: {ms.summary()}")
    print(f"   exact gap |log s1 - log s2| = {exact_gap:.4f}")

    # (A1)/(A2) are checked on the linearisation at the common fixed point 0,
    # which is where the linear theory applies verbatim.
    J0 = sys.jacobian(np.full(4, 1e-6))
    cert = L.certify_finest_decomposition(J0, partition)
    print(f"   linearisation at 0: {cert.summary()}")

    # ---- data -------------------------------------------------------------
    X, Z, dec = make_dataset(sys, n_obs=10, n_traj=256, T=25, rng=rng, radius=1.0)
    print(f"\n   data: X {X.shape}, latents {Z.shape}")

    # ---- modular fit ------------------------------------------------------
    print(f"\n   modular model, partition {partition}, {N_RESTARTS} restarts:")
    mcfg = ModelConfig(n_obs=10, d=4, partition=partition, decoder="linear", encoder="linear")
    tcfg = TrainConfig(steps=2500, lr=3e-3, batch=64, seed=SEED)
    fits = fit_many(X, mcfg, tcfg, n_restarts=N_RESTARTS)

    reports, assignments, losses = [], [], []
    for f in fits:
        r = MT.recovery_report(Z, f.z_fit, partition, partition)
        A = MT.fit_linear_relation(Z, f.z_fit)
        reports.append(r)
        assignments.append(MT.coordinate_pairing(A, partition))
        losses.append(f.fit_quality)
        print(f"     seed {f.seed:5d}: fit_quality {f.fit_quality:.3e}  {r.summary()}")

    uniq = MT.nonuniqueness_report(assignments, losses, rel_tol=2.0)

    # Restarts that failed to converge say nothing about identifiability, so the
    # recovery claim is scored on near-optimal fits only -- the same filter
    # experiment 2 uses.  All-restart numbers are reported alongside so the
    # filter cannot quietly manufacture the result.
    losses_arr = np.array(losses)
    near = losses_arr <= losses_arr.min() * 3.0
    on_block = np.array([r.on_block_fraction for r in reports])
    n_recovered = int(sum(r.recovered for r in reports))
    n_recovered_near = int(sum(r.recovered for r, k in zip(reports, near) if k))
    n_near = int(near.sum())

    print(f"\n   on-block fraction: mean {on_block.mean():.4f}, min {on_block.min():.4f} "
          f"(chance {reports[0].chance_level:.4f})")
    print(f"   converged restarts (within 3x of best loss): {n_near}/{N_RESTARTS}")
    print(f"   partition recovered: {n_recovered_near}/{n_near} converged, "
          f"{n_recovered}/{N_RESTARTS} overall")
    print(f"   non-converged restarts: fit_quality "
          f"{[f'{v:.1e}' for v, k in zip(losses, near) if not k]} vs best {losses_arr.min():.1e}")
    print(f"   {uniq.summary()}")

    # ---- unconstrained fit, same data -------------------------------------
    print("\n   unconstrained model (no partition to recover), same data:")
    ucfg = ModelConfig(n_obs=10, d=4, partition=None, decoder="linear", encoder="linear")
    ufit = fit(X, ucfg, TrainConfig(steps=2500, lr=3e-3, batch=64, seed=SEED))
    u_rep = MT.recovery_report(Z, ufit.z_fit, partition, partition)
    print(f"     fit_quality {ufit.fit_quality:.3e}  {u_rep.summary()}")
    print("     (its h is an arbitrary element of GL(d); a high on-block fraction here")
    print("      would be coincidence, and a low one is the expected outcome)")

    banner("VERDICTS")
    checks = [
        (ms.gap > 0.1, f"modules have separated Lyapunov spectra (gap {ms.gap:.4f})"),
        (cert.canonical, "linearisation at the fixed point satisfies (A1) and (A2)"),
        (
            n_recovered_near == n_near,
            f"modular fit recovers the partition in {n_recovered_near}/{n_near} converged "
            f"restarts ({n_recovered}/{N_RESTARTS} overall; mean on-block "
            f"{on_block.mean():.4f} vs chance {reports[0].chance_level:.4f})",
        ),
        (
            uniq.n_distinct == 1,
            f"the recovered grouping is unique across near-optimal restarts {uniq.counts} "
            "-- the contrast with experiment 2, which is non-unique on the same "
            "machinery, is the result",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    save(
        "exp03_modular_recovery",
        {
            "seed": SEED,
            "s_values": list(S_VALUES),
            "partition": partition,
            "lyapunov_spectra": [s.tolist() for s in ms.spectra],
            "lyapunov_gap": ms.gap,
            "exact_gap": float(exact_gap),
            "linearisation_canonical": cert.canonical,
            "linearisation_reasons": cert.reasons,
            "n_restarts": N_RESTARTS,
            "modular": {
                "fit_quality": losses,
                "reports": [r.to_dict() for r in reports],
                "assignments": [list(a) for a in assignments],
                "distinct_groupings": uniq.n_distinct,
                "grouping_counts": uniq.counts,
                "n_recovered": n_recovered,
                "n_converged": n_near,
                "n_recovered_converged": n_recovered_near,
                "mean_on_block": float(on_block.mean()),
                "min_on_block": float(on_block.min()),
            },
            "unconstrained": {"fit_quality": ufit.fit_quality, **u_rep.to_dict()},
            "all_passed": passed,
            "checks": [{"passed": ok, "claim": m} for ok, m in checks],
        },
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
