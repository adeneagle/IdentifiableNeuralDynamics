"""Calibration for exp17: does the harder fitted class reach the matched arm?

exp17 needs an MLP encoder *and* an MLP decoder (§11.7: under a linear encoder
the lattice alternative is outside the fitted class, so the test would be empty).
That is a strictly harder fit than exp16's linear/linear, and nothing in the repo
says the budget carries over -- §3.11's rule is that a step count does not
survive a change of observation model, and this is a change of *model class*,
which is worse.

So two things are calibrated here before the real run, because both are
preconditions rather than results:

  1. the MATCHED arm.  Two halves warm-started at the same representative must
     agree.  If they do not, every "the fits stayed apart" reading in exp17 is
     unattributable -- it could just be that this class cannot reproduce itself.
  2. arm B's exponents.  The repo default (0.90, 0.75, 0.60, 0.45) puts the
     fastest mode at 1e-11 after 30 steps, i.e. outside the data, and §3.13
     says an exponent read there is invented.  Arm B is scored on spectra, so
     that would decide the arm by an artefact.

Writes `results/exp17_armB_calibration.json`.  Cheap on purpose: 2 restarts.

--------------------------------------------------- what it said (2026-08-16) --

Arm B's exponents.  The repo default puts the fastest mode below any horizon;
the chosen set keeps every mode alive and still moves the per-module spectra by
twice `SPEC_TOL`:

    lams                     sep(spec)   slowest^T   fastest^T
    (0.90, 0.75, 0.60, 0.45)    0.2231    4.24e-02    3.95e-11   <- mode 4 is gone
    (0.97, 0.88, 0.79, 0.70)    0.1079    4.01e-01    2.25e-05   <- chosen
    (0.95, 0.85, 0.75, 0.65)    0.1252    2.15e-01    2.44e-06

The matched arm, mlp encoder + mlp decoder, 2 restarts a side:

    arm         steps  warmres   x-rot   x-spec  ->true rot  ->true spec     fitq
    A_spirals    3000   0.0027  0.0139   0.0170      0.1670       0.0189  2.69e-3
    C_cycles     3000   0.0002  0.0003   0.0755      0.0001       0.7230  4.28e-4
    B_regroup    3000   0.0035  0.0015   0.0123      0.0016       0.0402  7.08e-3
    A_spirals    8000   0.0020  0.0103   0.0102      0.1678       0.0462  2.05e-3
    C_cycles     8000   0.0001  0.0001   0.0690      0.0001       0.3605  3.31e-4

Two things decided `exp17`'s settings.

**The matched arm reproduces itself at 3000 steps**, so a later disagreement is
attributable to the treatment rather than to the class.

**More steps buy a better fit and not a better recovery** -- arm A at 8000 fits
1.3x better (`fitq` 2.69e-3 -> 2.05e-3) and recovers rotation *identically*
(0.1670 -> 0.1678) while its spectral error gets *worse* (0.0189 -> 0.0462).
That is §3.13(e) in a fourth regime, now with the causal arrow checked by
intervention, and it is why `STEPS` stays at 3000.

Arm A's `->true rot` of 0.167 is not a failure of the fit: it is the *donor*
module, whose rotation number is not in the data (§3.13b).  The module the
lattice map actually moves comes back to 3e-4.  That measurement is what made
`exp17` score on the informative modules rather than on the whole fingerprint.

**The B-at-8000 row is missing and the JSON was never written**: two torch
processes were thrashing each other's threads and the run was stopped in favour
of `exp17` itself.  Re-run this file to regenerate it -- nothing in `exp17`
depends on that row, since the 3000-step numbers are what its settings rest on.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp17_adversarial_init as E                  # noqa: E402
from idyn import metrics as M                       # noqa: E402
from idyn import systems as S                       # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "exp17_armB_calibration.json"


def matched_only(system, radius, seed, steps, restarts=2):
    """Fit two disjoint halves, both warm-started at R1.  Must agree."""
    r = np.random.default_rng(seed)
    z0 = E.annulus_z0(r, E.N_TRAJ, *radius)
    Z = system.simulate(z0, E.T_STEPS)
    dec = S.LinearDecoder.random(E.N_OBS, E.D, r)
    X = dec(Z)
    X = (X - X.mean(axis=(0, 1), keepdims=True)) / X.std()
    Z1 = E.whiten_modules(Z)

    cols = np.arange(E.N_OBS)
    r.shuffle(cols)
    ha, hb = np.sort(cols[: E.N_OBS // 2]), np.sort(cols[E.N_OBS // 2:])

    old = E.STEPS
    E.STEPS = steps
    try:
        f1 = [E.fit_half(X[:, :, ha], seed + 10 * i, Z1) for i in range(restarts)]
        f2 = [E.fit_half(X[:, :, hb], seed + 10 * i + 1, Z1) for i in range(restarts)]
    finally:
        E.STEPS = old

    nt = X.shape[1]
    a = [E.fitted_fingerprint(x, nt) for x in f1]
    b = [E.fitted_fingerprint(x, nt) for x in f2]
    tgt = E.fingerprint(system, E.annulus_z0(r, 120, *radius))

    rot, spec = [], []
    for fa in a:
        for fb in b:
            g = M.invariant_agreement(fa, fb, spec_tol=E.SPEC_TOL, rot_tol=E.ROT_TOL)
            rot.append(g.rotation_error)
            spec.append(g.spectrum_error)
    to_t = [M.invariant_agreement(fp, tgt, spec_tol=E.SPEC_TOL, rot_tol=E.ROT_TOL)
            for fp in a + b]
    return {
        "cross_rotation": float(np.median(rot)),
        "cross_spectrum": float(np.median(spec)),
        "to_truth_rotation": float(np.median([g.rotation_error for g in to_t])),
        "to_truth_spectrum": float(np.median([g.spectrum_error for g in to_t])),
        "warm_residual": float(np.median([x.warm_residual for x in f1 + f2])),
        "fit_quality": float(np.median([x.fit_quality for x in f1 + f2])),
        "steps": steps,
    }


def main() -> int:
    t0 = time.time()
    rec: dict = {"params": {"n_obs": E.N_OBS, "n_traj": E.N_TRAJ, "T": E.T_STEPS,
                            "warm_steps": E.WARM_STEPS}}

    print("=" * 78)
    print("1 -- arm B's exponents: is the discriminating separation measurable,")
    print("     and does every mode survive the horizon?")
    print("=" * 78)
    print(f"  {'lams':28} {'sep(spec)':>10} {'slowest^T':>10} {'fastest^T':>10}")
    cands = [(0.90, 0.75, 0.60, 0.45), (0.97, 0.88, 0.79, 0.70), (0.95, 0.85, 0.75, 0.65)]
    lam_rows = []
    for lams in cands:
        reg = S.regrouping_counterexample(lams=lams)
        r = np.random.default_rng(3)
        zr = E.annulus_z0(r, 120, 0.5, 1.2)
        f1 = E.fingerprint(reg["system"], zr)
        f2 = E.fingerprint(reg["system_tilde"], zr)
        g = M.invariant_agreement(f1, f2, spec_tol=E.SPEC_TOL, rot_tol=E.ROT_TOL)
        row = {"lams": list(lams), "sep_spectrum": float(g.spectrum_error),
               "slowest_decay": float(max(lams) ** E.T_STEPS),
               "fastest_decay": float(min(lams) ** E.T_STEPS)}
        lam_rows.append(row)
        print(f"  {str(lams):28} {row['sep_spectrum']:10.4f} "
              f"{row['slowest_decay']:10.2e} {row['fastest_decay']:10.2e}")
    rec["arm_b_lams"] = lam_rows

    print("\n" + "=" * 78)
    print("2 -- the matched arm at a mlp/mlp fitted class: can it reproduce itself?")
    print("=" * 78)
    systems = {
        "A_spirals": (E.spirals(0.35, 1.10), (0.5, 1.2)),
        "C_cycles": (S.torus_regrouping_counterexample()["system"], (0.8, 1.2)),
        "B_regroup": (S.regrouping_counterexample(lams=E.ARM_B_LAMS)["system"], (0.5, 1.2)),
    }
    out: dict = {}
    print(f"  {'arm':12} {'steps':>6} {'warmres':>8} {'x-rot':>8} {'x-spec':>8} "
          f"{'->true rot':>11} {'->true spec':>12} {'fitq':>10}")
    for steps in (3000, 8000):
        for nm, (sysm, rad) in systems.items():
            c = matched_only(sysm, rad, 4242 + steps, steps)
            out[f"{nm}|{steps}"] = c
            print(f"  {nm:12} {steps:6d} {c['warm_residual']:8.4f} "
                  f"{c['cross_rotation']:8.4f} {c['cross_spectrum']:8.4f} "
                  f"{c['to_truth_rotation']:11.4f} {c['to_truth_spectrum']:12.4f} "
                  f"{c['fit_quality']:10.2e}")
    rec["matched"] = out
    rec["runtime_s"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}  ({rec['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
