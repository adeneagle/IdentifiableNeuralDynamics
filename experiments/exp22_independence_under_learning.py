r"""exp22 -- does Route D survive learning?  (task 48)

`exp21` established Route D analytically: independence of the module marginals
rejects the §4.3 triangular conjugacy and the §7 lattice regrouping, the two
objects that block Theorems B and F, while being correctly blind to §3.1's.
Theorem D (§15.12) proves it at a contracting fixed point and Theorem D′ (§15.13)
on a cycle under a trivial-stabiliser hypothesis.

None of that says a **fitted** model finds the independent representative.  This
repo's own history says to expect trouble: three times now a structural
constraint has been satisfied in gauge rather than in structure (§3.12 the
optimiser shrank the block; §3.15 time-pooling erased a rotating signal; §13.4
the encoder flattened the phase).  So the question is whether the term bites.

--------------------------------------------------------------- the design ---

`exp18`'s system, seeds and settings verbatim, so the comparison is
within-experiment; the only new thing is ``w_independence``.  Every fit is
warm-started at a designated representative and then trained normally.

               | adversarial (warm-start R2) | matched (warm-start R1)
  asymmetric   |  does the term pull it off R2? | must STAY at R1
  symmetric    |  must NOT move -- the term is  |  (not run: the informative
               |  provably blind here (Thm D')  |   cell is the adversarial one)

**The symmetric cell is the control that makes the asymmetric one readable.**
With ``p_B`` rotationally symmetric the lattice image is genuinely
independence-preserving, so a term that moved the fit *there* would be acting on
something other than independence -- the §3.12 failure in a new place.  It must
come back SURVIVED.

---------------------------------------------------- pre-registered, split ---

1. asymmetric + adversarial: moves off R2 **toward R1**, not past it.  `exp18`
   overshot to 0.1101 against an R1-R2 separation of only 0.0796, which is what
   an evaded constraint looks like; a real one should reduce both distances.
2. asymmetric + matched: stays at R1, with fit quality within ~3x of the
   penalty-free arm.
3. symmetric + adversarial: does **not** move (the term is blind by Thm D′).
4. the fitted independence tracks the term: arms that end near R1 score low
   whitened dCor, arms that stay at R2 score high.

A wrong prediction is committed failing, per CLAUDE.md §8.

--------------------------------------------------------------- calibration --

Stage 0 sweeps ``w_independence`` on a *short* budget and reports, for each
weight, the fitted dCor and the fit-quality cost.  CLAUDE.md §3.12 is explicit
that a weight never survives a change of penalty definition, so inheriting
`exp18`'s ``W_BEHAVIOR`` would be meaningless here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import exp18_behaviour_vs_lattice as E                      # noqa: E402
from idyn import train as T                                  # noqa: E402
from idyn.metrics import procrustes_block_distance            # noqa: E402
from idyn.models import LatentDynamicsModel, ModelConfig     # noqa: E402

SEED = 20260826
N_RESTARTS = 3
CAL_STEPS = 800
W_GRID = (0.0, 0.3, 1.0, 3.0)
OUT = Path(__file__).resolve().parents[1] / "results" / "exp22_independence_under_learning.json"


def fitted_dcor(z: np.ndarray) -> float:
    """Whitened dCor between the two fitted modules, averaged over time."""
    import torch
    t = torch.as_tensor(np.ascontiguousarray(z), dtype=torch.float64)
    return float(np.mean([
        float(LatentDynamicsModel._whitened_dcor(t[:, k, :2], t[:, k, 2:]))
        for k in range(t.shape[1])]))


def _procrustes_distance(A, B, partition=E.PART) -> float:
    """Gauge-quotiented distance -- see ``metrics.procrustes_block_distance``.

    CLAUDE.md 3.17.  Per-module whitening leaves O(d_i), which 7 grants, so a
    raw distance reads a fit that is AT a representative up to a within-module
    rotation as a failure.  Measured in this experiment's first stage 0: raw
    d(R1) = 0.83 against d(R2) = 1.03, which is equally consistent with "moved
    off R2 to nowhere" and "arrived at R1 in rotated coordinates".
    """
    return procrustes_block_distance(
        E.whiten_modules(A), E.whiten_modules(B), partition)


def distances(z: np.ndarray, R1: np.ndarray, R2: np.ndarray) -> tuple[float, float]:
    """Gauge-invariant distance of the fitted latents to each representative."""
    return _procrustes_distance(z, R1), _procrustes_distance(z, R2)


def one_fit(X, U, warm, seed, w_ind, steps):
    cfg = ModelConfig(n_obs=E.N_OBS, d=E.D, partition=E.PART,
                      decoder="mlp", encoder="mlp")
    tc = T.TrainConfig(steps=steps, seed=seed, warm_steps=E.WARM_STEPS,
                       batch=E.BATCH, w_behavior=0.0,
                       inv_start=2, inv_stop=4, w_independence=w_ind)
    return T.fit(X, cfg, tc, U=U, warm_z=warm)


def main() -> int:
    t0 = time.time()
    OUT.parent.mkdir(exist_ok=True)
    rec: dict = {"seed": SEED, "n_restarts": N_RESTARTS, "w_grid": list(W_GRID),
                 "cal_steps": CAL_STEPS, "steps": E.STEPS,
                 "predictions": {
                     "1": "asymmetric+adversarial moves off R2 TOWARD R1",
                     "2": "asymmetric+matched stays at R1 with fit quality intact",
                     "3": "symmetric+adversarial does NOT move (Thm D' blindness)",
                     "4": "fitted dCor tracks which representative the fit ends at"}}

    data = {}
    for tag, kappa in (("asym", E.KAPPA_ASYM), ("sym", E.KAPPA_SYM)):
        rng = np.random.default_rng(E.SEED + int(kappa * 10) + 10)
        X, Z, U, _ = E.make_data(rng, kappa)
        data[tag] = {"X": X, "U": U,
                     "R1": E.whiten_modules(Z),
                     "R2": E.whiten_modules(E.lattice_map(Z))}
        d1, d2 = data[tag]["R1"], data[tag]["R2"]
        sep = _procrustes_distance(d2, d1)
        print(f"{tag}: R1-R2 separation {sep:.4f}   "
              f"data dCor R1 {fitted_dcor(d1):.4f} R2 {fitted_dcor(d2):.4f}")
        rec.setdefault("data", {})[tag] = {
            "separation": sep, "dcor_R1": fitted_dcor(d1), "dcor_R2": fitted_dcor(d2)}
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    # ------------------------------------------------- stage 0: calibrate w
    print(f"\n=== stage 0: calibrating w_independence ({len(W_GRID)} short fits) ===")
    cal = {}
    for w in W_GRID:
        res = one_fit(data["asym"]["X"], data["asym"]["U"], data["asym"]["R2"],
                      SEED + 11, w, CAL_STEPS)
        z = np.asarray(res.z_fit, float)
        r1, r2 = distances(z, data["asym"]["R1"], data["asym"]["R2"])
        cal[str(w)] = {"dcor": fitted_dcor(z), "fit_quality": float(res.fit_quality),
                       "d_R1": r1, "d_R2": r2}
        print(f"  w={w:<5} dCor {cal[str(w)]['dcor']:.4f}  fitq "
              f"{cal[str(w)]['fit_quality']:.3e}  d(R1) {r1:.4f}  d(R2) {r2:.4f}")
        rec["stage0_calibration"] = cal
        OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    # pick the largest w whose fit-quality cost stays within 5x of w=0
    base_q = cal[str(W_GRID[0])]["fit_quality"]
    usable = [w for w in W_GRID if w > 0 and cal[str(w)]["fit_quality"] <= 5 * base_q]
    w_use = max(usable) if usable else W_GRID[1]
    rec["w_used"] = w_use
    print(f"  -> using w_independence = {w_use} "
          f"(fit-quality cost {cal[str(w_use)]['fit_quality'] / base_q:.1f}x)")

    # --------------------------------------------------- stage 1: the cells
    cells: dict = {}
    plan = (("asym", "adversarial"), ("asym", "matched"), ("sym", "adversarial"))
    for tag, arm in plan:
        key = f"{tag}_{arm}"
        print(f"\n=== {key} ({N_RESTARTS} fits, w={w_use}) ===")
        warm = data[tag]["R2" if arm == "adversarial" else "R1"]
        rows = []
        for r in range(N_RESTARTS):
            seed = SEED + 1000 * (r + 1) + (7 if arm == "adversarial" else 13)
            res = one_fit(data[tag]["X"], data[tag]["U"], warm, seed, w_use, E.STEPS)
            z = np.asarray(res.z_fit, float)
            r1, r2 = distances(z, data[tag]["R1"], data[tag]["R2"])
            row = {"restart": r, "seed": seed, "d_R1": r1, "d_R2": r2,
                   "nearer": "R1" if r1 < r2 else "R2",
                   "dcor": fitted_dcor(z), "fit_quality": float(res.fit_quality)}
            rows.append(row)
            print(f"  r{r}  d(R1) {r1:.4f}  d(R2) {r2:.4f}  -> {row['nearer']}"
                  f"   dCor {row['dcor']:.4f}  fitq {row['fit_quality']:.3e}")
            cells[key] = rows
            rec["stage1_cells"] = cells
            OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- checks
    checks: list[dict] = []
    add = lambda n, ok, d: checks.append({"name": n, "pass": bool(ok), "detail": d})

    aa, am, sa = cells["asym_adversarial"], cells["asym_matched"], cells["sym_adversarial"]
    med = lambda rows, k: float(np.median([r[k] for r in rows]))

    add("pred 1: asymmetric+adversarial moves off R2",
        sum(r["nearer"] == "R1" for r in aa) > N_RESTARTS / 2,
        f"{sum(r['nearer'] == 'R1' for r in aa)}/{N_RESTARTS} nearer R1; "
        f"median d(R1) {med(aa, 'd_R1'):.4f} d(R2) {med(aa, 'd_R2'):.4f}")
    add("pred 1b: and TOWARD R1, not past it (exp18 overshot)",
        med(aa, "d_R1") < rec["data"]["asym"]["separation"],
        f"median d(R1) {med(aa, 'd_R1'):.4f} vs R1-R2 separation "
        f"{rec['data']['asym']['separation']:.4f}")
    add("pred 2: asymmetric+matched stays at R1",
        sum(r["nearer"] == "R1" for r in am) > N_RESTARTS / 2,
        f"{sum(r['nearer'] == 'R1' for r in am)}/{N_RESTARTS}")
    add("pred 3: symmetric+adversarial does NOT move -- Thm D' blindness",
        sum(r["nearer"] == "R2" for r in sa) > N_RESTARTS / 2,
        f"{sum(r['nearer'] == 'R2' for r in sa)}/{N_RESTARTS} still at R2")
    add("pred 4: fitted dCor tracks the representative",
        med(aa, "dcor") < med(sa, "dcor") or med(am, "dcor") < med(sa, "dcor"),
        f"asym adv {med(aa, 'dcor'):.4f}, asym matched {med(am, 'dcor'):.4f}, "
        f"sym adv {med(sa, 'dcor'):.4f}")
    add("the penalty did not wreck the fit",
        med(am, "fit_quality") < 5 * cal[str(0.0)]["fit_quality"],
        f"matched fitq {med(am, 'fit_quality'):.3e} vs w=0 calibration "
        f"{cal[str(0.0)]['fit_quality']:.3e}")

    rec["checks"] = checks
    rec["n_pass"] = sum(c["pass"] for c in checks)
    rec["n_check"] = len(checks)
    rec["runtime_s"] = time.time() - t0
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print()
    for c in checks:
        print(f"   [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}\n          {c['detail']}")
    print(f"\n{rec['n_pass']}/{rec['n_check']} checks pass -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
