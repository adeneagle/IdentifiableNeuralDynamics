"""exp18 calibration -- how strong must the behavioural penalty be here?

Not a result.  `W_BEHAVIOR = 1.0` is exp13's value, calibrated on a different
system (contracting blocks), a different modulation (variance) and a different
horizon.  CLAUDE.md §3.12 says a loss weight does not survive a change of penalty
definition; it does not survive a change of *regime* either, and the exp18 smoke
test showed why -- at ``w = 1.0`` the adversarially-started fit kept its
invariant block at u-dependence 1.02, against a true 0.28.  The penalty was
being ignored.

What has to be true of the chosen weight, both directions:

* **it bites**: started at R2 in the ASYMMETRIC cell, the fitted invariant
  block's u-dependence must come down toward the true block's value;
* **it does not wreck the fit**: started at R1, ``fit_quality`` must stay within
  a small factor of the ``w = 0`` fit, or "the fit left R2" is unattributable --
  it could just be a fit destroyed by its own regulariser.

Findings are recorded in this docstring after the run, in the exp17_calibrate
style, since the JSON of a calibration is not an artifact anyone should cite.

RESULTS (filled in after running):
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import exp18_behaviour_vs_lattice as E          # noqa: E402
from idyn import train as T                     # noqa: E402
from idyn.models import ModelConfig             # noqa: E402

WEIGHTS = (0.0, 1.0, 5.0, 20.0, 80.0)


def main() -> None:
    rng = np.random.default_rng(E.SEED + 5)
    X, Z, U, _ = E.make_data(rng, E.KAPPA_ASYM)
    R1 = E.whiten_modules(Z)
    R2 = E.whiten_modules(E.lattice_map(Z))
    tgt1, tgt2 = E.targets()
    keep, kind = E.informative_modules(tgt1, tgt2)

    true_b = E.udep(Z[:, -1, 2:], U)
    r2_b = E.udep(E.lattice_map(Z)[:, -1, 2:], U)
    print(f"informative modules {keep} on {kind}")
    print(f"invariant block u-dependence: true {true_b:.4f}   under R2 {r2_b:.4f}\n")
    print(f"{'w':>6} {'arm':>12} {'fitq':>10} {'inv-udep':>9} {'->R1':>8} {'->R2':>8}  where")

    for w in WEIGHTS:
        for tag, warm in (("adversarial", R2), ("matched", R1)):
            cfg = ModelConfig(n_obs=E.N_OBS, d=E.D, partition=E.PART,
                              decoder="mlp", encoder="mlp")
            tc = T.TrainConfig(steps=E.STEPS, seed=E.SEED + 31, warm_steps=E.WARM_STEPS,
                               w_behavior=w, inv_start=2, inv_stop=4,
                               behavior_whiten=True)
            t0 = time.time()
            res = T.fit(X, cfg, tc, U=U, warm_z=warm)
            fp = E.fitted_fingerprint(res, E.T_STEPS + 1)
            d1 = E.restricted_distance(fp, tgt1, keep, kind)
            d2 = E.restricted_distance(fp, tgt2, keep, kind)
            print(f"{w:6.1f} {tag:>12} {res.fit_quality:10.3e} "
                  f"{E.udep(res.z_fit[:, -1, 2:], U):9.4f} {d1:8.4f} {d2:8.4f}  "
                  f"{'R1' if d1 < d2 else 'R2'}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
