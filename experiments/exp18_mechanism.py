"""exp18 part 2 -- why the asymmetric behaviour-on cell kept the lattice representative.

`exp18`'s pre-registered check 5b predicted that once ``p_B``'s rotational
symmetry is broken -- so that behaviour CAN see the lattice regrouping (part 0:
detector 0.956 against a 0.057 floor) -- imposing the behavioural penalty would
push an adversarially-started fit back to R1.  It did not, in 4/4 restarts.  Two
explanations, and they lead to opposite conclusions:

(a) **Optimisation barrier.**  R1 is better on both counts and simply not
    reachable from R2's basin.  Then the fitted half of exp18 measures learning,
    not identifiability, and says nothing about Route B.

(b) **The encoder restores the symmetry.**  The symmetry was broken in the
    DATA, but a nonlinear encoder can map a concentrated phase back toward a
    uniform one.  Then ``p_B`` is symmetric again in the FITTED latent, the
    escape of part 0 reopens, and the penalty is satisfied while sitting at R2.

(b) is the stronger claim and it is the one the measurement supports.  The
discriminating quantity is the circular concentration ``|E e^{i phi}|`` of the
fitted invariant block: 0 is rotationally symmetric, 1 is a point mass.

    fit           concentration   by u              u-dep    fit_quality
    matched (R1)      0.8090      [0.8113, 0.8079]  0.1008     2.163e-03
    adversarial (R2)  0.2703      [0.3129, 0.4288]  0.4359     2.769e-03

    for reference, in the TRUE data: R1 block 0.8541, R2 block 0.3921

The matched fit reproduces the data's own concentration (0.809 against 0.854)
and is flat across u -- a genuinely invariant block.  The adversarial fit's is
**0.270, below even the R2 representation's own 0.392**: the encoder has
smeared the phase further than the regrouping already did, which is exactly the
direction that makes the block rotationally symmetric and the coupling
invisible.  The escape is not fully reached -- the per-u concentrations still
differ, leaving u-dep 0.436 -- and that partial hiding is what the 30% fit-quality
cost buys.

**Consequence, and it is the sharp one.**  Part 0's design rule -- "break
``p_B``'s rotational symmetry and behaviour can see the regrouping" -- is a
statement about the data, and it **does not survive learning**: the model class
can enlarge ``p_B``'s symmetry group in its own latent.  So the u-invariant
subspace is canonical only if ``p_B`` has trivial symmetry in *every
representation the model can reach*, which is a much stronger requirement than
having trivial symmetry in the data.  Third instance of the §3.12 pattern: a
structural constraint satisfied by moving in the gauge rather than by acquiring
the structure.
"""

from __future__ import annotations

import json
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

OUT = Path(__file__).resolve().parents[1] / "results" / "exp18_mechanism.json"
N_RESTARTS = 3


def concentration(block: np.ndarray) -> float:
    """``|E e^{i phi}|`` of a 2-D block: 0 = rotationally symmetric, 1 = a point mass.

    The right statistic here because the escape is a ROTATION: what decides
    whether behaviour can see it is exactly how far ``p_B`` is from being
    invariant under the circle action, and no second-moment quantity sees that
    (a whitened isotropic block and a whitened phase-concentrated block have the
    same covariance by construction).
    """
    z = block[..., 0] + 1j * block[..., 1]
    return float(abs(np.mean(np.exp(1j * np.angle(z)))))


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(E.SEED + int(E.KAPPA_ASYM * 10) + 10)
    X, Z, U, _ = E.make_data(rng, E.KAPPA_ASYM)
    R1 = E.whiten_modules(Z)
    R2 = E.whiten_modules(E.lattice_map(Z))

    rec = {
        "seed": E.SEED,
        "question": "did the fit stay at R2 because R1 is unreachable, or because "
                    "the encoder restored p_B's rotational symmetry?",
        "params": {"kappa_b": E.KAPPA_ASYM, "w_behavior": E.W_BEHAVIOR,
                   "batch": E.BATCH, "steps": E.STEPS, "warm_steps": E.WARM_STEPS,
                   "n_restarts": N_RESTARTS},
        "data_concentration": {
            "R1": concentration(Z[:, -1, 2:]),
            "R2": concentration(E.lattice_map(Z)[:, -1, 2:]),
        },
        "fits": [],
    }
    print(f"TRUE data invariant block, circular concentration at t=T: "
          f"R1 {rec['data_concentration']['R1']:.4f}   "
          f"R2 {rec['data_concentration']['R2']:.4f}")

    for r in range(N_RESTARTS):
        row = {"restart": r}
        for tag, warm in (("adversarial", R2), ("matched", R1)):
            seed = E.SEED + 1000 * (r + 1) + (7 if tag == "adversarial" else 13)
            cfg = ModelConfig(n_obs=E.N_OBS, d=E.D, partition=E.PART,
                              decoder="mlp", encoder="mlp")
            tc = T.TrainConfig(steps=E.STEPS, seed=seed, warm_steps=E.WARM_STEPS,
                               batch=E.BATCH, w_behavior=E.W_BEHAVIOR,
                               inv_start=2, inv_stop=4, behavior_whiten=True,
                               behavior_per_time=True)
            res = T.fit(X, cfg, tc, U=U, warm_z=warm)
            zf = np.asarray(res.z_fit, float)
            row[tag] = {
                "seed": seed,
                "fit_quality": float(res.fit_quality),
                "concentration": concentration(zf[:, -1, 2:]),
                "concentration_by_u": [concentration(zf[U == u, -1, 2:]) for u in (0, 1)],
                "u_dependence": E.udep(zf[:, -1, 2:], U),
            }
            c = row[tag]
            print(f"  r{r} {tag:12s} fitq {c['fit_quality']:.3e}  "
                  f"concentration {c['concentration']:.4f}  "
                  f"by u {np.round(c['concentration_by_u'], 4)}  "
                  f"u-dep {c['u_dependence']:.4f}")
        rec["fits"].append(row)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    adv = [f["adversarial"]["concentration"] for f in rec["fits"]]
    mat = [f["matched"]["concentration"] for f in rec["fits"]]
    rec["summary"] = {
        "adv_concentration_med": float(np.median(adv)),
        "matched_concentration_med": float(np.median(mat)),
        "ratio": float(np.median(mat) / max(np.median(adv), 1e-12)),
        # The claim: the adversarial fit is MORE rotationally symmetric than the
        # data's own R2 representation, i.e. the encoder actively flattened it.
        "adv_flatter_than_data_R2": bool(
            np.median(adv) < rec["data_concentration"]["R2"]),
        "matched_matches_data_R1": bool(
            abs(np.median(mat) - rec["data_concentration"]["R1"]) < 0.15),
        "runtime_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    s = rec["summary"]
    print(f"\nadversarial {s['adv_concentration_med']:.4f} vs matched "
          f"{s['matched_concentration_med']:.4f} ({s['ratio']:.1f}x); "
          f"flatter than the data's R2: {s['adv_flatter_than_data_R2']}; "
          f"matched reproduces R1: {s['matched_matches_data_R1']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
