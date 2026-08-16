"""exp17 -- adversarial initialisation: turn agreement into evidence (task 41).

`exp16` found the hole this closes.  Its arm C fitted two disjoint neuron halves
of a system whose non-identifiability is **proved** -- two limit cycles, rotation
vector pinned only up to GL(2,Z) -- and the two fits agreed to 0.0004.  Both
landed on the same lattice representative.  So the §10.3 protocol measures
**estimator reproducibility across neuron samples**, not identifiability, and a
"yes" from it carries much less than it appears to.

The repair is to stop letting the optimiser choose the representative:

    warm-start the two halves at *deliberately different* representatives,
    then train normally and see whether the data pulls them back together.

If the observations pin the representation, the adversarial fit abandons its
start and the invariants reconverge.  If they do not, it stays, and the protocol
finally returns the "not identifiable" that arm C should always have produced.
That converts a necessary condition into something close to sufficient.

---------------------------------------------------------------- the design --

Four arms.  Each supplies a *true* representative R1 and an *alternative* R2
reached by an explicit map h, and each has a known right answer:

  A  spirals    two contracting spirals, (F3) holds at +0.50.  h is the lattice
                regrouping z1 -> z1 z2/|z2|, an EXACT conjugacy (§11.6) --  but
                the donor decays to ~1e-8, so ||Dh|| ~ 1/|z2| ~ 1e8 and (F1)
                fails.       -> R2 should NOT survive
  C  cycles     two limit cycles, same h, donor stays at radius ~1, so ||Dh||
                stays O(1) and (F1) holds.  Non-identifiable, provably.
                             -> R2 SHOULD survive
  B  regroup    §3.1: four distinct exponents, h is a coordinate permutation,
                ||Dh|| = 1.  Non-identifiable, provably.
                             -> R2 SHOULD survive
  E  escape     arm A's data, h(z1,z2) = (z1 + c z2, z2).  Expressible by the
                model, and NOT a modular conjugacy.
                             -> R2 must NOT survive

E is the load-bearing control (§3.11: build the arm that should score perfectly
into the sweep).  Without it, "the fit stayed at R2" is unattributable -- it
could just be optimiser inertia, and then every positive reading is void.  E is
the arm where inertia is the *only* thing that could hold the fit in place.

A and C are the pair that matters, because they differ in exactly one respect --
whether the donor module decays -- and §11.6 predicts that respect, and not the
spectral gap, decides the outcome.  (F3) is *positive* for A and negative for C,
so a predictor built on (F3) gets this pair backwards.

------------------------------------------------------------- what is fitted --

Generating decoder **linear**; fitted encoder and decoder both **MLP**.  Three
deliberate choices:

* the fitted encoder must be nonlinear or the test is empty -- §11.7: under a
  linear encoder z_hat = L g(z) is linear in z, so the lattice alternative is
  outside the model class and the protocol could not detect it whatever the
  optimiser did.  That is what invalidated exp16's arm-C diagnosis.
* the fitted *decoder* must be nonlinear too, since it has to represent
  W . h^{-1}.
* the *generating* decoder is kept linear on purpose.  exp16 §11.3(c) showed the
  estimator degrades under a strong nonlinear observation map (arm D's rotation
  error 0.0413 against a 0.0681 null).  Mixing that in would confound "the data
  did not pin the representative" with "the estimator could not find it", and
  those are different claims.  The ambiguity stays live regardless, because it
  is the *fitted* class that has to express h.

Warm-start targets are whitened per module before use, so the whitening penalty
has no reason to move them; that is a within-module change of basis, which §7
declines to identify anyway.

Pre-registered predictions are in ``PREDICTIONS`` and were written before the
first run (CLAUDE.md §8).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idyn import metrics as M                       # noqa: E402
from idyn import spectra as SP                      # noqa: E402
from idyn import systems as S                       # noqa: E402
from idyn import train as T                         # noqa: E402
from idyn.models import LearnedSystem, ModelConfig  # noqa: E402

SEED = 20260816
PART = [2, 2]
D = 4
N_OBS = 160
N_TRAJ = 240
T_STEPS = 30
STEPS = 3000
WARM_STEPS = 800
N_RESTARTS = 3
SPEC_TOL = 0.05
ROT_TOL = 0.01
ESCAPE_C = 0.8
# Arm B's four exponents.  Not the repo default (0.90, 0.75, 0.60, 0.45): over a
# 30-step trial 0.45 decays to 1e-11, so the fastest mode leaves no trace in the
# data and its exponent is not measured but invented (§3.13a/b).  These are
# compressed enough that every mode survives the horizon and still separate
# enough that the regrouping moves the per-module spectra by 0.108 -- twice
# SPEC_TOL.  Calibrated, not guessed; see `results/exp17_armB_calibration.json`.
ARM_B_LAMS = (0.97, 0.88, 0.79, 0.70)
OUT = Path(__file__).resolve().parents[1] / "results" / "exp17_adversarial_init.json"

PREDICTIONS = {
    "1_C_keeps_its_adversarial_start": "arm C is provably non-identifiable, so the R2 fits stay at R2",
    "2_B_keeps_its_adversarial_start": "same for the §3.1 regrouping, where h is a permutation",
    "3_A_returns": "arm A's alternative has ||Dh|| ~ 1e8, so it cannot be held: the fits return to R1",
    "4_E_returns": "arm E's target is not a conjugacy at all, so the data must reject it",
    "5_min_radius_predicts_survival_and_F3_does_not": (
        "min|z_donor| separates A from C; filtration_gap orders them the other way"
    ),
    "6_matched_warm_starts_still_agree": "warm-starting BOTH halves at R1 reproduces exp16's agreement",
}


def banner(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


# ------------------------------------------------------- systems and their h --


def spirals(w1: float, w2: float) -> S.ModularSystem:
    """Two contracting spirals.  ``beta=0.0`` is explicit: the default is 0.6."""
    return S.ModularSystem([
        S.TwistBlock(s=0.92, omega=w1, beta=0.0),
        S.TwistBlock(s=0.55, omega=w2, beta=0.0),
    ])


def lattice_map(Z: np.ndarray) -> np.ndarray:
    """h(z1, z2) = (z1 z2/|z2|, z2) in complex coordinates -- the GL(2,Z) action.

    Bounded as a map (it only rotates z1 by z2's phase); its *derivative* is what
    blows up, like 1/|z2|.  That distinction is exactly (F1), and it is why the
    arm-A alternative is perfectly well defined numerically and still cannot be
    held by a finite-capacity encoder.
    """
    z1 = Z[..., 0] + 1j * Z[..., 1]
    z2 = Z[..., 2] + 1j * Z[..., 3]
    w = z1 * z2 / np.maximum(np.abs(z2), 1e-300)
    return np.stack([w.real, w.imag, z2.real, z2.imag], -1)


def escape_map(Z: np.ndarray) -> np.ndarray:
    """h(z1, z2) = (z1 + c z2, z2): expressible, invertible, NOT a conjugacy.

    It would be one only if the two blocks had the same linear part; arm A's
    differ in both rate and frequency, so no modular F~ satisfies h F = F~ h.
    Part 0 measures that defect rather than asserting it.
    """
    out = np.array(Z, dtype=float, copy=True)
    out[..., :2] = Z[..., :2] + ESCAPE_C * Z[..., 2:]
    return out


def arms() -> dict:
    """Each arm: the data-generating system, its alternative, and the map between."""
    tor = S.torus_regrouping_counterexample()
    reg = S.regrouping_counterexample(lams=ARM_B_LAMS)
    perm = reg["P"]
    return {
        "A_spirals": {
            "system": spirals(0.35, 1.10),
            "alt": spirals(0.35 + 1.10, 1.10),
            "h": lattice_map,
            "radius": (0.5, 1.2),
            "expect_survives": False,
            "why": "exact conjugacy, but (F1) fails: donor decays, ||Dh|| ~ 1e8",
        },
        "C_cycles": {
            "system": tor["system"],
            "alt": tor["system_tilde"],
            "h": tor["h"],
            "radius": (0.8, 1.2),
            "expect_survives": True,
            "why": "exact conjugacy with (F1) holding: donor stays on its cycle",
        },
        "B_regroup": {
            "system": reg["system"],
            "alt": reg["system_tilde"],
            "h": lambda Z: np.asarray(Z, float) @ perm.T,
            "radius": (0.5, 1.2),
            "expect_survives": True,
            "why": "h is a permutation: ||Dh|| = 1, the §3.1 counterexample",
        },
        "E_escape": {
            "system": spirals(0.35, 1.10),
            "alt": None,
            "h": escape_map,
            "radius": (0.5, 1.2),
            "expect_survives": False,
            "why": "not a modular conjugacy at all -- the data must reject it",
        },
    }


# ------------------------------------------------------------------ helpers --


def annulus_z0(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    out = []
    for _ in range(D // 2):
        th = rng.uniform(-np.pi, np.pi, n)
        r = rng.uniform(lo, hi, n)
        out.append(np.stack([r * np.cos(th), r * np.sin(th)], axis=-1))
    return np.concatenate(out, axis=-1)


def whiten_modules(Z: np.ndarray, partition=PART) -> np.ndarray:
    """Per-module whitening of a warm-start target.

    A block-diagonal change of basis, i.e. precisely the freedom §7 grants inside
    a module, so it moves no invariant.  Applied so the whitening penalty has
    nothing to complain about at step 0 -- otherwise ordinary training would
    start by rescaling the target, and any subsequent drift would be
    uninterpretable.
    """
    out = np.array(Z, dtype=float, copy=True)
    off = 0
    for k in partition:
        blk = out[..., off:off + k].reshape(-1, k)
        blk = blk - blk.mean(0, keepdims=True)
        cov = blk.T @ blk / max(blk.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov + 1e-10 * np.eye(k))
        W = vecs @ np.diag(vals ** -0.5) @ vecs.T
        out[..., off:off + k] = (blk @ W).reshape(out.shape[:-1] + (k,))
        off += k
    return out


def conjugacy_defect(system, alt, h, Z: np.ndarray) -> float:
    """Relative sup-norm of ``h(F(z)) - F~(h(z))``: is R2 a representative at all?"""
    if alt is None:
        return float("nan")
    lhs = h(system.step(Z))
    rhs = alt.step(h(Z))
    return float(np.abs(lhs - rhs).max() / max(np.abs(lhs).max(), 1e-300))


def min_donor_radius(Z: np.ndarray) -> float:
    """§11.6's checkable diagnostic: how close the donor module gets to zero."""
    return float(np.hypot(Z[..., 2], Z[..., 3]).min())


def fingerprint(system, z0s, T=400, warmup=100):
    return M.dynamical_fingerprint(system, z0s, T=T, warmup=warmup, T_rotation=T)


def fitted_fingerprint(res, n_t: int):
    """Read INSIDE the data horizon (§3.13a)."""
    dyn = res.model.double().dyn
    z0 = np.asarray(res.z_fit, float)[:, 0, :]
    warm = max(n_t // 4, 2)
    read = n_t - warm
    return M.dynamical_fingerprint(LearnedSystem(dyn, PART), z0,
                                   T=read, warmup=warm, T_rotation=read)


def fit_half(X: np.ndarray, seed: int, warm_z: np.ndarray | None):
    cfg = ModelConfig(n_obs=X.shape[-1], d=D, partition=PART,
                      decoder="mlp", encoder="mlp")
    tc = T.TrainConfig(steps=STEPS, seed=seed,
                       warm_steps=(WARM_STEPS if warm_z is not None else 0))
    return T.fit(X, cfg, tc, warm_z=warm_z)


def dist_to(fp, target_fp) -> tuple[float, float]:
    r = M.invariant_agreement(fp, target_fp, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
    return r.rotation_error, r.spectrum_error


def med(xs) -> float:
    return float(np.median(xs)) if len(xs) else float("nan")


def escape_offblock() -> float:
    """Exact off-block mass of ``H F H^{-1}`` for arm E's ``h``.

    Arm A's blocks are linear, so the whole computation is closed form:
    ``H = [[I, cI], [0, I]]`` gives ``H F H^{-1} = [[A1, c(A2 - A1)], [0, A2]]``.
    The cross block vanishes only if the two modules have the same linear part,
    which they do not -- so no modular F~ conjugates through this h, and arm E's
    target is genuinely not a representative.  Measured, not asserted.
    """
    sysm = spirals(0.35, 1.10)
    F = np.zeros((D, D))
    for j in range(D):
        e = np.zeros(D)
        e[j] = 1.0
        F[:, j] = sysm.step(e)
    H = np.eye(D)
    H[:2, 2:] = ESCAPE_C * np.eye(2)
    C = H @ F @ np.linalg.inv(H)
    off = np.linalg.norm(C[:2, 2:]) + np.linalg.norm(C[2:, :2])
    return float(off / np.linalg.norm(C))


# ------------------------------------------------------------- the protocol --


def run_arm(name: str, spec: dict, seed: int, rng: np.random.Generator) -> dict:
    system, alt, h = spec["system"], spec["alt"], spec["h"]
    lo, hi = spec["radius"]
    r = np.random.default_rng(seed)

    z0 = annulus_z0(r, N_TRAJ, lo, hi)
    Z = system.simulate(z0, T_STEPS)
    dec = S.LinearDecoder.random(N_OBS, D, r)
    X = dec(Z)
    X = (X - X.mean(axis=(0, 1), keepdims=True)) / X.std()

    Z1 = whiten_modules(Z)
    Z2 = whiten_modules(h(Z))

    cols = np.arange(N_OBS)
    r.shuffle(cols)
    ha, hb = np.sort(cols[: N_OBS // 2]), np.sort(cols[N_OBS // 2:])

    # half 1 always starts at the TRUE representative; half 2 starts at R1 in
    # the matched control and at R2 in the treatment.  Only the warm start
    # differs -- same data, same objective, same budget.
    f1 = [fit_half(X[:, :, ha], seed + 10 * i + 0, Z1) for i in range(N_RESTARTS)]
    f2_matched = [fit_half(X[:, :, hb], seed + 10 * i + 1, Z1) for i in range(N_RESTARTS)]
    f2_adv = [fit_half(X[:, :, hb], seed + 10 * i + 2, Z2) for i in range(N_RESTARTS)]

    nt = X.shape[1]
    fp1 = [fitted_fingerprint(x, nt) for x in f1]
    fp_m = [fitted_fingerprint(x, nt) for x in f2_matched]
    fp_a = [fitted_fingerprint(x, nt) for x in f2_adv]

    # analytic targets, for "where did the adversarial fit end up"
    z_read = annulus_z0(r, 120, lo, hi)
    tgt1 = fingerprint(system, z_read)
    tgt2 = fingerprint(alt, h(z_read)) if alt is not None else None

    def cross(a_list, b_list):
        rot, spec_ = [], []
        for a in a_list:
            for b in b_list:
                rr = M.invariant_agreement(a, b, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
                rot.append(rr.rotation_error)
                spec_.append(rr.spectrum_error)
        return {"rotation": med(rot), "spectrum": med(spec_), "n_pairs": len(rot)}

    # ---- which invariant to score on.  R1 and R2 differ in exactly one place:
    # rotation for the two oscillatory arms, spectra for the §3.1 regrouping.
    # Scoring on the other one would be exp16 §11.3(f)'s error -- a comparison
    # that cannot fail.  Arm E has no R2, so it inherits the arm it is built on.
    sep_rot, sep_spec = dist_to(tgt1, tgt2) if tgt2 is not None else (0.0, 0.0)
    disc = 0 if (tgt2 is None or sep_rot >= sep_spec) else 1
    sep = (sep_rot, sep_spec)[disc]

    d1 = [dist_to(fp, tgt1) for fp in fp_a]
    d2 = [dist_to(fp, tgt2) for fp in fp_a] if tgt2 is not None else []
    m1 = [dist_to(fp, tgt1) for fp in fp_m]

    # "returned" is measured against the *matched* fits' own distance to R1, not
    # against zero: that is the best this estimator does on this arm, so it is
    # the only fair floor.  ABS_FLOOR keeps a near-perfect matched arm from
    # setting an unreachable bar.
    ABS_FLOOR = 0.02
    base = max(3.0 * med([x[disc] for x in m1]), ABS_FLOOR)
    adv1 = med([x[disc] for x in d1])
    returned = bool(adv1 <= base)
    n_closer_R2 = int(sum(b[disc] < a[disc] for a, b in zip(d1, d2))) if d2 else 0

    gaps = [fp.filtration_gap for fp in fp1 + fp_m + fp_a]
    return {
        "why": spec["why"],
        "expect_survives": spec["expect_survives"],
        "discriminating_invariant": ("rotation", "spectrum")[disc],
        "conjugacy_defect": conjugacy_defect(system, alt, h, Z.reshape(-1, D)),
        "min_donor_radius": min_donor_radius(Z),
        "separation": {"rotation": sep_rot, "spectrum": sep_spec, "used": sep},
        "warm_residual_true": med([x.warm_residual for x in f1 + f2_matched]),
        "warm_residual_adv": med([x.warm_residual for x in f2_adv]),
        "fit_quality_matched": med([x.fit_quality for x in f1 + f2_matched]),
        "fit_quality_adv": med([x.fit_quality for x in f2_adv]),
        "matched": cross(fp1, fp_m),
        "adversarial": cross(fp1, fp_a),
        "matched_to_R1": {"rotation": med([x[0] for x in m1]), "spectrum": med([x[1] for x in m1])},
        "adv_to_R1": {"rotation": med([x[0] for x in d1]), "spectrum": med([x[1] for x in d1])},
        "adv_to_R2": ({"rotation": med([x[0] for x in d2]), "spectrum": med([x[1] for x in d2])}
                      if d2 else None),
        "return_threshold": base,
        "survived": (not returned),
        "verdict_correct": bool((not returned) == spec["expect_survives"]),
        "n_closer_to_R2": n_closer_R2,
        "n_adv_fits": len(fp_a),
        "filtration_gap_median": med(gaps),
        "duplicate_flagged": int(sum(bool(fp.duplicate_modules()) for fp in fp1 + fp_m + fp_a)),
        "fingerprints": {
            "half1_R1": [_fp_json(x) for x in fp1],
            "half2_R1": [_fp_json(x) for x in fp_m],
            "half2_R2": [_fp_json(x) for x in fp_a],
            "target_R1": _fp_json(tgt1),
            "target_R2": _fp_json(tgt2) if tgt2 is not None else None,
        },
    }


def _fp_json(fp) -> dict:
    return {"partition": list(fp.partition),
            "spectra": [list(map(float, s)) for s in fp.spectra],
            "rotations": [float(x) for x in fp.rotations],
            "coherences": [float(x) for x in fp.coherences]}


def reachability(spec: dict, seed: int, budgets=(200, 800, 3200)) -> dict:
    """Warm-start residual against warm-start budget, with no main training.

    Separates two readings of "arm A returned to R1" that are easy to conflate:

      (i)  the alternative representative is not reachable by this model class at
           all -- which is (F1) made empirical, and the correct verdict;
      (ii) the warm start was simply under-trained -- an artefact of the budget.

    They look identical in the main table.  They come apart here: (ii) improves
    with budget, (i) plateaus.  §3.11's rule ("sweep the budget before believing
    a structural number off a new regime") applied to the *precondition* rather
    than to the result.
    """
    r = np.random.default_rng(seed)
    lo, hi = spec["radius"]
    Z = spec["system"].simulate(annulus_z0(r, N_TRAJ, lo, hi), T_STEPS)
    X = S.LinearDecoder.random(N_OBS, D, r)(Z)
    X = (X - X.mean(axis=(0, 1), keepdims=True)) / X.std()
    Z1, Z2 = whiten_modules(Z), whiten_modules(spec["h"](Z))
    out = {"budgets": list(budgets), "at_R1": [], "at_R2": []}
    for b in budgets:
        cfg = ModelConfig(n_obs=X.shape[-1], d=D, partition=PART,
                          decoder="mlp", encoder="mlp")
        tc = T.TrainConfig(steps=1, seed=seed, warm_steps=b)
        out["at_R1"].append(T.fit(X, cfg, tc, warm_z=Z1).warm_residual)
        out["at_R2"].append(T.fit(X, cfg, tc, warm_z=Z2).warm_residual)
    return out


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    A = arms()
    rec: dict = {"seed": SEED, "predictions": PREDICTIONS, "params": {
        "partition": PART, "n_obs": N_OBS, "n_traj": N_TRAJ, "T": T_STEPS,
        "steps": STEPS, "warm_steps": WARM_STEPS, "n_restarts": N_RESTARTS,
        "spec_tol": SPEC_TOL, "rot_tol": ROT_TOL, "escape_c": ESCAPE_C,
        "generating_decoder": "linear", "fitted_encoder": "mlp", "fitted_decoder": "mlp",
    }}
    checks: list[tuple[str, bool]] = []

    # ==================================================================
    banner("PART 0 -- is each alternative a representative at all?  (analytic)")
    print("  No fitting.  A map that is not an exact modular conjugacy is not an")
    print("  alternative representation of the same observations, and an arm built")
    print("  on one would be testing nothing.\n")
    print(f"  {'arm':12} {'conj defect':>12} {'min|z_donor|':>13} {'sup||Dh||~':>11}  expectation")
    pre: dict = {}
    for name, spec in A.items():
        r = np.random.default_rng(SEED + 7)
        lo, hi = spec["radius"]
        Z = spec["system"].simulate(annulus_z0(r, 200, lo, hi), T_STEPS)
        cd = conjugacy_defect(spec["system"], spec["alt"], spec["h"], Z.reshape(-1, D))
        mr = min_donor_radius(Z)
        pre[name] = {"conjugacy_defect": cd, "min_donor_radius": mr}
        print(f"  {name:12} {cd:12.2e} {mr:13.2e} {1.0 / max(mr, 1e-300):11.1e}  "
              f"{'survives' if spec['expect_survives'] else 'must not'}")
    rec["part0_analytic"] = pre

    checks.append(("A and C's alternatives are exact conjugacies",
                   max(pre["A_spirals"]["conjugacy_defect"],
                       pre["C_cycles"]["conjugacy_defect"]) < 1e-10))
    checks.append(("the sec-3.1 permutation is an exact conjugacy",
                   pre["B_regroup"]["conjugacy_defect"] < 1e-12))
    checks.append(("(F1) separates A from C by >=6 orders of magnitude",
                   pre["A_spirals"]["min_donor_radius"]
                   < 1e-6 * pre["C_cycles"]["min_donor_radius"]))

    off = escape_offblock()
    pre["E_escape"]["offblock_mass"] = off
    print(f"\n  arm E, exactly: H F H^-1 carries off-block mass {off:.4f} of its"
          f" total,\n  so no modular F~ conjugates through that h.")
    checks.append(("arm E's map is provably not a modular conjugacy", off > 0.05))

    # ==================================================================
    banner("PART 1 -- is each alternative REACHABLE?  (warm start only, no fitting)")
    print("  'the fit returned to R1' and 'the fit never left R1' are different")
    print("  claims and look the same in the main table.  A budget that improves")
    print("  the residual means under-training; a plateau means the model class")
    print("  cannot hold that representative, which is (F1) as a measurement.\n")
    reach: dict = {}
    print(f"  {'arm':12} {'target':>7} " + " ".join(f"{b:>9d}" for b in (200, 800, 3200)))
    for i, (name, spec) in enumerate(A.items()):
        rr = reachability(spec, SEED + 500 + i)
        reach[name] = rr
        for tag in ("at_R1", "at_R2"):
            print(f"  {name:12} {tag[3:]:>7} "
                  + " ".join(f"{v:9.4f}" for v in rr[tag]))
    rec["part1_reachability"] = reach
    plateau = {k: v["at_R2"][-1] for k, v in reach.items()}
    checks.append(("arm A's alternative stays unreachable at 16x the warm budget",
                   plateau["A_spirals"] > 0.25))
    checks.append(("arms B, C, E reach theirs",
                   max(plateau[a] for a in ("B_regroup", "C_cycles", "E_escape")) < 0.10))

    # ==================================================================
    banner("PARTS 2-3 -- warm-start half 2 at R1 (matched) and at R2 (adversarial)")
    print("  `matched` replays exp16's protocol with both halves started at the")
    print("  true representative and must agree.  `advers` is the treatment.")
    print("  `->R1` / `->R2` say where the adversarial fit actually ended up, and")
    print("  `sep` is how far apart the two representatives are -- the only")
    print("  meaningful yardstick, since the question is whether the protocol can")
    print("  tell them apart at all.\n")
    results: dict = {}
    print(f"  {'arm':12} {'inv':>5} {'warmres':>8} {'matched':>9} {'advers':>9} "
          f"{'->R1':>9} {'->R2':>9} {'sep':>9} {'R2?':>5}  verdict")
    for i, (name, spec) in enumerate(A.items()):
        # Deterministic per-arm seed.  NOT `hash(name)`: str hashing is salted
        # per process unless PYTHONHASHSEED is set, so that would make the cell
        # seeds unrecoverable from the JSON, against CLAUDE.md §8.
        out = run_arm(name, spec, SEED + 1000 * (i + 1), rng)
        results[name] = out
        k = out["discriminating_invariant"]
        a2 = out["adv_to_R2"][k] if out["adv_to_R2"] else float("nan")
        sep_s = ("      n/a" if out["adv_to_R2"] is None
                 else f"{out['separation']['used']:9.4f}")
        print(f"  {name:12} {k[:4]:>5} {out['warm_residual_adv']:8.4f} "
              f"{out['matched'][k]:9.4f} {out['adversarial'][k]:9.4f} "
              f"{out['adv_to_R1'][k]:9.4f} {a2:9.4f} {sep_s} "
              f"{out['n_closer_to_R2']}/{out['n_adv_fits']:<3}  "
              f"{'STAYED' if out['survived'] else 'returned'}"
              f"{'' if out['verdict_correct'] else '   <-- WRONG'}")
    rec["arms"] = results

    # ==================================================================
    banner("PART 3 -- checks")

    print("  warm-start residuals (fraction of the target's own variance).  A")
    print("  fit that never reached R2 cannot testify that R2 failed to hold, so")
    print("  read this column before any verdict:")
    for k, v in results.items():
        print(f"    {k:12} at R1 {v['warm_residual_true']:7.4f}   "
              f"at R2 {v['warm_residual_adv']:7.4f}")

    checks.append(("the adversarial warm start actually took (arms B, C, E)",
                   max(results[a]["warm_residual_adv"]
                       for a in ("B_regroup", "C_cycles", "E_escape")) < 0.10))
    checks.append(("matched warm starts still agree, in every arm",
                   max(r["matched"][r["discriminating_invariant"]]
                       for r in results.values()) < 0.02))
    checks.append(("ESCAPE CONTROL: arm E abandons a non-conjugacy start",
                   not results["E_escape"]["survived"]))
    checks.append(("arm C keeps its adversarial representative (correctly)",
                   results["C_cycles"]["survived"]))
    checks.append(("arm B keeps its adversarial representative (correctly)",
                   results["B_regroup"]["survived"]))
    checks.append(("arm A returns to the true one (correctly)",
                   not results["A_spirals"]["survived"]))
    checks.append(("every surviving arm ends nearer R2 than R1",
                   all(r["n_closer_to_R2"] > r["n_adv_fits"] / 2
                       for r in results.values() if r["survived"])))
    checks.append(("the protocol's verdict is right in all four arms",
                   all(r["verdict_correct"] for r in results.values())))

    surv = {k: v["survived"] for k, v in results.items()}
    rad = {k: v["min_donor_radius"] for k, v in results.items()}
    gap = {k: v["filtration_gap_median"] for k, v in results.items()}
    print("\n  what predicts survival?")
    print(f"  {'arm':12} {'survived':>9} {'min|z_d|':>10} {'(F3) gap':>10}")
    for k in results:
        print(f"  {k:12} {str(surv[k]):>9} {rad[k]:10.2e} {gap[k]:+10.4f}")
    checks.append(("min|z_donor| predicts survival for the A/C pair and (F3) does not",
                   (rad["C_cycles"] > rad["A_spirals"])
                   and (gap["A_spirals"] > gap["C_cycles"])
                   and surv["C_cycles"] and not surv["A_spirals"]))
    rec["part3_predictor"] = {"survived": surv, "min_donor_radius": rad,
                              "filtration_gap": gap}

    banner("CHECKS")
    for nm, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}")
    rec["checks"] = [{"name": n, "ok": bool(o)} for n, o in checks]
    rec["n_pass"] = int(sum(o for _, o in checks))
    rec["n_checks"] = len(checks)
    rec["runtime_s"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}  ({rec['n_pass']}/{rec['n_checks']}, {rec['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
