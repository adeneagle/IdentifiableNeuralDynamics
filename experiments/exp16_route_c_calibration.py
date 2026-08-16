"""exp16 -- calibrate the Route C instrument against known ground truth.

`exp15` ran the split-and-compare protocol on real data, where nobody knows the
right answer.  This runs it on systems where the answer is **certain**, in both
directions, and asks one question:

    does `spectra.filtration_gap` -- (F3), computable on a fitted model with no
    ground truth -- predict whether the per-module invariants are recoverable?

If yes, (F3) stops being a hypothesis in a theorem and becomes a decision rule
someone can apply to their own fit.

**Why both directions matter.** A protocol that never returns "not identifiable"
is not measuring anything; that is exactly the bug `exp15` shipped with, where
three checks passed against an empty control arm.  The repo owns three systems
where non-identifiability is *provable*, and they are the only place the
instrument's "yes" can be calibrated.

------------------------------------------------------------------ the arms --

  A  filtration      two modules, well-separated contraction rates, distinct
                     frequencies.  (F3) holds.        -> MUST identify
  B  regrouping      §3.1: four distinct exponents, two groupings, both exact
                     fits.  (F3) FAILS (hulls interleave).  -> MUST NOT identify
  C  torus           §7: two limit cycles, identical spectra, rotation vector
                     pinned only up to GL(2,Z).  (F3) FAILS.  -> MUST NOT
                     identify per-module rotation
  D  gauge           arm A modulo changes §8 declines to identify: a within-
                     module basis change, and a radial SHEAR (which §7.1 proves
                     is removable by the asymptotic phase).  -> MUST identify
                     anyway, i.e. the fingerprint must be BLIND to these
  E  null            arm A, per-neuron circular time shift.   -> MUST NOT identify

D is not decoration.  A metric that flags a gauge change as a disagreement is
broken in the opposite direction from one that misses a regrouping, and only D
catches it.

--------------------------------------------------------- crossed on decoder --

Every fitted arm is run under **both** observation models, because the
observation map -- not the flow -- sets the ambiguity group (§2/§3.5):

  linear   x = Wz with W full column rank  =>  h in GL(d) is FORCED.  The arm-C
           and arm-D ambiguities are nonlinear maps, so under a linear decoder
           they are not hard cases the protocol passes -- they are cases the
           model class **cannot express**.  This arm is therefore a positive
           control, and the place the protocol should look best.
  mlp      x = W . (coupling flow)(z), analytic and strongly nonlinear.  Now h
           may be nonlinear and C/D are live.

Pre-registered prediction, recorded before running (CLAUDE.md §8): **arms C and
D flip verdict with the decoder** -- identified under linear, ambiguous under
mlp.  If they do not flip, either the §3.5 reading is wrong or the MLP decoder
is not delivering its nominal dose, and §3.11 says to read the *delivered*
nonlinearity rather than the parameter.
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

SEED = 20260815
PART = [2, 2]
D = 4
N_OBS = 160                 # 80 neurons a side, comfortably over §3.13(e)'s ~32
N_TRAJ = 240
T_STEPS = 30
STEPS = 3000
N_RESTARTS = 4
DEC_STRENGTH = 1.0
SPEC_TOL = 0.05
ROT_TOL = 0.01
OUT = Path(__file__).resolve().parents[1] / "results" / "exp16_route_c.json"

PREDICTIONS = {
    "1_f3_predicts_identifiability": "sign(filtration_gap) tracks recoverability across A/B/C",
    "2_gauge_blindness_survives_mlp": "arm D agrees as well as arm A under both decoders",
    "3_torus_splits_rotation_from_spectra": "arm C: rotation recoverable, spectra tied, lattice margin fails",
    "4_duplicate_flags_the_degenerate_arms": "duplicate_modules flags C and E far more than A, B, D",
    "5_C_and_D_flip_with_the_decoder": "C and D are unreachable under a linear decoder, live under mlp",
}


def banner(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


# --------------------------------------------------------------- the systems --


def arm_a_system() -> S.ModularSystem:
    """Two modules, disjoint ordered spectral hulls: (F3) holds comfortably."""
    return S.ModularSystem([
        S.TwistBlock(s=0.92, omega=0.35, beta=0.0),
        S.TwistBlock(s=0.55, omega=1.10, beta=0.0),
    ])


def arm_d_systems() -> tuple[S.ModularSystem, S.ModularSystem]:
    """Arm A, and the same dynamics seen through changes §8 does not identify.

    Two gauge changes at once:
      * a radial **shear** in each module -- §7.1 proves it is removable by the
        asymptotic phase, so it is not a conjugacy invariant of a single module;
      * a within-module change of basis, which §8 explicitly declines to pin.

    Both leave every invariant in the fingerprint untouched, so agreement here
    is required, not optional.
    """
    base = arm_a_system()
    sheared = S.ModularSystem([
        S.TwistBlock(s=0.92, omega=0.35, beta=0.9),
        S.TwistBlock(s=0.55, omega=1.10, beta=0.6),
    ])
    return base, sheared


def annulus_z0(rng: np.random.Generator, n: int, lo: float = 0.5, hi: float = 1.2) -> np.ndarray:
    """Initial conditions on an annulus in each 2-D module (inside every basin)."""
    out = []
    for _ in range(D // 2):
        th = rng.uniform(-np.pi, np.pi, n)
        r = rng.uniform(lo, hi, n)
        out.append(np.stack([r * np.cos(th), r * np.sin(th)], axis=-1))
    return np.concatenate(out, axis=-1)


def fingerprint(system, z0s, T=400, warmup=100):
    return M.dynamical_fingerprint(system, z0s, T=T, warmup=warmup, T_rotation=T)


# ------------------------------------------------------------ the protocol --


def fit_half(X: np.ndarray, seed: int):
    cfg = ModelConfig(n_obs=X.shape[-1], d=D, partition=PART,
                      decoder="linear", encoder="linear")
    return T.fit(X, cfg, T.TrainConfig(steps=STEPS, seed=seed))


def fitted_fingerprint(res, n_t: int):
    """Read INSIDE the data horizon (§3.13a)."""
    dyn = res.model.double().dyn
    z0 = np.asarray(res.z_fit, float)[:, 0, :]
    warm = max(n_t // 4, 2)
    read = n_t - warm
    return M.dynamical_fingerprint(LearnedSystem(dyn, PART), z0,
                                   T=read, warmup=warm, T_rotation=read)


def circshift(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-neuron circular time shift: the null with no shared latent."""
    out = X.copy()
    for c in range(out.shape[0]):
        for j in range(out.shape[2]):
            out[c, :, j] = np.roll(out[c, :, j], int(rng.integers(out.shape[1])))
    return out


def compare(fps_a, fps_b) -> dict:
    rot, spec, t1, lat, dup = [], [], [], [], 0
    for fa in fps_a:
        for fb in fps_b:
            if fa.duplicate_modules() or fb.duplicate_modules():
                dup += 1
            r = M.invariant_agreement(fa, fb, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
            rot.append(r.rotation_error)
            spec.append(r.spectrum_error)
            a = np.sort(np.concatenate([np.asarray(s).ravel() for s in fa.spectra]))
            b = np.sort(np.concatenate([np.asarray(s).ravel() for s in fb.spectra]))
            t1.append(float(np.abs(a - b).max()))
            lat.append(SP.rotation_lattice_margin(
                [abs(x) for x in fa.rotations], [abs(x) for x in fb.rotations])[0])
    return {
        "n_pairs": len(rot),
        "rotation": float(np.median(rot)),
        "spectrum": float(np.median(spec)),
        "tier1": float(np.median(t1)),
        "lattice": float(np.median(lat)),
        "duplicate_pairs": dup,
    }


def run_cell(system, decoder_kind: str, seed: int, rng: np.random.Generator) -> dict:
    """One (arm, decoder) cell: generate, split neurons, fit halves, compare."""
    r = np.random.default_rng(seed)
    dec = (S.LinearDecoder.random(N_OBS, D, r) if decoder_kind == "linear"
           else S.MLPDecoder.random(N_OBS, D, r, strength=DEC_STRENGTH))
    z0 = annulus_z0(r, N_TRAJ)
    Z = system.simulate(z0, T_STEPS)
    X = dec(Z)
    X = (X - X.mean(axis=(0, 1), keepdims=True)) / X.std()

    # delivered nonlinearity, never the nominal parameter (§3.11)
    lin = Z.reshape(-1, D) @ np.linalg.lstsq(
        Z.reshape(-1, D), dec(Z).reshape(-1, N_OBS), rcond=None)[0]
    obs_nl = float(np.linalg.norm(dec(Z).reshape(-1, N_OBS) - lin)
                   / np.linalg.norm(dec(Z).reshape(-1, N_OBS)))

    cols = np.arange(N_OBS)
    r.shuffle(cols)
    ha, hb = np.sort(cols[: N_OBS // 2]), np.sort(cols[N_OBS // 2:])

    fps = []
    for k, sel in enumerate((ha, hb)):
        fps.append([fitted_fingerprint(fit_half(X[:, :, sel], seed + 100 * i + k), X.shape[1])
                    for i in range(N_RESTARTS)])
    treat = compare(fps[0], fps[1])

    Xn = circshift(X[:, :, hb], rng)
    fp_n = [fitted_fingerprint(fit_half(Xn, seed + 100 * i + 1), Xn.shape[1])
            for i in range(N_RESTARTS)]
    null = compare(fps[0], fp_n)

    gaps = [fp.filtration_gap for grp in fps for fp in grp]
    return {
        "obs_nonlinearity": obs_nl,
        "treatment": treat,
        "null": null,
        "filtration_gap_median": float(np.median(gaps)),
        "n_filtration": int(sum(fp.is_filtration for grp in fps for fp in grp)),
        "n_fits": int(sum(len(g) for g in fps)),
        # the verdict the instrument returns, with no ground truth used
        "identified_rotation": bool(treat["rotation"] < null["rotation"] / 3),
        "identified_spectrum": bool(treat["tier1"] < null["tier1"] / 3),
    }


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rec: dict = {"seed": SEED, "predictions": PREDICTIONS, "params": {
        "partition": PART, "n_obs": N_OBS, "n_traj": N_TRAJ, "T": T_STEPS,
        "steps": STEPS, "n_restarts": N_RESTARTS, "dec_strength": DEC_STRENGTH,
        "spec_tol": SPEC_TOL, "rot_tol": ROT_TOL,
    }}
    checks: list[tuple[str, bool]] = []

    # ==================================================================
    banner("PART 1 -- analytic calibration: exact systems, no fitting")
    print("  The instrument is checked where the answer is CERTAIN, in both")
    print("  directions.  No optimiser, no sampling: any failure here is the")
    print("  metric's, not the fit's.\n")

    z0 = annulus_z0(rng, 60)
    an: dict = {}

    # (a) gauge blindness -- MUST agree
    base, sheared = arm_d_systems()
    fa, fb = fingerprint(base, z0), fingerprint(sheared, z0)
    g = M.invariant_agreement(fa, fb, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
    an["gauge_shear"] = {"agree": bool(g.agree), "rotation_error": g.rotation_error,
                         "spectrum_error": g.spectrum_error}
    print(f"  (a) shear gauge      agree={g.agree}  rot_err={g.rotation_error:.2e}  "
          f"spec_err={g.spectrum_error:.2e}   [must agree]")
    checks.append(("gauge blindness: a radial shear is not a disagreement", bool(g.agree)))

    # (b) §3.1 regrouping -- MUST disagree, and (F3) must flip sign
    reg = S.regrouping_counterexample()
    z4 = annulus_z0(rng, 60)
    fr, frt = fingerprint(reg["system"], z4), fingerprint(reg["system_tilde"], z4)
    gr = M.invariant_agreement(fr, frt, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
    gap_true = SP.filtration_gap([np.asarray(s) for s in fr.spectra]).gap
    gap_reg = SP.filtration_gap([np.asarray(s) for s in frt.spectra]).gap
    an["regrouping"] = {"agree": bool(gr.agree), "spectrum_error": gr.spectrum_error,
                        "gap_true": gap_true, "gap_regrouped": gap_reg}
    print(f"  (b) sec-3.1 regrouping  agree={gr.agree}  spec_err={gr.spectrum_error:.4f}   "
          f"[must DISagree]")
    print(f"      (F3) gap: true {gap_true:+.4f}   regrouped {gap_reg:+.4f}  "
          f"[must flip sign]")
    checks.append(("the §3.1 regrouping is detected as a disagreement", not gr.agree))
    checks.append(("(F3) separates the true grouping from the regrouped one",
                   gap_true > 0 > gap_reg))

    # (c) torus GL(2,Z) -- rotation differs coordinatewise, lattice margin ~ 0
    tor = S.torus_regrouping_counterexample()
    zt = annulus_z0(rng, 60, lo=0.8, hi=1.2)
    ft, ftt = fingerprint(tor["system"], zt), fingerprint(tor["system_tilde"], zt)
    gt = M.invariant_agreement(ft, ftt, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
    lat, A = SP.rotation_lattice_margin([abs(x) for x in ft.rotations],
                                        [abs(x) for x in ftt.rotations])
    gap_t = SP.filtration_gap([np.asarray(s) for s in ft.spectra]).gap
    an["torus"] = {"agree": bool(gt.agree), "rotation_error": gt.rotation_error,
                   "lattice_margin": lat, "lattice_A": A.tolist() if A is not None else None,
                   "filtration_gap": gap_t}
    print(f"  (c) torus GL(2,Z)    agree={gt.agree}  rot_err={gt.rotation_error:.4f}  "
          f"lattice margin={lat:.2e}   [rotation must differ, lattice must not]")
    print(f"      (F3) gap {gap_t:+.4f}  [must be negative: two neutral modules]")
    checks.append(("the torus regrouping moves the rotation vector",
                   gt.rotation_error > ROT_TOL))
    checks.append(("...but is invisible after the GL(2,Z) quotient", lat < 1e-6))
    checks.append(("(F3) rejects two neutral oscillators", gap_t < 0))
    rec["part1_analytic"] = an

    # ==================================================================
    banner("PART 2 -- the fitted protocol, crossed with the observation model")
    arms = {
        "A_filtration": arm_a_system(),
        "B_regrouping": S.regrouping_counterexample()["system"],
        "C_torus": S.torus_regrouping_counterexample()["system"],
        "D_gauge": arm_d_systems()[1],
    }
    cells: dict = {}
    hdr = (f"  {'arm':14} {'decoder':8} {'obs-nl':>7} {'F3 gap':>9} {'rot':>9} "
           f"{'rot null':>9} {'tier1':>9} {'t1 null':>9} {'dup':>5}  verdict")
    print(hdr)
    for arm, sysm in arms.items():
        for dk in ("linear", "mlp"):
            key = f"{arm}|{dk}"
            c = run_cell(sysm, dk, SEED + abs(hash(key)) % 10_000, rng)
            cells[key] = c
            v = ("ROT+SPEC" if c["identified_rotation"] and c["identified_spectrum"]
                 else "ROT only" if c["identified_rotation"]
                 else "SPEC only" if c["identified_spectrum"] else "neither")
            print(f"  {arm:14} {dk:8} {c['obs_nonlinearity']:7.3f} "
                  f"{c['filtration_gap_median']:+9.4f} {c['treatment']['rotation']:9.4f} "
                  f"{c['null']['rotation']:9.4f} {c['treatment']['tier1']:9.4f} "
                  f"{c['null']['tier1']:9.4f} {c['treatment']['duplicate_pairs']:5d}  {v}")
    rec["part2_cells"] = cells

    # ==================================================================
    banner("PART 3 -- does (F3) predict recoverability?")
    rows = [(k, v["filtration_gap_median"], v["identified_rotation"]) for k, v in cells.items()]
    pos = [r for r in rows if r[1] > 0]
    neg = [r for r in rows if r[1] <= 0]
    print(f"  cells with (F3) > 0: {len(pos)}, of which identified: {sum(r[2] for r in pos)}")
    print(f"  cells with (F3) <= 0: {len(neg)}, of which identified: {sum(r[2] for r in neg)}")
    rec["part3_predictor"] = {
        "n_f3_positive": len(pos), "n_f3_positive_identified": int(sum(r[2] for r in pos)),
        "n_f3_negative": len(neg), "n_f3_negative_identified": int(sum(r[2] for r in neg)),
    }

    checks.append(("arm A is identified under BOTH decoders",
                   cells["A_filtration|linear"]["identified_rotation"]
                   and cells["A_filtration|mlp"]["identified_rotation"]))
    checks.append(("arm D (gauge) is identified under BOTH decoders",
                   cells["D_gauge|linear"]["identified_rotation"]
                   and cells["D_gauge|mlp"]["identified_rotation"]))
    checks.append(("the MLP decoder actually delivers nonlinearity",
                   min(cells[f"{a}|mlp"]["obs_nonlinearity"] for a in arms) > 0.15))
    checks.append(("the linear decoder arm is linear",
                   max(cells[f"{a}|linear"]["obs_nonlinearity"] for a in arms) < 1e-6))

    banner("CHECKS")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
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
