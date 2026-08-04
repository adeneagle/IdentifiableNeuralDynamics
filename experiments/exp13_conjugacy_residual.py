"""Experiment 13 -- CLAUDE.md task 32: which Lemma D hypothesis does a fitted h break?

Theory and experiment disagreed, and both were sound:

  * Lemma D (identifiability.md 4.5) proves that an **exact** conjugacy with a
    one-sided spectral gap and an additive h_B is block-diagonal.
  * exp12 measures a **learned** h and finds only a filtration, degrading
    monotonically with observation nonlinearity (jac_diag 0.994 -> 0.567).

A fitted h need not satisfy the lemma's hypotheses, so the useful question is
*which* one it breaks.  Four candidates:

  (i)   h is not an exact conjugacy: ||h o F - F~ o h|| > 0.
  (ii)  the LEARNED transition's spectra have no one-sided gap, so Lemma C's
        hypothesis fails in the model even though it holds in the truth.
  (iii) h_B is not additive -- Lemma D's own open case (a).
  (iv)  the BEHAVIOURAL hypothesis was never imposed at all.

**The answer is (iv)**, and this experiment is built to demonstrate it rather
than to discover it -- see CLAUDE.md 3.12.  The old behavioural penalty scored
the pinned block's conditional moments on the raw block, so it fell like
eps^2/eps^4 when the block shrank: "be u-invariant" and "be small" were the same
instruction, and the optimiser took the cheap one.  The pinned block came out 21x
smaller than its partner, scored a raw u-dependence of 0.0015, and still carried
the u-varying latent at distance correlation 0.99.

So exp11 and exp12 never tested B+C.  Their dose-response is real -- exp12's
strength-0 control rules out a degenerate metric -- but it measures
dynamics-only fitting under a decoy penalty.

### What this experiment does

Two penalty variants at each dose, 8 restarts each:

  * ``whiten=False``  reproduces exp11/exp12 exactly (same seeds, same data).
  * ``whiten=True``   whitens the block by its own pooled covariance first,
                      making the penalty a GL(d_b) invariant -- the freedom 7
                      grants within a module -- so it can only be paid with
                      genuine distributional invariance.

Parts 1-2 establish that the constraint was absent and is now present.  Part 3
asks the substantive question for the first time: **with behaviour actually
imposed, does the forbidden cross-block stay suppressed at high dose?**  That is
B+C's load-bearing claim, and it has never been tested.  Candidates (i)-(iii)
are reported alongside for both variants, cheap now that the fits are running.

### Reading the checks

  1-2 PASS  -> (iv) confirmed; every behavioural conclusion in exp11/exp12 is
               void, and `approaches.md` B.1's B column has never been tested.
  3 PASS    -> the fix imposes what the old penalty only appeared to.
  4-5 PASS  -> with behaviour genuinely applied, B+C survives a nonlinear
               observation map after all, and exp12's headline needs retracting.
  4-5 FAIL  -> B+C fails even when its hypothesis IS imposed, which is a much
               stronger negative result than exp12 was entitled to claim.

Checks 6-8 are the other three candidates.  A FAIL there is informative, not bad.

Caveat carried from 3.11 and taken seriously here: the calibration that set
`W_WHITENED` used single restarts, and sd(jac_diag) ~ 0.1 in this regime.  Eight
restarts per cell is the minimum that can tell 0.91 from 0.57; report the
distribution, never the best.
"""

from __future__ import annotations

import numpy as np
import torch

from _common import banner, save, verdict
from idyn import behavior as BH
from idyn import metrics as M
from idyn import spectra as SP
from idyn import systems as S
from idyn import train as T
from idyn.linear import slices_of
from idyn.models import ModelConfig

# Must match exp12 exactly, or the whiten=False arm is a different model and its
# `jac_diag` cannot be compared with the values already on record.
SEED = 0
N_OBS = 12
U_LEVELS = np.array([0, 1, 2, 3])
N_PER_U = 250
T_STEPS = 15
STEPS = 4000
N_RESTARTS = 8
INV = (2, 4)
STRENGTHS = (0.0, 0.5, 1.5, 2.0)   # doses ~ (0.00, 0.31, 0.43, 0.60); read obs-nl

# 3.12: the whitened penalty is O(1) where the raw one was O(scale^2..^4), so
# w_behavior does not survive the change of definition.  These are set from the
# calibration sweep: the smallest weight at which the constraint is actually
# satisfied, since anything beyond that only distorts the reconstruction.
# Calibration sweep (single restart, doses 0.00 and 0.60), jac_diag at dose 0.60:
#   w = 0.0  0.580   w = 0.5  0.829   w = 2.0  0.901
#   w = 0.2  0.582   w = 1.0  0.912   w = 5.0  0.711  (fitq 0.103, over-constrained)
# 1.0 is where the constraint is satisfied (scale-normalised u-dependence 0.078,
# below the 0.155 floor) at the least cost to the fit.
#
# KNOWN WEAKNESS, and the leading suspect for the dose-0.31 anomaly (task 34):
# this sweep only visited the ENDPOINT doses, so the interior was never tuned.
# The 8-restart run lands triangular at dose 0.31 in 8/8 restarts (sd 0.029),
# which is far too tight for seed noise -- sweep w at strength=0.5 before
# looking for a deeper explanation.
W_RAW = 5.0        # what exp11/exp12 used
W_WHITENED = 1.0

PART = [2, 2]
N_PROBE = 2000
LYAP_T = 300
LYAP_WARMUP = 100
LYAP_N_Z0 = 3


def build(strength, seed):
    """Aligned system (varying block = spectrally dominant) + a decoder."""
    rng = np.random.default_rng(seed)
    sys_a = S.ModularSystem(
        [S.TwistBlock(s=0.90, omega=0.40, beta=0.6), S.TwistBlock(s=0.50, omega=1.10, beta=-0.5)]
    )
    dec = S.MLPDecoder.random(N_OBS, 4, rng, strength=strength)
    X, Z, U, dec = T.make_behavioural_dataset(
        sys_a, 2, 2, N_OBS, N_PER_U, T_STEPS, U_LEVELS, rng, mode="variance", decoder=dec
    )
    return sys_a, X, Z, U, dec


def observation_nonlinearity(dec, Z):
    """Fraction of g not captured by any affine map -- the dose, measured."""
    z = np.asarray(Z, float).reshape(-1, Z.shape[-1])
    x = dec(z)
    aug = np.hstack([z, np.ones((len(z), 1))])
    A, *_ = np.linalg.lstsq(aug, x, rcond=None)
    return float(np.linalg.norm(x - aug @ A) / np.linalg.norm(x))


def h_of(model, dec):
    """h = encoder o g, in float64.

    The double cast is not optional: `metrics.hessian_of` divides a four-point
    difference by eps^2, so at float32 the roundoff floor swamps the signal.  It
    also sharpens the first derivatives -- verified to reproduce exp12's
    `jac_diag` to the printed digits on the whiten=False arm.
    """
    def h(z):
        x = dec(np.asarray(z, float))
        with torch.no_grad():
            return model.encode(torch.as_tensor(x, dtype=torch.float64)).numpy()
    return h


class LearnedBlock:
    """One block of a fitted ModularTransition, as a numpy `spectra.HasJacobian`.

    The block MLP takes only its own coordinates -- what `ModularTransition`
    structurally guarantees -- so a block can be iterated alone, which is the
    object Lemma C's gap is a statement about.  Verified against
    `torch.autograd.functional.jacobian` to 7e-10.
    """

    def __init__(self, dyn, k: int):
        a, b = dyn.bounds[k]
        self.net = dyn.nets[k]
        self.dim = b - a

    def _f(self, Z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(Z, float), dtype=torch.float64)
            return (t + self.net(t)).numpy()

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        return self._f(np.atleast_2d(z)).reshape(z.shape)

    def jacobian(self, z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        z = np.asarray(z, float).reshape(self.dim)
        E = np.eye(self.dim) * eps
        out = self._f(np.vstack([z + E, z - E]))
        return ((out[: self.dim] - out[self.dim :]) / (2.0 * eps)).T


def learned_step(dyn):
    def F_tilde(z):
        with torch.no_grad():
            return dyn(torch.as_tensor(np.asarray(z, float), dtype=torch.float64)).numpy()
    return F_tilde


def learned_spectra(dyn, z_fit0: np.ndarray) -> list[np.ndarray]:
    """Lyapunov spectrum of each learned block, along orbits of the learned map.

    Initial conditions are the fitted latents at t = 0, so the exponents are
    measured where the model was trained.  Caveat worth keeping in view: the
    orbits still run past the T = 15 window the data covers, and for the
    dominated block the data carries almost no signal after a few steps (3.11's
    design tension), so these are the least well determined numbers here.  They
    are horizon-stable from n = 25 to n = 300, so they are not an artifact of the
    horizon -- but that is not the same as being well constrained by data.
    """
    out = []
    for k, sl in enumerate(slices_of(PART)):
        blk = LearnedBlock(dyn, k)
        out.append(SP.lyapunov_spectrum_averaged(
            blk, z_fit0[:LYAP_N_Z0, sl], T=LYAP_T, warmup=LYAP_WARMUP))
    return out


def run_arm(strength, whiten):
    sys_a, X, Z, U, dec = build(strength, SEED)
    pts = Z.reshape(-1, 4)
    sub = np.random.default_rng(SEED).choice(len(pts), min(N_PROBE, len(pts)), replace=False)
    probe = pts[sub]

    # Off-trajectory probe: same system, same decoder, same law, different
    # trajectories.  On training data h(F z_t) = h(z_{t+1}) is literally the
    # encoder's own next latent, so the residual there coincides with the
    # prediction loss being minimised.  Here it does not.
    _, _, Z_off, _, _ = build(strength, SEED + 9999)
    off_pts = Z_off.reshape(-1, 4)
    off = off_pts[np.random.default_rng(1).choice(len(off_pts), N_PROBE, replace=False)]

    Urep_true = np.repeat(U, Z.shape[1])
    floor = BH.block_u_dependence(pts[:, 2:], Urep_true, normalize=True).total
    varying = BH.block_u_dependence(pts[:, :2], Urep_true, normalize=True).total

    cfg = ModelConfig(n_obs=N_OBS, d=4, partition=PART, decoder="mlp", encoder="mlp")
    w = W_WHITENED if whiten else W_RAW

    rows = []
    for r in range(N_RESTARTS):
        tc = T.TrainConfig(steps=STEPS, seed=SEED + 100 * r, w_behavior=w,
                           inv_start=INV[0], inv_stop=INV[1], behavior_whiten=whiten)
        res = T.fit(X, cfg, tc, U=U)
        model = res.model.double()
        h = h_of(model, dec)

        jr = M.jacobian_block_report(h, probe, PART, PART, rate_order=[0, 1])
        # assignment[r] = true module that fitted block r matches; true 0 is the
        # slow/dominant (s=0.90) module, true 1 the fast dominated one (s=0.50).
        fit_slow = int(np.argmin(np.asarray(jr.assignment)))
        fit_fast = 1 - fit_slow

        zf = res.z_fit.reshape(-1, 4)
        Urep = np.repeat(U, res.z_fit.shape[1])
        pinned = zf[:, INV[0]:INV[1]]

        c_on = M.conjugacy_residual(h, sys_a.step, learned_step(model.dyn), probe)
        c_off = M.conjugacy_residual(h, sys_a.step, learned_step(model.dyn), off)
        spec = learned_spectra(model.dyn, res.z_fit[:, 0, :])

        rows.append({
            "restart": r,
            "fit_quality": float(res.fit_quality),
            "jac_diag": float(jr.on_block),
            "jac_lower": float(jr.lower_mass),
            "jac_upper": float(jr.upper_mass),
            # (iv): was the behavioural constraint actually imposed?
            "udep_raw": float(BH.block_u_dependence(pinned, Urep).total),
            "udep_norm": float(BH.block_u_dependence(pinned, Urep, normalize=True).total),
            "block_scale_ratio": float(zf[:, :2].std() / max(zf[:, 2:].std(), 1e-12)),
            # (i), (ii), (iii)
            "conj_on_rel_step": c_on.rel_step,
            "conj_off_rel_step": c_off.rel_step,
            "conj_on_rel_state": c_on.rel_state,
            "learned_gap": float(np.min(spec[fit_slow]) - np.max(spec[fit_fast])),
            "additivity_defect": float(
                M.additivity_defect(h, probe, PART, slices_of(PART)[fit_fast])),
            "assignment": [int(a) for a in jr.assignment],
            "learned_spec_slow": [float(v) for v in spec[fit_slow]],
            "learned_spec_fast": [float(v) for v in spec[fit_fast]],
        })

    a = {k: np.array([row[k] for row in rows], dtype=float)
         for k in rows[0] if isinstance(rows[0][k], float)}
    return {
        "strength": strength,
        "whiten": whiten,
        "w_behavior": w,
        "obs_nonlinearity": observation_nonlinearity(dec, Z),
        "udep_floor_true_invariant": float(floor),
        "udep_true_varying": float(varying),
        "n_restarts": N_RESTARTS,
        "per_restart": rows,
        **{f"{k}_median": float(np.median(v)) for k, v in a.items()},
        **{f"{k}_min": float(v.min()) for k, v in a.items()},
        **{f"{k}_max": float(v.max()) for k, v in a.items()},
        **{f"{k}_sd": float(v.std()) for k, v in a.items()},
    }


def corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def within_arm_corr(arms, key_x, key_y) -> float:
    """Correlation with the arm mean removed from both variables.

    Every diagnostic rises with the dose because everything gets worse in a
    nonlinear regime, so a dose trend alone identifies nothing.  This asks
    whether a diagnostic tracks the degradation at FIXED dose, which a mere
    co-symptom would not.
    """
    xs, ys = [], []
    for a in arms:
        x = np.array([r[key_x] for r in a["per_restart"]], float)
        y = np.array([r[key_y] for r in a["per_restart"]], float)
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    return corr(np.concatenate(xs), np.concatenate(ys))


def main() -> int:
    # ASCII only in anything printed: the Windows console is cp1252 (exp12 note).
    banner("EXPERIMENT 13 -- task 32: which Lemma D hypothesis does a fitted h break?")
    print(f"   {N_RESTARTS} restarts x {STEPS} steps at each of {len(STRENGTHS)} doses,")
    print(f"   x2 penalty variants (raw w={W_RAW}, whitened w={W_WHITENED}); float64 readouts\n")

    groups = {}
    for whiten in (False, True):
        tag = "whitened" if whiten else "raw"
        print(f"   --- behavioural penalty: {tag.upper()} "
              f"{'(reproduces exp11/exp12)' if not whiten else '(CLAUDE.md 3.12 fix)'}")
        print(f"   {'obs-nl':>6s} {'fitq':>8s} | {'udep_raw':>8s} {'udep_nrm':>8s} "
              f"{'scale':>6s} | {'jac_diag':>8s} {'lower':>6s} {'upper':>6s} {'upmax':>6s} "
              f"| {'conj':>6s} {'lrngap':>7s} {'adddef':>6s}")
        arms = []
        for s in STRENGTHS:
            a = run_arm(s, whiten)
            arms.append(a)
            print(f"   {a['obs_nonlinearity']:6.3f} {a['fit_quality_median']:8.4f} "
                  f"| {a['udep_raw_median']:8.4f} {a['udep_norm_median']:8.4f} "
                  f"{a['block_scale_ratio_median']:6.1f} | {a['jac_diag_median']:8.3f} "
                  f"{a['jac_lower_median']:6.3f} {a['jac_upper_median']:6.3f} "
                  f"{a['jac_upper_max']:6.3f} | {a['conj_on_rel_step_median']:6.3f} "
                  f"{a['learned_gap_median']:7.3f} {a['additivity_defect_median']:6.3f}")
        groups[tag] = arms
        print()

    raw, wht = groups["raw"], groups["whitened"]
    floor = raw[0]["udep_floor_true_invariant"]
    true_varying = raw[0]["udep_true_varying"]
    print(f"   reference u-dependence (scale-normalised): true INVARIANT block "
          f"{floor:.4f}, true VARYING block {true_varying:.4f}")

    r_conj = within_arm_corr(raw + wht, "conj_on_rel_step", "jac_diag")
    r_add = within_arm_corr(raw + wht, "additivity_defect", "jac_diag")
    r_gap = within_arm_corr(raw + wht, "learned_gap", "jac_diag")
    print(f"   within-arm corr with jac_diag: conj {r_conj:+.3f}, "
          f"additivity {r_add:+.3f}, learned_gap {r_gap:+.3f}")

    banner("VERDICTS")
    checks = [
        (
            all(a["udep_norm_min"] > 2.0 * floor for a in raw),
            f"under the RAW penalty the pinned block is u-DEPENDENT at every dose "
            f"and every restart once the gauge is removed: scale-normalised "
            f"u-dependence {[round(a['udep_norm_median'], 3) for a in raw]} against a "
            f"true-invariant floor of {floor:.3f} -- the constraint was never imposed",
        ),
        (
            all(a["udep_raw_median"] < floor for a in raw),
            f"and it LOOKED imposed on the raw score "
            f"{[round(a['udep_raw_median'], 4) for a in raw]}, which is the trap: the "
            f"two readings disagree in direction because the block was shrunk "
            f"(scale ratios {[round(a['block_scale_ratio_median'], 1) for a in raw]})",
        ),
        (
            all(a["udep_norm_max"] < floor for a in wht),
            f"the WHITENED penalty imposes what the raw one only appeared to: "
            f"scale-normalised u-dependence {[round(a['udep_norm_median'], 3) for a in wht]} "
            f"at or below the floor, scale ratios "
            f"{[round(a['block_scale_ratio_median'], 1) for a in wht]}",
        ),
        (
            wht[-1]["jac_diag_median"] > raw[-1]["jac_diag_median"] + 0.15,
            f"at the TOP dose, imposing behaviour for real recovers block structure "
            f"the raw penalty could not: jac_diag {raw[-1]['jac_diag_median']:.3f} -> "
            f"{wht[-1]['jac_diag_median']:.3f} -- exp12's headline (B+C degrades to a "
            f"filtration under nonlinearity) was an artifact of the broken penalty",
        ),
        (
            max(a["jac_upper_max"] for a in wht) < 0.15
            and wht[-1]["jac_lower_median"] < 0.10,
            f"and the result is BLOCK-DIAGONAL, not merely triangular: forbidden "
            f"upper <= {max(a['jac_upper_max'] for a in wht):.3f} over all doses and "
            f"restarts, and the ALLOWED lower mass is only "
            f"{wht[-1]['jac_lower_median']:.3f} at the top dose -- B+C's load-bearing "
            f"claim, tested here for the first time",
        ),
        (
            all(b >= a - 1e-9 for a, b in zip(
                [x["conj_on_rel_step_median"] for x in wht],
                [x["conj_on_rel_step_median"] for x in wht][1:])),
            f"(i) the conjugacy residual rises with dose under the fixed penalty: "
            f"{[round(a['conj_on_rel_step_median'], 3) for a in wht]} "
            f"(off-trajectory {[round(a['conj_off_rel_step_median'], 3) for a in wht]})",
        ),
        (
            min(a["learned_gap_min"] for a in wht) > 0.0,
            f"(ii) the LEARNED one-sided gap survives at every dose and restart "
            f"(min {min(a['learned_gap_min'] for a in wht):+.3f}; the true gap is "
            f"{float(np.log(0.90) - np.log(0.50)):.3f}) -- Lemma C still applies to "
            f"the fitted model",
        ),
        (
            wht[-1]["additivity_defect_median"] > 2.0 * wht[0]["additivity_defect_median"],
            f"(iii) h_B loses additivity with dose: mixed-curvature share "
            f"{[round(a['additivity_defect_median'], 4) for a in wht]} -- Lemma D's "
            f"open case (a) is on the critical path",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    print("""
   Reading: 1-3 PASS settles task 32 as candidate (iv) -- exp11/exp12 never
   imposed the behavioural hypothesis, so their B+C conclusions are void and
   approaches.md B.1's B column is untested rather than refuted.  Check 4 is then
   the first real test of B+C under a nonlinear observation map: PASS revives it,
   FAIL is a stronger negative than exp12 was entitled to claim.  5-7 report the
   other three candidates; a FAIL there is informative, not bad.""")

    save(
        "exp13_conjugacy_residual",
        {
            "seed": SEED, "n_obs": N_OBS, "steps": STEPS, "n_restarts": N_RESTARTS,
            "strengths": list(STRENGTHS), "w_behavior_raw": W_RAW,
            "w_behavior_whitened": W_WHITENED, "u_levels": U_LEVELS.tolist(),
            "n_per_u": N_PER_U, "T": T_STEPS, "n_probe": N_PROBE,
            "lyap_T": LYAP_T, "lyap_warmup": LYAP_WARMUP,
            "arms_raw": raw, "arms_whitened": wht,
            "udep_floor_true_invariant": floor, "udep_true_varying": true_varying,
            "corr_conj_diag_within_arm": r_conj,
            "corr_additivity_diag_within_arm": r_add,
            "corr_learned_gap_diag_within_arm": r_gap,
            "all_passed": passed,
            "checks": [{"passed": ok, "claim": m} for ok, m in checks],
        },
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
