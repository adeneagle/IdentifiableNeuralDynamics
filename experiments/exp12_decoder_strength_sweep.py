"""Experiment 12 -- CLAUDE.md task 29: does B∘C survive a *nonlinear* observation map?

!! SUPERSEDED IN PART (CLAUDE.md §3.12, task 32).  The behavioural penalty this
!! experiment trains against is gauge-dependent: it falls like eps^2/eps^4 when
!! the pinned block shrinks, so the optimiser satisfied it by making the block
!! 21x smaller rather than u-invariant, while the block still carried the
!! u-varying latent at distance correlation 0.99.  **No arm of this sweep ever
!! imposed Lemma D's behavioural hypothesis**, so nothing here is evidence about
!! B∘C, and the framing below -- "behaviour's half fails, Lemma C's holds" -- is
!! not supported by the data it reports.  What survives is the dose-response
!! itself: observation nonlinearity monotonically degrades the block structure a
!! *dynamics-only* fit recovers.  `exp13` re-runs this with the penalty fixed.

`exp11` recovers block-diagonal structure when observations are linear and only
triangular structure when they are nonlinear.  That is either a real limit of the
B∘C composition or an artifact of one seed at one nonlinearity strength.  This
experiment settles it by sweeping the strength and asking for **monotonicity**:

    if increasing observation nonlinearity monotonically drives `jac_diag` down
    and `jac_lower` up, the effect is the nonlinearity and not the seed.

Why monotonicity rather than a single contrast: with sd(jac_diag) ~ 0.1 in the
nonlinear regime (§3.11) any two-point comparison is within noise.  A monotone
trend across four strengths, each averaged over many restarts, is not.

`strength = 0` is the control: `MLPDecoder` is then an exactly linear map, so
this arm must reproduce `exp11`'s linear-decoder result (`jac_diag` ~ 0.99).
That the control and the treatment differ only in one scalar is the point --
`exp11` confounded the decoder change with nothing else, but it also could not
show a dose-response.

What each outcome means:
  * monotone decline           -> B∘C reaches only a filtration once h is genuinely
                                  nonlinear; `approaches.md` §B.1 is wrong as written
                                  and the partial-iVAE lemma is not worth proving.
  * flat, all near 0.99        -> `exp11`'s nonlinear run was an artifact; B∘C stands.
  * flat, all near 0.76        -> something other than nonlinearity broke it (look at
                                  conditioning: cond(J) grows fast with strength).
  * non-monotone / huge spread -> underpowered; raise N_RESTARTS before concluding.

Only the FIRST is a falsification, and only if the control arm passes.
"""

from __future__ import annotations

import numpy as np
import torch

from _common import banner, save, verdict
from idyn import metrics as M
from idyn import systems as S
from idyn import train as T
from idyn.models import ModelConfig

SEED = 0
N_OBS = 12
U_LEVELS = np.array([0, 1, 2, 3])
N_PER_U = 250
T_STEPS = 15
STEPS = 4000          # §3.11: 1200 undertrains a nonlinear decoder into a reversed result
N_RESTARTS = 8        # §3.11: sd(jac_diag) ~ 0.104, so 3 cannot resolve a trend
W_BEHAVIOR = 5.0
INV = (2, 4)
# Chosen by the DOSE they produce on the contracted trajectory data, not by the
# parameter value.  The first attempt swept (0, 0.25, 0.5, 1.0), which lands on
# doses (0.00, 0.31, 0.31, 0.35) -- effectively two levels, so the monotonicity
# test had no range to work with and failed for want of a treatment.  These give
# roughly (0.00, 0.31, 0.43, 0.60).  Latents contract toward the origin where
# tanh is near-linear, so the dose on real data is well below the design value:
# always read `obs-nl`, never `strength`.
STRENGTHS = (0.0, 0.5, 1.5, 2.0)


def build(strength, rng):
    """Aligned system (varying block = spectrally dominant) + a decoder."""
    sys_a = S.ModularSystem(
        [S.TwistBlock(s=0.90, omega=0.40, beta=0.6), S.TwistBlock(s=0.50, omega=1.10, beta=-0.5)]
    )
    dec = S.MLPDecoder.random(N_OBS, 4, rng, strength=strength)
    X, Z, U, dec = T.make_behavioural_dataset(
        sys_a, 2, 2, N_OBS, N_PER_U, T_STEPS, U_LEVELS, rng, mode="variance", decoder=dec
    )
    return X, Z, U, dec


def observation_nonlinearity(dec, Z):
    """Fraction of g not captured by any affine map -- the dose, measured."""
    z = np.asarray(Z, float).reshape(-1, Z.shape[-1])
    x = dec(z)
    aug = np.hstack([z, np.ones((len(z), 1))])
    A, *_ = np.linalg.lstsq(aug, x, rcond=None)
    return float(np.linalg.norm(x - aug @ A) / np.linalg.norm(x))


def h_of(res, dec):
    def h(z):
        x = dec(np.asarray(z, float))
        with torch.no_grad():
            return res.model.encode(torch.tensor(x, dtype=torch.float32)).numpy().astype(float)
    return h


def run_arm(strength):
    rng = np.random.default_rng(SEED)
    X, Z, U, dec = build(strength, rng)
    pts = Z.reshape(-1, 4)
    sub = np.random.default_rng(SEED).choice(len(pts), min(2000, len(pts)), replace=False)
    cfg = ModelConfig(n_obs=N_OBS, d=4, partition=[2, 2], decoder="mlp", encoder="mlp")

    rows = []
    for r in range(N_RESTARTS):
        tc = T.TrainConfig(steps=STEPS, seed=SEED + 100 * r, w_behavior=W_BEHAVIOR,
                           inv_start=INV[0], inv_stop=INV[1],
                           # Pinned to the OLD, gauge-dependent penalty so this
                           # script keeps reproducing the JSON on record.
                           # CLAUDE.md 3.12: it is satisfied by shrinking the
                           # block, so the behavioural reading below is void --
                           # the dose-response is real, but it measures
                           # dynamics-only fitting.  exp13 is the replacement.
                           behavior_whiten=False)
        res = T.fit(X, cfg, tc, U=U)
        jr = M.jacobian_block_report(h_of(res, dec), pts[sub], [2, 2], [2, 2], rate_order=[0, 1])
        rows.append({
            "fit_quality": float(res.fit_quality),
            "jac_diag": float(jr.on_block),
            "jac_lower": float(jr.lower_mass),
            "jac_upper": float(jr.upper_mass),
        })
    # conditioning of the observation map on the data it is actually applied to.
    # At high strength the coupling flow becomes ill-conditioned and the encoder
    # simply cannot invert it -- that would look exactly like "nonlinearity breaks
    # recovery" while really being a fit failure, so it is reported alongside.
    Jd = M.jacobian_of(dec, pts[sub[:200]])
    sv = np.linalg.svd(Jd, compute_uv=False)
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return {
        "strength": strength,
        "obs_nonlinearity": observation_nonlinearity(dec, Z),
        "decoder_cond": float(np.median(sv[:, 0] / sv[:, -1])),
        "n_restarts": N_RESTARTS,
        "per_restart": rows,
        **{f"{k}_median": float(np.median(v)) for k, v in a.items()},
        **{f"{k}_min": float(v.min()) for k, v in a.items()},
        **{f"{k}_max": float(v.max()) for k, v in a.items()},
        **{f"{k}_sd": float(v.std()) for k, v in a.items()},
    }


def main() -> int:
    # ASCII only in anything printed: the Windows console is cp1252, and a stray
    # 'B∘C' or '§' in a banner raises UnicodeEncodeError mid-run.  Docstrings are
    # safe (never printed); f-strings that reach stdout are not.
    banner("EXPERIMENT 12 -- task 29: B+C vs observation nonlinearity (dose-response)")
    print(f"   {N_RESTARTS} restarts x {STEPS} steps at each of {len(STRENGTHS)} strengths\n")
    print(f"   {'stren':>5s} {'obs-nl':>7s} {'condJ':>7s} {'fitq':>9s} "
          f"| {'jac_diag':>19s} {'sd':>6s} | {'lower':>6s} | {'upper':>6s} {'max':>6s}")

    arms = []
    for s in STRENGTHS:
        a = run_arm(s)
        arms.append(a)
        print(f"   {s:5.2f} {a['obs_nonlinearity']:7.3f} {a['decoder_cond']:7.1f} "
              f"{a['fit_quality_median']:9.2e} "
              f"| {a['jac_diag_median']:8.3f} [{a['jac_diag_min']:.3f},{a['jac_diag_max']:.3f}] "
              f"{a['jac_diag_sd']:6.3f} | {a['jac_lower_median']:6.3f} "
              f"| {a['jac_upper_median']:6.3f} {a['jac_upper_max']:6.3f}")

    doses = [a["obs_nonlinearity"] for a in arms]
    print(f"\n   doses actually delivered: {[round(d, 3) for d in doses]}")
    if max(doses) - min(d for d in doses if d > 0) < 0.15:
        print("   !! WARNING: dose range too narrow to test monotonicity -- widen STRENGTHS")

    diag = [a["jac_diag_median"] for a in arms]
    lower = [a["jac_lower_median"] for a in arms]
    upper_max = max(a["jac_upper_max"] for a in arms)
    control, treated = arms[0], arms[-1]

    banner("VERDICTS")
    checks = [
        (
            control["jac_diag_median"] > 0.95,
            f"the linear control arm (strength 0) reproduces exp11's linear-decoder "
            f"result: jac_diag {control['jac_diag_median']:.3f} -- without this the "
            f"sweep measures nothing, since every other arm is compared against it",
        ),
        (
            all(b <= a + 1e-9 for a, b in zip(diag, diag[1:])),
            f"jac_diag falls monotonically with observation nonlinearity: "
            f"{[round(d, 3) for d in diag]} -- a dose-response, not a seed effect",
        ),
        (
            all(b >= a - 1e-9 for a, b in zip(lower, lower[1:])),
            f"and the ALLOWED cross-block rises to absorb it: "
            f"{[round(x, 3) for x in lower]} -- the map is becoming triangular, "
            f"not merely worse",
        ),
        (
            upper_max < 0.15,
            f"the FORBIDDEN cross-block stays suppressed at every strength and in "
            f"every restart (max {upper_max:.3f}) -- Lemma C's half of the "
            f"composition survives nonlinearity; it is behaviour's half that does not",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    print(f"""
   Reading: all four PASS => B+C reaches only a FILTRATION once the observation
   map is genuinely nonlinear, and approaches.md B.1 needs revising (behaviour
   supplies its zero only when CLAUDE.md 3.5 has already forced h into GL(d)).
   Check 1 failing invalidates the whole sweep.  Checks 2-3 failing with large sd
   means underpowered, not refuted -- raise N_RESTARTS.""")

    save(
        "exp12_decoder_strength_sweep",
        {
            "seed": SEED, "n_obs": N_OBS, "steps": STEPS, "n_restarts": N_RESTARTS,
            "strengths": list(STRENGTHS), "w_behavior": W_BEHAVIOR,
            "u_levels": U_LEVELS.tolist(), "n_per_u": N_PER_U, "T": T_STEPS,
            "arms": arms,
            "diag_medians": diag, "lower_medians": lower, "upper_max": upper_max,
            "all_passed": passed,
            "checks": [{"passed": ok, "claim": m} for ok, m in checks],
        },
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
