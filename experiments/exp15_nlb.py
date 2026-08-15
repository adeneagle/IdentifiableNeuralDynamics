"""exp15 -- the empirical program on real data (CLAUDE.md tasks 39 and 40).

**The first experiment in this repo whose generator is not `make_dataset`.**
Everything before it observed a system we wrote down; this one observes a monkey
(Neural Latents Benchmark MC_Maze, dandiset 000128, 1721 train trials of macaque
M1+PMd during a delayed-reach maze task).

Two questions, in the order §1.2 splits them, and only the second is about
identifiability:

  part 2  (task 39)  ADEQUACY.  Is the structure there at all?  Nested ladder
                     unconstrained > triangular > modular, scored by held-out
                     *neuron* co-smoothing.  This is gauge-invariant by
                     construction, so it cannot answer identifiability -- it can
                     only say whether the constraint costs anything.

  part 3  (task 40)  IDENTIFIABILITY.  Fit independently on **disjoint neuron
                     subsets** and compare the fits to each other.  Identifiable
                     dynamics give different coordinates and the same
                     invariants.  Varies the *data*, not the seed, so unlike
                     restarts it excludes "artefact of this sample of neurons".

  part 4             NEGATIVE CONTROLS.  §3.11: a treatment arm is
                     uninterpretable without a control the metric can fail.
                     Two, both leaving one half untouched and altering the
                     other: time reversal (dynamics inverted, spectra negate)
                     and within-condition time shuffling (dynamics destroyed).

  part 5             Do the fitted models satisfy the hypotheses the theory
                     needs?  (F3) ordered separation, Tier-1 regularity,
                     GL(2,Z) margin.  Reported whether or not it is flattering:
                     on `exp14`'s synthetic cycles (F3) held in 0 of 24.

Protocol is CLAUDE.md §3.13(e), which is not optional here:
  * many restarts, and report the **fraction** of agreeing pairs plus the
    median, never a max;
  * screen on `duplicate_modules` -- **not** on fit quality or coherence, both
    of which were measured to be uninformative or wrong-signed;
  * report **per-invariant** (§3.13(b)): one boolean hides which half of the
    fingerprint the data actually constrained;
  * read invariants **inside the data horizon** (§3.13(a)): a fitted map
    iterated past its training trajectories invents a fixed point.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idyn import metrics as M                       # noqa: E402
from idyn import nlb, spectra as SP                 # noqa: E402
from idyn import train as T                         # noqa: E402
from idyn.models import LearnedSystem, ModelConfig  # noqa: E402

SEED = 20260814
DATASET = "mc_maze"
BIN_MS = 20.0
WINDOW_MS = (-250.0, 450.0)
SMOOTH_MS = 40.0
# Primary K = 2, and the reason is measured rather than aesthetic.  A conjugacy
# acts on H_1(T^K) = Z^K (task 23), so what two fits can agree on is the
# GL(K,Z) *orbit* of the rotation vector.  That quotient loses power fast with
# K: for random rotation vectors on this data's scale (|rho| <= 0.025) the
# median null margin is 0.0025 at K = 2 but 0.0009 at K = 3, where 53% of
# *random* pairs match to 1e-3.  At K = 3 the agreement would therefore be
# uninformative by construction.  K = 3 is still run, and reported as such.
PART = [2, 2]
D = sum(PART)
PART_SECONDARY = [2, 2, 2]
STEPS = 3000
N_SPLITS = 3                     # independent neuron partitions
N_RESTARTS = 6                   # fits per half per split
LADDER_RESTARTS = 5
# Recalibrated for this data, and the recalibration is not optional.  exp14's
# synthetic limit cycles have |rho| ~ 0.1-0.3; here the fitted rotation numbers
# are ~0.002-0.023, an order of magnitude smaller, so exp14's rot_tol = 0.01
# would call almost anything a match.  The thresholds below sit under the
# measured GL(2,Z) null median (0.0025), so clearing them means something.
# The primary readout is the error *distribution* against the controls, which
# needs no threshold at all; frac_agree is secondary.
SPEC_TOL = 0.02
ROT_TOL = 0.002
OUT = Path(__file__).resolve().parents[1] / "results" / "exp15_nlb.json"


def banner(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def prepare(td: nlb.TrialData, smooth_ms: float = SMOOTH_MS):
    """PSTHs -> the (C, T, N) tensor the fitter consumes.

    Soft-normalise per neuron before centring, so a single high-rate unit does
    not dominate a mean-squared reconstruction; that is a gauge choice on the
    observation side and is stated rather than hidden.
    """
    R, labels, n_per = td.condition_average(smooth_ms=smooth_ms)
    rng_ = R.max(axis=(0, 1)) - R.min(axis=(0, 1))
    X = R / (rng_ + 0.5)
    X = X - X.mean(axis=(0, 1), keepdims=True)
    return X / X.std(), labels, n_per


def split_half_reliability(td: nlb.TrialData, smooth_ms: float, seed: int) -> float:
    """Correlation between PSTHs built from disjoint halves of the trials.

    The noise ceiling.  Any variance-explained number below is read against
    this, never against 1.0 -- §3.9's rule that a floor is measured, not assumed.
    """
    rng = np.random.default_rng(seed)
    labels, inv = np.unique(td.condition, return_inverse=True)
    A, B = [], []
    for c in range(len(labels)):
        idx = np.flatnonzero(inv == c)
        rng.shuffle(idx)
        h = len(idx) // 2
        if h < 3:
            continue
        A.append(td.spikes[idx[:h]].mean(0))
        B.append(td.spikes[idx[h : 2 * h]].mean(0))
    A, B = np.array(A) / (td.bin_ms / 1e3), np.array(B) / (td.bin_ms / 1e3)
    if smooth_ms > 0:
        s = smooth_ms / td.bin_ms
        hw = int(np.ceil(3 * s))
        k = np.exp(-0.5 * (np.arange(-hw, hw + 1) / s) ** 2)
        k /= k.sum()

        def sm(Z):
            p = np.pad(Z, ((0, 0), (hw, hw), (0, 0)), mode="edge")
            return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, p)

        A, B = sm(A), sm(B)
    A, B = np.sqrt(np.maximum(A, 0)), np.sqrt(np.maximum(B, 0))
    a = (A - A.mean((0, 1))).ravel()
    b = (B - B.mean((0, 1))).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def fit_one(X: np.ndarray, seed: int, structure: str = "modular", part=None):
    part = list(PART if part is None else part)
    cfg_m = ModelConfig(
        n_obs=X.shape[-1], d=sum(part), partition=part, structure=structure,
        decoder="linear", encoder="linear",
    )
    return T.fit(X, cfg_m, T.TrainConfig(steps=STEPS, seed=seed))


def fingerprint_of(res, X: np.ndarray, part=None) -> M.DynamicalFingerprint:
    """Fingerprint a fitted transition, read INSIDE the data horizon (§3.13a)."""
    part = list(PART if part is None else part)
    dyn = res.model.double().dyn
    z0 = np.asarray(res.z_fit, float)[:, 0, :]
    n_t = X.shape[1]
    warm = max(n_t // 4, 2)
    read = n_t - warm
    return M.dynamical_fingerprint(
        LearnedSystem(dyn, part), z0, T=read, warmup=warm, T_rotation=read
    )


def lattice_null(K: int, scale: float, seed: int, n: int = 300) -> dict:
    """Null distribution of the GL(K,Z) margin for random rotation vectors.

    The reference scale for any lattice-margin claim.  §3.9's rule: a floor is
    measured, not assumed -- and here the floor is the whole story, because the
    quotient's power collapses as K grows.
    """
    rng = np.random.default_rng(seed)
    ms = [
        SP.rotation_lattice_margin(
            rng.uniform(0, scale, K).tolist(), rng.uniform(0, scale, K).tolist()
        )[0]
        for _ in range(n)
    ]
    ms = np.asarray(ms)
    return {
        "median": float(np.median(ms)),
        "p10": float(np.percentile(ms, 10)),
        "p90": float(np.percentile(ms, 90)),
        "scale": scale,
        "K": K,
    }


def tier1_spectrum(fp: M.DynamicalFingerprint) -> np.ndarray:
    """The **global** Lyapunov spectrum: all d exponents, pooled and sorted.

    This is the §1.2 Tier-1 object, and it is the one readout here that the
    task-23 ambiguity cannot touch.  A conjugacy acts on H_1(T^K) = Z^K and can
    move rotation numbers *between* modules, but the Lyapunov spectrum of the
    whole system is a conjugacy invariant of F however its factors are grouped.

    So it is the honest primary: Tier 1 costs no theorem (injective decoders
    alone give Ftilde = h F h^-1), and it is unambiguous.  Per-module rotation
    numbers are Tier 2 and are only pinned up to the lattice.
    """
    return np.sort(np.concatenate([np.asarray(s).ravel() for s in fp.spectra]))


def tier1_error(fa: M.DynamicalFingerprint, fb: M.DynamicalFingerprint) -> float:
    a, b = tier1_spectrum(fa), tier1_spectrum(fb)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max())


def rotation_separation(fp: M.DynamicalFingerprint) -> float:
    """Smallest gap between this fit's own module rotation numbers.

    The natural scale for a rotation *error*.  An absolute error of 0.002 is
    unreadable on its own -- it is excellent if the modules sit 0.011 apart and
    meaningless if they sit 0.002 apart.  Dividing by this asks the question the
    claim actually rests on: **can the two halves tell the modules apart?**
    Same role `order_margin` plays for the spectral ordering (§3.13(c)).
    """
    r = sorted(abs(x) for x in fp.rotations)
    if len(r) < 2:
        return float("inf")
    return float(min(b - a for a, b in zip(r, r[1:])))


def dump_fp(fp: M.DynamicalFingerprint) -> dict:
    """Everything needed to re-score offline without refitting (§3.13)."""
    return {
        "partition": list(fp.partition),
        "spectra": [np.asarray(s).ravel().tolist() for s in fp.spectra],
        "rotations": [float(r) for r in fp.rotations],
        "coherences": [float(c) for c in fp.coherences],
        "filtration_gap": float(fp.filtration_gap),
        "is_filtration": bool(fp.is_filtration),
        "duplicate_modules": [list(p) for p in fp.duplicate_modules()],
    }


def pair_stats(fps_a, fps_b, screen: bool) -> dict:
    """All cross-half fingerprint comparisons, optionally duplicate-screened."""
    rot, spec, agr, order, f3, lat, t1, rel = [], [], [], [], [], [], [], []
    n_skip = 0
    for fa in fps_a:
        for fb in fps_b:
            if screen and (fa.duplicate_modules() or fb.duplicate_modules()):
                n_skip += 1
                continue
            r = M.invariant_agreement(fa, fb, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
            rot.append(r.rotation_error)
            spec.append(r.spectrum_error)
            agr.append(bool(r.agree))
            order.append(bool(r.order_agrees))
            f3.append(bool(fa.is_filtration and fb.is_filtration))
            t1.append(tier1_error(fa, fb))
            sep = min(rotation_separation(fa), rotation_separation(fb))
            rel.append(r.rotation_error / sep if sep > 0 else float("inf"))
            lat.append(
                SP.rotation_lattice_margin(
                    [abs(x) for x in fa.rotations], [abs(x) for x in fb.rotations]
                )[0]
            )
    if not rot:
        return {"n_pairs": 0, "n_screened_out": n_skip, "n_pairs_ok": 0}
    return {
        "n_pairs": len(rot),
        "n_pairs_ok": len(rot),
        "n_screened_out": n_skip,
        "tier1_spectrum_error_median": float(np.median(t1)),
        "tier1_spectrum_error_iqr": [float(np.percentile(t1, 25)), float(np.percentile(t1, 75))],
        "rotation_error_median": float(np.median(rot)),
        "rotation_error_rel_median": float(np.median(rel)),
        "rotation_error_iqr": [float(np.percentile(rot, 25)), float(np.percentile(rot, 75))],
        "spectrum_error_median": float(np.median(spec)),
        "spectrum_error_iqr": [float(np.percentile(spec, 25)), float(np.percentile(spec, 75))],
        "lattice_margin_median": float(np.median(lat)),
        "frac_agree": float(np.mean(agr)),
        "frac_order_agree": float(np.mean(order)),
        "frac_both_filtration": float(np.mean(f3)),
    }


def cosmooth_score(X_in: np.ndarray, X_out: np.ndarray, res, ridge: float = 1e-3) -> float:
    """Held-out-NEURON co-smoothing: refit a fresh decoder from latents to X_out.

    The scored neurons were never seen by the model, so this cannot be won by
    memorising single-neuron noise.  **It is gauge-invariant** -- if zhat = h(z)
    the refitted decoder is D.h and the score is unchanged -- which is exactly
    why it is an adequacy gate and not an identifiability test.

    A *linear* held-out decoder is a deliberate choice, not a neutral one: it
    makes the score invariant only under linear h, so it implicitly rewards
    latents from which the population is linearly readable.
    """
    z = np.asarray(res.z_fit, float)
    d = z.shape[-1]                      # the model's own latent dim, not D
    z = z.reshape(-1, d)
    y = X_out.reshape(-1, X_out.shape[-1])
    n = z.shape[0]
    half = n // 2
    idx = np.arange(n)
    # fit the readout on half the (condition, time) points, score on the rest
    A = np.c_[z[idx[:half]], np.ones(half)]
    B = np.c_[z[idx[half:]], np.ones(n - half)]
    W = np.linalg.solve(A.T @ A + ridge * np.eye(d + 1), A.T @ y[idx[:half]])
    pred = B @ W
    truth = y[idx[half:]]
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    rec: dict = {
        "seed": SEED,
        "params": {
            "dataset": DATASET, "bin_ms": BIN_MS, "window_ms": list(WINDOW_MS),
            "smooth_ms": SMOOTH_MS, "partition": PART, "steps": STEPS,
            "n_splits": N_SPLITS, "n_restarts": N_RESTARTS,
            "spec_tol": SPEC_TOL, "rot_tol": ROT_TOL,
        },
    }
    checks: list[tuple[str, bool]] = []

    # ------------------------------------------------------------------
    banner("PART 1 -- the data, and what a model at this d can possibly explain")
    td = nlb.load_trials(DATASET, bin_ms=BIN_MS, window_ms=WINDOW_MS)
    print(" ", td.summary())
    X, labels, n_per = prepare(td)
    rel = split_half_reliability(td, SMOOTH_MS, SEED)
    print(f"  split-half PSTH reliability (noise ceiling): r = {rel:.3f}")

    flat = X.reshape(-1, X.shape[-1])
    sv = np.linalg.svd(flat - flat.mean(0), compute_uv=False)
    pca_ve = float(np.cumsum(sv**2 / (sv**2).sum())[D - 1])
    res0 = fit_one(X, SEED)
    import torch

    with torch.no_grad():
        L = res0.model.losses(torch.as_tensor(X.astype(np.float32)))
    model_ve = 1.0 - float(L["recon"])
    dyn_res = float(L["dyn"])
    print(f"  linear PCA ceiling at d={D}: VE = {pca_ve:.4f}")
    print(f"  modular model at d={D}:      VE = {model_ve:.4f}   dynamics residual = {dyn_res:.4f}")
    print(f"  -> modular constraint costs {100*(pca_ve-model_ve):+.2f} VE points against PCA")
    # How nonlinear is the latent dynamics?  This qualifies everything below and
    # is measured before any of it.  For a LINEAR system with distinct
    # complex-conjugate eigenvalue pairs, block-diagonality is generic -- the
    # eigen-decomposition supplies it -- so if the latent flow is essentially
    # linear then "modular costs nothing" is a fact about linear algebra, and
    # what a real-data result can validate is Theorem A (proved, `linear_case.md`)
    # rather than Theorem B.
    lin = {}
    for k in (4, 6, 8):
        Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)[2][:k]
        Z = ((flat - flat.mean(0)) @ Vt.T).reshape(X.shape[0], X.shape[1], k)
        Z = Z / Z.std()
        a_in, a_out = Z[:, :-1].reshape(-1, k), Z[:, 1:].reshape(-1, k)
        W = np.linalg.lstsq(a_in, a_out, rcond=None)[0]
        mse_lin = float(((a_out - a_in @ W) ** 2).mean())
        feats = np.concatenate(
            [a_in] + [a_in[:, i : i + 1] * a_in[:, j : j + 1]
                      for i in range(k) for j in range(i, k)], axis=1
        )
        W2 = np.linalg.lstsq(feats, a_out, rcond=None)[0]
        mse_quad = float(((a_out - feats @ W2) ** 2).mean())
        var = float((a_out**2).mean())
        lin[k] = {
            "linear_r2": 1 - mse_lin / var,
            "quadratic_r2": 1 - mse_quad / var,
            "residual_reduction": (mse_lin - mse_quad) / mse_lin,
        }
        print(
            f"  latent flow at d={k}: linear R2 {lin[k]['linear_r2']:.4f}, "
            f"+quadratic {lin[k]['quadratic_r2']:.4f} "
            f"(residual -{100*lin[k]['residual_reduction']:.0f}%)"
        )
    print("  -> the flow is overwhelmingly linear; see the write-up for what that")
    print("     does and does not let a modularity result claim.")

    rec["part1"] = {
        "data": td.record(), "n_conditions": int(X.shape[0]),
        "split_half_reliability": rel, "pca_ve_at_d": pca_ve,
        "modular_ve_at_d": model_ve, "dynamics_residual": dyn_res,
        "latent_linearity": lin,
    }
    # the modular fit must reach the linear ceiling: if it cannot, nothing below
    # is interpretable, because the latents would not describe the population.
    checks.append(("modular fit reaches the linear PCA ceiling", model_ve > pca_ve - 0.02))

    # ------------------------------------------------------------------
    banner("PART 2 -- task 39: co-smoothing over the nested ladder")
    rate = td.spikes.mean(axis=(0, 1)) / (BIN_MS / 1e3)
    parts = nlb.neuron_split(td.n_units, seed=SEED, n_parts=5, rate=rate)
    held = parts[0]                                   # ~20% held-out neurons
    seen = np.concatenate(parts[1:])
    X_seen, X_held = X[:, :, seen], X[:, :, held]
    print(f"  {len(seen)} neurons fitted, {len(held)} held out for co-smoothing")

    ladder: dict = {}
    for structure in ("unconstrained", "triangular", "modular"):
        scores, fq = [], []
        for r in range(LADDER_RESTARTS):
            res = fit_one(X_seen, SEED + 100 * r, structure=structure)
            scores.append(cosmooth_score(X_seen, X_held, res))
            fq.append(float(res.fit_quality))
        ladder[structure] = {
            "cosmooth_median": float(np.median(scores)),
            "cosmooth_min": float(np.min(scores)),
            "cosmooth_max": float(np.max(scores)),
            "cosmooth_all": [float(s) for s in scores],
            "fit_quality_median": float(np.median(fq)),
        }
        print(
            f"  {structure:14} co-smoothing R2 median {np.median(scores):.4f}  "
            f"range [{np.min(scores):.4f}, {np.max(scores):.4f}]  "
            f"fitq {np.median(fq):.4f}"
        )
    rec["part2_ladder"] = ladder
    gap = ladder["unconstrained"]["cosmooth_median"] - ladder["modular"]["cosmooth_median"]
    print(f"  -> unconstrained minus modular: {gap:+.4f} R2")
    rec["part2_ladder"]["unconstrained_minus_modular"] = float(gap)

    # §3.11: a flat ladder is unattributable without an arm the metric CAN
    # separate.  Latent dimension is that arm -- if co-smoothing moves with d
    # but not with structure, then "structure is free" is a finding rather than
    # a dead readout.
    dim_sweep = {}
    for part in ([2], [2, 2], [2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2, 2]):
        res = fit_one(X_seen, SEED, part=part)
        dim_sweep[sum(part)] = float(cosmooth_score(X_seen, X_held, res))
    print(f"  sensitivity control -- co-smoothing vs latent dim: "
          f"{ {k: round(v, 4) for k, v in dim_sweep.items()} }")
    rec["part2_ladder"]["dimension_sweep"] = dim_sweep
    span = max(dim_sweep.values()) - min(dim_sweep.values())
    restart_spread = max(
        ladder[s]["cosmooth_max"] - ladder[s]["cosmooth_min"]
        for s in ("unconstrained", "triangular", "modular")
    )
    checks.append((
        "co-smoothing separates latent dimension (so a flat ladder means something)",
        span > 10 * restart_spread,
    ))
    rec["part2_ladder"]["restart_spread"] = float(restart_spread)

    # ------------------------------------------------------------------
    banner("PART 3 -- task 40: invariant agreement across DISJOINT neuron sets")
    treat_fps: list[tuple[list, list]] = []
    for s in range(N_SPLITS):
        halves = nlb.neuron_split(td.n_units, seed=SEED + s, n_parts=2, rate=rate)
        fps = []
        for hidx, cols in enumerate(halves):
            fp_h = []
            for r in range(N_RESTARTS):
                res = fit_one(X[:, :, cols], SEED + 7919 * s + 100 * r + hidx)
                fp_h.append(fingerprint_of(res, X))
            fps.append(fp_h)
        treat_fps.append((fps[0], fps[1]))
        raw = pair_stats(fps[0], fps[1], screen=False)
        scr = pair_stats(fps[0], fps[1], screen=True)
        print(
            f"  split {s} ({len(halves[0])}/{len(halves[1])} neurons): "
            f"raw rot-err {raw['rotation_error_median']:.4f} "
            f"agree {raw['frac_agree']:.2f}  |  screened rot-err "
            f"{scr.get('rotation_error_median', float('nan')):.4f} "
            f"agree {scr.get('frac_agree', float('nan')):.2f} "
            f"({scr['n_screened_out']} pairs dropped)"
        )

    def pooled(pairs, screen):
        keys = (
            "tier1_spectrum_error_median", "rotation_error_median",
            "rotation_error_rel_median",
            "spectrum_error_median", "lattice_margin_median",
        )
        acc: dict[str, list[float]] = {k: [] for k in keys}
        agr, f3 = [], []
        for a, b in pairs:
            st = pair_stats(a, b, screen=screen)
            if st["n_pairs"] == 0:
                continue
            for k in keys:
                acc[k].append(st[k])
            agr.append(st["frac_agree"])
            f3.append(st["frac_both_filtration"])
        out = {k: (float(np.median(v)) if v else float("nan")) for k, v in acc.items()}
        out["n_pairs_ok"] = len(agr)
        out["frac_agree"] = float(np.mean(agr)) if agr else float("nan")
        out["frac_both_filtration"] = float(np.mean(f3)) if f3 else float("nan")
        return out

    rec["part3_treatment"] = {
        "raw": pooled(treat_fps, False),
        "screened": pooled(treat_fps, True),
        "per_split_raw": [pair_stats(a, b, False) for a, b in treat_fps],
        "per_split_screened": [pair_stats(a, b, True) for a, b in treat_fps],
        # §3.13: dumped so a matching or scoring rule can be re-evaluated
        # offline.  Refitting is ~45 min; a criterion is a one-line change, and
        # the two should never have been coupled.
        "fingerprints": [
            {"half_a": [dump_fp(f) for f in a], "half_b": [dump_fp(f) for f in b]}
            for a, b in treat_fps
        ],
    }
    print(f"  pooled raw:      {rec['part3_treatment']['raw']}")
    print(f"  pooled screened: {rec['part3_treatment']['screened']}")

    # the reference scale for any lattice claim: what a RANDOM pair of rotation
    # vectors of this magnitude scores.  Without it "margin = 0.001" is unreadable.
    rho_scale = float(
        np.percentile([abs(r) for a, b in treat_fps for fp in a + b for r in fp.rotations], 90)
    )
    null_k = lattice_null(len(PART), max(rho_scale, 1e-6), SEED)
    print(
        f"  GL({len(PART)},Z) null at |rho|<={rho_scale:.4f}: median {null_k['median']:.5f} "
        f"[p10 {null_k['p10']:.5f}, p90 {null_k['p90']:.5f}]"
    )
    rec["part3_treatment"]["lattice_null"] = null_k
    rec["part3_treatment"]["rho_scale_p90"] = rho_scale

    # ------------------------------------------------------------------
    banner("PART 4 -- negative controls: what SHOULD fail")
    controls: dict = {}
    def _circshift(Z: np.ndarray) -> np.ndarray:
        """Independently roll each neuron in time, within every condition.

        **The strong control, and the one that matters.**  It preserves each
        neuron's own time course exactly -- same smoothness, same
        autocorrelation, same marginal statistics, so the fit still finds a
        smooth low-dimensional trajectory -- while destroying the cross-neuron
        alignment that makes a *shared* latent exist.  That is precisely the
        null "these neurons have no common dynamics".

        The other two controls are weaker and are kept for contrast.  Time
        reversal in particular barely bites here: the fitted spectra sit at
        |lambda| ~ 0.99, so reversing them moves the exponents by ~0.02, which
        is the size of the fit noise.  Near-neutral dynamics is nearly
        reversible, so that control cannot be the one a claim rests on.
        """
        out = Z.copy()
        for c in range(out.shape[0]):
            for j in range(out.shape[2]):
                out[c, :, j] = np.roll(out[c, :, j], int(rng.integers(out.shape[1])))
        return out

    for name, transform in (
        ("neuron_circshift", _circshift),
        ("time_reversed", lambda Z: Z[:, ::-1, :].copy()),
        ("time_shuffled", None),
    ):
        ctrl_pairs = []
        for s in range(N_SPLITS):
            halves = nlb.neuron_split(td.n_units, seed=SEED + s, n_parts=2, rate=rate)
            Xb = X[:, :, halves[1]]
            if transform is None:
                perm = rng.permutation(Xb.shape[1])
                Xb = Xb[:, perm, :]
            else:
                Xb = transform(Xb)
            fp_b = [
                fingerprint_of(fit_one(Xb, SEED + 7919 * s + 100 * r + 1), Xb)
                for r in range(N_RESTARTS)
            ]
            ctrl_pairs.append((treat_fps[s][0], fp_b))
        controls[name] = {
            "raw": pooled(ctrl_pairs, False),
            "screened": pooled(ctrl_pairs, True),
            "fingerprints_b": [[dump_fp(f) for f in b] for _, b in ctrl_pairs],
        }
        print(f"  {name:14} raw {controls[name]['raw']}")
        print(f"  {name:14} scr {controls[name]['screened']}")
    rec["part4_controls"] = controls

    # The load-bearing comparison: treatment must beat both controls, and the
    # controls must genuinely fail.  A screen that improved the controls too
    # would be a filter flattering everything (§3.13(d)), so both are reported.
    # Read per-invariant (§3.13(b)) -- one boolean would hide which half of the
    # fingerprint the data actually constrained, and here they differ.
    def _get(d, k):
        """NaN must FAIL a check, never pass it.

        The first version mapped NaN to +inf so that "treatment < control"
        succeeded whenever the control had no pairs left.  The circular-shift
        control screens out completely -- every one of its fits is
        duplicate-flagged -- so three checks passed against nothing.  Same
        family as CLAUDE.md §3.9: a comparison that cannot fail is not a test.
        """
        v = d.get(k, float("nan"))
        return None if v != v else float(v)

    def _beats(t_arm, c_arm, key):
        a, b = _get(t_arm, key), _get(c_arm, key)
        return a is not None and b is not None and a < b

    print(f"\n  treatment duplicate-flag rate: "
          f"{rec['part3_treatment']['raw']['n_pairs_ok']} split(s) scored")
    for name in controls:
        # Compare LIKE FOR LIKE.  If either side's screened set is empty the
        # comparison falls back to raw on *both* sides; scoring a screened
        # treatment against an unscreened control would flatter the treatment.
        which = (
            "screened"
            if rec["part3_treatment"]["screened"]["n_pairs_ok"]
            and controls[name]["screened"]["n_pairs_ok"]
            else "raw"
        )
        t, c = rec["part3_treatment"][which], controls[name][which]
        # Only the circular-shift control is load-bearing; the other two are
        # reported but not asserted, because near-neutral dynamics is nearly
        # time-reversible and shuffling is trivially detectable.  Asserting
        # against a control the metric cannot fail would be self-congratulation.
        strong = name == "neuron_circshift"
        for label, key in (
            ("global Lyapunov spectrum", "tier1_spectrum_error_median"),
            ("rotation number", "rotation_error_median"),
            ("rotation, relative to module separation", "rotation_error_rel_median"),
            ("per-module spectra", "spectrum_error_median"),
            ("GL(K,Z) lattice margin", "lattice_margin_median"),
        ):
            ok = _beats(t, c, key)
            a, b = _get(t, key), _get(c, key)
            shown = f"{a:.5f} vs {b:.5f}" if a is not None and b is not None else "n/a"
            if strong:
                checks.append((f"{label}: treatment beats {name} [{which}] ({shown})", ok))
            else:
                print(f"  (not asserted) {label} vs {name} [{which}]: {shown}"
                      f" -> {'better' if ok else 'WORSE'}")
    # and the lattice margin has to beat what a random rotation vector scores,
    # or the agreement is a property of Z^K rather than of the data
    tl = _get(rec["part3_treatment"]["raw"], "lattice_margin_median")
    checks.append((
        f"treatment lattice margin beats the GL(K,Z) random null "
        f"({tl:.5f} vs {null_k['median']:.5f})",
        tl is not None and tl < null_k["median"],
    ))
    # the duplicate-module screen is itself a ground-truth-free discriminator:
    # it should flag the null far more often than the data
    dup_t = sum(
        1 for b in rec["part3_treatment"]["fingerprints"]
        for k in ("half_a", "half_b") for f in b[k] if f["duplicate_modules"]
    )
    n_t = sum(len(b[k]) for b in rec["part3_treatment"]["fingerprints"]
              for k in ("half_a", "half_b"))
    dup_c = sum(1 for g in controls["neuron_circshift"]["fingerprints_b"]
                for f in g if f["duplicate_modules"])
    n_c = sum(len(g) for g in controls["neuron_circshift"]["fingerprints_b"])
    print(f"  duplicate-module flag: treatment {dup_t}/{n_t}, "
          f"circshift control {dup_c}/{n_c}")
    rec["duplicate_flag_rates"] = {"treatment": [dup_t, n_t], "neuron_circshift": [dup_c, n_c]}
    checks.append((
        f"duplicate_modules flags the null far more than the data "
        f"({dup_c}/{n_c} vs {dup_t}/{n_t})",
        dup_c / max(n_c, 1) > 3 * dup_t / max(n_t, 1),
    ))

    # ------------------------------------------------------------------
    banner(f"PART 4b -- the same test at K=3 (partition {PART_SECONDARY})")
    print("  Reported for completeness; the GL(3,Z) null below is why its")
    print("  rotation agreement cannot be read as evidence either way.")
    sec_pairs = []
    for s in range(N_SPLITS):
        halves = nlb.neuron_split(td.n_units, seed=SEED + s, n_parts=2, rate=rate)
        fps = []
        for hidx, cols in enumerate(halves):
            fps.append([
                fingerprint_of(
                    fit_one(X[:, :, cols], SEED + 31 * s + 100 * r + hidx, part=PART_SECONDARY),
                    X, part=PART_SECONDARY,
                )
                for r in range(N_RESTARTS)
            ])
        sec_pairs.append((fps[0], fps[1]))
    sec_raw = pooled(sec_pairs, False)
    rho_scale3 = float(
        np.percentile([abs(r) for a, b in sec_pairs for fp in a + b for r in fp.rotations], 90)
    )
    null3 = lattice_null(len(PART_SECONDARY), max(rho_scale3, 1e-6), SEED)
    print(f"  K=3 pooled raw: {sec_raw}")
    print(
        f"  GL(3,Z) null at |rho|<={rho_scale3:.4f}: median {null3['median']:.5f} "
        f"[p10 {null3['p10']:.5f}, p90 {null3['p90']:.5f}]"
    )
    rec["part4b_K3"] = {
        "partition": PART_SECONDARY, "raw": sec_raw,
        "screened": pooled(sec_pairs, True),
        "lattice_null": null3, "rho_scale_p90": rho_scale3,
    }

    # ------------------------------------------------------------------
    banner("PART 5 -- do the fitted models satisfy the theory's hypotheses?")
    fp_all = [fp for a, b in treat_fps for fp in (a + b)]
    f3_ok = [fp.is_filtration for fp in fp_all]
    gaps = [fp.filtration_gap for fp in fp_all]
    dup = [fp.duplicate_modules() for fp in fp_all]
    margins = [
        M.invariant_agreement(a[0], b[0], spec_tol=SPEC_TOL, rot_tol=ROT_TOL).order_margin
        for a, b in treat_fps
    ]
    # How nonlinear is the fitted transition?  This qualifies BOTH results
    # above.  For a *linear* system with distinct complex-conjugate eigenvalue
    # pairs, block-diagonality is generic (the eigen-decomposition supplies it),
    # so a flat ladder would be close to trivial and the identifiability
    # question would be the one §3.5 says is already settled by the decoder.
    # The claim only has teeth to the extent the learned map is not linear.
    res_nl = fit_one(X, SEED)
    dyn_nl = res_nl.model.double().dyn
    ls_nl = LearnedSystem(dyn_nl, PART)
    zs = np.asarray(res_nl.z_fit, float).reshape(-1, D)
    nl_frac = []
    for k, blk in enumerate(ls_nl.blocks):
        a = sum(PART[:k])
        zk = zs[:, a : a + PART[k]]
        nxt = blk.step(zk)
        J0 = blk.jacobian(zk.mean(0))
        lin = (zk - zk.mean(0)) @ J0.T + blk.step(zk.mean(0))
        nl_frac.append(
            float(np.linalg.norm(nxt - lin) / max(np.linalg.norm(nxt - nxt.mean(0)), 1e-12))
        )
    print(f"  fitted transition, nonlinear residual per module: {np.round(nl_frac, 4)}")
    rec["part5_hypotheses_nonlinearity"] = [float(v) for v in nl_frac]

    print(f"  (F3) ordered separation holds in {sum(f3_ok)} of {len(f3_ok)} fits")
    print(f"  chain gap: median {np.median(gaps):+.4f}  range [{np.min(gaps):+.4f}, {np.max(gaps):+.4f}]")
    print(f"  duplicate-module flag raised in {sum(1 for d in dup if d)} of {len(dup)} fits")
    print(f"  order_margin across splits: {np.round(margins, 4)}")
    rec["part5_hypotheses"] = {
        "n_fits": len(fp_all),
        "n_filtration": int(sum(f3_ok)),
        "chain_gap_median": float(np.median(gaps)),
        "chain_gap_range": [float(np.min(gaps)), float(np.max(gaps))],
        "n_duplicate_flagged": int(sum(1 for d in dup if d)),
        "order_margins": [float(m) for m in margins],
        "spectra_median": [
            float(np.median([fp.spectra[i][j] for fp in fp_all]))
            for i in range(len(PART)) for j in range(PART[i])
        ],
        "rotations_median": [
            float(np.median([abs(fp.rotations[i]) for fp in fp_all])) for i in range(len(PART))
        ],
    }

    # ------------------------------------------------------------------
    banner("CHECKS")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    rec["checks"] = [{"name": n, "ok": bool(o)} for n, o in checks]
    rec["n_pass"] = int(sum(o for _, o in checks))
    rec["n_checks"] = len(checks)
    rec["runtime_s"] = round(time.time() - t_start, 1)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}  ({rec['n_pass']}/{rec['n_checks']} checks, {rec['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
