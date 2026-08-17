"""exp18 -- can Route B kill the ambiguity Route C provably cannot?  (tasks 27/31)

Route C is dead in the oscillatory regime and that is not a proof gap: two limit
cycles have identical spectral hulls, so (F3) fails outright, and the GL(2,Z)
lattice regrouping

    h(z_A, z_B) = (z_A,  z_B * z_A/|z_A|)                     [donor-first form]

is an EXACT modular conjugacy carrying omega_B -> omega_B + omega_A (task 23).
`exp17` confirmed it survives adversarial initialisation: warm-started at that
alternative, the fit stays there (fit-to-fit 0.1271 against 0.00018 matched).

Route B is the natural candidate to finish the job, and on paper it should:
Lemma D' needs **no spectral hypothesis at all**, only ``1 not in spec(f~_B)``,
so it applies exactly where Lemma C, Theorem F and (D1) are all dead.  Its own
witness is two modules with identical spectra.  So the question this experiment
asks has a clean shape:

    does a behavioural auxiliary variable reject the lattice representative
    that the dynamics alone cannot?

------------------------------------------------------------ what part 0 found --

The pre-flight answers most of it before a single model is fitted, and the answer
is **no, not in general** -- for a reason that is a symmetry, not an estimator
failure.  Two findings, in order:

**(1) Lemma D's own modulation hypothesis is transient here.**  (D3) is variance
modulation ``z_A = s(u) z~_A``.  A limit cycle attracts every radius to rho, so a
radial conditioning of the initial law is *forgotten*: measured, the modulated
block's u-dependence falls 0.79 -> 0.041 and the between-level mean-radius gap
falls 6.0e-1 -> 2.5e-5 over twelve steps.  What persists on a cycle is the
**phase**, which is not (D3).  So Lemma D as proved has little purchase in the
regime that matters -- the same shape as §3.13(b): a quantity is usable only if
the data still moves along it.

**(2) With the modulation that does persist, the regrouping is invisible.**  In
complex coordinates the lattice map ROTATES the invariant block by the donor's
phase.  A rotation is exactly `systems.nonadditive_behavioural_escape`, which
CLAUDE.md records as satisfying (D1)-(D4) with ``M_BA != 0``; what excluded it
there was Step 1, since at a *fixed point* ``theta . f_A - theta`` constant
forces theta constant.  **On a limit cycle that increment is exactly omega_A**,
so the escape becomes a genuine modular conjugacy -- and the detector reads
0.0298 for the regrouped block against a 0.0388 invariant floor, i.e. nothing.

The mechanism is that ``p_B`` is rotationally symmetric, so rotating it by *any*
independent angle leaves it exactly invariant: ``p(h_B | u) = p(z_B)`` for every
u, whatever the donor's phase does.  Behaviour cannot see a coupling that acts
by a symmetry of the invariant block's own law.

**(3) And that is the whole story, which makes it a design rule rather than a
dead end.**  Breaking the recipient's rotational symmetry -- concentrating its
phase *without* making it u-dependent, so it stays a legitimate invariant block
-- turns the detector back on, monotonically in the concentration:

    kappa_B        0.0     0.5     1.0     2.0     4.0
    B under R2   0.0298  0.2173  0.4405  0.7316  0.9349
    ratio to floor 0.77x  14.96x   7.27x  10.93x  16.10x

------------------------------------------------------------------ the design --

A 2x2 with a known answer in every cell, which is what part 0 buys.  Every fit is
warm-started at R2 (the lattice representative) and then trained normally; the
score is whether it ends nearer R1 or R2, on the modules the construction says
separate them (exp17's readout, for exp17's reasons -- fit-to-fit would inherit
the defect being repaired).

                     | behaviour OFF (w=0) | behaviour ON (w>0)
    symmetric  p_B   |   R2 survives       |   R2 survives      <- B is blind
    asymmetric p_B   |   R2 survives       |   R2 REJECTED      <- B bites

The two behaviour-OFF cells reproduce `exp17` arm C and confirm the recipient's
phase law is not by itself doing the work.  The symmetric behaviour-ON cell is
the load-bearing control in the other direction: it shows a *failure* to reject
is a property of the symmetry rather than of a penalty too weak to matter, since
the same penalty at the same weight rejects in the cell below it.

Each cell also runs a **matched** arm warm-started at R1.  Without it, "the fit
left R2" cannot be separated from "the penalty wrecked the fit": the matched arm
must stay at R1 *and* keep its fit quality (§3.11 -- build the arm that should
score perfectly into the sweep).

Pre-registered predictions are in ``PREDICTIONS``, written after part 0 (which is
analytic and involves no fitting) and before any fit ran.  CLAUDE.md §8: a result
that contradicts them gets committed, not tuned away.

------------------------------------------------------------------ the caveat --

This tests the Route B **mechanism** -- the u-invariant subspace is canonical, so
h must map it into itself -- and not Lemma D as proved.  Phase modulation is not
(D3) and a rotation of z_B is not an additive h_B (open item (a)).  Finding (1)
is the reason: (D3) is unavailable on a cycle, so a faithful test of the lemma in
this regime is not possible at all.  Stated in the write-up as such.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)   # exp17: measurably faster here, and makes cost predictable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idyn import metrics as M                       # noqa: E402
from idyn import spectra as SP                      # noqa: E402
from idyn import systems as S                       # noqa: E402
from idyn import train as T                         # noqa: E402
from idyn.behavior import block_u_dependence        # noqa: E402
from idyn.models import LearnedSystem, ModelConfig  # noqa: E402

SEED = 20260817
PART = [2, 2]
D = 4
N_OBS = 160
N_PER_U = 300                # trajectories per behaviour level, for the FITS
# Part 0 does no fitting, so its sample size is free -- and it needs to be large.
# `block_u_dependence` has a finite-sample floor: at 120/level the invariant
# block scores 0.08-0.45 on data where the truth is 0, which is the same order as
# the effect being measured, and one sweep row read 1.08x purely because its
# floor estimate spiked.  Part 0 checks that floor falls like n^-1/2 rather than
# assuming it (the discipline task 31 used for Lemma D').
N_ANALYTIC = 4000
U_LEVELS = (0, 1)
T_STEPS = 30
STEPS = 3000
WARM_STEPS = 800
N_RESTARTS = 4
SPEC_TOL = 0.05
ROT_TOL = 0.01

A_CYC = 0.30                 # limit-cycle contraction parameter
OM_A, OM_B = 0.50, 1.30      # donor / recipient rotation rates
R_LO, R_HI = 0.6, 1.4        # inside the basin (boundary is 2.08)
KAPPA_A = 2.0                # donor phase concentration, and it MOVES with u
PHASE_MU = (-0.9, 0.9)       # the donor's per-u phase centres
KAPPA_SYM, KAPPA_ASYM = 0.0, 4.0   # the recipient's phase concentration
W_BEHAVIOR = 1.0             # §3.12's recalibrated weight for the WHITENED penalty

OUT = Path(__file__).resolve().parents[1] / "results" / "exp18_behaviour_vs_lattice.json"

PREDICTIONS = {
    "1_variance_modulation_is_transient": (
        "on a limit cycle a radial (D3) conditioning is erased by the attractor, "
        "so Lemma D's own hypothesis is unavailable in the oscillatory regime"
    ),
    "2_symmetric_recipient_hides_R2": (
        "with p_B rotationally symmetric the behavioural detector cannot see the "
        "lattice regrouping, because rotation is a symmetry of p_B"
    ),
    "3_asymmetric_recipient_exposes_R2": (
        "concentrating the recipient's phase (u-independently) makes the same "
        "regrouping visible, monotonically in the concentration"
    ),
    "4_behaviour_off_keeps_R2_in_both_cells": (
        "with w_behavior=0 both cells reproduce exp17 arm C: the fit stays at R2"
    ),
    "5_behaviour_on_rejects_R2_only_when_asymmetric": (
        "the 2x2's only rejection is the asymmetric behaviour-on cell"
    ),
    "6_matched_arms_stay_at_R1_with_fit_quality_intact": (
        "the penalty is not a blunt instrument: warm-started at R1 every cell "
        "stays at R1 and keeps its fit quality"
    ),
}


def banner(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def checkpoint(rec: dict) -> None:
    """Write after every cell -- exp17's lesson about multi-hour all-or-nothing runs."""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")


# ------------------------------------------------------- systems and their h --


def system(om_a: float = OM_A, om_b: float = OM_B) -> S.ModularSystem:
    """Donor FIRST (block A, u-varying); recipient second (block B, u-invariant).

    The orientation is forced and getting it backwards silently empties the
    experiment.  Lemma D kills ``M_BA = d h_B / d z_A``, and the lattice map's
    only nonzero cross-derivative is the recipient's dependence on the donor.  So
    behaviour has to modulate the **donor**; with the roles swapped the coupling
    sits in the block Lemma D does not touch and every cell returns a null for a
    trivial reason.

    ``beta=0.0`` is the LimitCycleBlock default but is written out, because
    ``TwistBlock``'s is 0.6 and that trap has bitten once already (§11.7).
    """
    return S.ModularSystem([
        S.LimitCycleBlock(a=A_CYC, omega=om_a, beta=0.0),
        S.LimitCycleBlock(a=A_CYC, omega=om_b, beta=0.0),
    ])


def lattice_map(Z: np.ndarray) -> np.ndarray:
    """h(z_A, z_B) = (z_A, z_B z_A/|z_A|): the recipient borrows the donor's phase.

    exp17's `lattice_map` is the mirror image of this one (recipient first).  The
    two are the same GL(2,Z) action; only the labelling differs, and here the
    labelling is fixed by which block behaviour modulates.
    """
    za = Z[..., 0] + 1j * Z[..., 1]
    zb = Z[..., 2] + 1j * Z[..., 3]
    w = zb * za / np.maximum(np.abs(za), 1e-300)
    return np.stack([za.real, za.imag, w.real, w.imag], -1)


# ------------------------------------------------------------------ sampling --


def polar_ic(rng, n, phase_mu=None, kappa=0.0):
    """Initial conditions on an annulus inside the basin, with a phase law.

    ``kappa=0`` gives a uniform phase -- a rotationally SYMMETRIC law, which is
    what makes the lattice regrouping invisible to behaviour.  Positive kappa
    concentrates it (von Mises) and breaks that symmetry.

    Gaussian draws are deliberately not used: `LimitCycleBlock`'s basin ends at
    2.08 and a Gaussian tail crosses it, whereupon orbits diverge and the NaNs
    read like attractor collapse.  They are not the same thing.
    """
    r = rng.uniform(R_LO, R_HI, n)
    th = (rng.uniform(-np.pi, np.pi, n) if kappa <= 0
          else rng.vonmises(0.0 if phase_mu is None else phase_mu, kappa, n))
    return np.stack([r * np.cos(th), r * np.sin(th)], -1)


def make_data(rng, kappa_b: float, n_per_u: int | None = None):
    """Donor phase moves with u; recipient phase is fixed in u but maybe concentrated.

    Returns ``(X, Z, U, decoder)``.  The recipient's law is a legitimate invariant
    block at both kappa values -- it does not depend on u either way.  That is the
    point of the sweep: only its *symmetry* changes, not its invariance.
    """
    n = N_PER_U if n_per_u is None else n_per_u
    Zs, Us = [], []
    for k, mu in zip(U_LEVELS, PHASE_MU):
        za = polar_ic(rng, n, phase_mu=mu, kappa=KAPPA_A)
        zb = polar_ic(rng, n, phase_mu=0.0, kappa=kappa_b)
        Zs.append(np.concatenate([za, zb], axis=1))
        Us.append(np.full(n, k, dtype=int))
    Z0 = np.concatenate(Zs)
    U = np.concatenate(Us)
    Z = system().simulate(Z0, T_STEPS)
    if not np.isfinite(Z).all():
        raise RuntimeError("orbits left the basin -- lower R_HI")
    dec = S.LinearDecoder.random(N_OBS, D, rng)
    return dec(Z), Z, U, dec


# ------------------------------------------------------------------ helpers --


def whiten_modules(Z: np.ndarray, partition=PART) -> np.ndarray:
    """Per-module whitening of a warm-start target: a §7 within-module gauge change."""
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


def udep(w, U) -> float:
    """Scale-normalised u-dependence (§3.12 -- never the raw one on fitted latents)."""
    try:
        return float(block_u_dependence(np.asarray(w, float), U, normalize=True).total)
    except np.linalg.LinAlgError:
        return float("nan")


def block_udep_over_time(Z, U, sl) -> list[float]:
    return [udep(Z[:, t, sl], U) for t in range(Z.shape[1])]


def fingerprint(sysm, z0s, T=400, warmup=100):
    return M.dynamical_fingerprint(sysm, z0s, T=T, warmup=warmup, T_rotation=T)


# The two analytic targets are the same in every cell, and they are the expensive
# part of the run: 76 s each at 600 initial conditions, against ~6 s for a fit.
# They depend only on the exact systems -- every orbit in the basin converges to
# the same cycle, so the recipient's phase law (the thing the cells vary) moves
# no invariant of the TRUE system.  Computed once, from a fixed draw, and cached.
_TARGETS: dict = {}


def targets():
    if not _TARGETS:
        rng = np.random.default_rng(SEED + 77)
        z0 = np.concatenate([polar_ic(rng, 200), polar_ic(rng, 200)], axis=1)
        _TARGETS["R1"] = fingerprint(system(), z0)
        _TARGETS["R2"] = fingerprint(system(OM_A, OM_B + OM_A), z0)
    return _TARGETS["R1"], _TARGETS["R2"]


def fitted_fingerprint(res, n_t: int):
    """Read INSIDE the data horizon (§3.13a)."""
    dyn = res.model.double().dyn
    z0 = np.asarray(res.z_fit, float)[:, 0, :]
    warm = max(n_t // 4, 2)
    read = n_t - warm
    return M.dynamical_fingerprint(LearnedSystem(dyn, PART), z0,
                                   T=read, warmup=warm, T_rotation=read)


def informative_modules(tgt1, tgt2, tol: float = 1e-6):
    """Which modules separate R1 from R2, fixed by the construction (exp17's rule)."""
    r1 = [abs(float(x)) for x in tgt1.rotations]
    r2 = [abs(float(x)) for x in tgt2.rotations]
    d_rot = [0.0 if (np.isnan(a) or np.isnan(b)) else abs(a - b) for a, b in zip(r1, r2)]
    d_spec = [float(np.abs(np.sort(a) - np.sort(b)).max())
              for a, b in zip(tgt1.spectra, tgt2.spectra)]
    use_rot = max(d_rot) >= max(d_spec)
    d = d_rot if use_rot else d_spec
    keep = [k for k, v in enumerate(d) if v > max(max(d) / 2.0, tol)]
    return keep, ("rotation" if use_rot else "spectrum")


def restricted_distance(fp, tgt, keep, kind: str) -> float:
    """Fit-to-target distance, minimised over module permutations (the label gauge)."""
    K = len(tgt.partition)
    if kind == "rotation":
        f = [np.inf if np.isnan(x) else abs(float(x)) for x in fp.rotations]
        t = [abs(float(x)) for x in tgt.rotations]
        def d(pi):
            return max(abs(f[pi[k]] - t[k]) for k in keep)
    else:
        f = [np.sort(np.asarray(s, float)) for s in fp.spectra]
        t = [np.sort(np.asarray(s, float)) for s in tgt.spectra]
        def d(pi):
            return max(float(np.abs(f[pi[k]] - t[k]).max()) for k in keep)
    return float(min(d(pi) for pi in permutations(range(K))))


def med(xs) -> float:
    return float(np.median(xs)) if len(xs) else float("nan")


# --------------------------------------------------------------------- part 0 --


def part0_analytic(rng) -> dict:
    """No fitting: any failure here is the construction's, not the estimator's."""
    banner("PART 0 -- analytic pre-flight (no fitting)")
    F, Ft = system(), system(OM_A, OM_B + OM_A)

    probe = np.concatenate([polar_ic(rng, 6000), polar_ic(rng, 6000)], axis=1)
    lhs, rhs = lattice_map(F.step(probe)), Ft.step(lattice_map(probe))
    defect = float(np.abs(lhs - rhs).max() / max(np.abs(lhs).max(), 1e-300))
    print(f"  lattice map is an exact modular conjugacy : defect {defect:.3e}")

    # (a) is (D3) variance modulation available on a cycle?
    Zs, Us = [], []
    for k, s in zip(U_LEVELS, (0.7, 1.3)):
        za = polar_ic(rng, N_ANALYTIC) * s
        zb = polar_ic(rng, N_ANALYTIC)
        Zs.append(np.concatenate([za, zb], axis=1))
        Us.append(np.full(N_ANALYTIC, k, dtype=int))
    Zv, Uv = system().simulate(np.concatenate(Zs), T_STEPS), np.concatenate(Us)
    rad = np.abs(Zv[..., 0] + 1j * Zv[..., 1])
    gap = [float(abs(rad[Uv == 0, t].mean() - rad[Uv == 1, t].mean()))
           for t in range(T_STEPS + 1)]
    var_ud = block_udep_over_time(Zv, Uv, slice(0, 2))
    print(f"  (D3) radial modulation of block A: u-dep {var_ud[0]:.4f} -> {var_ud[-1]:.4f}, "
          f"mean-radius gap {gap[0]:.3e} -> {gap[-1]:.3e}   [TRANSIENT]")

    # (b) the detector's finite-sample FLOOR, measured not assumed.  The invariant
    # block's true u-dependence is 0, so whatever this reads is noise, and it must
    # fall like n^-1/2 or the sweep below is measuring the estimator.
    floor = {}
    for n in (120, 480, 1920, N_ANALYTIC):
        _, Zf, Uf, _ = make_data(np.random.default_rng(SEED + 4242), 0.0, n_per_u=n)
        floor[str(n)] = udep(Zf[:, -1, 2:], Uf)
    ns = np.array([float(k) for k in floor])
    slope = float(np.polyfit(np.log(ns), np.log(list(floor.values())), 1)[0])
    print(f"  detector floor on a truly invariant block: "
          f"{ {k: round(v, 4) for k, v in floor.items()} }  log-log slope {slope:+.2f} "
          f"(expect ~-0.5)")

    # (c) the symmetry sweep: does breaking p_B's rotational symmetry expose R2?
    sweep = {}
    for kb in (0.0, 0.5, 1.0, 2.0, 4.0):
        _, Z, U, _ = make_data(np.random.default_rng(SEED + 991), kb, n_per_u=N_ANALYTIC)
        H = lattice_map(Z)
        tru = udep(Z[:, -1, 2:], U)
        reg = udep(H[:, -1, 2:], U)
        sweep[f"kappa_b={kb}"] = {"B_true": tru, "B_under_R2": reg,
                                  "ratio": reg / max(tru, 1e-12),
                                  "A_varying": udep(Z[:, -1, :2], U)}
        print(f"  kappa_B={kb:4.1f}   B invariant (truth) {tru:.4f}   "
              f"B under R2 {reg:.4f}   ratio {reg / max(tru, 1e-12):6.2f}x")

    # (d) Route C is dead here, for the record
    z0 = np.concatenate([polar_ic(rng, 600), polar_ic(rng, 600)], axis=1)
    fp = fingerprint(F, z0)
    fg = SP.filtration_gap([np.asarray(s, float) for s in fp.spectra])
    print(f"  (F3) filtration gap {fg.gap:+.4f} (ordered={fg.ordered})   "
          f"-- Route C has no hypothesis here, which is why Route B is being asked")

    return {"conjugacy_defect": defect,
            "D3_variance_udep_by_t": var_ud,
            "D3_mean_radius_gap_by_t": gap,
            "detector_floor": floor, "detector_floor_slope": slope,
            "symmetry_sweep": sweep,
            "filtration_gap": float(fg.gap), "F3_ordered": bool(fg.ordered)}


# --------------------------------------------------------------------- part 1 --


def run_cell(name: str, kappa_b: float, w_behavior: float, rec: dict) -> dict:
    """One 2x2 cell: warm-start at R2 (and at R1 as control), fit, score."""
    banner(f"CELL {name}  (kappa_B={kappa_b}, w_behavior={w_behavior})")
    rng = np.random.default_rng(SEED + int(kappa_b * 10) + int(w_behavior * 100))
    X, Z, U, _ = make_data(rng, kappa_b)

    R1 = whiten_modules(Z)
    R2 = whiten_modules(lattice_map(Z))
    tgt1, tgt2 = targets()
    keep, kind = informative_modules(tgt1, tgt2)
    print(f"  informative modules {keep} on the {kind};  "
          f"true u-dep: A {udep(Z[:, -1, :2], U):.4f}  B {udep(Z[:, -1, 2:], U):.4f}  "
          f"B under R2 {udep(lattice_map(Z)[:, -1, 2:], U):.4f}")

    out = {"kappa_b": kappa_b, "w_behavior": w_behavior,
           "informative_modules": keep, "invariant": kind, "restarts": []}

    for r in range(N_RESTARTS):
        row = {"restart": r}
        for tag, warm in (("adversarial", R2), ("matched", R1)):
            seed = SEED + 1000 * (r + 1) + (7 if tag == "adversarial" else 13)
            cfg = ModelConfig(n_obs=N_OBS, d=D, partition=PART,
                              decoder="mlp", encoder="mlp")
            tc = T.TrainConfig(steps=STEPS, seed=seed, warm_steps=WARM_STEPS,
                               w_behavior=w_behavior, inv_start=2, inv_stop=4,
                               behavior_whiten=True)
            res = T.fit(X, cfg, tc, U=U, warm_z=warm)
            fp = fitted_fingerprint(res, T_STEPS + 1)
            d1 = restricted_distance(fp, tgt1, keep, kind)
            d2 = restricted_distance(fp, tgt2, keep, kind)
            row[tag] = {
                "seed": seed,
                "warm_residual": float(res.warm_residual),
                "fit_quality": float(res.fit_quality),
                "to_R1": d1, "to_R2": d2,
                "nearer": "R1" if d1 < d2 else "R2",
                "fitted_inv_block_udep": udep(res.z_fit[:, -1, 2:], U),
                "spectra": [np.asarray(s, float).tolist() for s in fp.spectra],
                "rotations": [float(x) for x in fp.rotations],
                "duplicate_modules": bool(fp.duplicate_modules),
            }
            print(f"    r{r} {tag:12s} warm {res.warm_residual:.4f}  "
                  f"fitq {res.fit_quality:.3e}  ->R1 {d1:.4f}  ->R2 {d2:.4f}  "
                  f"[{row[tag]['nearer']}]  inv-udep {row[tag]['fitted_inv_block_udep']:.4f}")
        out["restarts"].append(row)
        rec.setdefault("cells", {})[name] = out
        checkpoint(rec)

    adv = [row["adversarial"] for row in out["restarts"]]
    mat = [row["matched"] for row in out["restarts"]]
    sep = float(restricted_distance(tgt2, tgt1, keep, kind))
    est_err = med([m["to_R1"] for m in mat])       # the estimator's own error scale
    out["summary"] = {
        "adv_stayed_at_R2": sum(a["nearer"] == "R2" for a in adv),
        "matched_stayed_at_R1": sum(m["nearer"] == "R1" for m in mat),
        "adv_to_R1_med": med([a["to_R1"] for a in adv]),
        "adv_to_R2_med": med([a["to_R2"] for a in adv]),
        "matched_fitq_med": med([m["fit_quality"] for m in mat]),
        "adv_fitq_med": med([a["fit_quality"] for a in adv]),
        # The load-bearing diagnostic for a SURVIVED verdict.  If R2 is held while
        # the invariant block is still strongly u-dependent, the penalty was not
        # enforced and the cell says nothing about behaviour; if R2 is held with
        # the penalty *satisfied*, behaviour genuinely failed to exclude it.  Those
        # are different results and only this number separates them.
        "adv_inv_udep_med": med([a["fitted_inv_block_udep"] for a in adv]),
        "matched_inv_udep_med": med([m["fitted_inv_block_udep"] for m in mat]),
        "true_inv_udep": udep(Z[:, -1, 2:], U),
        "R2_inv_udep": udep(lattice_map(Z)[:, -1, 2:], U),
        # §3.13(e): a collapsed fit puts both modules on one factor and its
        # distance to either target is meaningless.  No ground truth needed.
        "adv_duplicate_modules": sum(a["duplicate_modules"] for a in adv),
        "separation": sep,
        # exp17: below ~2 an arm cannot answer its own question, whatever it says.
        "resolving_power": float(sep / max(est_err, 1e-12)),
        "verdict": "SURVIVED" if sum(a["nearer"] == "R2" for a in adv) > N_RESTARTS / 2
                   else "REJECTED",
    }
    s = out["summary"]
    print(f"  -> {s['verdict']}: R2 held in {s['adv_stayed_at_R2']}/{N_RESTARTS}; "
          f"matched held R1 in {s['matched_stayed_at_R1']}/{N_RESTARTS}; "
          f"adv inv-udep {s['adv_inv_udep_med']:.3f} "
          f"(true {s['true_inv_udep']:.3f}, R2 {s['R2_inv_udep']:.3f}); "
          f"resolving {s['resolving_power']:.1f}x")
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rec = {"seed": SEED, "predictions": PREDICTIONS,
           "params": {"partition": PART, "n_obs": N_OBS, "n_per_u": N_PER_U,
                      "T": T_STEPS, "steps": STEPS, "warm_steps": WARM_STEPS,
                      "n_restarts": N_RESTARTS, "omega": [OM_A, OM_B],
                      "kappa_a": KAPPA_A, "phase_mu": list(PHASE_MU),
                      "kappa_sym": KAPPA_SYM, "kappa_asym": KAPPA_ASYM,
                      "w_behavior": W_BEHAVIOR, "generating_decoder": "linear",
                      "fitted_encoder": "mlp", "fitted_decoder": "mlp"}}

    rec["part0_analytic"] = part0_analytic(rng)
    checkpoint(rec)

    for kappa_b, tag in ((KAPPA_SYM, "sym"), (KAPPA_ASYM, "asym")):
        for w in (0.0, W_BEHAVIOR):
            run_cell(f"{tag}_w{w:g}", kappa_b, w, rec)

    # ------------------------------------------------------------- scoring --
    banner("CHECKS")
    p0, cells = rec["part0_analytic"], rec["cells"]
    sweep = p0["symmetry_sweep"]
    checks = {
        "1_D3_transient": p0["D3_variance_udep_by_t"][-1] < 0.15 * p0["D3_variance_udep_by_t"][0],
        "2_symmetric_hides_R2": sweep["kappa_b=0.0"]["ratio"] < 2.0,
        "3_asymmetric_exposes_R2": sweep["kappa_b=4.0"]["ratio"] > 5.0,
        "4a_sym_w0_keeps_R2": cells["sym_w0"]["summary"]["verdict"] == "SURVIVED",
        "4b_asym_w0_keeps_R2": cells["asym_w0"]["summary"]["verdict"] == "SURVIVED",
        "5a_sym_wB_keeps_R2": cells["sym_w1"]["summary"]["verdict"] == "SURVIVED",
        "5b_asym_wB_rejects_R2": cells["asym_w1"]["summary"]["verdict"] == "REJECTED",
        "6_matched_arms_hold_R1": all(
            c["summary"]["matched_stayed_at_R1"] > N_RESTARTS / 2 for c in cells.values()),
        "7_penalty_did_not_wreck_the_fit": (
            cells[f"asym_w{W_BEHAVIOR:g}"]["summary"]["matched_fitq_med"]
            < 5.0 * cells["asym_w0"]["summary"]["matched_fitq_med"]),
        # The SURVIVED verdict in the symmetric behaviour-on cell only means
        # "behaviour is blind" if the penalty was actually enforced there.  If the
        # invariant block stays as u-dependent as R2 makes it, the cell measured a
        # weight that was ignored, not a symmetry -- and says nothing.
        "8_symmetric_cell_actually_enforced_the_penalty": (
            cells[f"sym_w{W_BEHAVIOR:g}"]["summary"]["adv_inv_udep_med"]
            < 2.0 * cells[f"sym_w{W_BEHAVIOR:g}"]["summary"]["true_inv_udep"]),
        "9_no_mode_collapse": all(
            c["summary"]["adv_duplicate_modules"] <= N_RESTARTS / 2 for c in cells.values()),
        "10_arms_can_resolve_their_own_question": all(
            c["summary"]["resolving_power"] > 2.0 for c in cells.values()),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    rec["checks"] = {k: bool(v) for k, v in checks.items()}
    rec["n_pass"] = int(sum(checks.values()))
    rec["n_checks"] = len(checks)
    rec["runtime_s"] = time.time() - t0
    checkpoint(rec)
    print(f"\n{rec['n_pass']}/{rec['n_checks']} checks passed "
          f"in {rec['runtime_s'] / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
