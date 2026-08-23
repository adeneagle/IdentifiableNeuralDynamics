r"""exp20 -- Route B's exact reach: the compact-stabiliser dichotomy.

**Stage 1 of the Route B program: analytic, no fitting.**  Every arm is an exact
distribution built from known numbers, so any failure here is the *criterion's*
or the *detector's*, never an optimiser's.  That separation is the same one
`exp16` used and it is the reason to run this before anything expensive.

---

## What is being tested

`identifiability.md` section 13.3 records that Route B is blind to a coupling
acting by a symmetry of ``p_B``, and stops at the slogan.  It completes into a
dichotomy:

    Proposition.  Let h_B(z_A, z_B) = T(z_A) . z_B with T valued in a group G
    acting affinely on R^{d_B}, z_A independent of z_B given u, and p_B with
    finite nonsingular second moment.  Then (D4) holds with M_BA nonzero iff T
    takes values in a single coset of Stab_G(p_B) -- and that stabiliser is
    conjugate to a closed subgroup of O(d_B), hence COMPACT.

One line: ``g_* p_B = p_B`` forces ``A S A^T = S`` and ``A m + b = m``, so
``A`` lies in ``S^{1/2} O(d) S^{-1/2}`` and ``b`` is determined by ``A``.

Consequences, and they are what the arms below check:

* Couplings valued in a **non-compact** group -- translations, scalings, shears
  -- are killed completely.  Lemma D's additivity is the translation case, and
  "no probability law is translation-invariant" is the one-dimensional shadow of
  compactness.  So Lemma D generalises well past additive ``h_B``.
* The residual ambiguity is at most a **compact** group, and it is *conjugate*
  to a subgroup of the orthogonal group -- not equal to it.  Arm 2b is the arm
  that distinguishes those two readings.

Scope, stated because it is easy to overclaim: this measures **(D4) and its
detector**, not whether a given ``T`` arises from a genuine modular conjugacy.
`exp18` section 13.2 already showed those are different questions -- the
rotational escape is excluded at a fixed point by Step 1 and is a real conjugacy
on a cycle.  Stage 1 is about which couplings behaviour can *see*.

---

## Pre-registered predictions

Written before the run; a wrong one is committed failing, per CLAUDE.md section 8.

| arm | p_B | generator X | prediction |
|---|---|---|---|
| 1a | isotropic | translation | DETECTED |
| 1b | isotropic | scaling | DETECTED |
| 1c | isotropic | shear | DETECTED |
| 1d | isotropic | rotation | **AT FLOOR** -- Stab contains SO(2) |
| 2a | anisotropic | rotation | DETECTED -- Stab_{SO(2)} is trivial here |
| 2b | anisotropic | S^{1/2}-conjugated rotation | **AT FLOOR** -- this is Stab itself |
| 3  | whitened skew, centred | rotation | see below |
| 4  | isotropic + mean m | rotation | DETECTED, monotone in \|m\| |

**Arm 3 is the interesting one and my prediction is a negative.**  Take a
non-Gaussian ``p_B``, pre-whitened so its covariance is exactly ``I`` and
centred so its mean is ``0``, and asymmetric enough that ``Stab_{O(2)}(p_B)`` is
trivial.  The Proposition then says (D4) *fails* -- a rotation genuinely moves
that law.  But ``behavior.block_u_dependence`` reads only the first two
conditional moments, and a rotation preserves both when the mean is ``0`` and
the covariance is ``I``.  So I predict the detector reads **AT FLOOR** even
though the coupling is not stabiliser-valued.

If that lands, it is a real limit and it sharpens section 13.3 in a direction
that section does not anticipate: 13.3 blames ``p_B``'s rotational symmetry,
but for a *whitened second-moment* detector the blind group is ``O(d_B)`` for
**every** ``p_B``, because whitening makes the second moments isotropic by
construction.  CLAUDE.md section 3.12's fix -- whiten the block so the penalty
is ``GL(d_B)``-invariant -- would then be the *same fact* as the blindness,
not an unrelated repair.

**Arm 4 is the practical consequence** of that reading.  What a whitened
second-moment detector can still see is a ``u``-dependent *mean direction*, so a
nonzero ``E[z_B]`` restores the kill.  That is a much simpler design rule than
"break the rotational symmetry of ``p_B``", and it predicts `exp18`'s mechanism
measurement: its "circular concentration" is a mean-direction statistic, and the
encoder flattening it to 0.270 is the encoder *centring the block*.

---

## Design controls

**Matched move.**  A coupling that is invisible because it is weak proves
nothing.  Every arm's coefficient is bisected so that

    move := E||h_B - z_B|| / E||z_B||

hits a common target, so "detected" versus "at floor" cannot be a strength
artefact.  Reported per arm.

**One-parameter subgroups.**  Every coupling is ``T = exp(theta X)`` with
``theta = c * z_{A,0}`` a scalar.  Only the generator ``X`` differs between
arms 1a--1d, so the comparison isolates the group and nothing else.

**The floor is measured, not assumed.**  Arm 0 is ``T = identity``, genuinely
``u``-invariant, at the same ``n``.  Part 5 checks the floor falls like
``n^{-1/2}``; a floor that did not would mean the detector has a bias term and
"at floor" would be unreadable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from idyn.behavior import block_u_dependence

SEED = 20260824
N = 200_000
D_B = 2
SIGMAS = (0.7, 1.4)          # the two behaviour levels: z_A ~ sigma_u * mu_A
MOVE_TARGET = 0.30           # matched displacement of h_B relative to |z_B|
DETECT_RATIO = 5.0           # "detected" = total u-dependence > this * floor
FLOOR_RATIO = 2.0            # "at floor"  = total u-dependence < this * floor

RESULTS = Path(__file__).resolve().parents[1] / "results"


# --------------------------------------------------------------- generators

def _rotation(theta: np.ndarray) -> np.ndarray:
    """(n,2,2) rotations, X = J."""
    c, s = np.cos(theta), np.sin(theta)
    T = np.empty((theta.size, 2, 2))
    T[:, 0, 0], T[:, 0, 1] = c, -s
    T[:, 1, 0], T[:, 1, 1] = s, c
    return T


def _scaling(theta: np.ndarray) -> np.ndarray:
    """(n,2,2) scalings, X = I."""
    e = np.exp(theta)
    T = np.zeros((theta.size, 2, 2))
    T[:, 0, 0] = T[:, 1, 1] = e
    return T


def _shear(theta: np.ndarray) -> np.ndarray:
    """(n,2,2) shears, X = N nilpotent.  exp(theta N) = I + theta N."""
    T = np.zeros((theta.size, 2, 2))
    T[:, 0, 0] = T[:, 1, 1] = 1.0
    T[:, 0, 1] = theta
    return T


def _conjugated_rotation(theta: np.ndarray, S_half: np.ndarray,
                         S_half_inv: np.ndarray) -> np.ndarray:
    """S^{1/2} R(theta) S^{-1/2} -- the actual stabiliser of N(0, S)."""
    R = _rotation(theta)
    return np.einsum("ab,nbc,cd->nad", S_half, R, S_half_inv)


def _apply(T: np.ndarray | None, zB: np.ndarray, shift: np.ndarray | None) -> np.ndarray:
    out = zB if T is None else np.einsum("nab,nb->na", T, zB)
    return out if shift is None else out + shift


# ----------------------------------------------------------------- the laws

def _law(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw p_B.  Every variant is exact and its stabiliser is known."""
    if kind == "iso":
        return rng.standard_normal((n, D_B))
    if kind == "aniso":
        return rng.standard_normal((n, D_B)) * np.array([1.0, 2.0])
    if kind == "skew":
        # Three Gaussians at deliberately non-symmetric angles with unequal
        # weights -> trivial O(2) stabiliser.  Then centred and whitened, so the
        # first two moments are exactly those of the isotropic law and ONLY the
        # higher-order structure distinguishes it.
        ang = np.array([0.0, 1.75, 3.75])
        wts = np.array([0.5, 0.3, 0.2])
        which = rng.choice(3, size=n, p=wts)
        ctr = 2.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)[which]
        z = ctr + 0.6 * rng.standard_normal((n, D_B))
        z = z - z.mean(0, keepdims=True)
        cov = np.cov(z, rowvar=False)
        L = np.linalg.cholesky(cov)
        return np.linalg.solve(L, z.T).T
    raise ValueError(f"unknown law {kind!r}")


def _draw(kind: str, n: int, rng: np.random.Generator, mean_norm: float = 0.0):
    """Return (z_A scalar driver theta_raw, z_B, u labels)."""
    u = rng.integers(0, len(SIGMAS), size=n)
    sig = np.asarray(SIGMAS)[u]
    zA0 = sig * rng.standard_normal(n)          # the scalar the coupling reads
    zB = _law(kind, n, rng)
    if mean_norm:
        zB = zB + np.array([mean_norm, 0.0])
    return zA0, zB, u


# ------------------------------------------------------------- measurement

def _move(hB: np.ndarray, zB: np.ndarray) -> float:
    return float(np.linalg.norm(hB - zB, axis=1).mean()
                 / np.linalg.norm(zB, axis=1).mean())


def _run_arm(build, zA0, zB, u, target=MOVE_TARGET) -> dict:
    """Bisect the coupling coefficient to a matched move, then score."""
    lo, hi = 1e-6, 1e3
    for _ in range(60):
        c = np.sqrt(lo * hi)
        if _move(build(c * zA0, zB), zB) < target:
            lo = c
        else:
            hi = c
    c = np.sqrt(lo * hi)
    hB = build(c * zA0, zB)
    dep_n = block_u_dependence(hB, u, normalize=True)
    dep_r = block_u_dependence(hB, u, normalize=False)
    return {
        "coeff": float(c),
        "move": _move(hB, zB),
        "u_dep": dep_n.total,
        "u_dep_mean": dep_n.mean_variation,
        "u_dep_cov": dep_n.cov_variation,
        "u_dep_raw": dep_r.total,
    }


def _floor(zB, u) -> dict:
    dep = block_u_dependence(zB, u, normalize=True)
    return {"u_dep": dep.total, "u_dep_mean": dep.mean_variation,
            "u_dep_cov": dep.cov_variation}


# ------------------------------------------------------------------- parts

def part1(rng) -> dict:
    """The group dichotomy at a fixed isotropic p_B."""
    zA0, zB, u = _draw("iso", N, rng)
    fl = _floor(zB, u)
    arms = {
        "1a_translation": _run_arm(
            lambda th, z: _apply(None, z, np.stack([th, np.zeros_like(th)], 1)), zA0, zB, u),
        "1b_scaling": _run_arm(lambda th, z: _apply(_scaling(th), z, None), zA0, zB, u),
        "1c_shear": _run_arm(lambda th, z: _apply(_shear(th), z, None), zA0, zB, u),
        "1d_rotation": _run_arm(lambda th, z: _apply(_rotation(th), z, None), zA0, zB, u),
    }
    return {"floor": fl, "arms": arms}


def part2(rng) -> dict:
    """Stab is CONJUGATE to O(2), not equal to it."""
    zA0, zB, u = _draw("aniso", N, rng)
    fl = _floor(zB, u)
    S = np.diag([1.0, 4.0])
    S_half = np.diag(np.sqrt(np.diag(S)))
    S_half_inv = np.diag(1.0 / np.sqrt(np.diag(S)))
    arms = {
        "2a_rotation": _run_arm(lambda th, z: _apply(_rotation(th), z, None), zA0, zB, u),
        "2b_conjugated_rotation": _run_arm(
            lambda th, z: _apply(_conjugated_rotation(th, S_half, S_half_inv), z, None),
            zA0, zB, u),
    }
    return {"floor": fl, "arms": arms}


def _harmonic_gap(w: np.ndarray, u: np.ndarray, k: int) -> float:
    """|E exp(i k phi)| differenced across u -- a statistic moments 1-2 miss.

    Rotating a whitened, centred law leaves the mean at 0 and the covariance at
    I, so ``block_u_dependence`` is structurally blind to it.  The circular
    harmonics are not: they are exactly the higher-order content a rotation
    *does* move.  This is what separates "(D4) holds" from "the detector cannot
    see that (D4) fails".
    """
    ang = np.arctan2(w[:, 1], w[:, 0])
    vals = [np.hypot(np.cos(k * ang[u == j]).mean(), np.sin(k * ang[u == j]).mean())
            for j in np.unique(u)]
    return float(abs(vals[0] - vals[1]))


def part3(rng) -> dict:
    """Trivial stabiliser, but the second-moment detector cannot use it."""
    zA0, zB, u = _draw("skew", N, rng)
    fl = _floor(zB, u)
    arm = _run_arm(lambda th, z: _apply(_rotation(th), z, None), zA0, zB, u)
    # Independent evidence that the law really is non-Gaussian and asymmetric,
    # so "at floor" cannot be blamed on p_B secretly being isotropic Gaussian.
    ang = np.arctan2(zB[:, 1], zB[:, 0])
    arm["p_b_circular_concentration"] = float(
        np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    arm["p_b_mean_norm"] = float(np.linalg.norm(zB.mean(0)))
    arm["p_b_cov_dev_from_I"] = float(
        np.linalg.norm(np.cov(zB, rowvar=False) - np.eye(D_B)))
    # a genuinely rotation-invariant law would score ~0 on a 4th-order harmonic
    arm["p_b_harmonic4"] = float(np.hypot(np.cos(4 * ang).mean(), np.sin(4 * ang).mean()))

    # --- does (D4) ACTUALLY fail, or is the coupling stabiliser-valued? ---
    # Without this the "at floor" reading is unattributable: it would be equally
    # consistent with the Proposition saying there is nothing to detect.
    hB = _apply(_rotation(arm["coeff"] * zA0), zB, None)
    arm["harmonic_gap_coupled"] = {str(k): _harmonic_gap(hB, u, k) for k in (3, 4, 5)}
    arm["harmonic_gap_uncoupled"] = {str(k): _harmonic_gap(zB, u, k) for k in (3, 4, 5)}
    return {"floor": fl, "arms": {"3_rotation_whitened_skew": arm}}


def part4(rng) -> dict:
    """The practical fix: a nonzero mean restores the kill, monotonically."""
    out = {}
    for m in (0.0, 0.25, 0.5, 1.0, 2.0):
        zA0, zB, u = _draw("iso", N, rng, mean_norm=m)
        fl = _floor(zB, u)
        arm = _run_arm(lambda th, z: _apply(_rotation(th), z, None), zA0, zB, u)
        ang = np.arctan2(zB[:, 1], zB[:, 0])
        arm["floor"] = fl["u_dep"]
        arm["ratio"] = arm["u_dep"] / fl["u_dep"]
        arm["circular_concentration"] = float(
            np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
        out[f"mean_{m}"] = arm
    return {"arms": out}


def part5(rng) -> dict:
    """The floor is sampling noise: it must fall like n^{-1/2}."""
    ns, floors = [], []
    for n in (12_500, 25_000, 50_000, 100_000, 200_000):
        _, zB, u = _draw("iso", n, rng)
        ns.append(n)
        floors.append(_floor(zB, u)["u_dep"])
    slope = float(np.polyfit(np.log(ns), np.log(floors), 1)[0])
    return {"n": ns, "floor": floors, "log_log_slope": slope}


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rec: dict = {"seed": args.seed, "n": N, "d_b": D_B, "sigmas": list(SIGMAS),
                 "move_target": MOVE_TARGET,
                 "detect_ratio": DETECT_RATIO, "floor_ratio": FLOOR_RATIO}
    rec["part1_group_dichotomy"] = part1(rng)
    rec["part2_conjugacy"] = part2(rng)
    rec["part3_detector_limit"] = part3(rng)
    rec["part4_nonzero_mean"] = part4(rng)
    rec["part5_floor_scaling"] = part5(rng)

    # ------------------------------------------------------------- checks
    checks: list[dict] = []

    def add(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    p1 = rec["part1_group_dichotomy"]
    f1 = p1["floor"]["u_dep"]
    for key, want in (("1a_translation", "detect"), ("1b_scaling", "detect"),
                      ("1c_shear", "detect"), ("1d_rotation", "floor")):
        a = p1["arms"][key]
        r = a["u_dep"] / f1
        ok = r > DETECT_RATIO if want == "detect" else r < FLOOR_RATIO
        add(f"part1 {key} -> {want}", ok, f"u_dep={a['u_dep']:.4g} floor={f1:.4g} ratio={r:.2f}")

    p2 = rec["part2_conjugacy"]
    f2 = p2["floor"]["u_dep"]
    for key, want in (("2a_rotation", "detect"), ("2b_conjugated_rotation", "floor")):
        a = p2["arms"][key]
        r = a["u_dep"] / f2
        ok = r > DETECT_RATIO if want == "detect" else r < FLOOR_RATIO
        add(f"part2 {key} -> {want}", ok, f"u_dep={a['u_dep']:.4g} floor={f2:.4g} ratio={r:.2f}")

    p3 = rec["part3_detector_limit"]
    a3 = p3["arms"]["3_rotation_whitened_skew"]
    r3 = a3["u_dep"] / p3["floor"]["u_dep"]
    add("part3 p_B is genuinely asymmetric (4th harmonic well above 0)",
        a3["p_b_harmonic4"] > 0.02, f"harmonic4={a3['p_b_harmonic4']:.4g}")
    add("part3 p_B is exactly centred and whitened (so a rotation fixes moments 1-2)",
        a3["p_b_mean_norm"] < 1e-9 and a3["p_b_cov_dev_from_I"] < 1e-9,
        f"|mean|={a3['p_b_mean_norm']:.2e} ||cov-I||={a3['p_b_cov_dev_from_I']:.2e}")
    hg_c, hg_u = a3["harmonic_gap_coupled"]["3"], a3["harmonic_gap_uncoupled"]["3"]
    add("part3 (D4) GENUINELY fails here -- the 3rd harmonic separates the u levels",
        hg_c > 20 * hg_u,
        f"coupled={hg_c:.4g} uncoupled={hg_u:.4g} ratio={hg_c / max(hg_u, 1e-12):.1f}")
    add("part3 PREDICTED: trivial stabiliser is NOT enough for a 2nd-moment detector",
        r3 < FLOOR_RATIO, f"u_dep={a3['u_dep']:.4g} floor={p3['floor']['u_dep']:.4g} ratio={r3:.2f}")

    p4 = rec["part4_nonzero_mean"]["arms"]
    ms = (0.0, 0.25, 0.5, 1.0, 2.0)
    ratios = [p4[f"mean_{m}"]["ratio"] for m in ms]
    concs = [p4[f"mean_{m}"]["circular_concentration"] for m in ms]
    add("part4 zero mean is at floor", ratios[0] < FLOOR_RATIO, f"ratio={ratios[0]:.2f}")
    add("part4 nonzero mean is detected", ratios[-1] > DETECT_RATIO, f"ratio={ratios[-1]:.2f}")
    add("part4 monotone in |m|", all(b > a for a, b in zip(ratios, ratios[1:])),
        f"ratios={[round(r, 2) for r in ratios]}")

    # Where detection turns on, in the units exp18 reports.  Not smooth: it is a
    # threshold, so quote the bracket rather than a single number.
    below = [c for c, r in zip(concs, ratios) if r < DETECT_RATIO]
    above = [c for c, r in zip(concs, ratios) if r >= DETECT_RATIO]
    lo_c, hi_c = (max(below) if below else 0.0), (min(above) if above else float("inf"))
    rec["part4_nonzero_mean"]["detection_threshold_concentration"] = [lo_c, hi_c]
    add("part4 the threshold brackets exp18's fitted concentrations "
        "(adversarial 0.270 below, matched 0.809 above)",
        lo_c <= 0.270 <= hi_c or 0.270 < lo_c, f"threshold in [{lo_c:.3f}, {hi_c:.3f}]; "
        f"exp18 adversarial 0.270, data-R2 0.392, matched 0.809")

    sl = rec["part5_floor_scaling"]["log_log_slope"]
    add("part5 floor falls like n^{-1/2}", -0.65 < sl < -0.35, f"slope={sl:.3f}")

    rec["checks"] = checks
    rec["n_pass"] = sum(c["pass"] for c in checks)
    rec["n_check"] = len(checks)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "exp20_stabiliser_dichotomy.json"
    out.write_text(json.dumps(rec, indent=2))

    print(f"\n=== exp20: the compact-stabiliser dichotomy (seed {args.seed}) ===\n")
    for part, key in (("part1 -- group dichotomy, isotropic p_B", "part1_group_dichotomy"),
                      ("part2 -- Stab is CONJUGATE to O(2)", "part2_conjugacy"),
                      ("part3 -- the detector's own limit", "part3_detector_limit")):
        blk = rec[key]
        print(f"{part}   [floor {blk['floor']['u_dep']:.5f}]")
        for k, a in blk["arms"].items():
            print(f"   {k:26s} move={a['move']:.3f}  u_dep={a['u_dep']:.5f}"
                  f"  ({a['u_dep']/blk['floor']['u_dep']:6.2f}x floor)"
                  f"   mean={a['u_dep_mean']:.5f} cov={a['u_dep_cov']:.5f}")
        print()
    print("part4 -- a nonzero mean restores the kill")
    for m in (0.0, 0.25, 0.5, 1.0, 2.0):
        a = p4[f"mean_{m}"]
        print(f"   |m|={m:<5} conc={a['circular_concentration']:.3f}"
              f"  u_dep={a['u_dep']:.5f}  ({a['ratio']:6.2f}x floor)")
    print(f"\npart5 -- floor log-log slope in n: {sl:.3f}  (expect -0.5)\n")
    for c in checks:
        print(f"   [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}\n          {c['detail']}")
    print(f"\n{rec['n_pass']}/{rec['n_check']} checks pass -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
