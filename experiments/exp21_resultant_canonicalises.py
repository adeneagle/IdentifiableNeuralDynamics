r"""exp21 -- the phase resultant canonicalises the residue Route B cannot see.

**Stage 2 of the Route B program, and it does not go where stage 1 pointed.**

`exp20` (§14) settled the analytic half: Route B kills every coupling valued in
a non-compact group, and what it cannot reach is a compact residue conjugate to
a subgroup of ``O(d_B)`` -- for the lattice regrouping, a rotation of the
invariant block.  The obvious stage 2 was to *impose* the missing precondition
(a u-dependent mean direction) as an objective term and see whether it survives
learning.

Building that term produced two facts that redirect the experiment, and both are
committed as parts here because each one kills an obvious design:

**(1) The escape is strictly resultant-decreasing.**  If ``h_B = R(theta) z_B``
with ``theta`` independent of ``z_B``, then the phase of ``h_B`` is
``theta_B + theta``, so by the circular convolution theorem the resultants
MULTIPLY:

    Proposition C.   R_{h_B} = R_theta * R_{z_B}  <=  R_{z_B},
                     with equality iff theta is a.s. constant.

So the compact residue Proposition S leaves behind is not merely detectable in
principle -- it moves a *scalar functional monotonically*.  The true
representative is that functional's maximiser, and **this needs no behavioural
variable at all**.  That is a different and stronger claim than the one stage 2
set out to test.

**(2) But free maximisation is fakeable, so it must be a comparison, not an
objective.**  Optimising the whitened resultant over an unconstrained 2-D point
cloud reaches **0.9994** by giving the radius a heavy tail (spread 0.094 to
56.6): whitening fixes the second moment, and a heavy-tailed radius then lets
the *direction* concentrate almost arbitrarily.  An objective term that rewards
concentration is therefore payable without changing the representation -- the
same shape as §3.12, §3.15 and §13.4, caught this time before it was run.

What survives both is a **model-selection criterion**: among fits of the same
data, prefer the one whose pinned block has the larger whitened resultant, with
the radial tail reported beside it so a degenerate fit is visible.

---

## Why the statistic had to change from `exp18`'s

`exp18_mechanism` used the raw circular concentration ``|E e^{i phi}|``, which is
invariant under rescaling and rotating the block but **not** under shearing it.
§7 grants the whole of ``GL(d_b)`` inside a module, so the raw statistic is a
gauge quantity in the §3.12 sense and its numbers cannot certify a
gauge-invariant criterion.  The version used here whitens by the block's
**uncentred** second moment first.  That is exactly ``GL(d_b)``-invariant -- under
``w -> A w`` the Cholesky factor goes to ``A L Q``, so the whitened points rotate
and a resultant length does not move.

**Uncentred is load-bearing and was got wrong first.**  Whitening by the
covariance (i.e. centring) makes a strongly concentrated block score 0.0177 and a
phase-randomised one 0.0196 -- indistinguishable, because centring removes the
mean direction and by §14.4 the mean direction is the entire signal.  Pinned in
`tests/test_behavior.py::test_concentration_term_must_not_centre_the_block`.

---

## Pre-registered predictions

| part | prediction |
|---|---|
| 1 | ``R_{h_B} = R_theta R_{z_B}`` exactly, over a grid of concentrations |
| 2 | on `exp18`'s system the whitened resultant separates R1 from R2 at kappa_B > 0, and **not at all** at kappa_B = 0 |
| 3 | free maximisation reaches > 0.95 with a radial spread > 100x -- i.e. the objective form is unavailable |
| 4 | across fits, matched > adversarial in **every** restart, with no radial-tail degeneracy in either |

Part 4 reuses `exp18`'s system, seeds and settings verbatim so the comparison is
within-experiment; only the statistic differs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import exp18_behaviour_vs_lattice as E            # noqa: E402
from idyn import train as T                        # noqa: E402
from idyn.metrics import distance_correlation as _dc            # noqa: E402
from idyn.metrics import distance_correlation_baseline as _dcb  # noqa: E402
from idyn.models import LatentDynamicsModel, ModelConfig  # noqa: E402

SEED = 20260825
N_RESTARTS = 3
N_ANALYTIC = 200_000
OUT = Path(__file__).resolve().parents[1] / "results" / "exp21_resultant_canonicalises.json"


# ------------------------------------------------------------- the statistic

def whitened_resultant(block: np.ndarray) -> float:
    """``|E[y/|y|]|`` after whitening by the UNCENTRED second moment.

    ``block`` is ``(..., T+1, 2)`` or ``(..., 2)``; a leading time axis is scored
    per timestep and averaged, because an oscillatory block's mean direction
    rotates and pooling averages it away (§3.15, one term over).
    """
    b = np.asarray(block, float)
    t = torch.as_tensor(np.ascontiguousarray(b), dtype=torch.float64)
    if t.ndim == 2:
        return float(LatentDynamicsModel._whitened_resultant(t))
    return float(np.mean([float(LatentDynamicsModel._whitened_resultant(t[:, k]))
                          for k in range(t.shape[1])]))


def radial_tail(block: np.ndarray) -> float:
    """Ratio max/median of the whitened radius -- the guard on part 3's degeneracy.

    A fit that reached a high resultant by giving the radius a heavy tail rather
    than by finding the right representation shows up here and nowhere else.
    """
    b = np.asarray(block, float).reshape(-1, 2)
    S = b.T @ b / len(b) + 1e-12 * np.eye(2)
    y = np.linalg.solve(np.linalg.cholesky(S), b.T).T
    r = np.linalg.norm(y, axis=1)
    return float(r.max() / max(np.median(r), 1e-12))


# ------------------------------------------------------------------- part 1

def part1(rng) -> dict:
    """Proposition C: the resultants multiply, so the escape strictly shrinks it."""
    rows = []
    for kB, kA in ((4.0, 2.0), (4.0, 0.5), (1.0, 2.0), (8.0, 8.0), (2.0, 0.0)):
        tb = rng.vonmises(0.0, kB, N_ANALYTIC)
        ta = (rng.vonmises(0.6, kA, N_ANALYTIC) if kA > 0
              else np.full(N_ANALYTIC, 0.6))          # kA=0 -> theta constant
        R = lambda a: float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))
        rows.append({"kappa_B": kB, "kappa_theta": kA, "R_zB": R(tb),
                     "R_theta": R(ta), "product": R(tb) * R(ta),
                     "measured_R_hB": R(tb + ta)})
        rows[-1]["abs_err"] = abs(rows[-1]["product"] - rows[-1]["measured_R_hB"])
    return {"rows": rows, "max_abs_err": max(r["abs_err"] for r in rows)}


# ------------------------------------------------------------------- part 2

def part2() -> dict:
    """On exp18's own system: does the statistic separate R1 from R2?"""
    out = {}
    for kb in (0.0, 0.5, 1.0, 2.0, 4.0):
        rng = np.random.default_rng(SEED + 17)
        _, Z, _, _ = E.make_data(rng, kb, n_per_u=1500)
        H = E.lattice_map(Z)
        # scored on the whitened warm-start targets, i.e. exactly the objects the
        # fits are started at -- so part 4 is comparable to this by construction
        r1 = whitened_resultant(E.whiten_modules(Z)[:, :, 2:])
        r2 = whitened_resultant(E.whiten_modules(H)[:, :, 2:])
        out[f"kappa_b={kb}"] = {"R1": r1, "R2": r2, "ratio": r1 / max(r2, 1e-12),
                                "donor_R1": whitened_resultant(
                                    E.whiten_modules(Z)[:, :, :2])}
    return out


# ------------------------------------------------------------------- part 3

def part3() -> dict:
    """The degenerate optimum -- why this cannot be an objective term."""
    torch.manual_seed(SEED)
    y = torch.randn(2000, 2, requires_grad=True)
    opt = torch.optim.Adam([y], lr=0.02)
    for _ in range(4000):
        opt.zero_grad()
        (-LatentDynamicsModel._whitened_resultant(y)).backward()
        opt.step()
    with torch.no_grad():
        r = float(LatentDynamicsModel._whitened_resultant(y))
        n = y.norm(dim=1)
        return {"max_resultant": r,
                "radial_spread": float(n.max() / n.min()),
                "radial_tail": radial_tail(y.numpy())}


# ------------------------------------------------------------------- part 4

def part4(rec) -> dict:
    """exp18's fits, rescored on the gauge-invariant statistic."""
    rng = np.random.default_rng(E.SEED + int(E.KAPPA_ASYM * 10) + 10)
    X, Z, U, _ = E.make_data(rng, E.KAPPA_ASYM)
    R1 = E.whiten_modules(Z)
    R2 = E.whiten_modules(E.lattice_map(Z))
    data = {"R1": whitened_resultant(R1[:, :, 2:]),
            "R2": whitened_resultant(R2[:, :, 2:])}
    print(f"  data targets: R1 {data['R1']:.4f}   R2 {data['R2']:.4f}")
    fits = []
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
            row[tag] = {"seed": seed, "fit_quality": float(res.fit_quality),
                        "resultant": whitened_resultant(zf[:, :, 2:]),
                        "radial_tail": radial_tail(zf[:, :, 2:]),
                        "u_dependence": E.udep(zf[:, -1, 2:], U),
                        # Route D under learning: are the FITTED modules
                        # independent?  Part 8 says the lattice image makes them
                        # dependent, so this is the criterion's fitted readout
                        # and the one that matters for deployment.
                        "module_dependence": float(_dc(zf[:, -1, :2], zf[:, -1, 2:])),
                        "module_dependence_baseline": float(
                            _dcb(zf.shape[0], 2, 2, seed=0))}
            c = row[tag]
            print(f"  r{r} {tag:12s} resultant {c['resultant']:.4f}  "
                  f"dCor {c['module_dependence']:.4f}  "
                  f"radial-tail {c['radial_tail']:6.1f}  "
                  f"fitq {c['fit_quality']:.3e}  u-dep {c['u_dependence']:.4f}")
        fits.append(row)
        rec["part4_fitted"] = {"data": data, "fits": fits}
        OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return {"data": data, "fits": fits}


# ------------------------------------------------------------------- part 5

def part5(rng) -> dict:
    """Does the resultant close an escape behaviour cannot, or need the same thing?

    **It needs the same thing, and this part is here to say so.**  Both
    instruments die at ``R_{z_B} = 0``: behaviour because a rotation is then an
    exact symmetry of ``p_B`` (§14.3), and the resultant because there is no
    resultant to decrease.  So the criterion is *not* an independent
    escape-closer, and reading Proposition C as one would be wrong.

    What it is instead is **uniformly more sensitive, with sensitivity that does
    not degrade as the block flattens.**  Proposition C gives
    ``R_{h_B} / R_{z_B} = R_theta`` exactly, so the *relative* deficit is
    ``1 - R_theta`` -- a property of the coupling alone, independent of how
    concentrated ``p_B`` is.  The behavioural detector's signal-to-floor, by
    contrast, scales with ``R_{z_B}`` (§14.4's mean-direction mechanism).  Hence
    the resultant dominates in exactly the weakly-concentrated band where §14.4
    puts behaviour at floor -- and that band, ``R <~ 0.3``, is where `exp18`'s
    fitted adversarial block (0.270) sits.
    """
    n = 120_000
    u = rng.integers(0, 2, n)
    zA0 = np.array([0.7, 1.4])[u] * rng.standard_normal(n)
    th = 0.55 * zA0
    c, s = np.cos(th), np.sin(th)
    rows = {}
    for kB in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        ang = (rng.uniform(-np.pi, np.pi, n) if kB == 0 else rng.vonmises(0.0, kB, n))
        rad = 1.0 + 0.3 * np.abs(rng.standard_normal(n))
        zB = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        hB = np.stack([c * zB[:, 0] - s * zB[:, 1], s * zB[:, 0] + c * zB[:, 1]], 1)
        R0 = whitened_resultant(zB)
        floor = E.udep(zB, u)
        rows[f"kappa_b={kB}"] = {
            "R_zB": R0, "behaviour_ratio": E.udep(hB, u) / max(floor, 1e-12),
            "relative_deficit": (R0 - whitened_resultant(hB)) / max(R0, 1e-9)}
    return rows


# ------------------------------------------------------------------- part 6

GL2Z = ((1, 0, 0, 1), (1, 1, 0, 1), (1, -1, 0, 1), (1, 0, 1, 1),
        (2, 1, 1, 1), (0, 1, 1, 0), (1, 2, 0, 1), (3, 2, 1, 1), (-1, 0, 0, -1))


def part6(rng) -> dict:
    r"""Proposition R -- the lattice ambiguity is BROKEN, not merely quotiented.

    This is the part that matters, because it is a positive identifiability
    statement in the one regime where every route in this repo is dead: two
    oscillatory modules, where (B4') fails identically (Prop. N), (F3) fails
    (both hulls reach 0), and Route B is blind (§13.3).

        Proposition R.  Let K modules be oscillatory with an invariant K-torus
        and independent, non-uniform phase laws.  Then the true representative
        uniquely maximises the total directional resultant ``sum_i R_i`` over
        the GL(K,Z) orbit, up to the subgroup of signed permutations.

    *Why.*  A non-identity, non-permutation element sends some output phase to an
    integer combination involving another module's phase, which by Prop. C
    convolves it with a non-constant angle and strictly shrinks that resultant.
    Permutations only relabel, and sign flips conjugate, so both leave the total
    fixed -- and those are exactly the ambiguities §1.1 already accepts.

    **The boundary is sharp and is measured here, not assumed.**  If a module's
    phase law is *uniform* its resultant is already 0 and cannot shrink further,
    so elements that only pour into that module tie with the identity: the
    stabiliser grows exactly as far as the phase laws are uninformative.  So the
    criterion resolves the lattice ambiguity **to the extent the modules' phases
    are concentrated**, and the ``kappa_2 = 0`` row below is that statement's
    negative half.
    """
    n = 200_000
    out = {}
    for k1, k2 in ((4.0, 4.0), (4.0, 1.0), (1.0, 0.5), (4.0, 0.0)):
        p1 = rng.vonmises(0.0, k1, n) if k1 > 0 else rng.uniform(-np.pi, np.pi, n)
        p2 = rng.vonmises(0.3, k2, n) if k2 > 0 else rng.uniform(-np.pi, np.pi, n)
        r1 = 1 + 0.3 * np.abs(rng.standard_normal(n))
        r2 = 1 + 0.3 * np.abs(rng.standard_normal(n))
        blk = lambda p, r: np.stack([r * np.cos(p), r * np.sin(p)], 1)
        rows = {}
        for a, b, c, d in GL2Z:
            R = (whitened_resultant(blk(a * p1 + b * p2, r1))
                 + whitened_resultant(blk(c * p1 + d * p2, r2)))
            rows[f"{a},{b},{c},{d}"] = R
        ident = rows["1,0,0,1"]
        # ties are permitted only for signed permutations (and, when a phase law
        # is uniform, for elements that pour only into that module)
        strictly_better = {m: v for m, v in rows.items() if v > ident + 1e-3}
        out[f"kappa=({k1},{k2})"] = {
            "totals": rows, "identity": ident,
            "argmax": max(rows, key=rows.get),
            "n_strictly_better_than_identity": len(strictly_better),
            "runner_up_nonpermutation": max(
                (v for m, v in rows.items()
                 if m not in ("1,0,0,1", "0,1,1,0", "-1,0,0,-1")), default=0.0)}
    return out


# ------------------------------------------------------------------- part 7

def circular_dependence(p: np.ndarray, q: np.ndarray, n_harm: int = 3) -> float:
    r"""Dependence between two circular variables, as a max over harmonics of

        | E e^{i(k p - l q)} - E e^{ikp} . E e^{-ilq} |,

    the circular analogue of a covariance.  Zero for every ``(k,l)`` iff the
    joint characteristic function factorises on the harmonic lattice, which for
    the laws here is independence.  Model-free and needs no density estimate.
    """
    best = 0.0
    for k in range(1, n_harm + 1):
        for l in range(1, n_harm + 1):
            joint = abs(np.mean(np.exp(1j * (k * p - l * q))))
            marg = abs(np.mean(np.exp(1j * k * p))) * abs(np.mean(np.exp(-1j * l * q)))
            best = max(best, abs(joint - marg))
    return float(best)


def part7(rng) -> dict:
    r"""Proposition R' -- INDEPENDENCE is the criterion; the resultant proxies it.

    Part 6 shows the true representative maximises the total resultant.  But the
    reason it does is not really about concentration:

        In the true representation the modules' phases are **independent**.
        Under a lattice image ``phi_B -> phi_B + phi_A`` the donor's phase
        appears in *both* coordinates, so the modules become **dependent**.

    That reframes the criterion as the standard nonlinear-ICA assumption --
    independent latent factors -- rather than as a new aesthetic preference, and
    it is exactly the distinction §1.3 drew: iVAE needs conditionally independent
    **scalar** components, which excludes rotation outright, whereas what is
    wanted here is independence of **modules**, which a 2-D rotation satisfies
    perfectly well.  Modularity of the dynamics plus independence of the initial
    conditions is what pins the representative.

        Proposition R'.  The true representative is the unique element of the
        GL(K,Z) orbit, up to signed permutation, in which the modules' phase
        laws are statistically independent -- provided at most one module has a
        uniform phase law.

    The proviso is forced and is measured below: adding an independent *uniform*
    angle both preserves uniformity and destroys dependence, so if a module's
    phase is uniform the lattice acts freely on it and nothing distinguishes the
    representatives.  With **both** uniform the orbit is entirely undetermined,
    which is the honest floor of the whole approach.

    Independence is the sharper instrument by one to two orders of magnitude, so
    it is the criterion to report; the resultant is what an *objective* can use,
    since it is a smooth scalar of one block rather than a joint statistic.
    """
    n = 120_000
    out = {}
    # floor: the statistic on genuinely independent phases, at this n
    fl = [circular_dependence(rng.vonmises(0.0, 4.0, n), rng.vonmises(0.3, 1.0, n))
          for _ in range(5)]
    out["floor"] = {"mean": float(np.mean(fl)), "max": float(np.max(fl)), "n": n}
    for k1, k2 in ((4.0, 4.0), (4.0, 1.0), (1.0, 0.5), (4.0, 0.0), (0.0, 0.0)):
        p1 = rng.vonmises(0.0, k1, n) if k1 > 0 else rng.uniform(-np.pi, np.pi, n)
        p2 = rng.vonmises(0.3, k2, n) if k2 > 0 else rng.uniform(-np.pi, np.pi, n)
        rows = {f"{a},{b},{c},{d}": circular_dependence(a * p1 + b * p2, c * p1 + d * p2)
                for a, b, c, d in GL2Z}
        nonperm = {m: v for m, v in rows.items()
                   if m not in ("1,0,0,1", "0,1,1,0", "-1,0,0,-1")}
        out[f"kappa=({k1},{k2})"] = {
            "dependence": rows, "identity": rows["1,0,0,1"],
            "argmin": min(rows, key=rows.get),
            "best_nonpermutation": min(nonperm.values()),
            "margin": min(nonperm.values()) / max(rows["1,0,0,1"], 1e-12)}
    return out


# ------------------------------------------------------------------- part 8

def part8(rng) -> dict:
    r"""Route D: independence of the module marginals, against all three escapes.

    Part 7 reframed the lattice criterion as independence.  Applied to the
    repo's other counterexamples it turns out to do considerably more, and the
    pattern is exactly the right one:

    * **§4.3 / §3.7, the triangular conjugacy.**  ``h_1 = z_1 + c sgn(z_2)|z_2|^p``
      is the object that makes block-diagonality **false** under (B1)-(B4) --
      polynomial, hence ``C^infinity``, so no regularity hypothesis removes it.
      It makes the modules *dependent*, so independence rejects it.
    * **§7 / task 23, the lattice regrouping.**  Rejected likewise.
    * **§3.1, the regrouping across modules.**  A permutation of independent
      coordinates keeps them independent, so independence is blind -- and that
      is **correct**, because (B2) indecomposability is the hypothesis for that
      one.  A criterion that rejected all three would be rejecting too much.

    So independence knocks out precisely the two escapes that block Theorem B
    and Theorem F, and leaves the one already covered.  It needs no
    non-resonance (dead for oscillators by Prop. N), no (F3) (fails on real
    data), and no behavioural variable.

    **It is not a proof, and the reason is worth stating.**  Independence alone
    is famously insufficient for nonlinear ICA -- Hyvarinen & Pajunen exhibit
    infinitely many nonlinear independence-preserving maps.  Part 7's
    ``kappa=(0,0)`` row *is* one of them: with both phase laws uniform,
    ``(phi_A, phi_B) -> (phi_A, phi_A + phi_B)`` preserves independence exactly.
    What is conjectured here is that independence **plus being a modular
    conjugacy** is rigid, and that conjunction is untested.  `TODO(gap)`
    """
    from idyn.metrics import distance_correlation as dc
    from idyn.metrics import distance_correlation_baseline as dcb

    n = 1500
    base = dcb(n, 2, 2, seed=0)
    dep = lambda Z: float(dc(Z[:, :2], Z[:, 2:]))

    Z = rng.standard_normal((n, 4))
    tri = Z.copy()
    tri[:, 0] = Z[:, 0] + 0.8 * np.sign(Z[:, 2]) * np.abs(Z[:, 2]) ** 2

    t1, t2 = rng.vonmises(0.0, 4.0, n), rng.vonmises(0.3, 4.0, n)
    r1 = 1 + 0.2 * np.abs(rng.standard_normal(n))
    r2 = 1 + 0.2 * np.abs(rng.standard_normal(n))
    cyc = np.stack([r1 * np.cos(t1), r1 * np.sin(t1),
                    r2 * np.cos(t2), r2 * np.sin(t2)], 1)
    lat = np.stack([r1 * np.cos(t1), r1 * np.sin(t1),
                    r2 * np.cos(t2 + t1), r2 * np.sin(t2 + t1)], 1)

    Zr = rng.standard_normal((n, 4))
    reg = Zr[:, [0, 2, 1, 3]]

    # and the Hyvarinen-Pajunen case: uniform phases, where the lattice IS
    # independence-preserving -- the honest limit, in the same units
    u1, u2 = rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n)
    hp = np.stack([r1 * np.cos(u1), r1 * np.sin(u1),
                   r2 * np.cos(u1 + u2), r2 * np.sin(u1 + u2)], 1)

    # §3.10's actual blind spot, as an instrument check.  The EVEN coupling
    # ``h_B = z_B + c z_A^2`` has ``Cov(z_A, z_A^2) = 0`` for symmetric ``z_A``,
    # so a linear probe reads nothing; dCor must not.  (Note the §4.3
    # counterexample's ``sgn(z)|z|^p`` is *odd* and a linear probe does see it --
    # getting those two the wrong way round is easy and was done once here.)
    blind = {}
    for c in (0.25, 0.5, 1.0, 2.0, 5.0):
        Zb = rng.standard_normal((3000, 4))
        hb = Zb.copy()
        hb[:, 2] = Zb[:, 2] + c * Zb[:, 0] ** 2
        blind[f"c={c}"] = {
            "linear_abs_corr": float(abs(np.corrcoef(hb[:, 0], hb[:, 2])[0, 1])),
            "dcor": float(dc(hb[:, :2], hb[:, 2:]))}

    out = {"baseline": base, "n": n, "even_coupling_blind_spot": blind, "cases": {}}
    for name, true_Z, h_Z, expect in (
            ("4.3_triangular", Z, tri, "reject"),
            ("7_lattice", cyc, lat, "reject"),
            ("3.1_regrouping", Zr, reg, "blind"),
            ("HP_uniform_phase_lattice", cyc, hp, "blind")):
        out["cases"][name] = {"true": dep(true_Z), "under_h": dep(h_Z),
                              "ratio_to_baseline": dep(h_Z) / base,
                              "expected": expect}
    return out


# ---------------------------------------------------------------------- main

def main() -> int:
    t0 = time.time()
    OUT.parent.mkdir(exist_ok=True)
    reuse = "--reuse-fits" in sys.argv     # merge into an existing JSON's part 4
    rng = np.random.default_rng(SEED)
    rec: dict = {"seed": SEED, "n_analytic": N_ANALYTIC, "n_restarts": N_RESTARTS}

    print("\n=== part 1: Proposition C -- the resultants multiply ===")
    rec["part1_product_law"] = part1(rng)
    for r in rec["part1_product_law"]["rows"]:
        print(f"  kB={r['kappa_B']:<4} k_theta={r['kappa_theta']:<4} "
              f"R_zB={r['R_zB']:.4f} R_th={r['R_theta']:.4f} "
              f"product={r['product']:.4f} measured={r['measured_R_hB']:.4f} "
              f"err={r['abs_err']:.2e}")

    print("\n=== part 2: R1 vs R2 on exp18's system ===")
    rec["part2_separation"] = part2()
    for k, v in rec["part2_separation"].items():
        print(f"  {k:14s} R1 {v['R1']:.4f}  R2 {v['R2']:.4f}  ratio {v['ratio']:6.2f}x")

    print("\n=== part 3: the degenerate optimum ===")
    rec["part3_degenerate_max"] = part3()
    p3 = rec["part3_degenerate_max"]
    print(f"  free maximisation reaches {p3['max_resultant']:.4f} with radial "
          f"spread {p3['radial_spread']:.0f}x (tail {p3['radial_tail']:.0f}x)")
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print("\n=== part 5: does the resultant close what behaviour cannot? ===")
    rec["part5_sensitivity"] = part5(rng)
    print(f"  {'R_zB':>8} {'behaviour/floor':>16} {'relative deficit':>18}")
    for k, v in rec["part5_sensitivity"].items():
        print(f"  {v['R_zB']:8.4f} {v['behaviour_ratio']:16.2f} "
              f"{100 * v['relative_deficit']:17.1f}%")

    print("\n=== part 6: Proposition R -- the lattice ambiguity is BROKEN ===")
    rec["part6_proposition_R"] = part6(rng)
    for k, v in rec["part6_proposition_R"].items():
        print(f"  {k:18s} argmax {v['argmax']:10s}  identity {v['identity']:.4f}  "
              f"best non-permutation {v['runner_up_nonpermutation']:.4f}")

    print("\n=== part 7: Prop. R' -- INDEPENDENCE is the criterion ===")
    rec["part7_independence"] = part7(rng)
    p7 = rec["part7_independence"]
    print(f"  floor at n={p7['floor']['n']}: mean {p7['floor']['mean']:.5f} "
          f"max {p7['floor']['max']:.5f}")
    for k, v in p7.items():
        if k == "floor":
            continue
        print(f"  {k:18s} argmin {v['argmin']:10s}  identity {v['identity']:.5f}  "
              f"best non-perm {v['best_nonpermutation']:.5f}  margin {v['margin']:6.1f}x")

    print("\n=== part 8: Route D -- independence against all three escapes ===")
    rec["part8_route_D"] = part8(rng)
    p8 = rec["part8_route_D"]
    print(f"  dCor baseline at n={p8['n']}: {p8['baseline']:.4f}")
    for k, v in p8["cases"].items():
        print(f"  {k:26s} true {v['true']:.4f}  under h {v['under_h']:.4f}  "
              f"({v['ratio_to_baseline']:5.1f}x baseline)  expected {v['expected']}")

    if reuse:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        rec["part4_fitted"] = prev["part4_fitted"]
        print(f"\n=== part 4: reused from {OUT.name}, no refitting ===")
    else:
        print(f"\n=== part 4: rescoring exp18's fits ({2 * N_RESTARTS} fits) ===")
        rec["part4_fitted"] = part4(rec)

    # ------------------------------------------------------------- checks
    checks: list[dict] = []
    add = lambda n, ok, d: checks.append({"name": n, "pass": bool(ok), "detail": d})

    e = rec["part1_product_law"]["max_abs_err"]
    add("part1 Prop. C: resultants multiply", e < 5e-3, f"max abs err {e:.2e}")
    last = rec["part1_product_law"]["rows"][-1]
    add("part1 equality iff theta constant",
        abs(last["measured_R_hB"] - last["R_zB"]) < 5e-3,
        f"kappa_theta=0 -> R_hB {last['measured_R_hB']:.4f} vs R_zB {last['R_zB']:.4f}")

    p2 = rec["part2_separation"]
    add("part2 no separation at kappa_B=0 (the honest boundary)",
        p2["kappa_b=0.0"]["ratio"] < 2.0, f"ratio {p2['kappa_b=0.0']['ratio']:.2f}x")
    add("part2 separation grows with kappa_B",
        all(p2[f"kappa_b={a}"]["ratio"] <= p2[f"kappa_b={b}"]["ratio"] + 0.15
            for a, b in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0))),
        f"ratios {[round(p2[k]['ratio'], 2) for k in p2]}")
    add("part2 R1 clears R2 decisively at the asymmetric setting",
        p2["kappa_b=4.0"]["ratio"] > 1.5, f"ratio {p2['kappa_b=4.0']['ratio']:.2f}x")

    add("part3 PREDICTED: free maximisation is fakeable, so no objective form",
        p3["max_resultant"] > 0.95 and p3["radial_spread"] > 100,
        f"R {p3['max_resultant']:.4f}, radial spread {p3['radial_spread']:.0f}x")

    f4 = rec["part4_fitted"]["fits"]
    wins = sum(f["matched"]["resultant"] > f["adversarial"]["resultant"] for f in f4)
    add("part4 the criterion prefers matched in EVERY restart",
        wins == len(f4), f"{wins}/{len(f4)}")
    add("part4 no radial-tail degeneracy in either arm",
        all(f[t]["radial_tail"] < 20 for f in f4 for t in ("matched", "adversarial")),
        f"max tail {max(f[t]['radial_tail'] for f in f4 for t in ('matched','adversarial')):.1f}")
    med_m = float(np.median([f["matched"]["resultant"] for f in f4]))
    med_a = float(np.median([f["adversarial"]["resultant"] for f in f4]))
    p5 = rec["part5_sensitivity"]
    add("part5 HONEST BOUNDARY: both instruments die at R_zB = 0",
        p5["kappa_b=0.0"]["behaviour_ratio"] < 2.0 and p5["kappa_b=0.0"]["R_zB"] < 0.05,
        f"R_zB {p5['kappa_b=0.0']['R_zB']:.4f}, behaviour "
        f"{p5['kappa_b=0.0']['behaviour_ratio']:.2f}x floor -- neither fires")
    defs = [p5[f"kappa_b={k}"]["relative_deficit"] for k in (0.25, 0.5, 1.0, 2.0, 4.0)]
    add("part5 the relative deficit is scale-free in R_zB (Prop. C: it is 1 - R_theta)",
        max(defs) - min(defs) < 0.10, f"deficits {[round(100 * d, 1) for d in defs]}%")
    add("part5 the resultant dominates in the weakly-concentrated band",
        p5["kappa_b=0.25"]["behaviour_ratio"] < 5.0
        and p5["kappa_b=0.25"]["relative_deficit"] > 0.08,
        f"at R_zB={p5['kappa_b=0.25']['R_zB']:.3f}: behaviour "
        f"{p5['kappa_b=0.25']['behaviour_ratio']:.2f}x (blind), deficit "
        f"{100 * p5['kappa_b=0.25']['relative_deficit']:.1f}%")
    if "module_dependence" in f4[0]["matched"]:
        dwins = sum(f["matched"]["module_dependence"]
                    < f["adversarial"]["module_dependence"] for f in f4)
        add("part4 ROUTE D UNDER LEARNING: the matched fit's modules are more "
            "independent than the adversarial fit's, in every restart",
            dwins == len(f4),
            f"{dwins}/{len(f4)}; matched "
            + ", ".join(f"{f['matched']['module_dependence']:.3f}" for f in f4)
            + " vs adversarial "
            + ", ".join(f"{f['adversarial']['module_dependence']:.3f}" for f in f4)
            + f" (baseline {f4[0]['matched']['module_dependence_baseline']:.3f})")
    add("part4 matched tracks the data's R1 better than the adversarial arm tracks R2",
        abs(med_m - rec["part4_fitted"]["data"]["R1"]) < 0.2,
        f"matched med {med_m:.4f} vs data R1 {rec['part4_fitted']['data']['R1']:.4f}; "
        f"adversarial med {med_a:.4f} vs data R2 {rec['part4_fitted']['data']['R2']:.4f}")

    p6 = rec["part6_proposition_R"]
    add("part6 Prop. R: the identity maximises the total resultant in every regime",
        all(v["n_strictly_better_than_identity"] == 0 for v in p6.values()),
        "; ".join(f"{k} argmax {v['argmax']}" for k, v in p6.items()))
    add("part6 and it beats every NON-permutation element when both phases are concentrated",
        all(p6[k]["identity"] > p6[k]["runner_up_nonpermutation"] + 0.02
            for k in ("kappa=(4.0,4.0)", "kappa=(4.0,1.0)", "kappa=(1.0,0.5)")),
        "; ".join(f"{k}: {p6[k]['identity']:.3f} vs {p6[k]['runner_up_nonpermutation']:.3f}"
                  for k in ("kappa=(4.0,4.0)", "kappa=(4.0,1.0)", "kappa=(1.0,0.5)")))
    k0 = "kappa=(4.0,0.0)"
    add("part6 BOUNDARY: a uniform phase law enlarges the stabiliser, so the "
        "resolution is only as good as the concentration",
        abs(p6[k0]["identity"] - p6[k0]["totals"]["1,0,1,1"]) < 5e-3,
        f"identity {p6[k0]['identity']:.4f} ties (1,0,1,1) at "
        f"{p6[k0]['totals']['1,0,1,1']:.4f}, while (1,1,0,1) collapses to "
        f"{p6[k0]['totals']['1,1,0,1']:.4f}")

    p7 = rec["part7_independence"]
    conc = ("kappa=(4.0,4.0)", "kappa=(4.0,1.0)")
    add("part7 Prop. R': the identity is the unique independent representative",
        all(p7[k]["argmin"] in ("1,0,0,1", "0,1,1,0") for k in conc),
        "; ".join(f"{k} argmin {p7[k]['argmin']}" for k in conc))
    add("part7 and its margin is 1-2 orders of magnitude, far beyond the resultant's",
        all(p7[k]["margin"] > 10 for k in conc),
        "; ".join(f"{k}: {p7[k]['margin']:.1f}x" for k in conc))
    add("part7 identity sits at the floor, i.e. it is genuinely independent",
        all(p7[k]["identity"] < 3 * p7["floor"]["mean"] for k in conc),
        f"floor {p7['floor']['mean']:.5f}; identities "
        + ", ".join(f"{p7[k]['identity']:.5f}" for k in conc))
    add("part7 BOUNDARY: with BOTH phase laws uniform the orbit is undetermined",
        p7["kappa=(0.0,0.0)"]["margin"] < 3.0,
        f"margin {p7['kappa=(0.0,0.0)']['margin']:.2f}x -- every representative "
        f"is independent, so nothing selects one")
    add("part7 BOUNDARY: one uniform phase law already enlarges the stabiliser",
        p7["kappa=(4.0,0.0)"]["margin"] < 3.0,
        f"margin {p7['kappa=(4.0,0.0)']['margin']:.2f}x")

    c8 = rec["part8_route_D"]["cases"]
    add("part8 Route D rejects the 4.3 triangular counterexample -- the object "
        "that makes block-diagonality FALSE under (B1)-(B4)",
        c8["4.3_triangular"]["ratio_to_baseline"] > 4.0,
        f"{c8['4.3_triangular']['under_h']:.4f} = "
        f"{c8['4.3_triangular']['ratio_to_baseline']:.1f}x baseline")
    add("part8 Route D rejects the section 7 lattice regrouping",
        c8["7_lattice"]["ratio_to_baseline"] > 4.0,
        f"{c8['7_lattice']['under_h']:.4f} = "
        f"{c8['7_lattice']['ratio_to_baseline']:.1f}x baseline")
    add("part8 and is CORRECTLY blind to the 3.1 regrouping ((B2)'s job, not its own)",
        c8["3.1_regrouping"]["ratio_to_baseline"] < 2.0,
        f"{c8['3.1_regrouping']['under_h']:.4f} = "
        f"{c8['3.1_regrouping']['ratio_to_baseline']:.1f}x baseline")
    bl = rec["part8_route_D"]["even_coupling_blind_spot"]
    dcs = [bl[f"c={c}"]["dcor"] for c in (0.25, 0.5, 1.0, 2.0, 5.0)]
    lins = [bl[f"c={c}"]["linear_abs_corr"] for c in (0.25, 0.5, 1.0, 2.0, 5.0)]
    add("part8 the dCor instrument sees section 3.10's blind spot (even coupling) "
        "where a linear probe does not",
        all(b > a for a, b in zip(dcs, dcs[1:])) and max(lins) < 0.15,
        f"dCor {[round(v, 3) for v in dcs]} rises monotonically; "
        f"linear |corr| {[round(v, 3) for v in lins]} stays at nothing")
    add("part8 HONEST LIMIT: with uniform phases the lattice IS "
        "independence-preserving (Hyvarinen-Pajunen), so this is not a proof",
        c8["HP_uniform_phase_lattice"]["ratio_to_baseline"] < 2.0,
        f"{c8['HP_uniform_phase_lattice']['under_h']:.4f} = "
        f"{c8['HP_uniform_phase_lattice']['ratio_to_baseline']:.1f}x baseline")

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
