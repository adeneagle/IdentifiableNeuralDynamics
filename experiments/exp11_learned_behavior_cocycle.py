"""Experiment 11 -- does the B∘C mechanism survive *learning*?  (the falsification gate)

!! SUPERSEDED IN PART (CLAUDE.md §3.12, task 32).  The behavioural penalty used
!! here is gauge-dependent -- it falls like eps^2/eps^4 when the pinned block
!! shrinks -- so the optimiser satisfied it by collapsing the block (21x smaller
!! than its partner) rather than by making it u-invariant, while that block still
!! carried the u-varying latent at distance correlation 0.99.  **The behavioural
!! arm never differed from the no-behaviour arm in the way it claims**, so the
!! ablation below cannot support any conclusion about B∘C, in either the linear
!! or the nonlinear decoder regime.  `exp13` re-runs it with the penalty fixed.

exp10 verified the B∘C composition with analytic candidate reparameterisations.
This one fits actual models from data and asks whether the block-diagonal recovery
holds when a nonlinear encoder/decoder and a modular transition are *learned*, and
whether it degrades when either ingredient is removed -- the project's standard of
evidence (CLAUDE.md §6 task 6: falsify before more theory).

Data: an autonomous modular system, block A slow/dominant (``s=0.9``) with its
initial law conditioned on behaviour ``u`` (variance-modulated), block B fast/
dominated (``s=0.5``) and u-invariant.  Observed through a random 12-D nonlinear
coupling-flow decoder (``systems.MLPDecoder``) -- the Theorem-B regime, and the one
that matters: a *linear* decoder forces h into GL(d) outright (§3.5), so the
triangular/block-diagonal distinction the ablations must show only exists once the
observation map itself is nonlinear.

Provenance, because it changes how the earlier numbers read: until 2026-08-03 this
file *said* it used a nonlinear decoder while actually calling the default
``LinearDecoder`` -- ``MLPDecoder`` did not exist, so no experiment in the repo had
ever exercised the Theorem B observation model.  The nonlinearity came only from
the fitted MLP encoder.  The measured observation nonlinearity is now printed on
every run so the claim cannot silently rot again.

Two readouts of the fitted latents ``z_fit`` vs the true ``z``:

  * ``block-B u-dependence`` (``behavior.block_u_dependence`` on the fitted invariant
    block) -- the M_BA signature.  A fitted invariant block that has picked up the
    varying block is u-dependent; behaviour is what drives this to zero.
  * ``filtration_report`` diag / lower / upper energy of the recovered linear
    relation -- ``upper`` is the forbidden cross-block Lemma C (dynamics) kills,
    ``lower`` the one behaviour kills, ``diag`` the block-diagonal mass.

Ablations (best of a few restarts each):
  full = modular + behaviour; dynamics-only = modular, no behaviour;
  behaviour-only = unconstrained transition + behaviour; neither = unconstrained.
Expected: only ``full`` is block-diagonal; dynamics-only leaves the behavioural
cross-block (block B stays u-dependent), behaviour-only leaves the dynamical one
(forbidden ``upper`` mass) -- neither ingredient suffices alone.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import behavior as BEH
from idyn import metrics as M
from idyn import systems as S
from idyn import train as T
from idyn.models import ModelConfig

SEED = 0
N_OBS = 12
U_LEVELS = np.array([0, 1, 2, 3])
N_PER_U = 250
T_STEPS = 15
# 1200 was tuned for a linear observation map and *undertrains* a nonlinear one
# into a reversed result (forbidden mass 0.177 at 1200, ~0.05 by 3000) -- §3.11.
STEPS = 4000
N_RESTARTS = 8  # §3.11: sd(jac_diag) is 0.104 here, so 3 restarts cannot resolve it
W_BEHAVIOR = 5.0
INV = (2, 4)  # the block designated invariant (coords 2..3)
DEC_STRENGTH = 1.0  # coupling-flow nonlinearity; ~0.45 nonlinear residual, cond(J) ~ 49


def nonlinearity(dec, Z):
    """Fraction of the decoder not captured by any affine map, on the actual data.

    Reported because the claim "nonlinear decoder" has to be *measured*, not
    asserted: the construction this replaced looked nonlinear and was 3% so.
    """
    z = np.asarray(Z, dtype=float).reshape(-1, Z.shape[-1])
    x = dec(z)
    aug = np.hstack([z, np.ones((len(z), 1))])
    A, *_ = np.linalg.lstsq(aug, x, rcond=None)
    return float(np.linalg.norm(x - aug @ A) / np.linalg.norm(x))


def all_fits(X, U, part, behav, seed0):
    """*Every* restart for one ablation condition, not the best one.

    CLAUDE.md §3.11: selecting by ``fit_quality`` does not control the structural
    readout.  Measured, ``corr(fitq, jac_diag)`` is -0.044 with a linear decoder
    and +0.279 with a nonlinear one -- no information, and the nonlinear sign
    means best-fit mildly *anti*-selects for diagonality.  Since ``jac_diag`` has
    sd 0.104 in that regime, a best-of-N point estimate is close to a coin flip,
    so the caller summarises the distribution instead.
    """
    cfg_model = ModelConfig(n_obs=N_OBS, d=4, partition=part, decoder="mlp", encoder="mlp")
    out = []
    for r in range(N_RESTARTS):
        tc = T.TrainConfig(
            steps=STEPS, seed=seed0 + 100 * r,
            w_behavior=(W_BEHAVIOR if behav else 0.0),
            inv_start=(INV[0] if behav else 0), inv_stop=(INV[1] if behav else 0),
            # Pinned to the OLD, gauge-dependent penalty so this script keeps
            # reproducing the JSON on record.  CLAUDE.md 3.12: that penalty is
            # satisfied by shrinking the block, so every behavioural conclusion
            # below is void.  exp13 is the replacement.
            behavior_whiten=False,
        )
        out.append(T.fit(X, cfg_model, tc, U=(U if behav else None)))
    return out


def summarise(per_restart):
    """Median + range of each readout across restarts, plus the worst case.

    ``*_max`` / ``*_min`` are kept because the only conclusion that survives §3.11
    is one that holds in *every* restart, not one that holds at the median.
    """
    keys = [k for k, v in per_restart[0].items() if isinstance(v, float)]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in per_restart], dtype=float)
        out[k] = float(np.median(vals))
        out[f"{k}_min"] = float(vals.min())
        out[f"{k}_max"] = float(vals.max())
        out[f"{k}_sd"] = float(vals.std())
    out["n_restarts"] = len(per_restart)
    return out


def rate_order_of(system):
    """Module indices slowest-contracting first, as ``filtration_report`` wants.

    Must be derived from the system, not hardcoded: part 2 swaps which block is
    spectrally dominant, and a stale ``[0, 1]`` there transposes the upper and
    lower triangles -- i.e. silently swaps the forbidden cross-block for the
    allowed one, which is the entire quantity under test.
    """
    top = [float(np.max(blk.lyapunov_spectrum_exact())) for blk in system.blocks]
    return [int(i) for i in np.argsort(top)[::-1]]


def make_h(res, dec):
    """The reparameterisation h: z_true -> z_fit, as a callable for the Jacobian.

    Observations are x = g(z) for the true decoder g, the fitted latents are
    z_fit = enc(x), so h = enc . g.  This is the map Lemma C constrains, and
    evaluating its Jacobian is the only way to see coupling the linear probe
    misses.  Calls ``dec`` rather than ``z @ dec.W.T`` so it is correct for the
    nonlinear decoder as well as the linear one.
    """
    import torch

    def h(z):
        x = dec(np.asarray(z, dtype=float))
        with torch.no_grad():
            zf = res.model.encode(torch.tensor(x, dtype=torch.float32))
        return zf.numpy().astype(float)

    return h


def readouts(res, Z, U, rate_order, dec):
    """u-dependence plus THREE block-structure readouts of the same h.

    ``filtration_report`` is retained for continuity with exp10, but it reads
    only the *linear* part of h and is provably blind to purely nonlinear
    cross-block coupling (metrics.py §"Nonlinear block structure"): for
    h = (z_A, z_B + 5 z_A^2) it reports 0.97 block-diagonal.  The Jacobian and
    dCor readouts are the ones the verdicts should rest on.
    """
    zf = res.z_fit
    u_rep = np.repeat(U, zf.shape[1])
    blockB = zf.reshape(-1, 4)[:, INV[0]:INV[1]]
    dep_b = BEH.block_u_dependence(blockB, u_rep).total

    part = [2, 2]
    fr = M.filtration_report(Z, zf, part, part, rate_order=rate_order)
    # The correspondence must be *matched*, not pinned: nothing makes the fit put
    # its block 0 where the true block 0 is, and it demonstrably does not -- the
    # raw dCor here is [[0.26, 0.96], [0.99, 0.28]], a clean permutation.  Pinning
    # identity reads that correct recovery as a total failure (diag 0.000).  The
    # raw coupling matrices are recorded so the pairing can be checked by eye.
    pts = Z.reshape(-1, 4)
    sub = np.random.default_rng(SEED).choice(len(pts), min(2000, len(pts)), replace=False)
    jr = M.jacobian_block_report(make_h(res, dec), pts[sub], part, part, rate_order=rate_order)
    dr = M.distance_correlation_block_report(Z, zf, part, part,
                                             rate_order=rate_order, seed=SEED)
    return {
        "fit_quality": float(res.fit_quality),
        "blockB_u_dependence": float(dep_b),
        "diag": float(fr.on_block),
        "lower": float(fr.lower_mass),
        "upper": float(fr.upper_mass),
        "jac_diag": float(jr.on_block),
        "jac_lower": float(jr.lower_mass),
        "jac_upper": float(jr.upper_mass),
        "dcor_diag": float(dr.on_block),
        "dcor_lower": float(dr.lower_mass),
        "dcor_upper": float(dr.upper_mass),
        "dcor_coupling": dr.coupling.tolist(),
        "jac_coupling": jr.coupling.tolist(),
        "jac_assignment": list(jr.assignment),
        "dcor_assignment": list(dr.assignment),
    }


def classify(r, u_tol=0.02, mass_tol=0.08):
    """Label the recovered structure from the readouts.

    Reads the *Jacobian* masses, not the linear ones: the linear probe cannot
    distinguish the two structures this function is here to name.

    ``M_BA`` is tested two ways, and the disjunction matters.  ``u``-dependence
    asks whether the invariant block moves with behaviour; ``jac_lower`` asks
    whether it depends on the varying block *at all*.  A fit can pass the first
    and fail the second -- the block tracks the parts of A's variation that are
    not u-driven -- and the misaligned run does exactly that (u-dep 0.0022 but
    lower mass 0.121).  An earlier version tested only ``upper``, so it could
    not detect a lower-triangular map at all: it called every such fit
    block-diagonal, which is precisely the distinction part 2 exists to make.
    """
    ba_leak = (r["blockB_u_dependence"] > u_tol) or (r["jac_lower"] > mass_tol)
    ab_leak = r["jac_upper"] > mass_tol              # M_AB (forbidden dynamical cross-block)
    if not ba_leak and not ab_leak:
        return "block-diagonal"
    if ba_leak and not ab_leak:
        return "triangular (M_BA leak)"
    if ab_leak and not ba_leak:
        return "triangular (M_AB leak)"
    return "mixed"


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 11 -- does B+C survive learning?  (falsification gate)")

    # aligned: varying block A is the slow/dominant one
    sys_aligned = S.ModularSystem(
        [S.TwistBlock(s=0.90, omega=0.40, beta=0.6), S.TwistBlock(s=0.50, omega=1.10, beta=-0.5)]
    )
    dec = S.MLPDecoder.random(N_OBS, 4, rng, strength=DEC_STRENGTH)
    X, Z, U, dec = T.make_behavioural_dataset(
        sys_aligned, 2, 2, N_OBS, N_PER_U, T_STEPS, U_LEVELS, rng,
        mode="variance", decoder=dec,
    )
    print(f"   data X{X.shape}, {len(U_LEVELS)} behaviour levels, "
          f"nonlinear (coupling-flow) decoder, strength {DEC_STRENGTH}")
    print(f"   observation nonlinearity: {nonlinearity(dec, Z):.3f} "
          f"(fraction of g not captured by any affine map)\n")

    banner("PART 1 -- ablations on the aligned system (varying = dominant)")
    print(f"   {'condition':>16s} {'u-dep':>7s} | {'LINdiag':>8s} {'LINup':>6s} "
          f"| {'JACdiag':>8s} {'JAClow':>7s} {'JACup':>6s} | {'DCdiag':>7s} {'DCup':>6s} "
          f"| {'structure':>22s}")
    conditions = [
        ("full (B+C)", [2, 2], True),
        ("dynamics-only", [2, 2], False),
        ("behaviour-only", None, True),
        ("neither", None, False),
    ]
    order_aligned = rate_order_of(sys_aligned)
    print(f"   rate order (slowest-first): {order_aligned}\n")
    rows = {}
    for name, part, behav in conditions:
        per = [readouts(f, Z, U, order_aligned, dec)
               for f in all_fits(X, U, part, behav, SEED)]
        r = summarise(per)
        r["per_restart"] = per
        r["structure"] = classify(r)
        rows[name] = r
        print(f"   {name:>16s} {r['blockB_u_dependence']:7.4f} "
              f"| {r['diag']:8.3f} {r['upper']:6.3f} "
              f"| {r['jac_diag']:8.3f} {r['jac_lower']:7.3f} {r['jac_upper']:6.3f} "
              f"| {r['dcor_diag']:7.3f} {r['dcor_upper']:6.3f} | {r['structure']:>22s}")
        print(f"   {'':>16s} {'(spread)':>7s} "
              f"| {'':8s} {'':6s} "
              f"| {r['jac_diag_min']:8.3f}-{r['jac_diag_max']:.3f} sd {r['jac_diag_sd']:.3f}, "
              f"upper max {r['jac_upper_max']:.3f}")

    full = rows["full (B+C)"]
    dyn = rows["dynamics-only"]
    beh = rows["behaviour-only"]
    print(f"""
   Reading: 'full' should be block-diagonal; removing behaviour should let the
   invariant block pick up the varying one (block-B u-dependence rises); removing
   the modular dynamics should let the forbidden 'upper' cross-block survive.
   LIN is retained only to show what it misses -- verdicts read JAC.""")

    # ---------------------------------------------------------------- part 2
    banner("PART 2 -- alignment under learning: varying = DOMINATED gives only triangular")
    print("""   Swap so the behaviour-varying block is the fast/dominated one. Then behaviour
   and the gap both target the same cross-derivative and the other survives, so
   even the full B+C fit should not reach block-diagonal (exp10 part 4, learned).\n""")
    sys_misaligned = S.ModularSystem(
        [S.TwistBlock(s=0.50, omega=0.40, beta=0.6), S.TwistBlock(s=0.90, omega=1.10, beta=-0.5)]
    )
    decm = S.MLPDecoder.random(N_OBS, 4, rng, strength=DEC_STRENGTH)
    Xm, Zm, Um, decm = T.make_behavioural_dataset(
        sys_misaligned, 2, 2, N_OBS, N_PER_U, T_STEPS, U_LEVELS, rng,
        mode="variance", decoder=decm,
    )
    order_mis = rate_order_of(sys_misaligned)
    print(f"   rate order (slowest-first): {order_mis}  -- swapped, as intended\n")
    per_mis = [readouts(f, Zm, Um, order_mis, decm)
               for f in all_fits(Xm, Um, [2, 2], True, SEED)]
    r_mis = summarise(per_mis)
    r_mis["per_restart"] = per_mis
    r_mis["structure"] = classify(r_mis)
    print(f"   misaligned full B+C fit:  blockB u-dep {r_mis['blockB_u_dependence']:.4f}  "
          f"JAC diag {r_mis['jac_diag']:.3f}  lower {r_mis['jac_lower']:.3f}  "
          f"upper {r_mis['jac_upper']:.3f}  (LIN diag {r_mis['diag']:.3f}) "
          f"-> {r_mis['structure']}")

    banner("VERDICTS")
    # All thresholds read medians, except where the claim is "in every restart" --
    # §3.11: with sd(jac_diag) ~ 0.1 a single number decides nothing, and the only
    # safe assertions are the ones no restart violates.
    full_bd = full["structure"] == "block-diagonal"
    dyn_leaks_ba = dyn["blockB_u_dependence"] > 3.0 * max(full["blockB_u_dependence"], 1e-4)
    beh_leaks_ab = beh["jac_upper"] > 2.0 * max(full["jac_upper"], 1e-3)
    checks = [
        (
            full_bd,
            f"the learned full B+C model recovers BLOCK-DIAGONAL structure: block-B "
            f"u-dependence {full['blockB_u_dependence']:.4f} (~0) and Jacobian diagonal "
            f"{full['jac_diag']:.3f} [{full['jac_diag_min']:.3f}, {full['jac_diag_max']:.3f}] "
            f"with forbidden upper {full['jac_upper']:.3f}"
            + ("" if full_bd else
               "  <-- EXPECTED TO FAIL under a nonlinear observation map: the "
               "conclusion weakens to TRIANGULAR (forbidden upper is suppressed, "
               "allowed lower is not). This is CLAUDE.md task 29, not a bug. "
               "It passes with DEC_STRENGTH=0 (linear decoder), where §3.5 forces "
               "h into GL(d) and the claim is untestable anyway."),
        ),
        (
            # the one nonlinear-regime claim that survives EVERY restart
            full["jac_upper_max"] < 0.10,
            f"the forbidden cross-block M_AB is suppressed in every one of "
            f"{full['n_restarts']} restarts (max {full['jac_upper_max']:.3f}) -- Lemma C's "
            f"half of the composition holds under learning even with a nonlinear decoder",
        ),
        (
            dyn_leaks_ba,
            f"removing behaviour lets the invariant block pick up the varying one: block-B "
            f"u-dependence rises to {dyn['blockB_u_dependence']:.4f} vs {full['blockB_u_dependence']:.4f} "
            f"with behaviour -- the modular dynamics alone leave M_BA (behaviour's job)",
        ),
        (
            beh_leaks_ab,
            f"removing the modular dynamics lets the forbidden cross-block survive: Jacobian "
            f"upper {beh['jac_upper']:.3f} vs {full['jac_upper']:.3f} with modular dynamics -- "
            f"behaviour alone leaves M_AB (the dynamics' job)",
        ),
        (
            r_mis["structure"] != "block-diagonal",
            f"alignment holds under learning: with the varying block dominated, the full fit "
            f"does not reach block-diagonal ({r_mis['structure']}) -- block-diagonal needs "
            f"varying = dominant",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    save(
        "exp11_learned_behavior_cocycle",
        {
            "seed": SEED, "n_obs": N_OBS, "u_levels": U_LEVELS.tolist(),
            "n_per_u": N_PER_U, "T": T_STEPS, "steps": STEPS, "n_restarts": N_RESTARTS,
            "w_behavior": W_BEHAVIOR, "invariant_block": list(INV),
            "aligned": rows,
            "misaligned_full": r_mis,
            "all_passed": passed,
            "checks": [{"passed": ok, "claim": m} for ok, m in checks],
        },
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
