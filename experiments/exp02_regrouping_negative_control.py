"""Experiment 2 -- the §3.1 regrouping NEGATIVE CONTROL (CLAUDE.md §4 step 6).

This is the falsification gate.  Data come from four *independent* systems that
we then insist on describing with K = 2 modules of dimension 2.  Because the
[2,2] partition is not the finest one, the grouping is not determined: all three
ways of pairing four coordinates give identical observations.

**The fit must be non-unique here.**  If restarts all agree on one grouping, the
assumptions are hiding something and the result must be reported, not tuned away.

Two parts:

* (a) exact -- construct all three groupings explicitly, linear and nonlinear,
  and verify each reproduces the observations bit for bit.  This does not
  depend on an optimiser and cannot be argued with.
* (b) empirical -- fit the modular model from many random initialisations and
  count how many distinct groupings show up among near-optimal fits.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from _common import banner, save, verdict
from idyn import linear as L
from idyn import metrics as MT
from idyn import systems as S
from idyn.models import ModelConfig
from idyn.train import TrainConfig, fit_many, make_dataset

SEED = 0
N_RESTARTS = 12


def pairings_of_four() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """The three ways to split {0,1,2,3} into two unordered pairs."""
    out, seen = [], set()
    for c in combinations(range(4), 2):
        rest = tuple(sorted(set(range(4)) - set(c)))
        key = frozenset({c, rest})
        if key not in seen:
            seen.add(key)
            out.append((c, rest))
    return out


def perm_for(grouping) -> np.ndarray:
    """Permutation matrix reordering coordinates so ``grouping`` becomes [2,2]."""
    order = list(grouping[0]) + list(grouping[1])
    return np.eye(4)[order]


# --------------------------------------------------------------------------
# (a) exact construction
# --------------------------------------------------------------------------


def part_a_linear(rng) -> dict:
    ce = S.regrouping_counterexample(seed=SEED)
    F, W = ce["F"], ce["decoder"].W
    z = rng.standard_normal((64, 4))

    rows = []
    for g in pairings_of_four():
        P = perm_for(g)
        F_t = P @ F @ P.T
        W_t = W @ P.T
        obs_equal = bool(np.allclose(W_t @ (P @ z.T), W @ z.T, atol=1e-12))
        block_diag = bool(
            np.allclose(F_t[:2, 2:], 0, atol=1e-12) and np.allclose(F_t[2:, :2], 0, atol=1e-12)
        )
        cert = L.certify_finest_decomposition(F_t, [2, 2])
        rows.append(
            {
                "grouping": [list(g[0]), list(g[1])],
                "observations_identical": obs_equal,
                "F_tilde_is_block_diagonal": block_diag,
                "spectra": [np.diag(F_t)[:2].tolist(), np.diag(F_t)[2:].tolist()],
                "certificate_canonical": cert.canonical,
            }
        )
        print(
            f"   grouping {g[0]}|{g[1]}: identical observations={obs_equal}, "
            f"F~ block diagonal={block_diag}, canonical={cert.canonical}"
        )
    return {"rows": rows, "all_valid": all(r["observations_identical"] and r["F_tilde_is_block_diagonal"] for r in rows)}


def part_a_nonlinear(rng) -> dict:
    """Same regrouping, four independent *nonlinear* 1-D maps (§3.1 is not linear-only)."""
    nl = S.nonlinear_regrouping_counterexample(seed=SEED)
    blocks, W = nl["blocks"], nl["decoder"].W
    sys = S.ModularSystem(blocks)

    z0 = S.sample_initial_conditions(4, 32, rng, radius=1.5)
    Z = sys.simulate(z0, 12)
    X = Z @ W.T

    rows = []
    for g in pairings_of_four():
        P = perm_for(g)
        order = list(g[0]) + list(g[1])
        sys_t = S.ModularSystem([blocks[i] for i in order])
        W_t = W @ P.T
        Zt = sys_t.simulate(z0 @ P.T, 12)
        Xt = Zt @ W_t.T
        err = float(np.abs(Xt - X).max())
        # the relabelled system really is modular for the [2,2] partition:
        # coordinates 0,1 evolve without reference to 2,3 and vice versa
        rows.append({"grouping": [list(g[0]), list(g[1])], "max_obs_error": err, "identical": err < 1e-12})
        print(f"   grouping {g[0]}|{g[1]}: max |x~ - x| = {err:.3e}")
    return {"rows": rows, "all_valid": all(r["identical"] for r in rows)}


# --------------------------------------------------------------------------
# (b) empirical: does fitting find one grouping or several?
# --------------------------------------------------------------------------


def part_b(rng) -> dict:
    nl = S.nonlinear_regrouping_counterexample(
        scales=(0.95, 0.90, 0.85, 0.80), gains=(1.0, 1.4, 0.8, 1.7), seed=SEED
    )
    sys = S.ModularSystem(nl["blocks"])
    X, Z, dec = make_dataset(sys, n_obs=8, n_traj=256, T=20, rng=rng, radius=1.5)
    print(f"   data: X {X.shape}, latents {Z.shape}")

    mcfg = ModelConfig(n_obs=8, d=4, partition=[2, 2], decoder="linear", encoder="linear")
    tcfg = TrainConfig(steps=1500, lr=3e-3, batch=64, seed=SEED)
    fits = fit_many(X, mcfg, tcfg, n_restarts=N_RESTARTS)

    assignments, losses = [], []
    for f in fits:
        A = MT.fit_linear_relation(Z, f.z_fit)
        assignments.append(MT.coordinate_pairing(A, [2, 2]))
        losses.append(f.fit_quality)
        print(f"   restart seed {f.seed:5d}: fit_quality {f.fit_quality:.3e}  grouping {assignments[-1]}")

    # Within 3x of the best fit.  All restarts land at recon+dyn MSE of a few
    # times 1e-3 on unit-variance latents, so the spread is optimiser noise
    # rather than distinct quality basins; a tighter filter would discard good
    # fits and understate the non-uniqueness.
    rep = MT.nonuniqueness_report(assignments, losses, rel_tol=2.0)
    everything = MT.nonuniqueness_report(assignments, losses, rel_tol=np.inf)
    print(f"\n   {rep.summary()}")
    print(f"   across all {N_RESTARTS} restarts: {everything.counts}")
    return {
        "n_restarts": N_RESTARTS,
        "assignments": [list(a) for a in assignments],
        "fit_quality": losses,
        "distinct_groupings": rep.n_distinct,
        "grouping_counts": rep.counts,
        "grouping_counts_all_restarts": everything.counts,
        "n_near_optimal": rep.n_near_optimal,
        "best_loss": rep.best_loss,
        "non_unique": rep.non_unique,
    }


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 2 -- §3.1 regrouping NEGATIVE CONTROL (the fit must be non-unique)")

    print("\n(a) exact construction, linear:")
    a_lin = part_a_linear(rng)
    print("\n(a) exact construction, nonlinear (four independent 1-D maps):")
    a_nl = part_a_nonlinear(rng)
    print(f"\n(b) empirical: {N_RESTARTS} restarts of a modular [2,2] fit")
    b = part_b(rng)

    banner("VERDICTS")
    checks = [
        (a_lin["all_valid"], "all 3 linear groupings reproduce the observations exactly and stay modular"),
        (a_nl["all_valid"], "all 3 nonlinear groupings reproduce the observations exactly -- §3.1 is not a linear artifact"),
        (
            b["non_unique"],
            f"fitting is NON-UNIQUE: {b['distinct_groupings']} distinct groupings among "
            f"{b['n_near_optimal']} near-optimal restarts {b['grouping_counts']}",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    if not b["non_unique"]:
        print(
            "\n  !! The negative control came back UNIQUE.  Per CLAUDE.md §4 step 6 this\n"
            "     means the setup is hiding an extra constraint (a symmetry-breaking\n"
            "     initialisation, the whitening penalty, or the optimiser).  Investigate\n"
            "     before trusting any positive recovery result."
        )

    save(
        "exp02_regrouping_negative_control",
        {"seed": SEED, "part_a_linear": a_lin, "part_a_nonlinear": a_nl, "part_b_fitting": b,
         "all_passed": passed, "checks": [{"passed": ok, "claim": m} for ok, m in checks]},
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
