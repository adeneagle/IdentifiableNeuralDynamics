"""Experiment 1 -- the linear base case, numerically certified (CLAUDE.md §4 step 2).

Question: with h forced into GL(d) by a full-column-rank decoder (§3.5), when
does ``S F S^{-1}`` block-diagonal force S to be a block permutation?

The claim (theory/linear_case.md): it does exactly when (A1) every block is
indecomposable and (A2) the block spectra are pairwise disjoint.

The certificate used here is the *intertwiner space* ``Hom(F, F~) = {S : SF = F~S}``.
That space is the complete solution set of the identifiability problem in the
linear case, so we do not need to fit anything -- we compute it exactly:

* under (A1)+(A2) it should equal ``(+)_i Hom(f_i, f~_sigma(i))``, i.e. every
  element is supported on matched blocks only, and its dimension should equal
  ``sum_i dim End(f_i)``;
* in the §3.1 counterexample it should be strictly larger and contain invertible
  elements that mix modules.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import linear as L
from idyn import systems as S

SEED = 0


def scaled_rotation(s: float, theta: float) -> np.ndarray:
    return s * S.rotation(theta)


def hom_support(H: np.ndarray, part_row, part_col) -> np.ndarray:
    """Total block-energy pattern of a whole intertwiner space."""
    E = np.zeros((len(part_row), len(part_col)))
    for B in H:
        E += L.block_energy_matrix(B, part_row, part_col)
    return E


def analyse(name: str, F: np.ndarray, F_tilde: np.ndarray, partition, rng) -> dict:
    cert = L.certify_finest_decomposition(F, partition)
    H = L.intertwiner_space(F, F_tilde)
    E = hom_support(H, partition, partition)
    En = E / max(E.sum(), 1e-300)

    # predicted dimension under (A1)+(A2): only matched blocks contribute
    end_dims = [L.intertwiner_space(B, B).shape[0] for B in L.blocks_of(F, partition)]
    predicted = int(sum(end_dims))

    # is the space confined to a single matched block per row/column?
    rep = L.block_permutation_report(En, partition, partition, tol=1e-9)
    confined = bool(rep.on_block_fraction > 1 - 1e-9)

    # sample invertible elements and test each for block-permutation structure
    n_mix, n_inv, worst = 0, 0, 1.0
    for _ in range(400):
        c = rng.standard_normal(H.shape[0])
        Ssamp = np.tensordot(c, H, axes=(0, 0))
        r = L.block_permutation_report(Ssamp, partition, partition, tol=1e-8)
        if r.invertible:
            n_inv += 1
            worst = min(worst, r.on_block_fraction)
            if not r.is_block_permutation:
                n_mix += 1

    print(f"\n-- {name}")
    print(f"   certificate       : {cert.summary()}")
    print(f"   dim Hom(F,F~)     : {H.shape[0]}   (sum_i dim End(f_i) = {predicted})")
    print(f"   block energy (norm):\n{np.array2string(En, precision=4, prefix='       ')}")
    print(f"   invertible samples: {n_inv}/400, of which mixing modules: {n_mix}")
    print(f"   worst on-block frac among invertible samples: {worst:.4f}")

    return {
        "name": name,
        "partition": list(partition),
        "A1": cert.A1,
        "A2": cert.A2,
        "canonical": cert.canonical,
        "separation": cert.separation,
        "reasons": cert.reasons,
        "dim_hom": int(H.shape[0]),
        "predicted_dim": predicted,
        "block_energy_normalised": En,
        "space_confined_to_matched_blocks": confined,
        "n_invertible_samples": n_inv,
        "n_module_mixing": n_mix,
        "worst_on_block_fraction": float(worst),
    }


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 1 -- linear base case: when is S forced to be a block permutation?")

    records, checks = [], []

    # ---------------------------------------------------------------- case 1
    # (A1) holds: each 2x2 block is a scaled rotation, indecomposable over R.
    # (A2) holds: different moduli, so disjoint spectra.
    A1_ = scaled_rotation(0.95, 0.40)
    A2_ = scaled_rotation(0.70, 1.10)
    F = np.block([[A1_, np.zeros((2, 2))], [np.zeros((2, 2)), A2_]])
    P = np.eye(4)[[2, 3, 0, 1]]  # swap the two modules wholesale: a legal h
    r = analyse("A1+A2 hold (scaled rotations, distinct moduli)", F, P @ F @ P.T, [2, 2], rng)
    records.append(r)
    checks.append(
        (
            r["canonical"] and r["n_module_mixing"] == 0 and r["dim_hom"] == r["predicted_dim"],
            "under (A1)+(A2): every invertible intertwiner is a block permutation, "
            f"dim Hom = {r['dim_hom']} = sum_i dim End(f_i)",
        )
    )

    # ---------------------------------------------------------------- case 2
    # §3.1: blocks are decomposable, so the [2,2] partition is not the finest.
    ce = S.regrouping_counterexample(seed=SEED)
    r = analyse("(A1) FAILS -- §3.1 regrouping counterexample", ce["F"], ce["F_tilde"], [2, 2], rng)
    records.append(r)
    checks.append(
        (
            (not r["A1"]) and r["n_module_mixing"] > 0,
            "with (A1) violated the intertwiner space contains invertible maps that "
            f"mix modules ({r['n_module_mixing']}/{r['n_invertible_samples']}) -- "
            "the conjecture is false as originally stated",
        )
    )

    # ---------------------------------------------------------------- case 3
    # (A1) holds but (A2) fails: two Jordan blocks sharing an eigenvalue.  This
    # is the second counterexample (theory/counterexamples.md §2) and shows
    # indecomposability alone is NOT enough -- a point the original brief missed.
    J = np.array([[0.8, 1.0], [0.0, 0.8]])
    F2 = np.block([[J, np.zeros((2, 2))], [np.zeros((2, 2)), J]])
    r = analyse("(A1) holds, (A2) FAILS -- J2(0.8) (+) J2(0.8)", F2, F2, [2, 2], rng)
    records.append(r)
    checks.append(
        (
            r["A1"] and (not r["A2"]) and r["n_module_mixing"] > 0,
            "indecomposability alone is insufficient: shared spectra let invertible "
            f"intertwiners mix modules ({r['n_module_mixing']}/{r['n_invertible_samples']})",
        )
    )

    # ---------------------------------------------------------------- case 4
    # The finest partition of the §3.1 system IS canonical -- which is the fix.
    r = analyse("§3.1 system under its FINEST partition [1,1,1,1]", ce["F"], ce["F"], [1, 1, 1, 1], rng)
    records.append(r)
    checks.append(
        (
            r["canonical"] and r["n_module_mixing"] == 0,
            "refining to the indecomposable partition [1,1,1,1] restores uniqueness -- "
            "this is the §3.1 fix, working",
        )
    )

    banner("VERDICTS")
    tags = [verdict(ok, msg) for ok, msg in checks]
    passed = all(t == "PASS" for t in tags)

    save(
        "exp01_linear_base_case",
        {"seed": SEED, "cases": records, "all_passed": passed,
         "checks": [{"passed": ok, "claim": m} for ok, m in checks]},
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
