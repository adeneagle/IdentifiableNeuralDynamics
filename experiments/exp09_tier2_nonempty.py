"""Experiment 9 -- is Route A's Tier 2 non-empty?  (theory/approaches.md §A.2)

Route A splits.  **Tier 1** assumes non-resonance on the *full* spectrum; there
Poincare linearises F outright, the identified object is a linear map, and the
tier is honestly billed as robustness of Theorem A rather than as nonlinear
identifiability.  **Tier 2** keeps *within-module* resonances, which survive
normal-form reduction as finitely many nonlinear terms whose coefficients are
conjugacy invariants.  All of Route A's nonlinear content sits in Tier 2.

So the tier split is only worth anything if Tier 2 is non-empty -- if some system
can satisfy Tier 2's cross-module non-resonance hypothesis *while* carrying a live
within-module resonance.  If not, Route A collapses to Tier 1 and the headline
"identification of nonlinear dynamics" is hollow.

The witness is  f(z_a, z_b) = (mu z_a, mu^2 z_b + c z_a^2),  whose linear part is
diag(mu, mu^2).  Parts:

  1. The homological operator at degree 2 has exactly ONE zero eigenvalue, at the
     monomial z_a^2 in the z_b slot -- because lam_b - lam_a^2 = mu^2 - mu^2 = 0.
     So c is not removable and the map does not linearise.  With c = 0 it does.
  2. The resonance shows up dynamically as a SECULAR term: b_n carries a factor
     of n that no linear map produces.
  3. The resonance is within-module, so cross-module non-resonance still holds --
     Tier 2's hypotheses are satisfiable.  Partners at mu^2 and mu^3 are rejected.
  4. Consequence for the learning machinery: the linear part diag(mu, mu^2) is
     DECOMPOSABLE (two distinct real eigenvalues), so the linearised (B2) test in
     selection.certify_fitted_model reports this module as splitting -- while the
     map admits no invariant curve tangent to e_a and so does not split at all.
     A false negative living exactly in Tier 2.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import linear as LIN
from idyn import normalform as NF
from idyn import selection as SEL
from idyn import spectra as SP
from idyn import systems as S

SEED = 0
MU, C, NU = 0.70, 0.90, 0.50
DEGREE = 2


def main() -> int:
    rng = np.random.default_rng(SEED)
    banner("EXPERIMENT 9 -- Route A Tier 2 is non-empty")

    w = S.tier2_witness(mu=MU, c=C, nu=NU)
    node, eigs = w["node"], [MU, MU**2]

    # ---------------------------------------------------------------- part 1
    banner("PART 1 -- the homological operator: one resonance, exactly")
    print(f"   f(z_a, z_b) = ({MU} z_a,  {MU**2:.4g} z_b + {C} z_a^2)")
    print(f"   linear part diag({MU}, {MU**2:.4g}); homological eigenvalues "
          f"lam_i - lam^m at degree {DEGREE}:\n")
    print(f"     {'monomial':>16s} {'lam_i - lam^m':>16s} {'resonant?':>11s}")
    spectrum = []
    for vm in NF.homological_eigenvalues(eigs, DEGREE):
        mono = "".join(
            f"z{k + 1}^{p}" if p > 1 else f"z{k + 1}" for k, p in enumerate(vm.exponent) if p
        )
        label = f"{mono} e{vm.component + 1}"
        print(f"     {label:>16s} {vm.value.real:16.10f} {str(vm.resonant()):>11s}")
        spectrum.append({"component": vm.component, "exponent": list(vm.exponent),
                         "value": vm.value.real, "resonant": vm.resonant()})

    res = NF.resonances_at_degree(eigs, DEGREE)
    rep = NF.linearization_obstruction(eigs, {(1, (2, 0)): C}, DEGREE)
    rep0 = NF.linearization_obstruction(eigs, {(1, (2, 0)): 0.0}, DEGREE)
    print(f"\n   {rep.summary()}")
    print(f"   control (c = 0): linearizable = {rep0.linearizable}")
    print(f"""
   The zero is not an accident of the tolerance: lam_b - lam_a^2 = mu^2 - mu^2 is
   identically 0 for EVERY mu, so this resonance is structural.  The nearest
   non-resonant eigenvalue is {rep.min_abs_eigenvalue:.6f} away, so there is no
   ambiguity about which monomial is obstructed.""")

    # ---------------------------------------------------------------- part 2
    banner("PART 2 -- the resonance is visible dynamically, as a secular term")
    z0 = np.array([0.8, 0.3])
    print(f"   b_n = mu^(2n) z_b + n c mu^(2(n-1)) z_a^2, so b_n/mu^(2n) grows "
          f"LINEARLY in n.\n")
    print(f"     {'n':>4s} {'b_n':>16s} {'b_n/mu^(2n)':>14s} {'first difference':>18s}")
    ns = np.arange(1, 9)
    norm = np.array([node.iterate_exact(z0, int(n))[1] / MU ** (2 * n) for n in ns])
    z = z0.copy()
    iter_err = 0.0
    for i, n in enumerate(ns):
        z = node.step(z)
        iter_err = max(iter_err, float(abs(z[1] - node.iterate_exact(z0, int(n))[1])))
        diff = "" if i == 0 else f"{norm[i] - norm[i - 1]:18.10f}"
        print(f"     {n:4d} {z[1]:16.10f} {norm[i]:14.6f} {diff:>18s}")

    slope = float(np.polyfit(ns.astype(float), norm, 1)[0])
    slope_exact = C * z0[0] ** 2 / MU**2
    flat = S.ResonantNodeBlock(mu=MU, c=0.0)
    norm0 = np.array([flat.iterate_exact(z0, int(n))[1] / MU ** (2 * n) for n in ns])
    print(f"\n   slope {slope:.10f} vs exact c z_a^2/mu^2 = {slope_exact:.10f} "
          f"(err {abs(slope - slope_exact):.1e})")
    print(f"   closed form matches iteration to {iter_err:.1e}")
    print(f"   control c = 0: the same quantity is constant (spread {np.ptp(norm0):.1e})")

    # ---------------------------------------------------------------- part 3
    banner("PART 3 -- cross-module non-resonance survives: Tier 2 is satisfiable")
    print(f"   module 1 (resonant node) spectrum {np.round(w['spectra'][0], 5)}")
    print(f"   the relation 2 log mu = log mu + log mu is WITHIN module 1 and must")
    print(f"   not be flagged -- keeping it is the definition of Tier 2.\n")
    print(f"     {'nu':>10s} {'log nu':>10s} {'cross-nonresonant?':>20s} {'note':>28s}")
    partners = []
    for nu, note in ((0.50, "default"), (0.55, ""), (0.60, ""),
                     (MU**2, "= mu^2, collides"), (MU**3, "= mu^3, true resonance")):
        spectra = [w["spectra"][0], np.array([np.log(nu)])]
        ok = SP.is_cross_module_nonresonant(spectra, max_order=4)
        print(f"     {nu:10.4f} {np.log(nu):10.5f} {str(ok):>20s} {note:>28s}")
        partners.append({"nu": float(nu), "cross_nonresonant": bool(ok), "note": note})
    default_ok = partners[0]["cross_nonresonant"]
    rejects_resonant = not partners[3]["cross_nonresonant"] and not partners[4]["cross_nonresonant"]

    # ---------------------------------------------------------------- part 4
    banner("PART 4 -- the linearised (B2) test is a FALSE NEGATIVE here")
    L = w["linear_part"]
    n_summands = LIN.n_indecomposable_summands(L)
    linear_says_decomposable = not LIN.is_indecomposable(L)
    print(f"   linear part diag({MU}, {MU**2:.4g}): eigenvalues "
          f"{np.round(np.linalg.eigvals(L), 6)}")
    print(f"     n_indecomposable_summands = {n_summands}  -> reads DECOMPOSABLE")
    print(f"""
   But a complementary invariant factor would be a curve z_b = phi(z_a) with
   phi(0) = phi'(0) = 0, i.e. z_b/z_a^2 constant along orbits.  The closed form
   gives b_n/a_n^2 = z_b/z_a^2 + n c/mu^2, which is UNBOUNDED:\n""")
    ratio = np.array([
        (lambda v: v[1] / v[0] ** 2)(node.iterate_exact(z0, int(n))) for n in range(1, 41)
    ])
    steps = np.diff(ratio)
    print(f"     {'n':>4s} {'z_b/z_a^2':>14s}")
    for n in (1, 5, 10, 20, 40):
        print(f"     {n:4d} {ratio[n - 1]:14.6f}")
    print(f"\n   drift per step is exactly c/mu^2 = {C / MU**2:.10f} "
          f"(max dev {np.max(np.abs(steps - C / MU**2)):.1e})")
    unbounded = bool(ratio[-1] > 50.0)
    drift_exact = bool(np.allclose(steps, C / MU**2, rtol=1e-9))

    ratio0 = np.array([
        (lambda v: v[1] / v[0] ** 2)(flat.iterate_exact(z0, int(n))) for n in range(1, 41)
    ])
    print(f"   control c = 0: the ratio is pinned (spread {np.ptp(ratio0):.1e}), and "
          f"there the map really does split.")
    print("""
   So the map is dynamically INDECOMPOSABLE while its linearisation splits.
   selection.certify_fitted_model linearises, so on a fitted model of this shape
   it would report the module as decomposable and invite an over-split partition
   -- and it would do so precisely in the regime carrying Route A's nonlinear
   content.  This is a gap in the learning machinery, not in the theory.""")

    # ---------------------------------------------------------------- part 5
    banner("PART 5 -- the nonlinear (B2) check closes the false negative")
    print("""   block_nonlinear_certificate takes the quadratic jet in the eigenbasis and
   asks whether a RESONANT monomial couples the two eigendirections.  A
   non-resonant cross term is removable and must NOT flag; a resonant one cannot
   be removed and must.  Three cases:\n""")
    cases = [
        ("resonant node (c=0.9)", S.ResonantNodeBlock(mu=MU, c=C), True),
        ("linear control (c=0)", S.ResonantNodeBlock(mu=MU, c=0.0), False),
    ]

    # a non-resonant cross term: same shape but nu != mu^2, so z_a^2 is removable
    class NonResonantCross:
        dim = 2

        def step(self, z):
            z = np.asarray(z, float)
            return np.stack([MU * z[..., 0], 0.60 * z[..., 1] + C * z[..., 0] ** 2], axis=-1)

        def linear_part(self):
            return np.diag([MU, 0.60])

    cases.append(("non-resonant cross (nu=0.6)", NonResonantCross(), False))

    print(f"     {'block':>28s} {'linear reads':>14s} {'nonlinear verdict':>18s} "
          f"{'coupling':>10s} {'correct?':>9s}")
    part5 = []
    for name, blk, should_be_indec in cases:
        L_blk = blk.linear_part()
        lin_indec = LIN.is_indecomposable(L_blk)
        chk = SEL.block_nonlinear_certificate(blk.step, np.zeros(2), L_blk,
                                              coeff_tol=1e-2, res_tol=1e-2)
        verdict_indec = lin_indec or chk.indecomposable
        ok = verdict_indec == should_be_indec
        print(f"     {name:>28s} {('indec' if lin_indec else 'DECOMP'):>14s} "
              f"{('indecomposable' if verdict_indec else 'decomposable'):>18s} "
              f"{chk.max_coupling:10.4f} {str(ok):>9s}")
        part5.append({"case": name, "linear_indecomposable": lin_indec,
                      "nonlinear_indecomposable": verdict_indec,
                      "max_coupling": chk.max_coupling, "correct": ok})
    part5_ok = all(r["correct"] for r in part5)
    print(f"""
   The resonant node is now correctly called indecomposable, the linear control
   decomposable, and -- the subtle one -- the non-resonant cross term
   decomposable despite carrying a literal z_a^2 term, because that term is
   removable.  Only the resonant coupling obstructs.""")

    # ---------------------------------------------------------------- part 6
    banner("PART 6 -- the criterion is graph CONNECTEDNESS, not 'any coupling'")
    print("""   route_a_assessment §4.1: a module is indecomposable iff its resonance-
   coupling graph is connected.  With >= 3 sub-blocks that differs from 'a
   coupling exists': a coupling on {0,1} leaves an uncoupled {2} splitting off.
   And that is the OVER-report direction the fit cannot catch -- a decomposable
   module fits its split exactly.  Three 3-block modules:\n""")

    def node3(nu, extra):
        """(mu z0, mu^2 z1 + c z0^2, nu z2 + extra(z0,z1,z2))."""
        def step(z):
            z = np.asarray(z, float)
            z0, z1, z2 = z[..., 0], z[..., 1], z[..., 2]
            return np.stack([MU * z0, MU**2 * z1 + C * z0**2, nu * z2 + extra(z0, z1, z2)], axis=-1)
        return step, np.diag([MU, MU**2, nu])

    three = [
        ("nu=0.5 non-resonant  -> {0,1}(+){2}", *node3(0.50, lambda a, b, c_: 0.0 * a), False, 2),
        ("nu=mu^3, z0*z1 edge  -> connected", *node3(MU**3, lambda a, b, c_: 0.7 * a * b), True, 1),
        ("nu=mu^4, z1^2 edge   -> connected", *node3(MU**4, lambda a, b, c_: 0.6 * b * b), True, 1),
    ]
    print(f"     {'module':>36s} {'n_components':>13s} {'verdict':>15s} {'expected':>15s} {'ok?':>5s}")
    part6 = []
    for name, step, L3, should_indec, exp_comp in three:
        chk = SEL.block_nonlinear_certificate(step, np.zeros(3), L3, coeff_tol=1e-2, res_tol=1e-2)
        ok = (chk.indecomposable == should_indec) and (chk.n_components == exp_comp)
        print(f"     {name:>36s} {chk.n_components:>13d} "
              f"{('indecomposable' if chk.indecomposable else 'decomposable'):>15s} "
              f"{('indecomposable' if should_indec else 'decomposable'):>15s} {str(ok):>5s}")
        part6.append({"module": name, "n_components": chk.n_components,
                      "indecomposable": chk.indecomposable, "expected_indecomposable": should_indec,
                      "correct": ok})

    # ground the 'decomposable' verdict: exhibit the actual invariant product
    dstep, _ = node3(0.50, lambda a, b, c_: 0.0 * a)
    rng = np.random.default_rng(SEED)
    split_exact = True
    for _ in range(500):
        z = rng.normal(size=3)
        if abs(dstep(z)[2] - 0.50 * z[2]) > 1e-12:
            split_exact = False
        if np.max(np.abs(dstep(z)[:2] - dstep(np.array([z[0], z[1], 0.0]))[:2])) > 1e-12:
            split_exact = False
    part6_ok = all(r["correct"] for r in part6) and split_exact
    print(f"""
   The first module DECOMPOSES: z2 -> 0.5 z2 is autonomous and {{0,1}} ignores z2
   (verified exactly over 500 points: {split_exact}).  'Any coupling' would have
   called it indecomposable; connectedness calls it decomposable, correctly.""")

    banner("VERDICTS")
    checks = [
        (
            len(res) == 1 and (res[0].component, res[0].exponent) == (1, (2, 0))
            and abs(res[0].value) < 1e-15,
            f"the homological operator at degree {DEGREE} has exactly one zero eigenvalue, "
            f"at z_a^2 e_b, because lam_b - lam_a^2 = mu^2 - mu^2 vanishes identically",
        ),
        (
            (not rep.linearizable) and rep0.linearizable,
            f"with c = {C} the quadratic term is obstructed (the surviving coefficient IS "
            f"the normal-form invariant); with c = 0 it is removable -- so the obstruction "
            f"tracks c and is not an artefact",
        ),
        (
            abs(slope - slope_exact) < 1e-9 and iter_err < 1e-12 and float(np.ptp(norm0)) < 1e-12,
            f"the resonance is dynamically visible: b_n/mu^(2n) drifts linearly at exactly "
            f"c z_a^2/mu^2 = {slope_exact:.6f}, and is constant when c = 0",
        ),
        (
            default_ok and rejects_resonant,
            f"cross-module non-resonance holds for the witness (mu = {MU}, nu = {NU}) while a "
            f"within-module resonance is live -- so **Tier 2 is non-empty** -- and partners at "
            f"mu^2 and mu^3 are correctly rejected",
        ),
        (
            linear_says_decomposable and n_summands == 2 and unbounded and drift_exact,
            "the linearised (B2) test reports this module DECOMPOSABLE (2 summands) while the "
            "map admits no invariant curve tangent to e_a -- a false negative in "
            "certify_fitted_model, sitting exactly in Tier 2",
        ),
        (
            part5_ok,
            "the nonlinear (B2) check closes it: the resonant node reads indecomposable, the "
            "linear control and the NON-resonant cross term both read decomposable -- only the "
            "resonant coupling obstructs, and coordinate-invariantly",
        ),
        (
            part6_ok,
            "the criterion is graph CONNECTEDNESS: a 3-block module coupled only on {0,1} is "
            "correctly read as decomposable (its {0,1}(+){2} split is an exact invariant "
            "product), while genuinely connected 3-block modules read indecomposable -- "
            "'any coupling' would over-report the first",
        ),
    ]
    tags = [verdict(ok, m) for ok, m in checks]
    passed = all(t == "PASS" for t in tags)

    save(
        "exp09_tier2_nonempty",
        {
            "seed": SEED, "mu": MU, "c": C, "nu": NU, "degree": DEGREE,
            "homological_spectrum": spectrum,
            "n_resonances": len(res),
            "obstruction": {"c": C, "linearizable": rep.linearizable,
                            "linearizable_with_c_zero": rep0.linearizable,
                            "min_abs_nonresonant_eigenvalue": rep.min_abs_eigenvalue},
            "secular": {"slope": slope, "slope_exact": slope_exact,
                        "closed_form_error": iter_err,
                        "control_spread": float(np.ptp(norm0))},
            "partners": partners,
            "blind_spot": {"linear_summands": n_summands,
                           "linear_reads_decomposable": linear_says_decomposable,
                           "ratio_drift_per_step": float(C / MU**2),
                           "ratio_final": float(ratio[-1]),
                           "drift_is_exact": drift_exact,
                           "control_ratio_spread": float(np.ptp(ratio0))},
            "nonlinear_check": part5,
            "coupling_graph": part6,
            "all_passed": passed,
            "checks": [{"passed": ok, "claim": m} for ok, m in checks],
        },
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
