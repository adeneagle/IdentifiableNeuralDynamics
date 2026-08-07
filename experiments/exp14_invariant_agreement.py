"""Experiment 14 -- CLAUDE.md task 40: does invariant agreement work as a test?

Every metric in this repo before now compares a fit to the GROUND TRUTH, which
is exactly what real recordings do not have.  Task 40's proposal is to compare
two fits to EACH OTHER: fit independently on disjoint subsets of the population,
and if the dynamics are identifiable the two models recover different
coordinates and the **same invariants** -- filtration order, per-module Lyapunov
spectra, rotation numbers.  Agreement confirms; disagreement falsifies.  No
ground truth anywhere.

That is the plan.  This experiment asks the prior question: **does the
measurement work at all?**  Taking an unvalidated structural readout to data
with no ground truth is the mistake 3.9, 3.10 and 3.11 each record, so the
metric is validated here, on synthetic systems where the answer is known,
before it is ever pointed at a recording.

### The two new objects

  * ``spectra.rotation_number`` -- the invariant the Lyapunov spectrum provably
    cannot see.  ``LimitCycleBlock(a=0.3)`` has spectrum ``{0, log|1-2a|}`` for
    *every* omega, so two oscillatory modules are never separated spectrally
    (task 23).  Until now omega existed in this repo only as a *generating*
    parameter; nothing ever measured it back off a model.
  * ``metrics.invariant_agreement`` -- fit-to-fit comparison, module labels
    treated as meaningless, filtration order reported separately from the
    invariants themselves because 3.7 says the ordering is the part the theory
    actually delivers.

### The controls, and what each one is for

Part 2 runs the metric on exact systems where the answer is known by
construction:

  2a  per-module nonlinear change of coordinates      must AGREE
  2b  module labels permuted                          must AGREE
  2c  the 3.7 TRIANGULAR conjugacy                    must AGREE
  2d  the 3.1 REGROUPING counterexample               must DISAGREE
  2e  identical spectra, different rotation numbers   must DISAGREE

2c and 2d are the pair that matters.  Both are reparameterisations that a
fit cannot distinguish from the truth by fit quality -- 3.1's three groupings
tie at 2.2e-16 -- but they are not the same kind of ambiguity.  2c changes the
coordinates while leaving the factors alone, which 3.7 proves is irreducible;
the fingerprint must be blind to it, or it is claiming more than is true.  2d
changes the factors themselves; the fingerprint must catch it, or it is
claiming nothing.  A metric that fails either direction is useless, and it is
easy to write one that passes only one.

2e is task 23 in miniature: a spectrum-only comparison calls it a match.

Part 3 turned out to be where the compute earned itself.  It asks whether the
invariants of a LEARNED map are measurable at all, and finds two limits, both
of the 3.9 kind -- a confident number measuring the wrong thing.  See CLAUDE.md
3.13; the short version is that a fitted map iterated past its data converges to
a spurious attractor, and that recoverability is per-INVARIANT rather than
per-model.

Part 4 is then task 40 proper, run the way part 3 says it has to be: two fits on
disjoint neuron subsets, read inside the data horizon, on a system of two limit
cycles.  That system is chosen so nothing decays (a cycle has a neutral exponent)
and so the two modules have IDENTICAL spectra -- which makes the rotation number
the only thing that can tell the fits apart, i.e. exactly task 23's case.

### Reading the checks

  1a-1c  the rotation number is exact, gauge-invariant, and sees what the
         spectrum cannot.  A FAIL here means the instrument is broken.
  2      the five exact-system controls.  2c and 2d are the pair that matters:
         blind to an IRREDUCIBLE coordinate ambiguity, not blind to a different
         DECOMPOSITION.  Passing only one direction is easy and worthless.
  6-7    the two part-3 limits, asserted rather than assumed.
  8-9    do disjoint neuron splits agree on the rotation numbers, under a linear
         and then a nonlinear observation map?  9 is the setting task 40 proposes
         for real data.
  10     the test can FAIL: a frequency change is detected.  Without this arm,
         8 and 9 are unattributable (3.11).
  11     rotation agrees where the transverse spectrum does not -- the
         per-invariant asymmetry, stated so it cannot be read past.

Checks are per-INVARIANT, not on the combined `agree` flag, because the two
halves of the fingerprint behave completely differently here and one boolean
would hide it.  A FAIL on 9 or 11 is informative, not bad: it bounds what this
test can deliver, which is what it exists to find out -- and better found here
than on a recording.

### History worth keeping

The first run of part 4 failed checks 8/9/11 with a rotation error of 0.1274 in
5 of 16 comparisons -- which is exactly |rho_1 - rho_2|, the signature of a
module SWAP rather than of any recovery failure.  Cause: `invariant_agreement`
paired modules across fits by spectral distance alone, and two limit cycles have
identical spectra, so the cost matrix was flat and the pairing was decided by
nothing.  The same lesson as 2e, one level up.  Fixed by matching on the full
invariant vector and by quantising the order key; both are asserted in tests.
"""

from __future__ import annotations

import numpy as np

from _common import banner, save, verdict
from idyn import metrics as M
from idyn import spectra as SP
from idyn import systems as S
from idyn import train as T
from idyn.models import ModelConfig

SEED = 0
PART = [2, 2]

# Part 1-2: exact systems, no fitting.
ROT_T = 1500
ROT_WARMUP = 200
LYAP_T = 1200
LYAP_WARMUP = 200
N_Z0 = 6

# Part 3-4: fitting.  N_OBS splits into two disjoint halves that must EACH still
# determine the latents, and how many neurons that takes is not a free choice --
# it was measured (scratchpad sweeps, numbers in CLAUDE.md 3.13(e)):
#
#   neurons/side   fits recovering the invariants   coherence
#         8              1/4                          0.599
#        16              0/4                          0.683
#        32              3/4                          0.841
#        64              2/4                          0.815
#
# 8/side is comfortable for the LINEAR decoder and nowhere near enough for a
# nonlinear one; recovery turns on at ~32 and then plateaus.  Training budget is
# not the lever -- 3000 -> 20000 steps improves fit_quality 4.2x and leaves
# rotation recovery unchanged.  64 total (32/side) is the smallest setting where
# part 4 measures the method rather than the sample.
N_OBS = 64
N_TRAJ = 400
T_STEPS = 30          # >= 2 periods of the slower cycle (omega = 0.5 -> 12.6 steps)
STEPS = 3000
N_RESTARTS = 4
# The asymptotic reading, kept only so part 3 can show it is CONTAMINATED.
FIT_LYAP_T = 300
FIT_LYAP_WARMUP = 100
# The reading actually used: inside the data horizon, many initial conditions.
# Short windows are noisy per orbit, so the sample size has to come from the
# ensemble rather than from the horizon.
FIT_N_Z0 = 60

# Tolerances.  spec_tol is loose relative to the exact-system errors (0 to 1e-16)
# and tight relative to the 3.1 regrouping separation (0.223); rot_tol is well
# under the 2e separation (0.064).  Part 3 needs its own, since a fitted
# spectrum carries optimiser error -- stated at the call site, not hidden here.
SPEC_TOL = 0.05
ROT_TOL = 0.01
FIT_SPEC_TOL = 0.15
FIT_ROT_TOL = 0.05


# ---------------------------------------------------------------------------
# gauge changes: the freedom section 7 grants, made explicit
# ---------------------------------------------------------------------------


class ShearConj:
    """h(x, y) = (x + c y^3, y) applied to one module.

    Nonlinear, invertible in closed form, analytic, and orientation-preserving.
    This is a *within-module* coordinate change -- exactly what section 7 says is
    never identified -- so every invariant must be blind to it.
    """

    def __init__(self, blk, c: float = 0.7):
        self.blk, self.c = blk, float(c)
        self.dim = blk.dim

    def _h(self, z):
        z = np.asarray(z, dtype=float)
        return np.stack([z[..., 0] + self.c * z[..., 1] ** 3, z[..., 1]], axis=-1)

    def _hinv(self, w):
        w = np.asarray(w, dtype=float)
        return np.stack([w[..., 0] - self.c * w[..., 1] ** 3, w[..., 1]], axis=-1)

    def step(self, w):
        return self._h(self.blk.step(self._hinv(w)))

    def jacobian(self, w):
        w = np.asarray(w, dtype=float).reshape(self.dim)
        z = self._hinv(w)
        Dh = np.array([[1.0, 3.0 * self.c * float(self.blk.step(z)[1]) ** 2], [0.0, 1.0]])
        Dhinv = np.array([[1.0, -3.0 * self.c * float(w[1]) ** 2], [0.0, 1.0]])
        return Dh @ self.blk.jacobian(z) @ Dhinv


def gauged(system, c: float = 0.7):
    return S.ModularSystem([ShearConj(b, c) for b in system.blocks])


def relabelled(system, perm):
    return S.ModularSystem([system.blocks[i] for i in perm])


def circle_z0s(rng, d, n=N_Z0, lo=0.6, hi=1.1):
    """Initial conditions on an annulus in each 2-D module (inside every basin)."""
    out = []
    for _ in range(d // 2):
        th = rng.uniform(-np.pi, np.pi, n)
        r = rng.uniform(lo, hi, n)
        out.append(np.stack([r * np.cos(th), r * np.sin(th)], axis=-1))
    return np.concatenate(out, axis=-1)


def fingerprint(system, z0s, T=LYAP_T, warmup=LYAP_WARMUP, T_rot=ROT_T):
    return M.dynamical_fingerprint(system, z0s, T=T, warmup=warmup, T_rotation=T_rot)


# ---------------------------------------------------------------------------
# part 3 plumbing: a fitted ModularTransition as a numpy modular system
# ---------------------------------------------------------------------------


class LearnedBlock:
    """One block of a fitted ModularTransition as a `spectra.HasJacobian`.

    Same construction as exp13: the block MLP sees only its own coordinates, so
    it can be iterated alone -- which is what makes a per-module spectrum and a
    per-module rotation number well posed on a fitted model at all.
    """

    def __init__(self, dyn, k: int):
        import torch  # local, so parts 1-2 run without touching torch

        self._torch = torch
        a, b = dyn.bounds[k]
        self.net = dyn.nets[k]
        self.dim = b - a

    def _f(self, Z):
        torch = self._torch
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(Z, float), dtype=torch.float64)
            return (t + self.net(t)).numpy()

    def step(self, z):
        z = np.asarray(z, float)
        return self._f(np.atleast_2d(z)).reshape(z.shape)

    def jacobian(self, z, eps: float = 1e-6):
        z = np.asarray(z, float).reshape(self.dim)
        E = np.eye(self.dim) * eps
        out = self._f(np.vstack([z + E, z - E]))
        return ((out[: self.dim] - out[self.dim :]) / (2.0 * eps)).T


class LearnedSystem:
    """`partition` + `blocks`, the only interface the fingerprint needs."""

    def __init__(self, dyn, partition):
        self.partition = list(partition)
        self.blocks = [LearnedBlock(dyn, k) for k in range(len(self.partition))]


def fit_fingerprint(X, seed, decoder_kind):
    """Fit one model to X and fingerprint its learned transition.

    Read **inside the data horizon** -- part 3 measures what happens otherwise.
    A fitted map iterated past the trajectories it was trained on is
    extrapolating, and here it extrapolates to a spurious attracting fixed
    point, so its asymptotic invariants describe the model's imagination.
    """
    cfg_m = ModelConfig(n_obs=X.shape[-1], d=sum(PART), partition=PART,
                        decoder=decoder_kind, encoder=decoder_kind)
    cfg_t = T.TrainConfig(steps=STEPS, seed=seed)
    res = T.fit(X, cfg_m, cfg_t)
    # float64 for every readout: the Jacobian is a central difference at eps=1e-6,
    # so a float32 model puts the roundoff floor on top of the signal.  Same cast
    # exp13 makes, and for the same reason.
    dyn = res.model.double().dyn
    z0 = np.asarray(res.z_fit, float)[:, 0, :]
    warm = T_STEPS // 4
    read = T_STEPS - warm
    fp = M.dynamical_fingerprint(
        LearnedSystem(dyn, PART), z0[:FIT_N_Z0], T=read, warmup=warm, T_rotation=read
    )
    return fp, float(res.fit_quality)


def main() -> int:
    rng = np.random.default_rng(SEED)
    rec: dict = {"seed": SEED, "params": {
        "partition": PART, "rot_T": ROT_T, "lyap_T": LYAP_T, "n_obs": N_OBS,
        "n_traj": N_TRAJ, "T_steps": T_STEPS, "steps": STEPS,
        "n_restarts": N_RESTARTS, "spec_tol": SPEC_TOL, "rot_tol": ROT_TOL,
        "fit_spec_tol": FIT_SPEC_TOL, "fit_rot_tol": FIT_ROT_TOL,
    }}
    checks: list[tuple[str, bool]] = []

    # -----------------------------------------------------------------
    banner("PART 1 -- the rotation number is exact, and the spectrum is blind to it")

    known = []
    for name, blk, want in [
        ("TwistBlock  s=0.90 w=0.40", S.TwistBlock(s=0.90, omega=0.40, beta=0.6), 0.40),
        ("TwistBlock  s=0.50 w=1.10", S.TwistBlock(s=0.50, omega=1.10, beta=-0.5), 1.10),
        ("LimitCycle  a=0.30 w=0.50", S.LimitCycleBlock(a=0.3, omega=0.5), 0.50),
        ("LimitCycle  a=0.30 w=1.30", S.LimitCycleBlock(a=0.3, omega=1.3), 1.30),
        ("LimitCycle  beta=0.4 w=0.50", S.LimitCycleBlock(a=0.3, omega=0.5, beta=0.4), 0.50),
        ("LinearBlock diag(0.9,0.5)", S.LinearBlock(np.diag([0.9, 0.5])), 0.0),
        ("LinearBlock diag(-0.8,0.5)", S.LinearBlock(np.diag([-0.8, 0.5])), np.pi),
    ]:
        z0 = circle_z0s(np.random.default_rng(SEED), 2)
        r = SP.rotation_number_averaged(blk, z0, T=ROT_T, warmup=ROT_WARMUP)
        target = want / (2.0 * np.pi)
        err = abs(abs(r.rho) - abs(target))
        known.append({"block": name, "rho": r.rho, "want": target, "err": err,
                      "coherence": r.coherence, "n_used": r.n_used})
        print(f"  {name:28s} rho={r.rho:+.6f}  want={target:+.6f}  err={err:.2e}  "
              f"coh={r.coherence:.4f}  n={r.n_used}")

    ok1a = max(k["err"] for k in known) < 1e-9
    checks.append(("1a known-answer rotation numbers exact", ok1a))
    verdict(ok1a, f"max error over 7 blocks = {max(k['err'] for k in known):.2e} (< 1e-9)")

    print("\n  Conjugacy invariance -- same system, nonlinear change of coordinates:")
    conj = []
    for name, blk in [("TwistBlock", S.TwistBlock(s=0.90, omega=0.40, beta=0.6)),
                      ("LimitCycle", S.LimitCycleBlock(a=0.3, omega=0.5))]:
        z0 = circle_z0s(np.random.default_rng(SEED), 2)
        a = SP.rotation_number_averaged(blk, z0, T=ROT_T, warmup=300)
        b = SP.rotation_number_averaged(ShearConj(blk), z0, T=ROT_T, warmup=300)
        conj.append({"block": name, "raw": a.rho, "conjugated": b.rho,
                     "diff": abs(a.rho - b.rho), "coherence_conj": b.coherence})
        print(f"    {name:12s} raw {a.rho:+.8f}   conjugated {b.rho:+.8f}   "
              f"|diff| {abs(a.rho - b.rho):.2e}  (coh {b.coherence:.4f})")
    ok1b = max(c["diff"] for c in conj) < 1e-3
    checks.append(("1b rotation number survives a nonlinear gauge change", ok1b))
    verdict(ok1b, f"max drift {max(c['diff'] for c in conj):.2e} (< 1e-3; a distorted "
                  "cycle averages over a partial period, so this is O(1/T), not exact)")

    print("\n  Task 23 -- two limit cycles, identical spectra, different rotation:")
    t23 = []
    for w in (0.5, 0.9, 1.3):
        blk = S.LimitCycleBlock(a=0.3, omega=w)
        z0 = circle_z0s(np.random.default_rng(SEED), 2, lo=0.9, hi=1.1)
        sp = SP.lyapunov_spectrum_averaged(blk, z0, T=LYAP_T, warmup=LYAP_WARMUP)
        r = SP.rotation_number_averaged(blk, z0, T=ROT_T, warmup=ROT_WARMUP)
        t23.append({"omega": w, "spectrum": sp.tolist(), "rho": r.rho})
        print(f"    omega={w}: spectrum={np.round(sp, 6)}  rho={r.rho:+.6f}")
    spec_spread = float(np.abs(np.diff(np.array([t["spectrum"] for t in t23]), axis=0)).max())
    rho_spread = float(np.abs(np.diff([t["rho"] for t in t23])).min())
    ok1c = spec_spread < 1e-9 and rho_spread > 0.01
    checks.append(("1c spectra identical, rotation numbers separated", ok1c))
    verdict(ok1c, f"spectrum spread {spec_spread:.2e} (blind), "
                  f"min rotation separation {rho_spread:.4f} (sees it)")
    rec["part1"] = {"known": known, "conjugacy": conj, "task23": t23,
                    "spectrum_spread": spec_spread, "rho_separation": rho_spread}

    # -----------------------------------------------------------------
    banner("PART 2 -- invariant_agreement on exact systems, where the answer is known")

    sys_a = S.ModularSystem([S.TwistBlock(s=0.90, omega=0.40, beta=0.6),
                             S.TwistBlock(s=0.55, omega=1.10, beta=0.3)])
    Z4 = circle_z0s(rng, 4)
    fp_a = fingerprint(sys_a, Z4)
    print(f"  reference: {fp_a.summary()}")

    part2: dict = {}

    def control(tag, other_fp, want_agree, note):
        r = M.invariant_agreement(other_fp, fp_a, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
        good = (r.agree == want_agree)
        print(f"\n  [{tag}] {note}")
        print(f"        {r.summary()}")
        for n in r.notes:
            print(f"        note: {n}")
        part2[tag] = {"agree": r.agree, "want_agree": want_agree,
                      "spectrum_error": r.spectrum_error, "rotation_error": r.rotation_error,
                      "order_agrees": r.order_agrees, "order_margin": r.order_margin,
                      "matching": r.matching, "notes": r.notes}
        return good

    g2a = control("2a", fingerprint(gauged(sys_a), Z4), True,
                  "per-module nonlinear change of coordinates -> must AGREE")
    g2b = control("2b", fingerprint(relabelled(sys_a, [1, 0]), Z4[:, [2, 3, 0, 1]]), True,
                  "module labels permuted -> must AGREE")

    # 2c: the section 3.7 triangular conjugacy.  h(z1,z2) = (z1 + c sgn(z2)|z2|^p, z2)
    # with p = log(mu1)/log(mu2) satisfies h o F = F o h exactly while being
    # provably NOT block-diagonal.  So it is a reparameterisation that mixes the
    # modules and that no amount of extra hypothesis can rule out -- the
    # irreducible ambiguity of 3.7.  The invariants must survive it, or the
    # fingerprint is claiming more than the theory delivers.
    tri = S.triangular_conjugacy_counterexample()
    sys_t, h, h_inv = tri["system"], tri["h"], tri["h_inv"]
    probe = np.stack([rng.uniform(0.3, 1.0, 512), rng.uniform(0.3, 1.0, 512)], axis=-1)

    m12 = float(np.abs(tri["cross_derivative"](probe[:, 1])).max())      # h not block-diagonal
    conj_res = float(np.abs(h(sys_t.step(probe)) - sys_t.step(h(probe))).max())  # but a conjugacy

    class ConjugatedBlock:
        """Coordinate k of G = h o F o h^-1, iterated alone.

        Legitimate only because ``off_block`` below is verified to be zero: G is
        modular in the h-coordinates even though h itself is not block-diagonal.
        """

        def __init__(self, k: int):
            self.k, self.dim = k, 1

        def _full(self, w1: float, w2: float) -> np.ndarray:
            w = np.array([[w1, w2]], dtype=float)
            return h(sys_t.step(h_inv(w)))[0]

        def step(self, z):
            z = np.asarray(z, dtype=float)
            flat = z.reshape(-1)
            args = [(v, 0.0) if self.k == 0 else (0.0, v) for v in flat]
            return np.array([self._full(a, b)[self.k] for a, b in args]).reshape(z.shape)

        def jacobian(self, z, eps: float = 1e-6):
            v = float(np.asarray(z, dtype=float).reshape(1)[0])
            return np.array([[(float(self.step(np.array([v + eps]))[0])
                               - float(self.step(np.array([v - eps]))[0])) / (2 * eps)]])

    # does G depend on the other coordinate?  If h were not a conjugacy, it would.
    off = []
    for w in probe[:64]:
        e = 1e-5
        d0 = (ConjugatedBlock(0)._full(w[0], w[1] + e)[0]
              - ConjugatedBlock(0)._full(w[0], w[1] - e)[0]) / (2 * e)
        d1 = (ConjugatedBlock(1)._full(w[0] + e, w[1])[1]
              - ConjugatedBlock(1)._full(w[0] - e, w[1])[1]) / (2 * e)
        off.append(max(abs(d0), abs(d1)))
    off_block = float(max(off))

    sys_g = S.ModularSystem([ConjugatedBlock(0), ConjugatedBlock(1)])
    Zt = rng.uniform(0.4, 1.0, (N_Z0, 2))
    fp_f, fp_g = fingerprint(sys_t, Zt), fingerprint(sys_g, Zt)
    r_t = M.invariant_agreement(fp_f, fp_g, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
    print(f"\n  [2c] the 3.7 triangular conjugacy -- the ambiguity that cannot be removed")
    print(f"        h is NOT block-diagonal:   max |M12| = {m12:.4f}  (nonzero by construction)")
    print(f"        h IS an exact conjugacy:   max |h(Fz) - F(hz)| = {conj_res:.2e}")
    print(f"        G = h F h^-1 is modular:   max off-block |DG| = {off_block:.2e}")
    print(f"        F: {fp_f.summary()}")
    print(f"        G: {fp_g.summary()}")
    print(f"        {r_t.summary()}")
    part2["2c"] = {"agree": r_t.agree, "want_agree": True, "max_M12": m12,
                   "conjugacy_residual": conj_res, "off_block_jacobian": off_block,
                   "spectrum_error": r_t.spectrum_error, "rotation_error": r_t.rotation_error}
    g2c = r_t.agree and m12 > 0.1 and conj_res < 1e-12

    rg = S.regrouping_counterexample()
    Zr = rng.uniform(0.5, 1.0, (N_Z0, 4))
    fp_r1, fp_r2 = fingerprint(rg["system"], Zr), fingerprint(rg["system_tilde"], Zr)
    r_r = M.invariant_agreement(fp_r1, fp_r2, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
    print(f"\n  [2d] the 3.1 regrouping: same observations, DIFFERENT decomposition")
    print(f"        A: {fp_r1.summary()}")
    print(f"        B: {fp_r2.summary()}")
    print(f"        {r_r.summary()}")
    part2["2d"] = {"agree": r_r.agree, "want_agree": False,
                   "spectrum_error": r_r.spectrum_error, "rotation_error": r_r.rotation_error}
    g2d = not r_r.agree

    sysL1 = S.ModularSystem([S.LimitCycleBlock(a=0.3, omega=0.5),
                             S.LimitCycleBlock(a=0.3, omega=1.3)])
    sysL2 = S.ModularSystem([S.LimitCycleBlock(a=0.3, omega=0.5),
                             S.LimitCycleBlock(a=0.3, omega=0.9)])
    Zl = circle_z0s(rng, 4, lo=0.9, hi=1.1)
    fp_l1, fp_l2 = fingerprint(sysL1, Zl), fingerprint(sysL2, Zl)
    r_l = M.invariant_agreement(fp_l1, fp_l2, spec_tol=SPEC_TOL, rot_tol=ROT_TOL)
    print(f"\n  [2e] task 23: identical spectra, different rotation numbers")
    print(f"        A: {fp_l1.summary()}")
    print(f"        B: {fp_l2.summary()}")
    print(f"        {r_l.summary()}")
    for n in r_l.notes:
        print(f"        note: {n}")
    print(f"        spectrum_error {r_l.spectrum_error:.3e} -- a spectrum-only test AGREES")
    print(f"        rotation_error {r_l.rotation_error:.6f} -- this is what catches it")
    part2["2e"] = {"agree": r_l.agree, "want_agree": False,
                   "spectrum_error": r_l.spectrum_error, "rotation_error": r_l.rotation_error,
                   "order_margin": r_l.order_margin, "notes": r_l.notes}
    g2e = (not r_l.agree) and r_l.spectrum_error < 1e-6

    ok2 = all([g2a, g2b, g2c, g2d, g2e])
    checks.append(("2 all five exact-system controls behave as specified", ok2))
    verdict(ok2, f"2a={g2a} 2b={g2b} 2c={g2c} 2d={g2d} 2e={g2e} "
                 "(2c blind to an irreducible coordinate ambiguity, 2d not blind to a "
                 "different decomposition -- both directions required)")
    rec["part2"] = part2

    # -----------------------------------------------------------------
    banner("PART 3 -- reading invariants off a LEARNED map: where they are measurable")

    print("  Two findings first, because they decide how part 4 has to be run.\n"
          "  Both were found here rather than assumed, and both are of the 3.9 kind:\n"
          "  a confident number that is measuring the wrong thing.\n")

    def population(system, strength, seed, n_obs=N_OBS, T_steps=T_STEPS):
        r = np.random.default_rng(seed)
        d = system.dim
        dec = (S.LinearDecoder.random(n_obs, d, r) if strength == 0.0
               else S.MLPDecoder.random(n_obs, d, r, strength=strength))
        X, Z, dec = T.make_dataset(system, n_obs, N_TRAJ, T_steps, r, decoder=dec)
        return X, Z, dec

    # 3.1 -- the learned map, iterated past the data, converges to a spurious attractor
    sys_c = S.ModularSystem([S.TwistBlock(s=0.90, omega=0.40, beta=0.6),
                             S.TwistBlock(s=0.55, omega=1.10, beta=0.3)])
    Xc, Zc, _ = population(sys_c, 0.0, SEED)
    res_c = T.fit(Xc[..., : N_OBS // 2],
                  ModelConfig(n_obs=N_OBS // 2, d=4, partition=PART,
                              decoder="linear", encoder="linear"),
                  T.TrainConfig(steps=STEPS, seed=SEED))
    ls_c = LearnedSystem(res_c.model.double().dyn, PART)
    z0c = np.asarray(res_c.z_fit, float)[:, 0, :]

    norms = []
    zz = z0c[0, :2].copy()
    for _ in range(400):
        norms.append(float(np.linalg.norm(zz)))
        zz = ls_c.blocks[0].step(zz)
    stall = float(norms[-1])
    print(f"  [3.1] learned block 0 orbit norm: t=0 {norms[0]:.3e}  "
          f"t={T_STEPS} {norms[T_STEPS]:.3e}  t=399 {stall:.3e}")
    print(f"        The true block contracts to 0; the learned one STALLS at "
          f"{stall:.3e}.\n"
          f"        Past the data horizon the fit is extrapolating, and it has "
          f"invented a\n        spurious attracting fixed point.  Every asymptotic "
          f"invariant read there\n        describes the extrapolation, not the data.")

    fp_true_asym = fingerprint(sys_c, Zc[:6, 0, :])
    fp_lrn_asym = M.dynamical_fingerprint(ls_c, z0c[:3], T=FIT_LYAP_T,
                                          warmup=FIT_LYAP_WARMUP, T_rotation=FIT_LYAP_T)
    warm = T_STEPS // 4
    fp_true_hor = M.dynamical_fingerprint(sys_c, Zc[:60, 0, :], T=T_STEPS - warm,
                                          warmup=warm, T_rotation=T_STEPS - warm)
    fp_lrn_hor = M.dynamical_fingerprint(ls_c, z0c[:60], T=T_STEPS - warm,
                                         warmup=warm, T_rotation=T_STEPS - warm)
    r_asym = M.invariant_agreement(fp_true_asym, fp_lrn_asym, spec_tol=FIT_SPEC_TOL,
                                   rot_tol=FIT_ROT_TOL)
    r_hor = M.invariant_agreement(fp_true_hor, fp_lrn_hor, spec_tol=FIT_SPEC_TOL,
                                  rot_tol=FIT_ROT_TOL)
    print(f"\n        true, asymptotic : {fp_true_asym.summary()}")
    print(f"        learned, asymptotic: {fp_lrn_asym.summary()}")
    print(f"          -> {r_asym.summary()}")
    print(f"        true, data horizon : {fp_true_hor.summary()}")
    print(f"        learned, horizon   : {fp_lrn_hor.summary()}")
    print(f"          -> {r_hor.summary()}")

    # Tested on the ROTATION, which is where the effect is unambiguous.  A spurious
    # attracting FIXED POINT has perfectly plausible exponents and no rotation at
    # all, so extrapolation destroys rho (it reads exactly 0) while leaving the
    # spectrum looking reasonable.  Requiring both to improve would test the
    # insensitive probe alongside the sensitive one and report a null when the
    # fit is good enough that only rho is damaged.  Spectrum reported alongside.
    ok6 = r_hor.rotation_error < r_asym.rotation_error
    checks.append(("6 reading at the data horizon beats reading asymptotically", ok6))
    verdict(ok6, f"rotation error {r_asym.rotation_error:.3f} -> {r_hor.rotation_error:.3f} "
                 f"(spectrum {r_asym.spectrum_error:.3f} -> {r_hor.spectrum_error:.3f}); "
                 "never iterate a fitted map past its data")

    # 3.2 -- a module that has decayed carries no information about its own invariants
    per_mod = []
    fa, fb = fp_true_hor.reordered(), fp_lrn_hor.reordered()
    for k in range(fa.K):
        retained = float(np.exp(2.0 * fa.spectra[k][0] * T_STEPS))  # variance at t = T
        per_mod.append({
            "module": k,
            "retained_variance_fraction": retained,
            "spectrum_error": float(np.abs(fa.spectra[k] - fb.spectra[k]).max()),
            "rotation_error": float(abs(abs(fa.rotations[k]) - abs(fb.rotations[k]))),
            "coherence": fb.coherences[k],
        })
        print(f"\n  [3.2] module {k}: retains {retained:.2e} of its variance by t={T_STEPS}"
              f"  ->  lambda error {per_mod[-1]['spectrum_error']:.4f}, "
              f"rho error {per_mod[-1]['rotation_error']:.4f}")
    dom, sub = per_mod[0], per_mod[-1]
    print(f"\n        The dominant module is recovered; the dominated one is not.  That is\n"
          f"        3.11's design tension as a HARD LIMIT on this test: a module whose\n"
          f"        signal is gone before the trial ends has invariants nothing can read.")
    ok7 = (dom["spectrum_error"] < FIT_SPEC_TOL and dom["rotation_error"] < FIT_ROT_TOL
           and sub["retained_variance_fraction"] < 1e-6)
    checks.append(("7 the module that retains signal is recovered; the decayed one is not", ok7))
    verdict(ok7, f"dominant: lambda err {dom['spectrum_error']:.4f}, rho err "
                 f"{dom['rotation_error']:.4f}; dominated retains "
                 f"{sub['retained_variance_fraction']:.1e} of its variance")

    rec["part3"] = {
        "spurious_attractor_norm": stall,
        "asymptotic": {"spectrum_error": r_asym.spectrum_error,
                       "rotation_error": r_asym.rotation_error},
        "horizon": {"spectrum_error": r_hor.spectrum_error,
                    "rotation_error": r_hor.rotation_error},
        "per_module": per_mod,
    }

    # -----------------------------------------------------------------
    banner("PART 4 -- task 40 proper: two fits on DISJOINT neuron subsets")

    print("  System: two LIMIT CYCLES with different frequencies.  Chosen deliberately.\n"
          "  A cycle carries a neutral exponent 0, so nothing decays and part 3's signal\n"
          "  limit does not bite -- and the two modules have IDENTICAL spectra, so the\n"
          "  ROTATION NUMBER is the only thing that can tell the fits apart.  This is\n"
          "  task 23's case: exactly where Lemma C has no gap to use.\n")

    cyc = (0.50, 1.30)
    sys_p = S.ModularSystem([S.LimitCycleBlock(a=0.3, omega=cyc[0]),
                             S.LimitCycleBlock(a=0.3, omega=cyc[1])])
    sys_q = S.ModularSystem([S.LimitCycleBlock(a=0.3, omega=cyc[0]),
                             S.LimitCycleBlock(a=0.3, omega=0.90)])

    def split_arm(tag, system, strength, decoder_kind, other=None):
        """Fit on neurons [:half] and [half:] independently; compare fingerprints."""
        X, _, _ = population(system, strength, SEED)
        half = N_OBS // 2
        Xs = [X[..., :half], X[..., half:]]
        if other is not None:  # negative control: the second half is a DIFFERENT system
            Xo, _, _ = population(other, strength, SEED)
            Xs[1] = Xo[..., half:]
        fps, quals = [], []
        for side, Xi in enumerate(Xs):
            for k in range(N_RESTARTS):
                fp, q = fit_fingerprint(Xi, seed=SEED + 100 * side + k,
                                        decoder_kind=decoder_kind)
                fps.append((side, fp))
                quals.append(q)
        def clean(f):
            """No two modules carrying the same factor -- the one degeneracy that
            is visible without ground truth."""
            return not f.duplicate_modules(spec_tol=FIT_SPEC_TOL, rot_tol=FIT_ROT_TOL)

        cross, screened = [], []
        for sa, fa in fps:
            for sb, fb in fps:
                if sa != 0 or sb != 1:
                    continue
                c = M.invariant_agreement(fa, fb, spec_tol=FIT_SPEC_TOL, rot_tol=FIT_ROT_TOL)
                cross.append(c)
                if clean(fa) and clean(fb):
                    screened.append(c)
        frac = float(np.mean([c.agree for c in cross]))
        specs = [c.spectrum_error for c in cross]
        rots = [c.rotation_error for c in cross]
        s_rots = [c.rotation_error for c in screened] or [float("nan")]
        n_flagged = sum(not clean(f) for _, f in fps)
        print(f"\n  [{tag}] {len(cross)} cross-split comparisons "
              f"({N_RESTARTS} x {N_RESTARTS} restarts)")
        print(f"        example fingerprint  {fps[0][1].summary()}")
        print(f"        agree           {frac:.3f}")
        print(f"        spectrum_error  median {np.median(specs):.4f}  "
              f"range [{min(specs):.4f}, {max(specs):.4f}]")
        print(f"        rotation_error  median {np.median(rots):.4f}  "
              f"range [{min(rots):.4f}, {max(rots):.4f}]")
        print(f"        fit_quality     median {np.median(quals):.4e}")
        print(f"        degenerate fits {n_flagged}/{len(fps)} (two modules on one factor)")
        print(f"        SCREENED        n={len(screened):2d}  rot median "
              f"{np.median(s_rots):.5f}  frac<{FIT_ROT_TOL} "
              f"{np.mean(np.array(s_rots) < FIT_ROT_TOL):.2f}")
        return {"agree_fraction": frac,
                "n_flagged": int(n_flagged),
                "screened": {"n": len(screened),
                             "rotation_median": float(np.median(s_rots)),
                             "rotation_frac_agree": float(
                                 np.mean(np.array(s_rots) < FIT_ROT_TOL)),
                             "all": [float(x) for x in s_rots]},
                "spectrum_error": {"median": float(np.median(specs)),
                                   "min": float(min(specs)), "max": float(max(specs))},
                "rotation_error": {"median": float(np.median(rots)),
                                   "min": float(min(rots)), "max": float(max(rots)),
                                   "all": [float(x) for x in rots]},
                "fit_quality_median": float(np.median(quals)),
                "n_comparisons": len(cross),
                # Every fingerprint, so this arm can be re-analysed without
                # refitting.  25 fits is ~30 minutes; a matching rule is a
                # one-line change, and the two should not be coupled.
                "fingerprints": [
                    {"side": side, "partition": fp.partition,
                     "spectra": [s.tolist() for s in fp.spectra],
                     "rotations": fp.rotations, "coherences": fp.coherences}
                    for side, fp in fps
                ]}

    a_lin = split_arm("4a linear decoder", sys_p, 0.0, "linear")
    a_nl = split_arm("4b nonlinear decoder", sys_p, 1.5, "mlp")
    a_neg = split_arm("4c different frequencies (negative control)", sys_p, 0.0,
                      "linear", other=sys_q)

    # Checks are per-INVARIANT, not on the combined `agree` flag, because the two
    # halves of the fingerprint behave completely differently here and a single
    # boolean would hide it.  See the write-up under check 13.
    ok8 = a_lin["rotation_error"]["max"] < FIT_ROT_TOL
    checks.append(("8 rotation numbers agree across neuron splits (linear decoder)", ok8))
    verdict(ok8, f"max rotation error {a_lin['rotation_error']['max']:.5f} over "
                 f"{a_lin['n_comparisons']} cross-split pairs (< {FIT_ROT_TOL})")

    # SCREENED, and distributional -- neither is a lowered bar.  A per-restart
    # failure survives at every population size and every training budget, so a
    # `max` over raw pairs would assert a reliability the method does not have.
    # The screen is the one degeneracy visible without ground truth (two modules
    # on one factor); coherence and fit_quality both fail as screens (r = -0.48
    # and +0.24, the latter with the wrong sign).  Check 12 verifies the screen
    # is not merely a filter that flatters everything.
    med_nl = a_nl["screened"]["rotation_median"]
    ok9 = med_nl < FIT_ROT_TOL
    checks.append(("9 rotation numbers agree in most splits under a nonlinear map", ok9))
    verdict(ok9, f"screened median rotation error {med_nl:.5f} over "
                 f"{a_nl['screened']['n']}/{a_nl['n_comparisons']} surviving pairs "
                 f"({a_nl['screened']['rotation_frac_agree']:.0%} agree); raw median was "
                 f"{a_nl['rotation_error']['median']:.5f}. This is the setting task 40 "
                 "proposes for real data")

    true_sep = abs(cyc[1] - 0.90) / (2.0 * np.pi)
    ok10 = a_neg["rotation_error"]["min"] > FIT_ROT_TOL
    checks.append(("10 the test can fail: a frequency change is detected", ok10))
    verdict(ok10, f"min rotation error {a_neg['rotation_error']['min']:.5f} against a true "
                  f"frequency separation of {true_sep:.5f}; the spectra are identical by "
                  "construction, so nothing but rho could have caught this")

    # The asymmetry, stated as its own check so it cannot be read past.  Measured
    # on the LINEAR arm only, and deliberately: an arm whose fits do not
    # reproduce the attractor cannot speak to which invariants are recoverable.
    # Mixing the two would have let fit quality masquerade as identifiability.
    # A RATIO, because the two errors are in different units and have different
    # tolerances -- an earlier version compared a spectral error against the
    # ROTATION tolerance and reported a null at a 187x gap.  What the claim needs
    # is that rotation is recovered (below its own tolerance) and that the
    # spectrum is worse by a wide margin on the same fits.
    spec_bad = a_lin["spectrum_error"]["median"]
    rot_good = a_lin["rotation_error"]["median"]
    ratio = spec_bad / max(rot_good, 1e-12)
    ok11 = rot_good < FIT_ROT_TOL and ratio > 20.0
    checks.append(("11 rotation agrees where the transverse spectrum does not", ok11))
    verdict(ok11, f"median spectrum error {spec_bad:.4f} vs median rotation error "
                  f"{rot_good:.5f} -- a {ratio:.0f}x gap ON THE SAME FITS")

    # Attribution for whatever the nonlinear arm did, via the metric's own quality
    # flag rather than by assertion.  A rotation number with low coherence
    # describes nothing (spectra.RotationNumber), so if the nonlinear arm's
    # readings are incoherent its disagreement is a statement about the FITS.
    # Does the screen earn its keep?  A filter that raises every arm is measuring
    # nothing; one that raises the treatment and leaves the negative control
    # rejecting is removing a real defect.  Same discipline as 3.13(d).
    gain_nl = a_nl["rotation_error"]["median"] / max(a_nl["screened"]["rotation_median"], 1e-12)
    neg_before = a_neg["rotation_error"]["median"]
    neg_after = a_neg["screened"]["rotation_median"]
    ok12 = gain_nl > 2.0 and neg_after > FIT_ROT_TOL
    checks.append(("12 screening on duplicate invariants helps the treatment, not the control",
                   ok12))
    verdict(ok12, f"nonlinear median {a_nl['rotation_error']['median']:.5f} -> "
                  f"{a_nl['screened']['rotation_median']:.5f} ({gain_nl:.0f}x better); "
                  f"negative control {neg_before:.5f} -> {neg_after:.5f} (still rejecting). "
                  "A screen that improved both would be a filter, not a fix")

    coh_lin = min(min(f["coherences"]) for f in a_lin["fingerprints"])
    coh_nl = min(min(f["coherences"]) for f in a_nl["fingerprints"])
    fitq_ratio = a_nl["fit_quality_median"] / a_lin["fit_quality_median"]
    col_lin, col_nl = a_lin["n_flagged"], a_nl["n_flagged"]
    print(f"\n  [attribution] worst rotation coherence: linear {coh_lin:.3f}, "
          f"nonlinear {coh_nl:.3f}")
    print(f"                fit_quality: nonlinear is {fitq_ratio:.0f}x worse than linear")
    print(f"                mode-collapsed fits (two modules on one factor): "
          f"linear {col_lin}/{len(a_lin['fingerprints'])}, "
          f"nonlinear {col_nl}/{len(a_nl['fingerprints'])}")
    print("                Collapse is the failure mode behind the nonlinear arm's\n"
          "                disagreements, and the only ground-truth-free way found to\n"
          "                see it: coherence correlates with recovery error at -0.48\n"
          "                and fit_quality at +0.24 (wrong sign, 3.11 again), so\n"
          "                neither can be used to screen fits.  Duplicate invariants\n"
          "                can -- they are a property of the fitted model alone.")
    attribution = {"worst_coherence_linear": coh_lin,
                   "worst_coherence_nonlinear": coh_nl,
                   "fit_quality_ratio": fitq_ratio,
                   "mode_collapsed_linear": col_lin,
                   "mode_collapsed_nonlinear": col_nl}
    print("\n        Why: an invariant is recoverable only where the DATA GOES.  On a\n"
          "        limit cycle the rotation number and the neutral exponent live ON the\n"
          "        attractor, which is where every trajectory spends its time -- and they\n"
          "        come back to 1e-4.  The transverse exponent lives OFF it, and orbits\n"
          "        collapse onto the cycle in ~4 steps of a 30-step trial, so almost no\n"
          "        data constrains it.  3.8's support caveat with a sharp edge: it is not\n"
          "        that the fit is bad, it is that this number was never measured.")

    rec["part4"] = {"attribution": attribution,
                    "omegas": list(cyc), "negative_omega": 0.90,
                    "true_rho_separation": true_sep,
                    "linear": a_lin, "nonlinear": a_nl, "negative_control": a_neg,
                    "asymmetry": {"median_spectrum_error": spec_bad,
                                  "median_rotation_error": rot_good}}

    # -----------------------------------------------------------------
    banner("SUMMARY")
    for name, ok in checks:
        verdict(ok, name)
    rec["checks"] = [{"name": n, "pass": bool(o)} for n, o in checks]
    n_pass = sum(o for _, o in checks)
    rec["n_pass"], rec["n_checks"] = int(n_pass), len(checks)
    path = save("exp14_invariant_agreement", rec)
    print(f"\n  {n_pass}/{len(checks)} checks passed; wrote {path}")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
