# Identifiability of Modular Nonlinear Latent Dynamics

Does constraining latent dynamics to be **modular** — a direct product of
independently evolving subsystems — make the latent representation
**identifiable**?

Motivation: LFADS-style models for neural population analysis learn latents
defined only up to arbitrary invertible reparameterisation. If modular structure
pins down the latents, it gives a principled route to interpretable latent
dynamical models.

**The novelty claim is application, not method.** The deliverable is *a simple,
identifiable latent dynamics model for a neural population* — novel in the
**neuroscience** literature, where LFADS-style latents come with no statement of
what about them is real. It need not be novel in dynamical systems or nonlinear
ICA. So published machinery is a resource to cite rather than an obligation to
discharge, and the claim to aim at is the one a neuroscientist can act on: the
partition, module dimensions, and per-module dynamical invariants — *not*
block-diagonality of the reparameterisation. Rigour does not relax; the scope of
novelty does. See [CLAUDE.md](CLAUDE.md) §1.0.

**Scope.** The target is an **autonomous, single-area** nonlinear latent
dynamical system. Modules are dynamical factors *within one population* —
separate timescales, distinct oscillatory components — **not brain regions**.
Multi-region models and inter-area communication are out of scope; input drive
is a nice-to-have. See [CLAUDE.md](CLAUDE.md) §1.1.

Start with [CLAUDE.md](CLAUDE.md) for the full context and the to-do list.

---

## Where things stand

The original conjecture is **false as stated**, and its proof sketch has a real
error. Both are diagnosed, repaired where possible, and the repairs are
certified numerically.

| | status |
|---|---|
| **Global conjugacy (Tier 1)** | **Free.** Injective decoders + exact fit force $\hat F = h F h^{-1}$, so latent dimension, fixed points and their stability, Lyapunov spectrum, rotation number and attractor topology are identified with *no theorem and no auxiliary variable*. The **decomposition** is what costs — that is Tier 2, and every open obligation lives there |
| **Linear decoder (Theorem A)** | **Proved and numerically certified.** Sharp: two independent hypotheses, neither removable |
| Perturbation, linear case | **Measured.** $O(\epsilon/\mathrm{gap})$ confirmed; log-log slope $1.0000$ |
| **Nonlinear decoder (Theorem B)** | Stated form (weak hypotheses) is **false**; **block-diagonal recovered under $C^\infty$ + cross-module non-resonance**, assembled and reduced to a classical estimate (fixed-point regime) — see below |
| Matching lemma (B3) | A *conclusion* on the normal-form route (falls out of Theorem L); linear case proved |
| **Behavioural route (Lemma D)** | **Proved** for additive $h_B$ + linear modules: a behavioural auxiliary kills the cross-derivative the spectral gap provably cannot. Two behaviour levels suffice |
| **Does learning recover it?** | **Partly — and the earlier "no" is retracted.** That penalty was gauge-dependent and the optimiser paid it by shrinking the block instead of making it $u$-invariant, so no arm ever imposed the hypothesis (§3.12). Fixed, `exp13` kills the forbidden cross-block at every dose and recovers block-diagonality at 3 of 4 — the exception is consistent across all 8 restarts, and is the front line |
| Literature positioning | Drafted in `theory/literature.md`, provenance-tagged |
| **Identifiability test (no ground truth)** | **Built and validated on synthetic systems** (`exp14`, task 40). Compares two fits *to each other* — module count, dimensions, filtration order, per-module Lyapunov spectra, and **rotation numbers**, the invariant the spectrum provably cannot see. Blind to a within-module gauge change and to the §3.7 triangular conjugacy; catches the §3.1 regrouping and a frequency change. Under a **linear** observation map, disjoint neuron splits agree to $2.5\times10^{-4}$ in 16/16 comparisons, with signs and module order differing freely — the gauge quotiented out. Under a **strong nonlinear** map it is limited by **per-restart reliability**, not by data or training: $3\times$ the budget makes recovery *worse*, population size helps only to ~32 neurons/side, and what remains is fits that put two modules on one factor. That failure is invisible to `fit_quality` ($r=+0.24$, wrong sign) and to `coherence` ($r=-0.48$) but visible as duplicate invariants, which needs no ground truth — screening on it takes the median error $0.0633\to0.0032$ while leaving the negative control untouched at $0.0636$. `exp14` passes 11/11 under that protocol |
| **Real data** | **None — everything here is synthetic** (`make_dataset`). The adequacy half of the empirical program is still unbuilt: co-smoothing over a nested model ladder (task 39). `CLAUDE.md` §6 |

### The main finding

**The target conclusion is false, not merely hard to prove.**

First, the correction proposed for the proof's cocycle error, *even executed
correctly*, cannot reach it: block-diagonality needs both cross-derivative blocks
to vanish, requiring $\lambda_{\max}(f_2) < \lambda_{\min}(f_1)$ **and**
$\lambda_{\max}(f_1) < \lambda_{\min}(f_2)$, and chaining these is a
contradiction. The two decay rates are exact negatives — measured summing to
$8.7\times10^{-10}$.

Then, worse: with $f_i(z_i) = \mu_i z_i$ and $0 < \mu_1 < \mu_2 < 1$,

$$h(z_1,z_2) = \left(z_1 + c\,\mathrm{sgn}(z_2)|z_2|^{p},\; z_2\right), \qquad p = \tfrac{\log\mu_1}{\log\mu_2},$$

is an exact conjugacy satisfying *every* hypothesis of Theorem B as stated, and
is triangular but **not** block-diagonal. Setting $\mu_1 = \mu_2^m$ makes $h$
polynomial, so $C^\infty$ regularity does not rescue it either — cross-module
**non-resonance** turns out to be a necessary hypothesis.

So a spectral gap buys a **triangular** $h$ (a skew product), and that is sharp.
See [theory/counterexamples.md](theory/counterexamples.md) §3 and §5, and
[theory/identifiability.md](theory/identifiability.md) §4.3.

**Where this leaves the project.** Three live results:

- **Block-diagonal recovery is not dead** — the counterexample needs a
  measure-zero *resonance*. Strengthening the hypotheses to **real-analytic $h$**
  (free with `tanh`-MLP decoders) plus cross-module non-resonance recovers it near
  an attracting fixed point, and the theorem is **assembled and clean**
  ([theory/identifiability.md](theory/identifiability.md) §5.3–5.4): analyticity
  makes the formal-to-smooth step the *identity theorem* (a block-diagonal Taylor
  jet is a block-diagonal map), so the whole fixed-point case reduces to classical
  Poincaré–Dulac + the identity theorem + a formal lemma — **nothing unwritten,
  no paywalled sources**. (A $C^\infty$ fallback without the analyticity
  assumption still works, via the lemma **(FLAT-D)** and the wave-operator
  construction $h=\lim\Psi^{-n}\Phi^n$ verified in `exp07`, at the cost of one
  textbook $C^k$ distortion bound.)
- **The triangular / filtration structure is proved unconditionally** and is
  recorded as the safe fallback: an ordered filtration of dynamical factors, a
  slow component evolving autonomously and a faster one driven by it. Its formal
  write-up is deferred while Route A (block-diagonal) is pushed to completion.
  It also reaches **further** than the block-diagonal result: the cocycle
  argument never uses the fixed point, and it holds at attracting periodic
  orbits ([identifiability.md](theory/identifiability.md) §4.4, certified in
  `exp08`) — precisely the regime where normal forms fail, since a limit cycle's
  neutral exponent puts it in the Siegel domain. Two structural riders come with
  it: an oscillatory module is always the **top** of the filtration, and a
  filtration can hold **at most one** of them.

- **A behavioural auxiliary closes the direction the dynamics provably cannot**
  ([identifiability.md](theory/identifiability.md) §4.5, **Lemma D**). §3.7 shows
  the spectral gap can never kill both cross-derivatives, so a gap alone gives
  only a triangular $h$ — that is sharp. But conditioning on behaviour does kill
  the survivor: a conjugacy forces the coupling to be a *semiconjugacy* between
  modules, only cross-module *resonant* degrees survive, and **the gap itself
  forces every such degree to be $\ge 2$** — so the coupling scales as
  $\sigma^{p}$ and two levels of variance modulation detect it. The one escape,
  a scale-invariant (degree-0) coupling, needs $1 \in \operatorname{spec}(\tilde f_B)$,
  which the gap forbids. **Two levels**, where iVAE needs $nk+1$ — this avoids
  Khemakhem et al.'s assumption (iv) rather than repairing it.

**The open front is learning — and the last answer there was wrong.**
`exp11`/`exp12` fit these models from data and reported that a nonlinear
observation map degrades the recovered structure to triangular, which read as
*behaviour failing to supply its kill*. It was not. The behavioural penalty
those runs optimised scored the pinned block's conditional moments on the raw
block, so it fell like $\varepsilon^2$–$\varepsilon^4$ as the block shrank: the
optimiser made the block **21× smaller** instead of $u$-invariant, and it still
carried the $u$-varying latent at distance correlation $0.99$. **The hypothesis
was never imposed in any arm** (CLAUDE.md §3.12), so neither experiment tested
the composition.

This is the fourth defect of the same family in this repo — after a rate fitted
to a noise floor (§3.9), a linear probe blind to nonlinear coupling (§3.10), and
a point estimate standing in for a distribution (§3.11) — and the first in the
*objective* rather than a readout, where it does not merely misreport but
steers. The fix is the same move each time: make the quantity invariant under
the gauge §7 already grants. With the penalty whitened and its weight
recalibrated, `exp13` (8 restarts × 4 doses) kills the forbidden cross-block
*uniformly* — $\le 0.081$ everywhere, against $0.316$ — and recovers
block-diagonality at the two highest doses ($0.567 \to 0.815$ at the top). The
negative result is retracted. It is not a clean positive either: at one
intermediate dose the fit is triangular in all 8 restarts, too consistently for
seed noise, which is the open question.

The audit of the assembled theorem is [theory/route_a_assessment.md](theory/route_a_assessment.md);
the route comparison is [theory/approaches.md](theory/approaches.md).

---

## Quick start

Create the pinned environment (CPU-only torch — deliberate, see
[`environment.yml`](environment.yml)):

```bash
conda env create -f environment.yml && conda activate idyn
```

Then the acceptance test — **expect 295 passing**:

```bash
python -m pytest -q
```

```bash
python experiments/run_all.py
```

Each experiment writes a JSON record — seed, every parameter, every measured
number — to `results/`. The JSON is the artifact; console output is a summary.
Everything runs on CPU. `run_all.py` covers exp01–exp10 and takes ~30 minutes,
dominated by exp06; exp11–exp14 are unregistered and run individually (reasons
in that file).

> On the machine this was developed on the interpreter is
> `C:/Users/adene/miniconda3/envs/torch/python.exe`, which is not on `PATH`.
> If you have moved the repo, see CLAUDE.md §4.1 "Moving this repo to another
> machine" — the checklist is four steps and the test suite is the real gate.

---

## The experiments

| | what it establishes |
|---|---|
| `exp01_linear_base_case` | The linear theorem, certified exactly via intertwiner spaces. Four cases: both hypotheses, each dropped in turn, and the repair |
| `exp02_regrouping_negative_control` | **The falsification gate.** Data from four independent systems forced into two 2-D modules. All three groupings reproduce the observations bit-for-bit, and fitting genuinely finds several |
| `exp03_modular_recovery` | Positive control. Two nonlinear oscillators with separated exponents: partition recovered in 5/5 converged restarts, on-block $0.9647$ vs chance $0.5$ |
| `exp04_perturbation` | $\epsilon$-coupling sweep. Slope $1.0000$, gap-independent constant, breakdown located at $\epsilon \approx \mathrm{sep}$ |
| `exp05_cocycle_and_spectra` | The two proof-level defects, plus the new two-sided obstruction |
| `exp06_partition_lattice` | Learning the module partition from data: lattice search + fitted-model certification. Fit and uniqueness each cover the other's blind spot |
| `exp07_flat_tangency` | (FLAT-D), Route A's last lemma: the telescoping conjugacy $h=\lim\Psi^{-n}\Phi^n$ converges ($C^0/C^1$), is exact, and is flat-tangent — across linear/nonlinear $\Phi$ and two flatness rates |
| `exp08_attractor_cocycle` | Lemma C off the fixed point: the rate holds on a limit-cycle attractor to $2.4\times10^{-14}$, with the threshold exactly at $\|1-2a\|$. Also the measurement trap that hid it — a $\sigma_{\min}$ bound that is noise past $n\approx39$ and can *invent* a spectral gap |
| `exp09_tier2_nonempty` | Route A's nonlinear claim is not hollow: an explicit system satisfying cross-module non-resonance whose within-module resonance $\mu^2-\mu^2=0$ obstructs linearisation. Finds a false negative in the linearised (B2) test, closes it with a quadratic-jet test, and shows the right verdict is resonance-coupling-graph *connectedness* (not "any coupling") — the degree-2 (B2) indecomposability criterion, proved in `route_a_assessment.md` §4.1a |
| `exp10_behavior_cocycle` | The B∘C composition: behaviour kills one latent cross-derivative (the $u$-invariant subspace is canonical), the cocycle kills the complementary one under a one-sided gap, together block-diagonal. Truth table (only block-diagonal passes both); the alignment condition (block-diagonal needs varying = dominant block); and the Prop-1 caveat (variance vs mean modulation) |
| `exp11_learned_behavior_cocycle` | Does B∘C survive *learning*? Fits encoder/decoder/transition from data and ablates behaviour and modularity. **Behavioural conclusions superseded (§3.12)** — the penalty was gauge-dependent, so the behaviour arm never differed from the ablation in the way it claims. What stands is the tooling it forced: a Jacobian block metric (the linear probe is blind here, §3.10) and distributions over restarts rather than best-of-$N$ (§3.11) |
| `exp12_decoder_strength_sweep` | CLAUDE.md task 29. Sweeps observation nonlinearity and finds a **confirmed dose-response**: `jac_diag` falls monotonically $0.994 \to 0.730 \to 0.702 \to 0.567$ across doses $(0.00, 0.31, 0.43, 0.60)$, the forbidden cross-block violated in $0/8$ restarts at low dose but $8/8$ at high. Not fit failure — better fits are *more* coupled. **But the B∘C reading is retracted (§3.12)**: no arm imposed the behavioural hypothesis, so this measures *dynamics-only* fitting. Two checks fail **by design** |
| `exp13_conjugacy_residual` | CLAUDE.md tasks 32–33. Asks which Lemma D hypothesis a fitted $h$ breaks and finds the answer is none of them — the behavioural one was never imposed (§3.12). Re-runs the exp12 dose sweep under both the old and the whitened penalty, reporting the other three candidates alongside: the conjugacy residual $\lVert h\circ F-\tilde F\circ h\rVert$, the *learned* spectral gap, and the additivity defect of $h_B$ |
| `exp14_invariant_agreement` | CLAUDE.md task 40 — the identifiability test that needs **no ground truth**: fit on disjoint neuron subsets and compare the fits to each other. Validates the new machinery (`spectra.rotation_number`, `metrics.invariant_agreement`) on exact systems first, where the answer is known: blind to a within-module gauge change *and* to the §3.7 triangular conjugacy, not blind to the §3.1 regrouping or a frequency change. Then measures learned models, and the useful part is what it found there — a fitted map iterated past its data converges to a **spurious attractor**, and recoverability is **per-invariant** (§3.13). Checks are per-invariant rather than one boolean, for that reason |

Read `exp02` and `exp03` together. `exp02` shows the method reports
non-uniqueness when the truth is non-unique, so `exp03`'s positive result is not
just an easily-satisfied metric.

`exp08` part 1 is worth reading on its own as a cautionary tale: the discarded
estimator did not fail loudly, it returned a plausible negative rate that would
have certified the central lemma in a case where the lemma's hypothesis fails.

---

## Layout

```
CLAUDE.md              context, resources, to-dos      <- start here
docs/brief_v0.md       the original brief, verbatim
theory/
  identifiability.md   corrected statement: Theorem A / Theorem B
  linear_case.md       the linear theorem, proved
  counterexamples.md   every counterexample, each one asserted in tests/
  approaches.md        the three routes (A/B/C): claims, costs, decision doc
  route_a_assessment.md  primary-source audit of the Route A / Tier-2 proof
  literature.md        positioning + recon; every claim provenance-tagged
src/idyn/
  systems.py           modular maps, oscillators, coupling, counterexamples;
                         LinearDecoder (Thm A) + MLPDecoder (Thm B, coupling
                         flow) -- the *data* decoders
  linear.py            indecomposability, primary decomposition, intertwiners
  spectra.py           Lyapunov / dichotomy spectra, module gaps, rotation
                         number (the invariant the spectrum cannot see)
  cocycle.py           the corrected §3.3 argument, measurable
  normalform.py        Poincaré–Dulac: homological operator, resonant monomials
  behavior.py          Route B: u-conditioned sampling, invariant-subspace detector
  models.py            torch: modular vs unconstrained, 2 decoder settings
  train.py             fitting, restarts, datasets
  metrics.py           partition recovery, non-uniqueness diagnostics,
                         Jacobian + distance-correlation block structure,
                         dynamical_fingerprint + invariant_agreement (fit-to-fit,
                         the only metric here needing no ground truth)
  selection.py         partition-lattice search, fitted-model certification
experiments/           exp01..exp14, each writing results/<name>.json
                         (exp11/exp12 report failing checks by design, and their
                          behavioural readings are superseded -- see CLAUDE.md 3.12;
                          run_all.py covers exp01..exp10 only, exp13/exp14 are
                          excluded on cost -- their machinery is covered by tests/)
environment.yml        pinned env; requirements.txt is the pure-pip equivalent
tests/                 295 tests
```

## Conventions

- Partitions are `list[int]` of block dimensions summing to `d`.
- Everything numerical takes an explicit `rng` or `seed`; no global RNG state.
- `float64` for linear algebra and spectra; `float32` only inside torch models.
- **A numerical result that contradicts a theory claim gets committed and
  reported, not tuned away.** That is what `exp02` is for.
