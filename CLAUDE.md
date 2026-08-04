# CLAUDE.md — Identifiability of Modular Nonlinear Latent Dynamics

Context for AI assistants working in this repo. Read fully before editing theory
files, writing proofs, or running simulations.

This file supersedes the original brief (`docs/brief_v0.md`). Sections 1–3 and
7–8 are carried over from it; sections 4–6 are new (resources, layout, to-dos).

---

## 1. What this project is

We are trying to determine whether constraining the latent dynamics of a
nonlinear latent variable model to be **modular** (a direct product of
independently evolving subsystems) makes the latent representation
**identifiable**.

Motivation: LFADS-style models for neural population analysis learn expressive
latents that are only defined up to arbitrary invertible reparameterization.
If modular structure pins down the latents (up to module permutation and
within-module coordinate change), it gives a principled route to interpretable
latent dynamical models for neural data.

### 1.0 The novelty claim is APPLICATION, not method (governing, 2026-08-04)

**Read this before deciding what to work on.** The deliverable is *a simple,
identifiable latent dynamics model for a neural population*. That is novel in
the **neuroscience** literature — LFADS and its descendants produce latents
defined only up to arbitrary reparameterisation, and nobody hands a
neuroscientist a fitted model plus a statement of what about it is real. It
does **not** need to be novel in dynamical systems or in nonlinear ICA.

Three consequences, and they reverse earlier priorities:

1. **Published machinery is a resource, not a threat.** Block-identifiability
   (§1.3 of `literature.md`; tasks 38, 8) means the behavioural half can be
   *cited* rather than proved. Discovering that a lemma already exists is
   progress, not a setback — it removes an obligation. Assemble from available
   parts and say so.
2. **"Essentially proved, needs writing up" is a finished state.** Route C is
   the asset (task 11). Its value here is that it is *correct and applicable*,
   not that it is new.
3. **Prefer the claim a neuroscientist can act on.** That is task 37 — the
   partition, module dimensions, and per-module invariants (Lyapunov spectrum,
   rotation number, attractor topology) — *not* block-diagonality of $h$. §7 has
   always said coordinates are unidentifiable; under this framing that ceases to
   be a caveat and becomes the specification.

What still has to be true: the identifiability statement must be **correct**,
its hypotheses **checkable on real data**, and the recovery **validated against
synthetic ground truth**. Rigour is not what relaxes — scope of novelty is.

### 1.1 Scope — the minimal objective (governing constraint)

**The target is an autonomous, single-area, nonlinear latent dynamical system
that is identifiable.** Everything else is optional. Deliberately in and out:

| | |
|---|---|
| **In** | one population from one area; autonomous dynamics $z_{t+1} = F(z_t)$; nonlinear decoder $x_t = g(z_t)$; modular $F$ |
| **Nice to have** | input drive $u_t$; noise/distributional equivalence |
| **Out** | multi-region models, inter-area communication, anything where modules are *regions* |

**A module is a dynamical factor inside one population, not a brain area.**
Concretely: separate timescales, distinct oscillatory components, subspaces that
evolve without reference to each other. This is what makes the *filtration*
reading (§3.7, ordering modules by Lyapunov spectrum) natural rather than a
consolation prize — a slow component driving a fast one is the expected
structure within a single area, whereas a symmetric partition is not.

**Consequence for the old §3.8 autonomy worry:** it is resolved by scoping, not
by proof. Input-driven latents were the main disanalogy with LFADS and the main
threat to the applied claim; restricting to autonomous single-area models
removes it from the critical path. Do not re-open it as a blocker.

### Model

Latent state partitioned into $K$ modules, $z_t = (z_t^{(1)}, \dots, z_t^{(K)})$
with $z_t^{(i)} \in \mathbb{R}^{d_i}$, each evolving autonomously:

$$z_{t+1}^{(i)} = f_i(z_t^{(i)}), \qquad F = f_1 \oplus \cdots \oplus f_K$$

Observations via a linear decoder:

$$x_t = W z_t + \epsilon_t$$

### Target identifiability statement

Any invertible $h$ with $h \circ F = \tilde F \circ h$, where $\tilde F$ is also
modular and $\tilde W \tilde z_t = W z_t$, should factor as

$$h = h_1 \oplus \cdots \oplus h_K \quad \text{up to permutation of modules.}$$

**As written this is false** — see §3.1. The repaired statement lives in
[`theory/identifiability.md`](theory/identifiability.md).

---

## 2. Status

The original draft contained a conjecture and proof sketch. **The conjecture as
written is false and the proof sketch has a real error.** Do not build on either
without applying the fixes in §3.

Current state of this repo:

| Piece | Where | State |
|---|---|---|
| Corrected theorem statement | `theory/identifiability.md` | Restated with indecomposability |
| Linear base case (§4 step 2) | `theory/linear_case.md`, `src/idyn/linear.py` | **Proved**, sharp, numerically certified |
| Counterexamples | `theory/counterexamples.md`, `src/idyn/systems.py` | Constructive, all asserted in tests |
| Dichotomy/Lyapunov spectra | `src/idyn/spectra.py` | Implemented, exact on known cases |
| Cocycle argument (§3.3) | `src/idyn/cocycle.py` | Instantiated; rate matches theory to 1e-12 |
| **Lemma C off the fixed point** | `theory/identifiability.md` §4.4, `exp08` | **Done: extends to periodic attractors** (Lemma C′), rate exact to 2.4e-14. Chaotic/non-uniform case open |
| **Route A Tier 2 non-empty** | `theory/approaches.md` §A.2.1, `src/idyn/normalform.py`, `exp09` | **Done.** Witness $(\mu z_a,\ \mu^2 z_b + c z_a^2)$: structural resonance $\mu^2-\mu^2=0$ obstructs linearisation while cross-module non-resonance holds. Route A's nonlinear claim is not hollow |
| **(B2) linearised test is a false negative** | `theory/approaches.md` §A.2.2, `exp09` | **Found and fixed.** Witness has decomposable linear part + indecomposable map. Nonlinear jet test `selection.block_nonlinear_certificate` (`certify_fitted_model(nonlinear=True)`) closes it at degree 2; near-resonance / higher degree open |
| **Tier-2 (B2) indecomposability = graph connectedness** | `theory/route_a_assessment.md` §4.1a, `normalform.resonance_coupling_components`, `exp09` | **Proved at degree 2 (distinct eigenvalues).** A module is indecomposable iff its resonance-coupling graph is connected. Fixed an over-report in the certifier (≥3 sub-blocks). General-degree connectedness-invariance open |
| **Route B∘C composition** | `theory/approaches.md` §B.1, `src/idyn/behavior.py`, `exp10` | **Mechanism built + verified.** Behaviour kills $M_{BA}$ (invariant subspace canonical), Lemma C kills $M_{AB}$ (one-sided gap) → block-diagonal. New **alignment condition**: needs varying block = spectrally dominant, else only triangular. Dynamics half proved; partial-iVAE lemma is the one open obligation |
| **Lemma D — behavioural kill** | `theory/identifiability.md` §4.5, `systems.lemma_d_witness`, `tests/test_behavior.py` | **PROVED** for additive $h_B$ with linear modules. Behaviour kills $M_{BA}$, the cross-derivative §3.7 proves the gap can *never* reach. Mechanism: the conjugacy makes the coupling a semiconjugacy $\psi\circ f_A = \tilde f_B\circ\psi$; only *resonant* degrees survive; **the gap itself forces those degrees $\ge2$**; a degree-$p\ge2$ homogeneous $\psi$ scales as $\sigma^p$, so variance modulation detects it. **Two behaviour levels suffice** (vs iVAE's $nk+1$) — this is the partial-iVAE obligation discharged *dynamically*, sidestepping the assumption-(iv) obstruction rather than confronting it. Open: non-additive $h_B$, anisotropic modulation |
| **Two-sided cocycle obstruction** | `theory/counterexamples.md` §3, §5 | **Settled: the conclusion is FALSE — see §3.7** |
| Matching lemma (§3.2) | `theory/identifiability.md` §6 | `TODO(gap)` — open (proved for linear); route in `literature.md` §3.3 |
| Nonlinear decoder (Thm B) | `theory/identifiability.md` §5.3–5.4 | **Assembled** for the fixed-point regime: analytic decoders + non-resonance close it via Poincaré–Dulac + identity theorem (see §3.7). Non-fixed-point attractors open |
| Literature positioning (step 8) | `theory/literature.md` | Drafted, provenance-tagged; needs write-up into §8 |
| Numerical falsification | `experiments/` | exp01–exp10 pass, all JSONs current. **`exp11` and `exp12` intentionally report FAILING checks** — they encode predictions the *dynamics-only* regime refuted. Per §8 those are committed, not tuned away, so **neither is registered in `run_all.py`**, which covers exp01–exp10 only. Both are also **superseded in part by §3.12** and carry a banner saying so; read `exp13` instead for anything behavioural. Read their JSONs directly |
| Learning the partition from data | `src/idyn/selection.py`, `exp06` | Lattice search + fitted-model certification; fit and uniqueness each cover the other's blind spot |
| **The behavioural penalty was a decoy** | **§3.12** (canonical), `models._behavioural_penalty`, `exp13` | **Found and fixed.** The penalty was gauge-dependent, so the optimiser paid it by shrinking the pinned block 21× rather than making it $u$-invariant. **No arm of `exp11`/`exp12` ever imposed Lemma D's behavioural hypothesis**, which voids both rows below. Fix: whiten the block → invariant under $GL(d_b)$, the §7 gauge group. Numbers and the four-defect pattern in §3.12 |
| **B∘C under learning — linear decoder** | `exp11` | **Void as a B∘C result (§3.12)** — behaviour was never imposed, so it is dynamics-only. Measurement itself is sound (`jac_diag` $\in[0.988,0.999]$, sd $0.004$). Doubly uninformative anyway: §3.5 forces $h\in GL(d)$ here |
| **B∘C under learning — nonlinear decoder** | `exp11`, `exp12` → superseded by `exp13` | **Headline retracted (§3.12), then reversed.** `exp12`'s monotone decline $0.994\to0.730\to0.702\to0.567$ is real but measures *dynamics-only* fitting. With the penalty fixed, `exp13` gets $0.815$ at the top dose and kills the forbidden `upper` to $\le0.081$ at **every** dose and restart (from $0.316$). Not uniform — see tasks 33/34 |
| **Nonlinear block-structure metrics** | `src/idyn/metrics.py` | `jacobian_block_report` (= $M_{ij}$ itself, standardised) + `distance_correlation_block_report` (model-free). The linear `filtration_report` is provably blind here — §3.10 |
| **Nonlinear data decoder** | `src/idyn/systems.py` `MLPDecoder` | Affine coupling flow, invertible in closed form, analytic, ~45% nonlinear residual. Until 2026-08-03 **no experiment generated nonlinear observations at all** — §3.11 |
| Test suite | `tests/` | 236 tests, all passing |

**Headline result of the build:** the §3.3 fix is correct but insufficient — it
yields a *triangular* h, not a block-diagonal one, and provably cannot do better.
**And the target is not merely unreachable, it is false:** there is an explicit
$C^1$ (indeed $C^\infty$ in the resonant case) conjugacy satisfying every stated
hypothesis that is triangular and not block-diagonal. See §3.7.

---

## 3. Known problems — do not reintroduce these

### 3.1 The conjecture is false without a minimality condition (BLOCKING)

Counterexample. Let $d = 4$, $F = \mathrm{diag}(\lambda_1,\lambda_2,\lambda_3,\lambda_4)$
with distinct eigenvalues, $K = 2$, $d_1 = d_2 = 2$, so
$f_1 = \mathrm{diag}(\lambda_1,\lambda_2)$, $f_2 = \mathrm{diag}(\lambda_3,\lambda_4)$.

Let $h = P$ be the permutation swapping coordinates 2 and 3, with
$\tilde W = W P^{-1}$. Then:

- $\tilde W \tilde z_t = W z_t$ exactly — observations identical;
- $\tilde F = P F P^{-1} = \mathrm{diag}(\lambda_1,\lambda_3) \oplus \mathrm{diag}(\lambda_2,\lambda_4)$
  — still modular, same $K$, same block dimensions;
- all five draft assumptions hold.

Yet $h$ moves a coordinate across modules, so it is not $h_1 \oplus h_2$ up to
module permutation. **Cause:** a decomposition into non-indecomposable blocks is
not unique — invariant subspaces can be regrouped freely. Not a linear artifact;
four independent 1D nonlinear maps regrouped into two 2D modules fails the same way.

**Fix:** require each $f_i$ to be dynamically *indecomposable* (no further
invariant splitting), and require the alternative representation to have the same
$K$ and the same multiset of block dimensions. Prove uniqueness of the *finest*
modular decomposition, not block-separability of $h$.

> Implemented as `systems.regrouping_counterexample()` and asserted in
> `tests/test_counterexamples.py`. It is the negative control for experiment 2.

### 3.2 The spectral assumption is on the wrong pair (BLOCKING)

The Sylvester step needs $\mathrm{spec}(D\tilde f_1) \cap \mathrm{spec}(Df_2) = \emptyset$
— transformed system 1 against original system 2. The draft assumes disjointness
only among the $f_i$. In the §3.1 counterexample $\mathrm{spec}(\tilde f_1) = \{\lambda_1,\lambda_3\}$
overlaps $\mathrm{spec}(f_2) = \{\lambda_3,\lambda_4\}$, which is exactly why it fails.
This cannot be bootstrapped without first knowing the module correspondence.

**Fix:** a matching lemma pairing indecomposable blocks across representations by
conjugacy invariants, established *before* the spectral hypothesis is used.

> Still open. Flagged `TODO(gap)` in `theory/identifiability.md` §4.
> `linear.py` proves the linear instance of it (spectra are complete invariants
> for semisimple blocks); the nonlinear case is not done.

### 3.3 The proof sketch drops an argument shift (BLOCKING)

Differentiating $h_1(f_1(z_1), f_2(z_2)) = \tilde f_1(h_1(z_1,z_2))$ w.r.t. $z_2$ gives

$$M(F(z))\, Df_2(z_2) = D\tilde f_1(h_1(z))\, M(z), \qquad M := \partial h_1 / \partial z_2$$

The left $M$ is evaluated at $F(z)$, not $z$. The draft writes both at the same
point, turning a **cocycle relation** into a pointwise Sylvester equation. As
written the argument is valid only at fixed points of $F$ — it is a linearization
result at equilibria.

**Fix:** iterate to $M(F^n z)\, Df_2^{(n)} = D\tilde f_1^{(n)} M(z)$ and use
growth-rate mismatch under a spectral gap to force $M \equiv 0$. Requires $M$
bounded (compact invariant set + $C^1$ conjugacy).

> `src/idyn/cocycle.py` instantiates the iterated relation numerically and
> measures the decay rate of $\|M\|$; `experiments/exp05_cocycle_and_spectra.py`
> shows it decays at the predicted rate with a gap and stalls without one.
> **This fix is correct but not sufficient — see §3.7.**

### 3.4 Assumption 4 is not usable as stated

"Jacobian spectra generically distinct" — pointwise Jacobian spectra vary with $z$
in a nonlinear system and will generically cross somewhere.

**Fix:** replace with disjoint **dichotomy (Sacker–Sell) / Lyapunov spectra**.

> `src/idyn/spectra.py`. Use `lyapunov_spectrum()` / `module_spectral_gap()`.
> Never write an assumption in terms of pointwise `eig(Df(z))`.

### 3.5 The linear decoder already collapses $h$ to linear

If $W, \tilde W$ both have full column rank $d$ and $W z_t = \tilde W \tilde z_t$,
then $\tilde z_t = (\tilde W^+ W) z_t$, so $h \in GL(d)$ — forced, before dynamics
enter. All nonlinear conjugacy machinery in the draft is therefore unnecessary for
the model as specified, and the motivating nonlinear reparameterization ambiguity
does not arise at all.

**Fix:** split into two theorems — linear decoder (base case) and nonlinear
decoder $x_t = g(z_t)$ (the setting the motivation actually requires).

> Done: Theorem A / Theorem B in `theory/identifiability.md`. `models.py` takes
> `decoder="linear"` or `decoder="mlp"` to keep the two settings separable in
> code as well.

### 3.7 The cocycle fix cannot close both directions (BLOCKING — new)

**Not in the original brief.** Found while building this repo; it is the main
obstacle between the current state and Theorem B.

Block-diagonality of $h$ needs *both* cross-derivative blocks to vanish. By
Lemma C (§3.3 done properly), $M_{ij} \equiv 0$ iff
$\lambda_{\max}(f_j) < \lambda_{\min}(\tilde f_i)$. Once the modules are matched,
$\tilde f_i \sim f_i$, so for $K = 2$ the two requirements are

$$\lambda_{\max}(f_2) < \lambda_{\min}(f_1) \quad\text{and}\quad \lambda_{\max}(f_1) < \lambda_{\min}(f_2),$$

and chaining them gives
$\lambda_{\max}(f_2) < \lambda_{\min}(f_1) \le \lambda_{\max}(f_1) < \lambda_{\min}(f_2) \le \lambda_{\max}(f_2)$
— a contradiction. **They can never both hold**, for any $K \ge 2$ and any pair.

Measured (`exp05` part 2c): the two rates are exact negatives of each other,
summing to $8.7 \times 10^{-10}$ across four spectral gaps.

**Consequence.** A spectral gap buys a *triangular* $h$ — a skew product where
$h_1 = h_1(z_1)$ but $h_2$ still depends on both — not a block-diagonal one.

**Update — it is worse than a proof gap: the conclusion is FALSE.**
Take $f_i(z_i) = \mu_i z_i$ with $0 < \mu_1 < \mu_2 < 1$ and

$$h(z_1, z_2) = \left(z_1 + c\,\mathrm{sgn}(z_2)|z_2|^{p},\; z_2\right), \qquad p = \frac{\log \mu_1}{\log \mu_2} > 1.$$

This is an exact conjugacy (verified to $1.3\times10^{-15}$), is $C^1$ with
bounded derivative and unit Jacobian determinant, and satisfies (B1)–(B4)
**exactly as written** in `theory/identifiability.md` — (B4) asks only for
*disjoint* spectra. Yet it is triangular and not block-diagonal.

No contradiction with Lemma C: the oriented gap Lemma C needs
($\lambda_{\max}(f_2) < \lambda_{\min}(\tilde f_1)$) fails here, and the
orientation that does hold kills $M_{21}$, which is indeed zero. The point is
that (B4) as stated is strictly weaker than the oriented gap.

Setting $\mu_1 = \mu_2^m$ makes $p = m$ an integer and $h$ polynomial, hence
$C^\infty$ — so raising the regularity of $h$ does **not** rescue it.
**Cross-module non-resonance is a necessary hypothesis.**

> Built as `systems.triangular_conjugacy_counterexample()`, asserted in
> `tests/test_counterexamples.py`, written up in `theory/counterexamples.md` §5.

**Consequences.** The triangular conclusion of §4.2 is *sharp*. Both candidate
routes in `theory/identifiability.md` §5 are dead at $C^1$ regularity. **Do not
attempt a proof of block-diagonality under (B1)–(B4); a counterexample exists.**

**The live route** is to strengthen the hypotheses rather than the argument. The
decisive strengthening is **real-analyticity of $h$**, which is *free* with
`tanh`-MLP decoders: $g, \tilde g$ are analytic, so $h = \tilde g^{-1}\circ g$ is
analytic. Plus cross-module non-resonance (measure-zero, generic — but necessary,
since the §3.7 counterexample is polynomial hence analytic).

Under analyticity the fixed-point case closes on **classical Poincaré–Dulac
theory + the identity theorem** — no unwritten steps, no paywalled sources:

- **Per-module normal forms** are analytic (Poincaré–Dulac in the Poincaré domain
  = contractions; no small divisors).
- **Formal $\Rightarrow$ analytic is the identity theorem.** The formal lemma
  (proved, `route_a_assessment.md` §2.3) gives a block-diagonal $\infty$-jet; a
  real-analytic map equals its Taylor series, so a block-diagonal jet **is** a
  block-diagonal map. This is the step that needed **(FLAT-D)** + a $C^k$
  distortion bound in the $C^\infty$ category; analyticity makes it one line.
- **(B1) weakens**: an analytic $h$ is pinned down by a set with a limit point,
  which a contracting orbit provides — so $\mathrm{int}\,\Omega\neq\emptyset$
  relaxes to "$\Omega$ has a limit point".

The assembled analytic theorem is `identifiability.md` §5.3–5.4. Tier 1 (full
non-resonance) is still just **robustness of Theorem A** (Poincaré linearizes $F$);
the nonlinear content is Tier 2, where Poincaré–Dulac keeps finitely many analytic
resonant terms. The $C^\infty$ route (FLAT-D, the wave-operator construction
$h=\lim\Psi^{-n}\Phi^n$, `exp07`) remains valid as the fallback if analytic
activations are not assumed; its one unwritten step is the $C^k$ bound.

**Still open:** non-fixed-point attractors (limit cycles, chaos) push into the
*Siegel* domain, where small divisors return and analyticity alone is not enough.
The finite-dimensionality of the MLP class is the natural lever there, unproven.

That covers the `exp05` regime; non-fixed-point attractors remain open.

**Two corrections to how that hypothesis must be phrased** (`counterexamples.md`
§6, both $C^\infty$ counterexamples, both asserted in tests):

1. **Pairwise non-resonance is not enough.** $\mu_1 = \mu_2\mu_3$ admits the
   polynomial conjugacy $h = (z_1 + c z_2 z_3, z_2, z_3)$ while every pairwise
   log-ratio stays $\ge 0.26$ from an integer. Quantify over **multi-indices**.
2. **Rotation angles do not protect.** Phases cancel in $x^2+y^2$, so a repeated
   exponent $\{\log s, \log s\}$ — which *every* `TwistBlock` has — resonates
   with any module at $2\log s$. Check radial rates **with multiplicity**.

> Use `spectra.is_cross_module_nonresonant()` to guard any new test system.
> `s = (0.95, 0.9025)` fails it — $0.9025 = 0.95^2$ — and looks innocuous.

See `theory/route_a_assessment.md` for the full analysis, including the tier
split (§A.2 of `approaches.md`): under *full* non-resonance Sternberg linearizes
$F$ outright, so that tier is best billed as **robustness of Theorem A** against
decoder ambiguity rather than as nonlinear identifiability. The nonlinear content
lives in the cross-only tier.

### 3.8 Secondary issues

- **§7.1 too narrow.** Failure needs only a *shared factor* (common semiconjugate
  quotient), not $f_1 = f_2$ or full conjugacy.
- **Noise dropped.** $\epsilon_t$ appears in the model but equivalence is defined
  pathwise as $W z_t = \tilde W \tilde z_t$. Should be equality of observation
  *distributions*, with the attendant latent-scale / noise-variance tradeoffs.
- **Support.** All conclusions hold only on the closure of the visited region. If
  trajectories collapse to a low-dimensional attractor, $h$ is unconstrained off it.
  State explicitly — this is the regime real recordings are in. **This is now a
  hypothesis, not just a caveat:** the normal-form route needs
  $\mathrm{int}\,\Omega \neq \emptyset$, and on a thin $\Omega$ (a single orbit)
  the conclusion is *false* — see `theory/route_a_assessment.md` §3.5, folded into
  (B1) in `identifiability.md` §4.
- ~~**Autonomy is the main disanalogy with LFADS.**~~ **Out of scope by decision
  (§1.1).** Shared inputs $u_t$ do couple modules and destroy block-diagonality,
  which is why input-driven models are excluded from the minimal objective
  rather than accommodated. If inputs are added later, this becomes live again
  and the right starting point is Vahidi et al. 2024 (§4.3) on separating
  intrinsic from input-driven dynamics.

### 3.9 Never fit a rate to $\sigma_{\min}$ of an accumulated product

**Not a theory defect — a measurement defect that nearly became a theory defect.**

$\sigma_{\min}(Df^{(n)})$ stops being measurable once
$\mathrm{cond}(Df^{(n)}) = e^{n(\lambda_{\max}-\lambda_{\min})}$ outruns float64,
i.e. past $n \approx 36/\mathrm{spread}$. Beyond that the SVD returns its noise
floor, and the *slope* of that floor is $\lambda_{\max}$, not $\lambda_{\min}$.
Factoring a scalar out of the running product (which `jacobian_product_logs`
does) cures underflow but **not conditioning**.

Why it went unnoticed: every module in `exp05` is a `TwistBlock`, whose spectrum
$\{\log s, \log s\}$ is *repeated* — spread $0$, horizon infinite,
$\mathrm{cond}(Df^{(400)}) \approx 10^2$. Those numbers were always sound. A
`LimitCycleBlock` has $\{0, \log|1-2a|\}$, spread $0.92$, horizon $n\approx39$,
$\mathrm{cond}(Df^{(400)}) \approx 10^{300}$ — and `cocycle_bound`'s default
`fit_from=0.5` fits $n\in[200,400)$, entirely inside the noise.

The **defect** is a rate that reads the wrong exponent: the noise floor's slope is
$\lambda_{\max}$, so the fit lands on
$\lambda_{\max}(f_j) - \lambda_{\max}(\tilde f_i)$ instead of
$\lambda_{\max}(f_j) - \lambda_{\min}(\tilde f_i)$ — **off by exactly the block's
spread** ($0.92$ on the `exp08` cycle: $-1.204$ against a true $-0.288$). In one
no-gap case it returned $-0.60$ where the truth is $+0.12$, which via
`forces_M_zero` is a **false certification that the block-separation step
closes**. That is the failure mode to fear — it does not look like an error, it
looks like a result.

> **Do not test for this via the rate's *instability*.** Whether the noise floor
> also wanders is a BLAS detail, not a property of the defect. It moved by $2.7$
> across `n_max` and $0.7$ across initial conditions on numpy 2.4; on numpy 2.5 /
> scipy 1.18 it is stable to $0.011$ and converges cleanly **to the wrong answer**
> — the more dangerous version, since a stable wrong number reads as a
> measurement. Two checks written against the old symptom broke on the env move
> and now assert the *bias* instead: `exp08` part 1 and
> `tests/test_spectra_and_cocycle.py::test_naive_sigma_min_route_is_noise_on_a_limit_cycle`.
> Assert distance from the truth (and closeness to the $\lambda_{\max}$ limit),
> never variance.

**Do:** use `spectra.inverse_jacobian_product_logs` ($\sigma_{\max}$ of the
propagated inverse cocycle — the well-determined end of a product, stable to
$10^{-14}$ at $n=400$). `cocycle_bound` now does this and reports the discarded
`naive_rate` plus `n_resolvable` so the discrepancy stays visible.
`spectra.resolvable_horizon(spread)` is the guard.

**Do not** blame non-normality without checking. The shear $\beta$ was the
first suspect and is innocent: on the cycle the polar Jacobian is
$\left(\begin{smallmatrix}1-2a & 0\\ \beta & 1\end{smallmatrix}\right)$, so
$\sigma_{\max}(J^n) \to \sqrt{(\beta/2a)^2+1}$ — $\beta$ moves the **intercept**
and leaves the rate exactly invariant (verified to $10^{-9}$ for
$\beta \in \{0, 0.6, 1.5\}$).

**`propagate_M` had the same disease** and is fixed the same way. Solving against
the accumulated target product projects onto its dominant singular direction once
that product is numerically rank-deficient, so the rate drifts from
$\lambda_{\max}(f_j) - \lambda_{\min}(\tilde f_i)$ to
$\lambda_{\max}(f_j) - \lambda_{\max}(\tilde f_i)$ — on a limit cycle an error of
$0.93$, reported without complaint. It now propagates the inverse cocycle;
single-step Jacobians are well conditioned, so they are safe to invert one at a
time, and the ill-conditioned product is only ever multiplied. (An earlier
`lstsq` fallback there suppressed the *exception* while still returning the wrong
number — the worse of the two outcomes.)

> Certified in `exp08` part 1; regression tests in
> `tests/test_spectra_and_cocycle.py`. Related: `CocycleBound.forces_M_zero`
> now tests `rate < -zero_tol`, not `rate < 0` — a fitted rate of $-10^{-17}$ is
> zero, and the sign of a rounding error must not decide whether the argument
> reports as closing.

### 3.10 Never measure block structure with a linear probe

**A second measurement defect of the §3.9 kind — found in `exp11`.**
`metrics.filtration_report` fits the *linear* relation between true and fitted
latents. Once the decoder or encoder is nonlinear that is a first-order proxy
(its own docstring says so), and the blind spot is total, not partial:

$$h(z) = (z_A,\; z_B + c\,z_A^{2}), \qquad \mathrm{Cov}(z_A, z_A^2) = 0 \text{ for symmetric } z_A.$$

At $c=5$ the cross term carries $\sim 25\times$ the variance of $z_B$'s own
contribution — $h$ is overwhelmingly triangular — and the linear probe reports
**on_block $=0.97$**. In the *forbidden* direction it is worse: a map that is
$89\%$ upper coupling reports `upper_mass` $=0.002$. A metric that cannot
separate block-diagonal from triangular cannot answer this project's only
question.

**Do:** use `metrics.jacobian_block_report` (entry $(i,j)$ is
$\mathbb{E}_z\|\partial h_i/\partial z_j\|_F^2$ — the object Lemma C forces to
zero, evaluated rather than proxied) and cross-check with
`metrics.distance_correlation_block_report` (model-free, no derivatives). They
answer the same question by different routes; **disagreement is a signal**, and
it was: on `exp11` the raw Jacobian energies differed by $3\times10^3$ between
two blocks that dCor scored $0.96$ and $0.99$.

Three traps inside the fix, all live:

1. **Standardise.** Raw Jacobian energy is not invariant under rescaling a
   block, but §7 grants exactly that freedom, so unstandardised energy measures
   the gauge. Contraction guarantees the disparity — a whitened encoder applies
   huge gain to a fast (low-variance) module and tiny gain to a slow one.
   `standardize=True` is the default; turn it off only to inspect gains.
2. **Match, do not pin, when the fit may permute.** Nothing makes a fitted
   block 0 correspond to true block 0. `exp11`'s fit permutes: raw dCor
   $[[0.26, 0.96], [0.99, 0.28]]$. Pinning identity read that *correct* recovery
   as `on_block` $=0.000$ — a total false negative. Conversely the max-energy
   matching *inverts* when off-block coupling exceeds on-block, reporting a
   maximally triangular map as block-diagonal. Pass `assignment` when the
   correspondence is known, else read the raw `coupling` matrix and check it.
3. **Block-diagonal needs the *lower* mass too.** Testing only `upper` cannot
   detect a lower-triangular map, so every such fit reports as block-diagonal —
   which silently deletes the triangular/block-diagonal distinction. `exp11`'s
   misaligned control has u-dependence $0.0022$ (looks clean) and lower mass
   $0.121$ (is not): the Jacobian catches an $M_{BA}$ leak the behavioural test
   misses, because the block tracks the non-$u$-driven part of $z_A$.

> Regression tests in `tests/test_metrics_and_models.py` (§"Nonlinear block
> structure"), including the known-answer triangular family and the
> rescaling-invariance check. `dCor` is biased upward in finite samples — use
> `distance_correlation_baseline` and compare against it, never against zero.

### 3.11 A fitted structural readout is a distribution, not a number

**Third measurement defect, same family as §3.9 and §3.10.** Any claim read off
a *fitted* model inherits the optimiser's randomness, and for block structure
that randomness is not small.

**Selecting the restart by fit quality does not control the structure.**
Measured over 8 restarts: $\mathrm{corr}(\texttt{fitq}, \texttt{jac\_diag}) =
-0.044$ (linear decoder), $+0.279$ (nonlinear). Essentially no information —
and the nonlinear sign means best-fit mildly *anti*-selects for diagonality.
`fitq` spread was $2.2$–$2.6\times$ across restarts while it explained none of
the structural variance. A best-of-$N$ point estimate is close to a coin flip.

**How large the spread is depends on the regime, so measure it, do not assume
it.** Linear decoder: `jac_diag` $\in[0.988,0.999]$, sd $0.004$ — a point
estimate is fine. Nonlinear decoder: $[0.562,0.900]$, sd $0.104$ — **26× worse**,
and any single number is meaningless. The same experiment, the same metric, the
same seed: only the observation map changed.

**Undertraining fakes a structural result, and fakes it in the wrong
direction.** At `STEPS=1200` (tuned for the linear decoder) the nonlinear run
reported forbidden mass $0.177$; at 3000+ it collapses to $\approx0.05$ and
stays. Read literally, the undertrained number said *behaviour-only beats
full B∘C at suppressing the forbidden cross-block* — backwards from the theory
and briefly reported as such. Sweep the budget before believing any structural
number off a new regime.

> **Do:** report median + range over $\ge 8$ restarts, and state the sd. Assert
> only what survives every restart — the nonlinear-decoder conclusion rests on
> `upper` $\le 0.082$ in *all* 8, not on any median. **Do not** report best-of-$N$
> for a fitted structural quantity; **do not** carry a step count across a change
> of observation model.

**Corollary: a treatment arm is uninterpretable without a control the metric can
pass.** `exp11`'s nonlinear-decoder numbers looked equally consistent with "B∘C
degrades to triangular" and with "the measurement is degenerate" — the forbidden
`upper` mass was near zero in *every* condition including the negative control,
which reads like a metric that cannot see anything. What settled it was `exp12`'s
strength-0 arm: same system, same horizon, same metric, and it returns
`jac_diag` $0.994$ with lower **exactly** $0.000$. A degenerate measurement
cannot produce that, so the degradation is real. Build the arm that *should*
score perfectly into the sweep; without it, a null is unattributable.

**And check the dose was actually delivered.** `exp12` run 1 swept `strength`
$(0,0.25,0.5,1.0)$ and got observation nonlinearity $(0.00,0.31,0.31,0.35)$ —
two distinct levels, so its monotonicity test failed for want of a treatment
rather than for want of an effect. Cause: contraction pulls latents toward the
origin, where `tanh` is nearly linear, so a decoder's *delivered* nonlinearity on
trajectory data is well below its value on spread data ($0.35$ vs $0.46$ at
strength $1.0$). Sweep the measured dose, never the parameter that nominally
controls it; `exp12` now prints the delivered doses and warns when their range is
too narrow to support a trend.

**Related design tension, not yet resolved.** Lemma C *needs* a spectral gap, and
a gap makes the faster module decay exponentially: at $s_B=0.5$ over $T=15$ its
variance runs $0.985 \to 9.6\times10^{-4}\ (t{=}5) \to 9.2\times10^{-10}\ (t{=}15)$,
so $12$ of $16$ timesteps carry no block-B signal and the fitted latents are
driven overwhelmingly by block A (Jacobian column mass $0.89$–$0.99$ on the slow
block in every condition). This is `approaches.md` cost 2, quantified. It does not
invalidate the readings above — the strength-0 control proves that — but it caps
how much can be asked of this system, and any *finer* block claim should probe
$h$ at early timesteps, where the A/B variance ratio is $1.3\times$ rather than
$4.8\times$. Legitimate under §3.8: $t=0$ points are in the visited region.

**This is also why the learned per-module spectra are the least trustworthy
number in `exp13`** (§3.12's candidate (ii)): the dominated block's transition is
fit on data that carries almost no block-B signal after a few steps, so its
learned Lyapunov exponents are weakly constrained. They are *horizon*-stable
(unchanged from $n=25$ to $n=300$), which rules out an extrapolation artifact —
but horizon-stable is not the same as data-constrained, and they disagree with
the true exponents even for fits whose block structure is essentially perfect.

### 3.12 A structural *penalty* can be gauge-dependent too — and then the optimiser games it (BLOCKING for exp11/exp12)

**Fourth defect of the §3.9 family, and the first one in the objective rather
than the readout.** §3.10 caught a metric that measured the gauge; this is the
same error one level down, where it is worse, because a training loss does not
merely misreport — it *steers*.

`models._behavioural_penalty` scored the between-$u$ spread of the pinned
block's first two conditional moments on the **raw** block. Under $w \mapsto
\varepsilon w$ the mean term falls like $\varepsilon^2$ and the covariance term
like $\varepsilon^4$. So "make the block $u$-invariant" and "make the block
small" are the same instruction, and the second is far cheaper.

**The fit took the cheap one, in every arm.** At `w_behavior = 5.0` the fitted
"invariant" block came out **21× smaller** than its partner, scored a raw
$u$-dependence of $0.0015$ — and still carried the $u$-varying latent at
**distance correlation 0.99**. Rescaled to unit variance its $u$-dependence is
**1.07**, against $0.15$ for a genuinely invariant block and $1.09$ for the true
*varying* one. The block the penalty was supposed to purify was
indistinguishable from the thing it was supposed to exclude.

**Consequence, and it is not small: exp11's and exp12's behavioural conclusions
are void.** Not mismeasured — *unimposed*. Lemma D's hypothesis is that
$p(h_B \mid u)$ does not move with $u$; no arm of either experiment ever
satisfied it. So neither experiment tested B∘C, and neither is evidence that
"behaviour fails to supply its zero once $h$ is nonlinear". The dose-response in
`exp12` is real (§3.11's strength-0 control proves the metric works) but it
measures *dynamics-only* fitting under a decoy penalty. **This is the answer to
task 32**: the discrepancy between Lemma D and `exp12` is not that the fit
breaks a hypothesis, it is that the experiment never applied one.

**Fix:** whiten the block by its own pooled covariance before scoring. The
penalty is then invariant under all of $GL(d_b)$ — exactly the freedom §7 grants
within a module — so it can only be paid with genuine distributional invariance.
`TrainConfig.behavior_whiten` defaults to `True`; `False` reproduces the old
runs. Measured effect at matched weight ($w=5$): pinned-block scale-normalised
$u$-dependence $1.07 \to 0.037$, block scale ratio $21.4\times \to 1.4\times$.
At the recalibrated $w=1$ it lands at $0.08$–$0.11$ — still under the $0.155$
floor, i.e. the constraint is satisfied, without paying $w=5$'s distortion.

**Do:** for anything read off a fit, use `behavior.block_u_dependence(...,
normalize=True)`. **Do not** compare raw $u$-dependence across blocks whose
scales you have not checked.

**Blast radius is bounded — `exp10` is unaffected, and this was checked, not
assumed.** Its candidates are analytic maps $h_B = z^B + \varepsilon z^A$ at
comparable scale (output std $1.00 \to 1.14$ across the whole sweep), so the raw
detector was never being asked the question it gets wrong. Re-run with
`normalize=True` the net $u$-dependence is monotone in $\varepsilon$ with the
same shape ($0.000, -0.001, 0.001, 0.028, 0.079, 0.215$ against raw $0.000,
-0.001, 0.001, 0.031, 0.091, 0.287$). The B∘C *mechanism* result stands; only
the *learned* results (`exp11`, `exp12`) are void.

**And recalibrate the weight after the fix.** The whitened penalty is $O(1)$
where the raw one was $O(\text{scale}^{2..4})$, so `w_behavior = 5.0` is
effectively a much larger weight now — at that value `fit_quality` degrades
$8.5\times$. Same lesson as §3.11's step-count warning: **a loss weight does not
survive a change of penalty definition**, any more than a step count survives a
change of observation model.

> Regression tests in `tests/test_behavior.py` (§"the behavioural constraint is
> a GAUGE quantity"): the shrinking exploit, $GL(d_b)$-invariance of the fix, and
> that the fix is neither vacuous (a genuine leak still scores) nor unsatisfiable
> (a genuinely invariant block still scores ~0).

**Scoreboard on the other three candidates — all three were poor bets, and the
one I ranked first was the worst.** `exp13` measures each per restart and
correlates it with `jac_diag` *within arm* (dose partialled out, so a mere
co-symptom of nonlinearity cannot score):

| candidate | rises with dose? | within-arm corr with `jac_diag` |
|---|---|---|
| (i) conjugacy residual $\lVert h\circ F-\tilde F\circ h\rVert$ | yes, $0.23\to0.39$ | **$+0.05$ — nothing** |
| (iii) additivity defect of $h_B$ | yes, $0.001\to0.122$ | $-0.17$ |
| (ii) learned one-sided gap | — | $+0.34$, and the sign Lemma C predicts |

I called (i) "the decisive cheap experiment". It is the *least* informative of
the three: it tracks the dose cleanly and carries essentially zero information
about whether any particular fit recovered the structure. **A diagnostic that
correlates with the treatment is not thereby a mechanism** — that is what
partialling out the arm is for, and it should be the default for any claim of
the form "X explains the degradation".

**(ii) is genuinely live, with a caveat that cannot be waved off.** The learned
one-sided gap is *negative* at 3 of 4 doses (medians $+0.09, -0.33, -0.10,
-0.22$; min $-1.17$) against a true gap of $+0.588$ — so Lemma C's hypothesis
does **not** hold in the fitted transition even where the recovered structure is
good. But per §3.11's design tension these are the least well-determined numbers
in the repo: the dominated block's transition is fit on data carrying almost no
block-B signal after a few steps. Treat as "unresolved and probably
unmeasurable in this system", not as "Lemma C fails".

---

## 4. Resources

### 4.1 Environment

- **Python:** conda env `torch` — `C:\Users\adene\miniconda3\envs\torch\python.exe`
  (Python 3.13.14). Use this interpreter explicitly; it is not on `PATH` by default.
- **Installed and verified (2026-08-03):** torch 2.13.0+**cpu**, numpy 2.5.1,
  scipy 1.18.0, matplotlib 3.11.1, scikit-learn 1.9.0, pytest 9.1.1.
- **Do not install packages** without asking. Everything here runs on the above.
- All experiments are small (d ≤ 16, T ≤ 10⁴); **CPU is the default and is
  sufficient**. `--device cuda` exists but is not needed and is not the tested path.
  This env has **no CUDA build** — that is deliberate, not a gap to fix.

> **History.** An earlier revision recorded `C:\Users\alexa\...` with
> torch 2.7.1+cu118 / numpy 2.4.3. That was a *different machine*; the repo was
> moved. If a path under `\Users\alexa\` appears anywhere, it is stale.
> Two tests were environment-fragile across the move and are now written against
> the invariant rather than its old symptom — see `tests/test_spectra_and_cocycle.py`
> `test_naive_sigma_min_route_is_noise_on_a_limit_cycle` (asserts the §3.9 defect's
> *magnitude*, since on numpy 2.5 the noise floor converges to the wrong answer
> instead of wandering to it).

Run anything with:

```bash
C:/Users/adene/miniconda3/envs/torch/python.exe -m pytest -q
```

### 4.2 Sibling project (prior art, same author)

`C:\AdenCode\IdentifiableCommunication` — implementation of Hälvä et al.,
*Disentangling Identifiable Features from Noisy Data with Structured Nonlinear
ICA* (arXiv 2106.09620). Independent-latent SLDS prior + nonlinear mixing +
structured VAE, with an identifiability check by correlation to true latents.

Relevance: despite the folder name, it is not about inter-area communication —
it is the **conditioning-on-temporal-structure** baseline that step 8 says we
must position against, and it remains the closest prior art under the narrowed
scope (§1.1). Its `src/identifiability.py` (MCC-style correlation to
ground-truth latents) is the right shape for our `metrics.py`, but our claim is
about the **partition/filtration**, not the coordinates — do not copy its metric
wholesale. See §7 scope note.

### 4.3 Literature — local copies

> **Not on this machine (verified 2026-08-03).** The table below describes the
> *previous* machine's `C:\Users\alexa\Downloads\`, which does not exist here —
> consistent with §4.1's warning that any `\Users\alexa\` path is stale. None of
> these files are present under `C:\Users\adene\Downloads\` either. Treat the
> table as a **reading list with provenance notes**, not as local paths: every
> claim sourced from them is already written up in `theory/literature.md` with
> its provenance tag, so nothing is blocked, but do not try to open them.

Formerly in `C:\Users\alexa\Downloads\`:

| File | Paper | Why it matters |
|---|---|---|
| `1603.06277v5.pdf` | **Johnson et al., structured VAE** (*not* TCL — see below) | Prior-art inference machinery; relevant to the sibling project |
| `2106.09620v2.pdf` + `NeurIPS-2021-disentangling-...-Paper.pdf` | Hälvä et al., structured nonlinear ICA (SNICA) | Independent latents + SLDS; closest existing result |
| `snicainference.pdf` | SNICA inference details | Inference machinery if we go variational |
| `mrlfads.pdf` | multi-region LFADS | **Out of scope (§1.1)** — modules here are dynamical factors within one area, not regions. Keep only as contrast |
| `vahidi-et-al-2024-...pdf` | Vahidi et al. 2024 | **Deferred (§1.1)** — the entry point *if* input drive is added later |

**Inventory correction (verified by opening the file).** `1603.06277v5.pdf` is
*not* Hyvärinen & Morioka TCL, as an earlier revision of this table claimed —
it is the structured-VAE paper (its text is about GMM SVAE and spiral cluster
data). TCL is arXiv **1605.06336** and is *not* held locally. The earlier entry
was written from recollection of an arXiv ID without opening the PDF; do not
repeat that. Every row above has now been checked against file contents.

Not held locally (needed for §8 positioning): TCL (1605.06336), Hyvärinen &
Morioka permutation-contrastive (PCL), Hälvä & Hyvärinen HMM-nonlinear-ICA,
Khemakhem et al. iVAE. All were read online for `theory/literature.md`; see the
provenance tags there for which claims rest on full text and which on abstracts.

### 4.4 Mathematical background the proofs lean on

- **Krull–Schmidt / primary decomposition** — uniqueness of the finest invariant
  splitting in the linear case (`theory/linear_case.md`).
- **Sylvester / Rosenblum theorem** — $AX - XB = 0 \Rightarrow X = 0$ when
  $\mathrm{spec}(A) \cap \mathrm{spec}(B) = \emptyset$. The linear base case.
- **Sacker–Sell spectral theory / exponential dichotomies** — the nonlinear
  replacement for eigenvalue separation (§3.4).
- **Oseledets / MET** — justifies Lyapunov spectra as a.e.-defined invariants.

---

## 5. Repo layout

```
IdentifiableDynamics/
├── CLAUDE.md                  # this file
├── README.md                  # short orientation + how to run
├── docs/brief_v0.md           # original brief, verbatim, for provenance
├── theory/
│   ├── identifiability.md     # corrected statement: Theorem A / Theorem B
│   ├── linear_case.md         # §4 step 2, proved
│   ├── counterexamples.md     # §3.1 regrouping, (A2), the §3.7 counterexample
│   └── literature.md          # step 8 positioning + recon; provenance-tagged
├── src/idyn/
│   ├── systems.py             # modular maps, oscillators, coupling, counterexample,
│   │                          #   LinearDecoder (Thm A) + MLPDecoder (Thm B, flow)
│   ├── linear.py              # finest invariant decomposition, block-permutation test
│   ├── spectra.py             # Lyapunov / dichotomy spectra, module gaps
│   ├── cocycle.py             # §3.3 iterated cocycle relation
│   ├── normalform.py          # Poincaré–Dulac: homological operator, resonances
│   ├── behavior.py            # Route B: u-conditioned sampling, invariant-subspace detector
│   ├── models.py              # torch: modular vs unconstrained latents, 2 decoders
│   ├── train.py               # fitting loop
│   ├── metrics.py             # partition + filtration recovery, non-uniqueness
│   └── selection.py           # partition-lattice search, fitted-model certification
├── experiments/               # exp01..exp13 + run_all.py; each writes to results/
│                              #   run_all covers exp01..exp10 only (see §2)
├── results/                   # generated JSON records, one per experiment
└── tests/                     # pytest
```

---

## 6. To-dos

Ordered. Numbering matches the original §4 next-steps list; the tracker IDs are
in brackets. **Numerical falsification (6) gates further theory** — if the
negative control comes back unique, stop and re-examine assumptions.

| # | Task | Status |
|---|---|---|
| 0a | Integrated CLAUDE.md + resource inventory `[1]` | done |
| 0b | Scaffold package, tests, env check `[2]` | done |
| 0c | Modular systems library `[3]` | done |
| 1 | **Restate theorem with indecomposability**; prove uniqueness of the finest modular decomposition | done for linear (`linear_case.md` Thm L(i)); nonlinear (B2) = `TODO(gap)` |
| 2 | **Settle the linear case completely.** With $h \in GL(d)$, when does $AFA^{-1}$ block-diagonal force $A$ block-permutation? | **done.** Iff (A1) blocks indecomposable + (A2) disjoint spectra; both necessary. Matching block dimensions come out as a *conclusion*, not a hypothesis |
| 3 | **Swap in dichotomy-spectrum separation** (§3.4) and redo the Section 5 proof as a cocycle argument (§3.3) | done; measured rate matches theory to $1.5\times10^{-12}$ — **but see §3.7** |
| 4 | **Prove the matching lemma** closing the $\tilde f$ vs $f$ gap (§3.2) | linear case **done** (`linear_case.md` §5, Cor. M), and it establishes $\sigma$ *before* the spectral hypothesis is used, exactly as §3.2 demands. Nonlinear **open** |
| 5 | **Move to a nonlinear decoder** $x_t = g(z_t)$, $g$ injective immersion | **model *and now data* both built.** `ModelConfig(decoder="mlp")` is the fitted side; `systems.MLPDecoder` (task 28) is the generating side, which did not exist before 2026-08-03 — so every earlier "nonlinear decoder" result was a *linear* observation map read through a nonlinear encoder. Theory still blocked by §3.7 |
| 6 | **Numerical falsification before more theory.** **§3.1 regrouping is the negative control — the fit must be non-unique there** | **done, and the gate held.** Negative control: all 3 groupings exact to $2.2\times10^{-16}$, fitting finds several. Positive control: 5/5 converged restarts recover the partition, on-block $0.9647$ vs chance $0.5$ |
| 7 | **Perturbation result.** $\epsilon$-coupling; target "recovered partition within $O(\epsilon)$ provided gap $> C\epsilon$" | **linear case confirmed and sharp**: log-log slope $1.0000$, gap-independent constant, breakdown at $\epsilon \approx \mathrm{sep}$. Nonlinear theorem open |
| 8 | **Position against the literature** (Hyvärinen & Morioka, Hälvä & Hyvärinen, Khemakhem et al.). Be explicit about what modular *dynamics* adds beyond conditioning on a time index | drafted in `theory/literature.md` §1; needs write-up into `identifiability.md` §8. **CORRECTION (2026-08-04):** an earlier revision claimed "those results all need non-stationarity or an auxiliary variable, and we have neither — autonomous + stationary is exactly the uncovered case". **That is wrong, and it conflates an autonomous *system* with a stationary *process*.** The dynamics are time-invariant, but the process $\{z_t\}$ is *not* stationary unless $z_0$ is drawn from the invariant measure — and `make_dataset` deliberately spreads initial conditions over an annulus, so every dataset in this repo is non-stationary by construction. Measured: with $u=t$ the blocks' scale-normalised $t$-dependence is $1.32$ and $4.20$, i.e. strongly non-stationary. The positioning has to be rewritten around *what kind* of non-stationarity, not its absence — see task 35 |
| **9** | **NEW: resolve the two-sided cocycle obstruction (§3.7)** | **resolved negatively.** The conclusion is *false* under (B1)–(B4), not merely unprovable — counterexample in `counterexamples.md` §5. Both candidate routes are dead |
| **10** | **Route A — diagonality under $C^\infty$ + cross-module non-resonance** | open; proof plan in `literature.md` §2.2. **Not dead** — the §5 counterexample needs resonance, a measure-zero condition |
| **11** | **Route C — filtration identifiability.** Assemble `identifiability.md` §4.2 into a standalone theorem | open; **essentially proved already**, needs writing up |
| **12** | **Route B — hybrid: behavioural auxiliary + one-sided gap** | open; mechanism verified numerically. Needs a *partial* iVAE theorem |
| **13** | **Learn an indecomposable model** — certify the fitted model, then search the partition lattice | **largely done** (`selection.py`, `exp06`): fitted-model certification + lattice search recover the finest partition from data. Finding: fit rejects splitting an indecomposable block, uniqueness breaks ties among equal-fit regroupings — each covers the other's blind spot. The linearised certifier's Tier 2 false negative is now fixed at the fixed point (task 25, degree-2 jet). Off-fixed-point (genuine-attractor) indecomposability still open |
| **22** | **Does Lemma C extend to genuine attractors?** (the filtration off the fixed point) | **resolved positively** for *periodic* attractors — Lemma C′, `identifiability.md` §4.4, certified in `exp08` (rate exact to 2.4e-14, threshold exact at $\|1-2a\|$; uniformity over the basin measured to 2.8e-16 across a 100× radius range). Uses only (B1)'s bounded-derivative clause — **not** $\mathrm{int}\,\Omega\neq\emptyset$, not (B2), not (B3). Prerequisite finding: the naive $\sigma_{\min}$ bound is unusable here (§3.9). **Open:** attractors with non-uniform exponents (chaotic) |
| **23** | **NEW, raised by 22: two oscillatory modules cannot be separated spectrally** | open. A limit cycle's neutral exponent 0 makes it undominatable, so an oscillatory module is always the **top** of the filtration and a filtration holds **at most one**. This collides with §1.1's own definition of a module ("distinct oscillatory components") and frequency does not help — Lyapunov exponents are blind to rotation number. Needs a finer conjugacy invariant (rotation number), which is a different argument. See `identifiability.md` §4.4. **Candidate answer found 2026-08-04: task 35 (Route B′, $u=t$).** Conditioning on the time index is sensitive to rotation number by construction. Measured: two limit cycles with identical spectra $\{0,-0.9163\}$ — where Lemma C has no gap to use — are cleanly separated by their $u=t$ phase signature when $\omega$ differs ($1.571$ rad) and not when it does not ($0.006$). This is the first mechanism in the repo that can see past the neutral exponent |
| **24** | **Route A: is Tier 2 non-empty?** (does the nonlinear claim have content, or collapse to Theorem A?) | **resolved positively.** Witness `systems.tier2_witness()`, machinery `src/idyn/normalform.py`, certified in `exp09` (5/5). $(\mu z_a,\ \mu^2 z_b + c z_a^2)$: the resonance $\lambda_b-\lambda_a^2=\mu^2-\mu^2$ vanishes *identically in $\mu$*, so it is structural; $c$ survives as a normal-form invariant while cross-module non-resonance holds. Honest limit: $c$ rescales to 1, so the invariant is binary (linearisable or not); a continuous modulus needs two resonant coefficients |
| **25** | **NEW, raised by 24: the linearised (B2) test has a false negative** | **fixed at degree 2** (`selection.block_nonlinear_certificate`, `normalform.coupling_resonances`/`quadratic_jet_coefficients`, `certify_fitted_model(nonlinear=True)`, `exp09` part 5). Reads the quadratic jet in the eigenbasis and flags a *resonant* cross-eigendirection monomial; correctly ignores *non-resonant* cross terms (removable) and is coordinate-invariant. Opt-in, so linear-only callers unchanged. **Open:** higher-degree resonances (need the coefficient post-normalisation) and near-resonances in *fitted* models (eigenvalues carry fit error — `res_tol`/`coeff_tol` knobs exposed) |
| **26** | **NEW: the (B2) criterion is graph *connectedness***, and it is proved at degree 2 | **done** (`route_a_assessment.md` §4.1a, `normalform.resonance_coupling_components`, `exp09` part 6). A module is indecomposable iff its resonance-coupling graph is connected; proved both directions at degree 2 for distinct eigenvalues (degree-2 resonant coefficients are normal-form invariants — homological operator vanishes on them). Fixed the task-25 certifier, which flagged *any* coupling and so **over-reported** indecomposability with ≥3 sub-blocks (the direction the fit can't catch). **Open:** connectedness-invariance at higher degree; repeated-eigenvalue sub-blocks |
| **27** | **NEW (active): Route B∘C** — behaviour + one-sided gap = block-diagonal | **mechanism built + verified** (`behavior.py`, `exp10`, `approaches.md` §B.1). Behaviour kills $M_{BA}$ (canonical invariant subspace), Lemma C kills $M_{AB}$ (gap); together block-diagonal, inheriting C's attractor reach. Found the **alignment condition**: block-diagonal needs varying block = spectrally dominant, else only triangular. Prop-1 caveat operational (variance vs mean modulation). **The one open obligation: the partial-iVAE lemma** — $u$-invariant complement identified as a subspace; not a corollary of Khemakhem et al. (§6.1). This is the B∘C front line |
| **28** | **NEW: nonlinear data decoder** — the Theorem B observation model had never been run | **done.** `systems.MLPDecoder`, an affine coupling flow: invertible in closed form for arbitrary $s,t$, analytic (tanh/exp, which §3.7 needs), ~45% nonlinear residual, `strength=0` is an exact linear control. Threaded through `make_dataset`/`make_behavioural_dataset`. **Discarded first attempt** — the contractive $z+\epsilon m(z)$ construction caps at **3%** nonlinearity even at $\epsilon\,\mathrm{Lip}=0.99$ and *worsens* with depth, because $\mathrm{Lip}\le\prod\|W_k\|_2$ is a worst case while tanh is far gentler; a decoder within 3% of linear does not test Theorem B. See §3.11 |
| **29** | **does *learning* recover B∘C under a nonlinear observation map?** | **ANSWER RETRACTED (§3.12).** The sweep is sound as a measurement but tested *dynamics-only* fitting: no arm imposed the behavioural constraint, so it never bore on B∘C. Re-asked properly as task 33. What stands unchanged: the dose-response, the strength-0 control, and the two methodological lessons (read `obs-nl` not `strength`; report distributions). Original entry, for provenance: ~~**DONE — answer is no, with a confirmed dose-response.**~~ `exp12` run 2, doses $(0.00,0.31,0.43,0.60)$: `jac_diag` falls **monotonically** $0.994\to0.730\to0.702\to0.567$; forbidden `upper` violated in $0/8$ restarts at low dose, $6/8$ at $0.43$, **$8/8$ at $0.60$** — a threshold, not a tail. Not fit failure: $\mathrm{corr}(\texttt{fitq},\texttt{upper})=-0.58$ at the top dose, so *better* fits are more coupled. The strength-0 control ($0.994$, sd $0.009$, lower exactly $0.000$) is what makes it readable. **Run 1 failed for want of a treatment, not an effect** — strengths $(0,0.25,0.5,1.0)$ deliver doses $(0.00,0.31,0.31,0.35)$, two levels not four, because latents contract toward the origin where tanh is near-linear: **always read `obs-nl`, never `strength`**. **Scope, and it is the whole point: this bounds LEARNING, not identifiability** — a fitted $h$ need not satisfy Lemma C/D's hypotheses (exact conjugacy, additive form, gap in the *learned* spectra). With task 31 proved, the honest reading is a **learning** gap. Successor is task 32 |
| **31** | **NEW: Lemma D — behaviour kills $M_{BA}$** (`identifiability.md` §4.5) | **PROVED** for additive $h_B$ + linear modules; witness `systems.lemma_d_witness`, 5 tests. Replaces the partial-iVAE obligation of task 27 with a dynamical argument. **Reconciliation with task 29 superseded**: the apparent conflict was not a learning gap at all — `exp12` never imposed Lemma D's behavioural hypothesis (§3.12, task 32), so there was nothing to reconcile. **Open:** (a) non-additive $h_B$ — the graded reduction survives but Step 4's characteristic-function factorisation needs independence; (b) anisotropic variance modulation; (c) nonlinear modules (needs §5.3 normal forms) |
| **32** | **close the gap between Lemma D and `exp12`** | **RESOLVED — and not by any of the three candidates I listed.** The answer is **(iv): the experiment never imposed the behavioural hypothesis.** `models._behavioural_penalty` scored the pinned block's conditional moments on the *raw* block, so it falls like $\varepsilon^2/\varepsilon^4$ when the block shrinks; the optimiser satisfied it by making the block **21× smaller** than its partner, while that block still carried the $u$-varying latent at **dCor 0.99** (scale-normalised $u$-dependence **1.07**, against $0.15$ for a genuinely invariant block and $1.09$ for the true *varying* one). So the discrepancy was never a hypothesis the fit *breaks* — it is one the experiment never *applied*. Full write-up §3.12; fix is `TrainConfig.behavior_whiten=True` (default), which restores the constraint (u-dep $\to 0.037$ at matched weight, scale ratio $\to 1.4\times$). **Consequence: exp11 and exp12's behavioural conclusions are void**, and `approaches.md` §B.1's B column is *untested*, not refuted. Successor is task 33 |
| **33** | **does B∘C survive a nonlinear observation map once behaviour is *actually* imposed?** | **ANSWERED, PARTLY YES** (`exp13`, 8 restarts × 4 doses × 2 penalties). **(a) The forbidden direction is now uniformly killed.** `upper` $\le 0.081$ over *every* dose and *every* restart, against $0.316$ raw — Lemma C's half is clean once behaviour stops being paid off with scale. **(b) Block-diagonality improves sharply at the high doses**: `jac_diag` $0.702\to0.893$ at dose $0.43$ and $0.567\to0.815$ at $0.60$. **exp12's "degrades to a filtration" is refuted.** **(c) But it is not uniform, and the exception is sharp** — at dose $0.31$ the whitened fit lands *triangular in all 8 restarts* (`jac_diag` $0.546$, `lower` $0.416$, **sd $0.029$**). Deterministic, so not underpowering. So the defensible claim is: behaviour genuinely supplies its kill, and B∘C reaches block-diagonal at 3 of 4 doses — not at all of them. Successor is task 34 |
| **38** | **NEW: the "partial iVAE lemma" is PUBLISHED — reposition the behavioural half** | open, and it **changes what Route B can claim** (`literature.md` §1.3). The obligation task 27 called the B∘C front line — "$u$-invariant complement identified as a subspace, not a corollary of Khemakhem et al." — is the literature's **block-identifiability** (von Kügelgen et al. 2021, Def. 4.1: $\hat c = h(c)$ for invertible $h$ — verbatim our target and verbatim §7). Two theorems in *our* setting, an auxiliary variable indexing a law rather than paired views: **Kong et al. arXiv 2306.06510 Thm 4.2** and **Sun et al. arXiv 2208.14161 Prop. 4.2**, both giving the invariant block up to invertible transformation, both allowing invariant/varying dependence, and Kong's componentwise-monotonic domain action $z_s=f_u(\tilde z_s)$ **exactly matches our variance modulation** $z_A=s(u)\tilde z_A$. **Lemma D is not redundant** — it needs **two** behaviour levels against Kong's $2n_s+1$ and Sun's $2\ell+1$, and buys that economy with the dynamics (the gap forces coupling degree $\ge2$). That economy is now the defensible novelty of the behavioural half, and it should be stated that way rather than as "we proved a lemma nobody had". **Blocking check first:** Kong's A2 is *componentwise* conditional independence, which our within-module coordinates violate — read from summaries, A2/A3 look like they serve the finer $z_s$ conclusion while the block conclusion rests on A1 + A4 (a cylinder-set condition, not a factorisation), but **this was not read from the proof.** Verify before citing |
| **35** | **Route B′ — auxiliary variable = the time index $t$** | open, **promoted: it matches the target model class and it is free.** Autonomous LFADS is random $g_0$ + deterministic generator, so all trial-to-trial randomness is in the initial condition; at fixed $t$ across trials that is an **ensemble**, exactly what TCL/iVAE consume. Path-conditioning (PCL/SNICA) is excluded by the target itself, not by a modelling choice. Available because the process is non-stationary (task 8 correction). **Not a cheaper $u$ for Lemma D** — Lemma D needs one block *flat* in $u$, and time moves every block ($t$-dep $1.32$/$4.20$ vs behaviour's $1.14$/$\mathbf{0.14}$), so there is no $t$-invariant subspace. What it powers is the TCL/iVAE **variability** condition: per-module natural parameters go like $s_i^{-2t}$, linearly independent across modules **iff the contraction rates differ** — i.e. satisfied exactly when Lemma C's gap holds. Same hypothesis, second extraction. **It may answer task 23**, the route's sharpest limitation: $u=t$ sees the **rotation number**, which Lyapunov exponents provably cannot. Measured — two limit cycles, $\omega=0.5$ vs $1.3$, *identical* spectra $\{0,-0.9163\}$ so Lemma C is dead, yet phase signatures separate by $1.571$ rad; at $\omega_1=\omega_2=0.9$ separation collapses to $0.006$ (**equal rotation numbers are B′'s resonance analogue**). Three costs: **(i)** at a fixed point only $\approx9$–$10$ usable $t$ levels before the fast block underflows, against Khemakhem's $nk+1=9$ — right at the line; **(ii)** on a *cycle* the variability is **persistent**, not transient ($t$-dep $1.489$ early, $1.493$ late, coherence held $0.881$; uniform initial phase → $0.057$, correctly dead), so B′ is **stronger off the fixed point than on it** — I first claimed the opposite and was corrected by measurement. Caveat: coherence held exactly because the model is noiseless with $\beta=0$; real dephasing sets a finite horizon, longer than (i)'s but not infinite; **(iii)** cost 3 — if TCL-style identification works, modularity may do no work. Defence: within-module coordinates are dependent, so plain TCL does not apply and a block version is needed |
| **36** | **add process noise to the modules — but it is a CHANGE OF TARGET, not a fidelity fix** | open, and **demoted 2026-08-04 after checking LFADS.** I originally justified this as "determinism was chosen for convenience, not the science — LFADS latents are stochastic." **That justification is wrong.** In an autonomous LFADS model one samples $g_0$ from the prior and then simulates a **deterministic** RNN forward; the only per-timestep stochasticity is the *inferred inputs* $u_t$, which §1.1 scopes out. So under this project's own scope, LFADS is exactly **random initial condition + deterministic flow** — which is precisely what `make_dataset` builds. Determinism here is *faithful to the target*. What remains true: determinism is what makes PCL/SNICA inapplicable (`literature.md` §1.2 point 1 — the pair $(z_t,z_{t-1})$ lives on the graph of $f_i$), and adding noise would open that family, close §3.8's distributional-equivalence `TODO(gap)`, and make the sibling SNICA implementation reusable. But that is **moving to a model class where identifiability is easier**, not modelling LFADS more accurately, and it should be argued on those terms. Note the link to §3.8: if input drive $u_t$ is ever brought back in scope, LFADS's stochastic inputs *would* supply a per-timestep stochastic drive, and this task merges with that one |
| **37** | **NEW: re-target from block-diagonality to dynamical invariants** | open. §7 already concedes that what is identified is the **partition plus each $f_i$'s conjugacy class**, never coordinates. The applied goal may need strictly less than block-diagonality of $h$: the number of modules, their dimensions, and per-module invariants (Lyapunov spectrum, rotation number, attractor topology). That reads as *"this population carries a slow 2-D rotation at 8 Hz and a fast 3-D decaying component"* — usable by a neuroscientist, and testable. Lemma C (filtration) plus task 35 (rotation number) may deliver it **without** ever proving $h$ block-diagonal, i.e. without needing tasks 33/34 to close |
| **34** | **NEW (ACTIVE): why does the dose-$0.31$ arm land triangular in every restart?** | open — **the current front line.** It is the one arm where imposing behaviour *lowers* `jac_diag` (whitened $0.546$ vs raw $0.730$), and the tightness (sd $0.025$/$0.029$ over 8 restarts) rules out optimiser noise: it is a property of the objective at that dose, not of the seed. Leading hypothesis, and it is my error to check first: **`W_WHITENED = 1.0` was calibrated on the endpoint doses only** ($0.00$ and $0.60$), so the interior was never tuned — sweep $w$ at `strength=0.5` before looking for anything deeper. Note the decoders form a *one-parameter family* (same rng draw, scaled), so this is not a decoder-draw artifact. Second hypothesis: `obs-nl` is non-monotone in `strength` in a way that makes $0.31$ special beyond its dose |
| **30** | **NEW: `exp11` methodology — report distributions, not best-of-$N$** | open, unambiguous (§3.11). Selection by `fit_quality` carries no structural information ($\mathrm{corr}=-0.044$ / $+0.279$); with sd $0.104$ the current point estimate is near a coin flip. Also raise `STEPS` — 1200 was tuned for the linear decoder and undertrains the nonlinear one into a *reversed* result. Do this before re-running exp11 as a gate |

### Which route

[`theory/approaches.md`](theory/approaches.md) is the decision document — the
comparison table, what each route claims and costs, and what single fact would
settle it. The trade in one line each:

- **A** (normal forms) — the original prize; two unproved lemmas, and it
  structurally cannot leave the fixed point (limit cycles are Siegel).
- **B** (behavioural auxiliary) — supplies the cross-derivative the gap provably
  cannot; the conclusion is now **published** as block-identifiability (task 38),
  so it is a citation, not an obligation.
- **C** (filtration) — proved, unconditionally nonlinear, reaches periodic
  attractors, immune to the learning gap. Weaker: it commits to a hierarchy.

C is a strict weakening of A; B composes with either.

**DECISION (2026-08-04, under §1.0).** Priority is **C + task 37**: the
filtration, plus per-module invariants, as the applied claim. Rationale, which
changed this session:

1. **C is what is left that is both proved and unduplicated.** Task 38 moved B's
   conclusion into the literature; §1.0 says that is a *gain* (cite it), but it
   means B is no longer where our own content sits. Lemma C has no ICA analogue.
2. **C covers oscillations and A does not** (task 22 — Lemma C′ holds at
   attracting periodic orbits). Oscillations are the least exotic thing in a
   neural population, so this is decisive for the applied goal, not a tiebreak.
3. **Task 37 needs strictly less than block-diagonality.** Lemma C gives the
   ordering; task 35 ($u=t$) gives rotation number, which Lyapunov exponents
   provably cannot see and which may settle task 23. Neither needs $h$
   block-diagonal, so tasks 33/34 stop being on the critical path.

**Route B status.** Its two halves are both discharged in restricted settings —
Lemma C (dynamics) and Lemma D (behaviour, additive $h_B$ + linear modules).
Under learning, `exp13` kills the forbidden direction uniformly and reaches
block-diagonal at 3 of 4 doses (task 33); the exception is task 34, now
demoted. Lemma D's residual value is its **two-level economy** over the
published results (task 38), not the conclusion itself.

**Route A is parked** mid-push: the fixed-point theorem is essentially
assembled; open are higher-degree (B2), the FLAT-D $C^k$ write-up, and the
Siegel extension it cannot reach.

> Historical note, to stop it being re-litigated: `exp11`/`exp12` once reported
> that B∘C degrades to a filtration under a nonlinear observation map. That was
> withdrawn — those runs never imposed the behavioural hypothesis (§3.12). The
> pivot to B∘C (2026-07-29) and its apparent refutation are both superseded by
> the decision above.

Route A's remaining cost is now small and named (see `route_a_assessment.md`):
1. **(FLAT-D) flat-tangency residual** — the existence half is located and
   source-verified (Chaperon 1986, Thm 2(i)); only the flat-tangency clause of
   the conjugacy remains, reduced to a one-page check against Chaperon's
   construction. **This is the current front line.**
2. **(B3) nonlinear matching lemma** — `literature.md` §3.3 argues it does not
   need complete conjugacy invariants; the orbit-separation-rate / filtration
   argument establishes the ordered correspondence before any spectral hypothesis.
3. **Assemble Tier 2** into a standalone theorem once 1 and 2 land.

Step 13 (learning an indecomposable model) is **done** for the linearisable
regime (`selection.py`, `exp06`); only the off-fixed-point nonlinear test is open.

Then step 4 (nonlinear matching lemma — note `literature.md` §3.3 argues it does
*not* need complete conjugacy invariants, which sidesteps the obstructions in
§3.2 there), then step 8 write-up.

Note that step 7's linear half came out *supporting* the interpretive claims
rather than evaporating: the $O(\epsilon/\mathrm{gap})$ scaling is real and
sharp. The concern recorded in the original brief — that an exact theorem might
not survive $\epsilon > 0$ — does not apply to the linear case.

---

## 7. Scope note for interpretation claims

Even on success, within-module $h_i$ is an arbitrary diffeomorphism. What is
identified is the **partition** plus each $f_i$'s **conjugacy class** — fixed-point
structure, attractor topology, Lyapunov spectrum. Not coordinates. Nothing here
licenses reading "motor primitive" off a latent axis. Keep claims in the draft
calibrated to this.

Given §3.7, the currently defensible claim is weaker still: an **ordered
filtration** of dynamical factors, not a symmetric partition. Under the §1.1
scope this is the natural object anyway — within one population it reads as "a
slow autonomous component and a faster component driven by it", which is a
testable claim about timescale structure. It is *not* a claim that the factors
correspond to anatomically or functionally labelled subpopulations.

This is why `metrics.py` reports partition-level quantities first and MCC second.
A high MCC with a wrong partition is a **failure**, not a partial success.

---

## 8. Conventions

- $K$ = number of modules; $d_i$ = dimension of module $i$; $d = \sum_i d_i$.
- $F$ = full latent transition; $f_i$ = module transition; $W$ = linear decoder.
- $h$ = candidate reparameterization; $\tilde{\cdot}$ denotes the alternative
  representation.
- $M := \partial h_1 / \partial z_2$ throughout the cross-derivative argument.
- Flag any claim that depends on an unproved lemma inline as `TODO(gap)`.

### Code conventions

- Partitions are `list[int]` of block dimensions, e.g. `[2, 2]`, always summing to `d`.
- Everything numerical takes an explicit `rng: np.random.Generator` or `seed: int`.
  No global seeding, no hidden state — experiments must be exactly reproducible.
- `float64` for all linear-algebra / spectrum code. `float32` only inside torch models.
- Experiments write a JSON record to `results/<name>.json` including the seed and
  every parameter. The JSON is the artifact; console output is a summary.
  No figures are generated yet — add them only if a number is hard to read as text.
- A numerical result that *contradicts* a theory claim gets committed and reported,
  not tuned away. That is the entire point of step 6.
