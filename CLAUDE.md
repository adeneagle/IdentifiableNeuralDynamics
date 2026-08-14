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

### 1.2 Two tiers of claim — and Tier 1 is free (2026-08-04)

**Tier 1 — the global conjugacy class, free.** If two models both have
*injective decoders* and reproduce the observations on the visited region, then
$\hat z = \tilde g^{-1}\!\circ g(z) =: h(z)$ with $h$ invertible, hence

$$\hat F = h \circ F \circ h^{-1}.$$

The fitted transition is **automatically conjugate to the true one**, so every
conjugacy invariant of the whole system is identified with *no theorem and no
auxiliary variable*: latent dimension, number and stability type of fixed
points, global Lyapunov spectrum, rotation number, attractor topology. This is
§3.5 stated for a general injective decoder and read as a **result** rather than
as an obstruction. Caveats are ordinary: correct latent dimension, exact fit,
visited region only (§3.8). Real data gives *approximate* conjugacy and the
perturbation statement is sharp only in the linear case (task 7).

**Tier 2 — the decomposition, not free.** Tier 1 gives nothing that is not a
conjugacy invariant: not coordinates, not axes, and **not the splitting into
parts** — $h$ may mix factors (§3.1 regrouping, §3.7 triangular). Route C
supplies the ordered filtration plus each factor's own invariants. Every open
obligation in this repo lives here.

**Why this matters for the write-up.** Tier 1 is correct, checkable, already
true of what is built, and not stated explicitly in the neuroscience literature.
State it first; it costs nothing and it frames what Tier 2 adds.

### 1.3 The identifiability content is dynamical, not distributional

The model class is **not** committed to a variational sequential autoencoder.
LFADS is one instantiation; BRAID/PSID/DPAD-style fits are another. Three
consequences, and they retire a line of work rather than opening one:

1. **iVAE/TCL machinery is prior-based, and we do not need it.** Those theorems
   derive identifiability from assumptions on $p(z\mid u)$ — they exist because
   the latent has a prior to constrain. A deterministic latent map with an
   injective decoder has no such prior; its identifiability content is conjugacy
   plus spectra. Route B′ (task 35, $u=t$) was an attempt to import the
   distributional theory, and the import is what failed — without the VAE it was
   never needed. Note *why* it fails, since the reason is instructive: iVAE needs
   conditionally independent **scalar** components given $u$, i.e. every latent
   its own 1-D system, which excludes rotation outright. Modularity does not —
   a 2-D rotation is a perfectly good module.
2. **Pathwise equivalence is the right notion.** §3.8 records that observation
   equivalence "should be distributional". For a deterministic generator it
   should not; that obligation dissolves rather than needing work.
3. **The argument touches only the decoder.** $h=\tilde g^{-1}\!\circ g$ needs
   both decoders injective; encoders never appear. Every claim here is therefore
   about the **model class**, not the fitting procedure, and covers a sequential
   VAE, a subspace-ID fit, or a plain regression equally. State that as a
   strength.

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
| **Lemma D′ — the gap was never needed** | `theory/identifiability.md` §4.5a, `systems.gapless_resonant_coupling` | **NEW (2026-08-12).** (D1)'s only job is excluding a degree-0 (scale-invariant) $\psi$, which needs merely **$1\notin\operatorname{spec}(\tilde f_B)$** — no gap, no ordering, no contraction. Step 3's $\lvert m\rvert\ge2$ is **not load-bearing**: Step 4's iteration runs at every degree $\ge1$. Witness: two modules at $s\,R(\omega)$, *identical* spectra (so `spectral_gap` $=0$, (F3) unordered, Lemma C dead, (D1) fails), degree-1 resonant $\psi=cI$, exact conjugacy to $6.7\times10^{-16}$ — and behaviour still kills it. **This reaches the two-oscillator case Theorem F provably cannot** (§6.5 rider 2), with a theorem rather than a measurement. For equal-rate oscillators the resonance is exactly $\omega_A=\pm\omega_B$: `sylvester_kernel_dim` is $2$ there and $\mathbf{0}$ otherwise, so either no coupling exists or two levels remove it. **Found while checking it: Step 4 assumes $\psi$ has a *single* degree, and Step 2 permits several. Pre-existing (it holds with (D1) too), previously unflagged** — **now closed, see the next row** |
| **Lemma D″ — several degrees at once** | `theory/identifiability.md` §4.5b, `systems.multidegree_resonant_coupling` / `two_level_tie_threshold` / `required_behaviour_levels` / `surviving_degree_bound` | **NEW (2026-08-14). Closes the `TODO(gap)` the row above opened.** Step 4's characteristic-function iteration is replaced by a **second-moment** argument: by (D3), $\operatorname{Var}\langle t,h_B\rangle = \operatorname{Var}\langle t,z_B\rangle + V_t(\sigma_u)$ with $V_t(\sigma)=\vec s(\sigma)^\top C\vec s(\sigma)$, $\vec s(\sigma)_p=\sigma^p$, $C$ the Gram matrix of the degrees' contributions. (D4) makes $V_t$ constant across levels; it is a polynomial supported on the **sumset** $P+P$ with **no constant term** (that is (D1′)'s entire job), so by Descartes $\lvert P+P\rvert+1$ levels force $V_t\equiv0$, hence $C=0$, hence $\psi\equiv0$. $P$ is finite from the spectra alone ($p\le\log\rho_{\min}(\tilde f_B)/\log\rho(f_A)$) — no gap. **My recorded conjecture ("$k$ degrees, $k+1$ levels, Vandermonde") was the right shape and the wrong count**: the Vandermonde at $k+1$ levels only kills *first* moments. **The two-level economy mostly survives** — (R2) opposite parities under symmetric $\mu_A$ give $L=2$; (R3) for $\lvert P\rvert=2$ the sharp criterion is a correlation threshold $\operatorname{corr}^2\ge D_{2p}D_{2q}/D_{p+q}^2$ lying in $[2\sqrt{pq}/(p+q),1)$, so hiding needs near-perfect anticorrelation. **Design rule that fell out: spread the behaviour levels** — the threshold rises to $1$ with separation ($0.9989$ at $\sigma_2/\sigma_1=20$). Honest limit: a tie defeats the *argument*, not the lemma — at the constructed tie the variance matches to $3\times10^{-17}$ while skewness moves $-0.267\to-1.770$, so (D4) still fails. `TODO(gap)` on whether two levels always suffice for (D4) itself |
| **Lemma D‴ — nonlinear modules, via Koopman eigenfunctions** | `theory/identifiability.md` §4.5c, `systems.koopman_coupling_witness` | **NEW (2026-08-14). Closes open item (c), and *not* through §5.3** — which matters, since §5.3 is Poincaré-domain only and would have forfeited exactly the reach Lemma D′ was proved to gain. **Half the item is vacuous:** additive $h_B$ *forces* $\tilde f_B$ affine (Step 1's split needs additivity; measured defect $1.17$ for $\tilde f_B = \mu w + 0.3w^2$ against $2.2\times10^{-16}$ affine), so nonlinear $\tilde f_B$ is outside the class, not a gap in it. **The other half needs no hypothesis on the dynamics at all:** Step 1 reads $\psi\circ f_A = \tilde B\psi$, which componentwise says **each component of $\psi$ is a Koopman eigenfunction of $f_A$** with eigenvalue in $\operatorname{spec}(\tilde B)$ — for linear $f_A$ these are the monomials $z^m$ at $\lambda_A^m$, i.e. Step 2 verbatim. So **Step 2 holds for arbitrary $f_A$: limit cycle, chaos, anything.** Lemma D‴ then kills $\psi$ with (D1″) $0$ is $\tilde f_B$'s only fixed point + (D2″) the level set has a **limit point**, the proof never touching $f_A$. **Ties to the torus work:** for a limit cycle $e^{i\Theta}$ (asymptotic phase) is a unimodular Koopman eigenfunction, verified to $2.0\times10^{-15}$ at $\beta=0,0.5,1.2$ — §7's regrouping was built from Koopman eigenfunctions all along, which is why shear never touched it. **The price is level richness and it is necessary:** $k$ eigenfunctions tie $k$ levels exactly ($5.6\times10^{-14}$ at $k{=}3$, $2.8\times10^{-15}$ at $k{=}4$, $\lVert c\rVert\ne0$), so no finite count independent of $\dim z_B$ can work — a Koopman eigenfunction of a nonlinear map is not homogeneous, so $V_t$ stops being a polynomial and there is nothing to count. **The trade: structure in the dynamics and richness in the behaviour are interchangeable.** Mild for the applied claim — a continuous behavioural covariate is *more* natural than two discrete conditions |
| **Lemma D — behavioural kill** | `theory/identifiability.md` §4.5, `systems.lemma_d_witness`, `tests/test_behavior.py` | **PROVED** for additive $h_B$ with linear modules. Behaviour kills $M_{BA}$, the cross-derivative §3.7 proves the gap can *never* reach. Mechanism: the conjugacy makes the coupling a semiconjugacy $\psi\circ f_A = \tilde f_B\circ\psi$; only *resonant* degrees survive; **the gap itself forces those degrees $\ge2$**; a degree-$p\ge2$ homogeneous $\psi$ scales as $\sigma^p$, so variance modulation detects it. **Two behaviour levels suffice** (vs iVAE's $nk+1$) — this is the partial-iVAE obligation discharged *dynamically*, sidestepping the assumption-(iv) obstruction rather than confronting it. Open: non-additive $h_B$, anisotropic modulation |
| **Two-sided cocycle obstruction** | `theory/counterexamples.md` §3, §5 | **Settled: the conclusion is FALSE — see §3.7** |
| Matching lemma (§3.2) | `theory/identifiability.md` §6 | `TODO(gap)` — open (proved for linear); route in `literature.md` §3.3 |
| Nonlinear decoder (Thm B) | `theory/identifiability.md` §5.3–5.4 | **Assembled** for the fixed-point regime: analytic decoders + non-resonance close it via Poincaré–Dulac + identity theorem (see §3.7). Non-fixed-point attractors open |
| Literature positioning (step 8) | `theory/literature.md` | Drafted, provenance-tagged; needs write-up into §8 |
| Numerical falsification | `experiments/` | exp01–exp10 pass, all JSONs current. **`exp11` and `exp12` intentionally report FAILING checks** — they encode predictions the *dynamics-only* regime refuted. Per §8 those are committed, not tuned away, so **neither is registered in `run_all.py`**, which covers exp01–exp10 only. Both are also **superseded in part by §3.12** and carry a banner saying so; read `exp13` instead for anything behavioural. Read their JSONs directly. **`exp14` is likewise unregistered, on cost (~25 fits) not outcome** — its parts 1–2 *are* the metric validation, run in seconds, and every claim in them is asserted in `tests/`, so the regression gate for the new machinery is the test suite |
| Learning the partition from data | `src/idyn/selection.py`, `exp06` | Lattice search + fitted-model certification; fit and uniqueness each cover the other's blind spot |
| **The behavioural penalty was a decoy** | **§3.12** (canonical), `models._behavioural_penalty`, `exp13` | **Found and fixed.** The penalty was gauge-dependent, so the optimiser paid it by shrinking the pinned block 21× rather than making it $u$-invariant. **No arm of `exp11`/`exp12` ever imposed Lemma D's behavioural hypothesis**, which voids both rows below. Fix: whiten the block → invariant under $GL(d_b)$, the §7 gauge group. Numbers and the four-defect pattern in §3.12 |
| **B∘C under learning — linear decoder** | `exp11` | **Void as a B∘C result (§3.12)** — behaviour was never imposed, so it is dynamics-only. Measurement itself is sound (`jac_diag` $\in[0.988,0.999]$, sd $0.004$). Doubly uninformative anyway: §3.5 forces $h\in GL(d)$ here |
| **B∘C under learning — nonlinear decoder** | `exp11`, `exp12` → superseded by `exp13` | **Headline retracted (§3.12), then reversed.** `exp12`'s monotone decline $0.994\to0.730\to0.702\to0.567$ is real but measures *dynamics-only* fitting. With the penalty fixed, `exp13` gets $0.815$ at the top dose and kills the forbidden `upper` to $\le0.081$ at **every** dose and restart (from $0.316$). Not uniform — see tasks 33/34 |
| **Nonlinear block-structure metrics** | `src/idyn/metrics.py` | `jacobian_block_report` (= $M_{ij}$ itself, standardised) + `distance_correlation_block_report` (model-free). The linear `filtration_report` is provably blind here — §3.10 |
| **The rotation number does not pin the splitting** | `theory/counterexamples.md` §7, `systems.torus_regrouping_counterexample`, `spectra.rotation_lattice_margin` | **Task 23 answered NO (2026-08-12).** Two cycles span an invariant torus, a conjugacy acts on $H_1=\mathbb{Z}^2$, and $h(z_1,z_2)=(z_1z_2/|z_2|,z_2)$ realises it exactly ($6.7\times10^{-16}$), moving $ho$ from $(0.0796,0.2069)$ to $(0.2865,0.2069)$. Only the $GL(2,\mathbb{Z})$ **orbit** is identified. Consistent with Theorem F, which never applied here. Shear in the donor block obstructs only the naive-angle form; the asymptotic phase restores it at every shear, so the ambiguity is intrinsic |
| **Rotation number** | `spectra.rotation_number`, `exp14` part 1 | **Built and exact.** The conjugacy invariant the Lyapunov spectrum provably cannot see (task 23): `LimitCycleBlock(a=0.3)` has spectrum $\{0,\log\|1-2a\|\}$ for *every* $\omega$. Machine-precision on all known-answer blocks, survives a nonlinear gauge change, carries a `coherence` and a §3.9-style underflow horizon. Until now $\omega$ existed only as a *generating* parameter — nothing ever measured it back |
| **(F3) on a fitted model** | `metrics.DynamicalFingerprint.filtration_gap` / `.is_filtration` | Does a fit satisfy Theorem F's hypothesis? **Not the same question as `order_margin`** (leading exponents vs whole intervals) — §3.14. `invariant_agreement` emits a note when either fingerprint fails it, and **deliberately does not change `agree`**: (F3) is sufficient, not necessary, and folding it into the score would turn `exp14`'s positive result into a failure |
| **Fit-to-fit invariant agreement** | `metrics.dynamical_fingerprint` / `invariant_agreement`, `exp14` | **Built and validated on exact systems.** The first metric here that needs **no ground truth** — it compares two fits to each other (task 40). Blind to a within-module gauge change and to the §3.7 triangular conjugacy; catches the §3.1 regrouping ($0.223$) and the task-23 frequency change ($\rho$ only, spectra tie at $10^{-18}$). Both directions are required and it is easy to get only one |
| **Nonlinear data decoder** | `src/idyn/systems.py` `MLPDecoder` | Affine coupling flow, invertible in closed form, analytic, ~45% nonlinear residual. Until 2026-08-03 **no experiment generated nonlinear observations at all** — §3.11 |
| **Theorem F — the filtration, written up** | `theory/identifiability.md` §6, `spectra.filtration_gap` | **DONE (task 11).** Standalone statement, no `TODO(gap)` in its dynamics. Tier 1 (Prop. T1) stated first with a regularity ledger — which invariants are free topologically and which need the $C^1$ bounds. **Hypothesis changed in the writing: (F3) ordered separation, not (B4) disjointness** — the module spectra must occupy disjoint *intervals*. Two things fell out as **conclusions**: the ordering half of the matching lemma, and identification of every head system $F_{\le i}$ (not just $f_1$). Open: only the coarsening, = nonlinear (B2) |
| Test suite | `tests/` | **329 tests, all passing** (was 274 with one environment failure). `test_naive_sigma_min_route_is_noise_on_a_limit_cycle` is fixed per §4.1's own prescription — it now asserts only the clause true on *both* numpy 2.4 and 2.5, namely that the naive rate misses the truth by more than half the spread. New datum from the rewrite: at `n_max=200` the naive rate is **+0.73 against a true −0.29**, i.e. it returns the wrong *sign* — a false **negative** on Lemma C, mirroring the known false positive |

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
  **Largely dissolved by §1.3:** for a deterministic generator with an injective
  decoder, pathwise equivalence *is* the right notion. The distributional version
  becomes necessary only if process noise is added — which is task 36, and a
  change of target rather than a fidelity fix.
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

**And the linear probe was not even a *fit*.** `fit_linear_relation` solved
without an intercept. Nothing pins the mean of a learned latent — the whitening
penalty constrains the covariance, and an MLP encoder carries biases — so on an
off-centre `z_fit` the solve is misspecified and scores **$R^2 = -1.38$**, below
the mean baseline, while still handing the block-energy readouts a matrix to
split. Fixed by centring both sides (an intercept fitted and discarded, which is
right because $h$ is only defined up to translation); the same fit then scores
$0.879$. **Blast radius is small and was checked:** exp02/03/06 use linear
decoders with zero-mean latents, where centring is a no-op — which is why it
stayed invisible. Gate any readout built on it with
`metrics.linear_relation_r2`. Same family as §3.9/§3.10: a misspecified readout
returns a plausible number, not an error.

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

> **Scope this correctly — it is about restarts, not about model classes.** The
> correlations above compare fits of the *same* class to the *same* data. They
> are **not** an argument against ordinary model comparison: for **nested**
> classes with a *true* constraint, held-out fit legitimately favours the
> constrained model (same bias, lower variance), and that is a valid test of
> whether the structure is present. What fit cannot do is choose *among*
> representations inside a class, and §3.1's regrouping is the clean
> demonstration — three different module decompositions fit to
> $2.2\times10^{-16}$, machine precision, all of them. The tie is **exact**, so
> no fit criterion breaks it. **Fit selects the class; it cannot select the
> representative.** Both stages are needed; see §6's empirical program.

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
learned Lyapunov exponents are weakly constrained. They disagree with the true
exponents even for fits whose block structure is essentially perfect.

> **CORRECTION (2026-08-04, `exp14`).** This paragraph used to argue that the
> learned spectra being *horizon*-stable — unchanged from $n=25$ to $n=300$ —
> "rules out an extrapolation artifact". **That inference is backwards.**
> Horizon-stability is the *signature* of one: `exp14` §3.13 measures the orbit
> and finds the learned block converging to a **spurious attracting fixed
> point** outside the data's support, and an orbit sitting on a fixed point is
> stable at every horizon by construction. Same trap as §3.9 — a stable wrong
> number reads as a measurement. The rest of the paragraph stands, and `exp14`
> supplies the mechanism plus the fix (read inside the data horizon).

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
unmeasurable in this system", not as "Lemma C fails". **§3.13(a) now supplies
the mechanism: those spectra were read at a spurious attractor.**

### 3.13 Never read an *asymptotic* invariant off a fitted map — and recoverability is per-invariant

**Fifth defect of the §3.9 family, found in `exp14`.** Three things, all the same
underlying point: **a fitted model is only a model where the data went.**

**(a) Past the data horizon a fitted map is inventing, and it invents a fixed
point.** Measured directly: the true block contracts to $0$; the learned block's
orbit **stalls at $\|z\| = 0.0204$** and sits there forever — a spurious
attracting fixed point outside the training support. Every asymptotic invariant
read there describes the extrapolation. The rotation number is the cleanest
casualty: it reads **exactly $0$** (a fixed point does not rotate) against a true
$0.1751$, at **coherence $1.00$**. A confident, stable, 100% error that looks
like a measurement. Reading inside the data horizon instead takes the rotation
error $0.1751 \to 0.1324$ overall, and to **$0.0020$** on the module that still
carries signal.

> **This corrects §3.11.** That section argued the learned spectra being
> *horizon-stable* ($n=25$ to $n=300$) "rules out an extrapolation artifact".
> **Backwards** — an orbit parked on a spurious fixed point is horizon-stable
> *by construction*. Exactly §3.9's trap: a stable wrong number reads as a
> measurement, and is more dangerous than an unstable one.

**Do:** read at $T =$ trial length with a modest warmup, and take the sample size
from the **ensemble** (many initial conditions), never from the horizon.

**(b) Recoverability is per-invariant, not per-model, and it tracks where the
orbits actually spend time.** Same fit, same horizon, two modules:

| module | variance retained at $t=T$ | $\lambda$ error | $\rho$ error |
|---|---|---|---|
| dominant | $2.7\times10^{-3}$ | $0.0035$ | $0.0020$ |
| dominated | $2.6\times10^{-16}$ | $0.391$ | $0.132$ |

Thirteen orders of magnitude of signal difference; a $100\times$ difference in
recovery.

**And the split runs *inside a single module*, between its own two exponents.**
On a limit cycle:

| quantity | lives | recovered |
|---|---|---|
| rotation number | on the attractor | $2.5\times10^{-4}$ across disjoint neuron splits |
| neutral exponent | on the attractor | $5\times10^{-3}$ (true $0$) |
| **transverse exponent** | **off it** | **$-0.56$ to $-0.70$ against a true $-0.916$ — 24–39% error** |

Same module, same fit, same horizon. Orbits collapse onto the cycle in $\approx4$
steps of a 30-step trial, so the transverse rate is determined by a handful of
early samples and the other two by everything.

§3.8's support caveat with a sharp edge. The failure is not that the fit is bad;
it is that **the number was never measured**. Consequence for task 40: report
**per-invariant** agreement. One boolean hides which half of the fingerprint the
data constrained.

**(c) Match modules across fits on the full invariant vector, not on spectra
alone.** The pairing has to happen before any comparison, and matching by
spectral distance is degenerate in exactly the case the rotation number exists
to handle — two limit cycles have identical spectra, so the cost matrix is flat
and the pairing is decided by nothing. Measured before the fix: `exp14` part 4a
paired the wrong modules in **5 of 16** comparisons, each returning a rotation
error of $0.1274$ — which is $|\rho_1 - \rho_2|$, the signature of a *swap*, not
of a recovery failure. The same trap one level up from §3.10 trap 2, and it fails
in the direction that reads as a null result.

Two parts to the fix, both in `metrics.invariant_agreement`: the Hungarian cost
carries a rotation term, and `_module_sort_key` **quantises** the spectral keys
(`ORDER_TOL = 1e-2`) so a near-tie falls through to the $|\rho|$ tie-break
instead of letting $10^{-3}$ of estimator noise order two neutral exponents.
Read `order_margin` alongside any ordering claim — it reports how far apart the
leading exponents actually are, so a tie is visible rather than implied.

**(d) Do not score an *undetermined* quantity as a disagreement.** The same run
exposed this one level further in: `agree` required the two fits to list their
modules in the same filtration order, but for two limit cycles the spectrum
**cannot** order them — `order_margin` was $0.0011$. So a comparison whose
rotation numbers matched to $5\times10^{-4}$ was still scored a disagreement, on
the strength of a hierarchy neither fit was entitled to claim. `agree` now drops
the order requirement when `order_margin <= spec_tol` and says so in the notes;
`order_agrees` is still reported, for callers whose claim *is* the hierarchy.

Measured effect, re-derived from the saved fingerprints without refitting:
linear arm $0.62 \to 1.00$, **negative control unchanged at $0.00$**. That second
number is what makes it a fix rather than a relaxation — a change that raises the
positive arm and leaves the negative arm rejecting is removing a false
constraint, not lowering a bar. Check the negative control before believing any
loosened criterion.

> **Every arm's fingerprints are now written to the JSON**, so a matching or
> scoring rule can be re-evaluated offline. Twenty-five fits is half an hour; a
> criterion is a one-line change, and the two should never have been coupled.

### 3.13(e) The remaining failure is per-restart mode collapse, and only one thing detects it

Under a **nonlinear** observation map the invariants come back badly, and the
first two explanations are both wrong. Ruling them out took two sweeps and both
are worth keeping, because each is the obvious first guess.

**Not undertraining.** Budget $3000 \to 20000$ steps improves `fit_quality`
$4.2\times$ and leaves rotation recovery *unchanged* ($0.080 \to 0.116$, median
over cross-split pairs), with coherence plateauing at $0.695$. **A better fit is
not a better recovery** — §3.11's correlation result, now with the causal arrow
checked by intervention rather than inferred from a correlation.

**Population size matters, up to a point, and then stops.** Per-fit $|\rho|$
error against ground truth, 4 fits each:

| neurons/side | fits recovering ($<10^{-2}$) | median coherence | `fit_quality` |
|---|---|---|---|
| 8 | 1/4 | 0.599 | $1.6\times10^{-2}$ |
| 16 | 0/4 | 0.683 | $1.3\times10^{-2}$ |
| **32** | **3/4** | 0.841 | $2.4\times10^{-3}$ |
| 64 | 2/4 | 0.815 | $3.5\times10^{-3}$ |

Recovery turns on near 32 neurons per side and then plateaus. (Caveat kept
because it is real: each row draws its own decoder, so the delivered dose varies
$0.489$–$0.624$ and is *lowest* at the 32 row — part of that jump may be an
easier observation map. §3.11's "read the delivered dose" applies to this table
too.)

**What survives at every size is a per-restart failure, and it is mode
collapse.** At 32/side over 12 restarts, 2 fits put **both modules on the same
factor** — duplicating one cycle and missing the other. Their error against
sorted truth is $|\rho_1-\rho_2|$ to three digits, which is the arithmetic
signature of exactly that.

**Nothing generic detects it**, and this is the load-bearing negative:

$$\mathrm{corr}(\texttt{coherence}, \log \text{err}) = -0.48, \qquad
\mathrm{corr}(\log \texttt{fit\_quality}, \log \text{err}) = +0.24$$

One collapsed fit scored coherence $0.961$, **above several good ones**; a gate
at $0.90$ lifts precision only $83\% \to 90\%$. And `fit_quality` is again
uninformative with the wrong sign — §3.11 in a third regime, so treat that as
settled rather than as a quirk of `exp11`.

**What does detect it: duplicate invariants.** `DynamicalFingerprint.duplicate_modules`
flags module pairs whose spectra and rotation numbers coincide. It needs no
ground truth — it is a property of the fitted model alone — and
`invariant_agreement` now emits a note when either fingerprint is flagged, so a
disagreement can be attributed to a collapsed fit instead of to the data. It is
a **flag, not a verdict**: a system genuinely can carry two identical factors,
and then duplication is the correct answer.

**Consequence for task 40 on real data.** Fit many restarts; screen on duplicate
invariants, **not** on fit quality or coherence; report the *fraction* of
cross-split pairs that agree together with the median, never a max. Per-restart
reliability is the binding constraint, and no amount of data or training removes
it.

**Measured effect of the screen, with the control that makes it a fix** (`exp14`
check 12, at 32 neurons/side):

| arm | flagged | raw median $\rho$ error | screened |
|---|---|---|---|
| linear | 0/8 | $0.00022$ | $0.00022$ (unchanged) |
| nonlinear | **1/8** | $0.0633$ | **$0.0032$** — $20\times$ better, 67% of pairs agree |
| negative control | 0/8 | $0.0636$ | $0.0636$ (**still rejecting**) |

Same discipline as (d): a screen that improved the negative control too would be
a filter flattering everything, not a defect being removed. The residual 33% is
the *other* failure mode — a fit that misses a factor outright rather than
duplicating one — which nothing here detects without ground truth. That is the
honest boundary of the method as it stands.

### 3.14 `spectral_gap` is (B4) and it is the wrong hypothesis — use `filtration_gap`

**Sixth defect of the §3.9 family, and the first one in a *hypothesis* rather
than a readout or a loss.** §3.12 caught an objective that measured the gauge;
this is a *check* that certifies the wrong condition, and it certifies it
positively — the worst direction.

`spectra.spectral_gap` returns the minimum distance between exponents belonging
to different modules. That is hypothesis **(B4)**: no two modules *share* an
exponent. Lemma C does not need that. It needs the **oriented** gap
$\lambda_{\max}(f_j) < \lambda_{\min}(\tilde f_i)$, and chaining those across
$K$ modules requires the module spectra to occupy **disjoint intervals** —
hypothesis **(F3)**, `spectra.filtration_gap`. `identifiability.md` §4.3 has said
(B4) is too weak since the triangular counterexample was found; what was missing
was a computed quantity for the condition that is actually wanted.

**The gap between the two is exactly the §3.1 regrouping counterexample.** With
$\lambda = (0.90, 0.75, 0.60, 0.45)$:

| grouping | `spectral_gap` (B4) | `filtration_gap` (F3) |
|---|---|---|
| true, $\{\lambda_1\lambda_2\}\{\lambda_3\lambda_4\}$ | $+0.2231$ | $+0.2231$ ✓ |
| regrouped, $\{\lambda_1\lambda_3\}\{\lambda_2\lambda_4\}$ | **$+0.1823$** ✓ | **$-0.2231$** ✗ |

The regrouping keeps every exponent distinct while *interleaving* the hulls
$[-0.5108,-0.1054]$ and $[-0.7985,-0.2877]$. So (B4) reports the repo's own
negative control as satisfying the separation hypothesis. (F3) rejects it.

**And (F3) is not merely stricter, it is *right*** — it reproduces a measured
threshold with no free parameter. On `exp08`'s sweep (attracting invariant
circle, hull $[-0.9163, 0]$, against a contracting partner) `filtration_gap > 0`
agrees with the measured `forces_M_zero` at **every** point on both sides of the
crossing, $s_{\text{fast}} \in \{0.20, 0.25, 0.30, 0.35, 0.38\}$ forcing and
$\{0.42, 0.50\}$ not, the crossing at $\log|1-2a|$ to the digit.

**A consequence that is easy to miss: width, not speed, is what disqualifies.**
A module with a *wide* hull can swallow a narrower one lying inside it, and then
neither cross-derivative is forced even though one module plainly has the larger
$\lambda_{\max}$. A limit cycle's hull $[\lambda_{\text{transverse}}, 0]$ is as
wide as they come, so `identifiability.md` §4.4's rider "an oscillatory module
sits at the top" does **not** make it separable from everything below it.

**Do:** gate any filtration claim on `filtration_gap(...).ordered`, or on
`DynamicalFingerprint.is_filtration` for a fitted model. **Do not** read
`spectral_gap > 0` as licence for Lemma C — it is the hypothesis of a theorem
this repo has a counterexample to. Regression tests in
`tests/test_spectra_and_cocycle.py` (§"Ordered separation").

**And it is not the same question as `order_margin`.** That property compares
only *leading* exponents — how robustly the modules can be **ordered**. (F3)
compares whole intervals — whether they form a **filtration** at all. A wide
hull can swallow a narrow one while still leading comfortably: cycle
$[-0.9163, 0]$ against a block at $[-0.6931,-0.6931]$ gives `order_margin`
$=+0.6931$ (decisively ordered) and chain gap $=-0.2232$ (no filtration).

> **Applying the new gate to what is already in `results/` — 0 of 24.** `exp14`
> part 4's saved fingerprints (re-scored offline, no refit) satisfy (F3) in
> **none** of 24 fits, median chain gap $-0.65$, across all three arms. Correct
> and expected — the system is two limit cycles, chosen precisely because the
> spectral route is dead there — but it means **the repo's best empirical result
> sits outside its best theorem.** (F3) is sufficient, not necessary; see
> `identifiability.md` §6.5 for the full reading, including the caveat that the
> model class imposes modularity so only the *cross-split agreement* and the
> *negative control* carry information.

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

> **MOVED AGAIN (2026-08-09) — the block above is stale, and it is the `adene`
> path that is now wrong.** This checkout is on a machine where
> `C:\Users\adene\miniconda3\envs\torch\python.exe` **does not exist**; the working
> interpreter is `C:\Users\alexa\miniconda3\envs\torch\python.exe` with
> **numpy 2.4.3, torch 2.7.1+cu118** (CUDA build present, still not needed). So
> the environment has moved *back* to the one the note below calls stale.
>
> **RESOLVED 2026-08-12.** This machine is now the primary one, so
> `test_naive_sigma_min_route_is_noise_on_a_limit_cycle` was rewritten against
> the §3.9 invariant alone, exactly as the note here prescribed: it asserts that
> the naive rate misses the truth by more than half the spread (true on both
> numpy 2.4 and 2.5) and no longer asserts that the noise floor *converges* to
> the wrong limit (true only on 2.5). **Baseline is now 285 passed / 0 failed.**
> Both floor behaviours are recorded in the docstring with numbers, so a future
> move has the comparison rather than a symptom.
>
> **History.** An earlier revision recorded `C:\Users\alexa\...` with
> torch 2.7.1+cu118 / numpy 2.4.3. That was a *different machine*; the repo was
> moved. Read the two blocks together: the lesson is that *neither* username is
> a fact about the repo.
> Two tests were environment-fragile across the move and are now written against
> the invariant rather than its old symptom — see `tests/test_spectra_and_cocycle.py`
> `test_naive_sigma_min_route_is_noise_on_a_limit_cycle` (asserts the §3.9 defect's
> *magnitude*, since on numpy 2.5 the noise floor converges to the wrong answer
> instead of wandering to it).

Run anything with:

```bash
C:/Users/adene/miniconda3/envs/torch/python.exe -m pytest -q
```

#### Moving this repo to another machine

The repo has been moved once already (see History above), and the *only* thing
that broke was paths written into prose. So the environment is now pinned in
[`environment.yml`](environment.yml) (or [`requirements.txt`](requirements.txt))
and the checklist is short:

0. **Check you are in the right checkout.** See the nesting warning below.
1. `conda env create -f environment.yml && conda activate idyn`
2. `python -m pytest -q` — **expect 329 passing**, on numpy 2.4 *or* 2.5; the
   one test that used to split across those versions no longer does (§4.1's
   RESOLVED note). This is the real acceptance test for a move: §3.9's
   regression tests are exactly the ones that changed behaviour last time the
   numerics moved under them, so if anything breaks, expect it there.
3. Update the interpreter path in §4.1 and in `README.md`'s quick start, and the
   repo root wherever it appears (§4.2, §5). **Nothing in `src/`, `tests/` or
   `experiments/` contains an absolute path** — verified, and worth keeping true.
4. Optionally re-run `experiments/run_all.py` (exp01–exp10, ~30 min, dominated by
   exp06) to confirm the JSONs reproduce. Not required: the test suite covers the
   same machinery and is seconds rather than minutes.

**Any path under `\Users\alexa\` or `\Users\adene\` in this file is a fact about
some past machine, not an instruction.** Treat it the way §4.3 already treats the
literature table.

> **Nesting hazard — resolved 2026-08-10, do not recreate it.** This repo was for
> a time cloned *inside* a stale July snapshot of itself, at
> `C:\AdenCode\IdentifiableDynamics\IdentifiableDynamics`. The outer copy had no
> `.git`, a CLAUDE.md half the size, and a to-do list naming a front line that had
> since closed twice over — so it read as authoritative and was not. A session was
> spent re-proving the partial-iVAE lemma from that stale brief before the nesting
> was noticed; the result was a rediscovery of published block-identifiability
> (task 38) via the very assumption-(iv) route Lemma D exists to avoid.
>
> The git checkout is now at `C:\AdenCode\IdentifiableDynamics` with nothing
> nested inside it; the snapshot is archived at
> `C:\AdenCode\_ARCHIVED_IdentifiableDynamics_stale_2026-07-31` and holds nothing
> unique but a regenerable `.codegraph/` index and that superseded work.
> **Before starting: confirm `git rev-parse --show-toplevel` is the directory you
> are editing, and skim §2's status table** — the cheap check that would have
> caught it.

### 4.2 Sibling project (prior art, same author)

> **Separate repo — check it exists before relying on it.** It does not travel
> with this one, so after a machine move the path below may be as stale as the
> §4.3 table. Nothing here depends on it: it is prior art to position against,
> not a dependency.

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
├── environment.yml            # pinned env (requirements.txt = pure-pip twin)
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
│   ├── spectra.py             # Lyapunov / dichotomy spectra, module gaps,
│   │                          #   filtration_gap = (F3) ordered separation, NOT
│   │                          #     spectral_gap = (B4) disjointness -- see 3.14
│   │                          #   rotation number (the invariant spectra cannot see)
│   ├── cocycle.py             # §3.3 iterated cocycle relation
│   ├── normalform.py          # Poincaré–Dulac: homological operator, resonances
│   ├── behavior.py            # Route B: u-conditioned sampling, invariant-subspace detector
│   ├── models.py              # torch: modular vs unconstrained latents, 2 decoders
│   ├── train.py               # fitting loop
│   ├── metrics.py             # partition + filtration recovery, non-uniqueness,
│   │                          #   dynamical_fingerprint + invariant_agreement
│   │                          #   (fit-to-fit, the only ground-truth-free metric)
│   └── selection.py           # partition-lattice search, fitted-model certification
├── experiments/               # exp01..exp14 + run_all.py; each writes to results/
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
| **11** | **Route C — filtration identifiability.** Assemble `identifiability.md` §4.2 into a standalone theorem | **DONE — `identifiability.md` §6.** Tier 1 stated first (Prop. T1) with a **regularity ledger**: dimension, fixed-point count, stability type, attractor topology, entropy and rotation number are free *topologically*; only the Lyapunov spectrum needs the $C^1$ derivative bounds. Then Theorem F under (F1)–(F4). **One hypothesis changed in the writing and it matters: (B4) disjointness is replaced by (F3) ordered separation** — disjoint *intervals*, not merely distinct exponents. That is not cosmetic: the §3.1 regrouping counterexample **passes (B4) at $+0.1823$** and **fails (F3) at $-0.2231$**, so (F3) is what rejects it, and `spectra.filtration_gap` computes it. Two things came out as *conclusions* rather than hypotheses — (a) the **ordering half of the matching lemma** (§3.2) is free, since both sides must partition the same Tier-1 spectrum into consecutive groups, leaving only the *coarsening* freedom = nonlinear (B2); (b) **every head system $F_{\le i}$ is identified**, not just $f_1$, so the whole chain of quotients is pinned, with $\Lambda(f_i)$ recovered by set difference. Validated: (F3) computed from spectra alone reproduces `exp08`'s measured `forces_M_zero` threshold at **every** sweep point on both sides of the crossing, no free parameter |
| **12** | **Route B — hybrid: behavioural auxiliary + one-sided gap** | open; mechanism verified numerically. Needs a *partial* iVAE theorem |
| **13** | **Learn an indecomposable model** — certify the fitted model, then search the partition lattice | **largely done** (`selection.py`, `exp06`): fitted-model certification + lattice search recover the finest partition from data. Finding: fit rejects splitting an indecomposable block, uniqueness breaks ties among equal-fit regroupings — each covers the other's blind spot. The linearised certifier's Tier 2 false negative is now fixed at the fixed point (task 25, degree-2 jet). Off-fixed-point (genuine-attractor) indecomposability still open |
| **22** | **Does Lemma C extend to genuine attractors?** (the filtration off the fixed point) | **resolved positively** for *periodic* attractors — Lemma C′, `identifiability.md` §4.4, certified in `exp08` (rate exact to 2.4e-14, threshold exact at $\|1-2a\|$; uniformity over the basin measured to 2.8e-16 across a 100× radius range). Uses only (B1)'s bounded-derivative clause — **not** $\mathrm{int}\,\Omega\neq\emptyset$, not (B2), not (B3). Prerequisite finding: the naive $\sigma_{\min}$ bound is unusable here (§3.9). **Open:** attractors with non-uniform exponents (chaotic) |
| **23** | **NEW, raised by 22: two oscillatory modules cannot be separated spectrally** | **RESOLVED NEGATIVELY 2026-08-12 — the rotation number does not rescue it.** `counterexamples.md` §7 / `systems.torus_regrouping_counterexample`: for two shear-free attracting cycles, $h(z_1,z_2)=(z_1z_2/|z_2|,\,z_2)$ is an **exact** modular conjugacy (residual $6.7\times10^{-16}$, invertible to $8.9\times10^{-16}$, (F1) satisfied on an annulus with $\sup\|Dh\|=2.72$, cross-block $\ge0.55$ everywhere) carrying $(\omega_1,\omega_2)\mapsto(\omega_1+\omega_2,\omega_2)$. Two cycles span an invariant torus and a conjugacy acts on $H_1(T^2)=\mathbb{Z}^2$, so **only the $GL(2,\mathbb{Z})$ orbit of the rotation vector is identified** — this is §3.1's regrouping in oscillatory form. Does **not** contradict Theorem F, whose (F3) fails here anyway (chain gap $-0.9163$). **The shear escape is closed (same day).** Donor shear breaks only the *naive-angle* form ($7.8\times10^{-2}$ at $\beta_2=0.3$); the object that advances rigidly is the **asymptotic phase** $\Theta=\theta+\beta\sum_k(g^k(r)-\rho)$, and rebuilding $h$ with it is exact at every shear ($\le1.8\times10^{-15}$ over $\beta_1\in\{0,0.5\}$, $\beta_2\in\{0,0.3,0.8\}$, inverse to $1.7\times10^{-15}$). Forced in hindsight: **shear is not a conjugacy invariant of a single cycle** — $(r,\Theta)$ removes it — so it could never protect one, and the regrouped module comes out shear-free even when the original is not. So the $GL(2,\mathbb{Z})$ ambiguity is intrinsic, not an artefact of an idealised oscillator. `counterexamples.md` §7.1. **Note $\beta=0$ is `LimitCycleBlock`'s default and `exp14` part 4's setting**, so the repo's own two-oscillator system is the vulnerable one. Original entry: open. A limit cycle's neutral exponent 0 makes it undominatable, so an oscillatory module is always the **top** of the filtration and a filtration holds **at most one**. This collides with §1.1's own definition of a module ("distinct oscillatory components") and frequency does not help — Lyapunov exponents are blind to rotation number. Needs a finer conjugacy invariant (rotation number), which is a different argument. See `identifiability.md` §4.4. **Candidate answer found 2026-08-04: task 35 (Route B′, $u=t$).** Conditioning on the time index is sensitive to rotation number by construction. Measured: two limit cycles with identical spectra $\{0,-0.9163\}$ — where Lemma C has no gap to use — are cleanly separated by their $u=t$ phase signature when $\omega$ differs ($1.571$ rad) and not when it does not ($0.006$). This is the first mechanism in the repo that can see past the neutral exponent |
| **24** | **Route A: is Tier 2 non-empty?** (does the nonlinear claim have content, or collapse to Theorem A?) | **resolved positively.** Witness `systems.tier2_witness()`, machinery `src/idyn/normalform.py`, certified in `exp09` (5/5). $(\mu z_a,\ \mu^2 z_b + c z_a^2)$: the resonance $\lambda_b-\lambda_a^2=\mu^2-\mu^2$ vanishes *identically in $\mu$*, so it is structural; $c$ survives as a normal-form invariant while cross-module non-resonance holds. Honest limit: $c$ rescales to 1, so the invariant is binary (linearisable or not); a continuous modulus needs two resonant coefficients |
| **25** | **NEW, raised by 24: the linearised (B2) test has a false negative** | **fixed at degree 2** (`selection.block_nonlinear_certificate`, `normalform.coupling_resonances`/`quadratic_jet_coefficients`, `certify_fitted_model(nonlinear=True)`, `exp09` part 5). Reads the quadratic jet in the eigenbasis and flags a *resonant* cross-eigendirection monomial; correctly ignores *non-resonant* cross terms (removable) and is coordinate-invariant. Opt-in, so linear-only callers unchanged. **Open:** higher-degree resonances (need the coefficient post-normalisation) and near-resonances in *fitted* models (eigenvalues carry fit error — `res_tol`/`coeff_tol` knobs exposed) |
| **26** | **NEW: the (B2) criterion is graph *connectedness***, and it is proved at degree 2 | **done** (`route_a_assessment.md` §4.1a, `normalform.resonance_coupling_components`, `exp09` part 6). A module is indecomposable iff its resonance-coupling graph is connected; proved both directions at degree 2 for distinct eigenvalues (degree-2 resonant coefficients are normal-form invariants — homological operator vanishes on them). Fixed the task-25 certifier, which flagged *any* coupling and so **over-reported** indecomposability with ≥3 sub-blocks (the direction the fit can't catch). **Open:** connectedness-invariance at higher degree; repeated-eigenvalue sub-blocks |
| **27** | **NEW (active): Route B∘C** — behaviour + one-sided gap = block-diagonal | **mechanism built + verified** (`behavior.py`, `exp10`, `approaches.md` §B.1). Behaviour kills $M_{BA}$ (canonical invariant subspace), Lemma C kills $M_{AB}$ (gap); together block-diagonal, inheriting C's attractor reach. Found the **alignment condition**: block-diagonal needs varying block = spectrally dominant, else only triangular. Prop-1 caveat operational (variance vs mean modulation). **The one open obligation: the partial-iVAE lemma** — $u$-invariant complement identified as a subspace; not a corollary of Khemakhem et al. (§6.1). This is the B∘C front line |
| **28** | **NEW: nonlinear data decoder** — the Theorem B observation model had never been run | **done.** `systems.MLPDecoder`, an affine coupling flow: invertible in closed form for arbitrary $s,t$, analytic (tanh/exp, which §3.7 needs), ~45% nonlinear residual, `strength=0` is an exact linear control. Threaded through `make_dataset`/`make_behavioural_dataset`. **Discarded first attempt** — the contractive $z+\epsilon m(z)$ construction caps at **3%** nonlinearity even at $\epsilon\,\mathrm{Lip}=0.99$ and *worsens* with depth, because $\mathrm{Lip}\le\prod\|W_k\|_2$ is a worst case while tanh is far gentler; a decoder within 3% of linear does not test Theorem B. See §3.11 |
| **29** | **does *learning* recover B∘C under a nonlinear observation map?** | **ANSWER RETRACTED (§3.12).** The sweep is sound as a measurement but tested *dynamics-only* fitting: no arm imposed the behavioural constraint, so it never bore on B∘C. Re-asked properly as task 33. What stands unchanged: the dose-response, the strength-0 control, and the two methodological lessons (read `obs-nl` not `strength`; report distributions). Original entry, for provenance: ~~**DONE — answer is no, with a confirmed dose-response.**~~ `exp12` run 2, doses $(0.00,0.31,0.43,0.60)$: `jac_diag` falls **monotonically** $0.994\to0.730\to0.702\to0.567$; forbidden `upper` violated in $0/8$ restarts at low dose, $6/8$ at $0.43$, **$8/8$ at $0.60$** — a threshold, not a tail. Not fit failure: $\mathrm{corr}(\texttt{fitq},\texttt{upper})=-0.58$ at the top dose, so *better* fits are more coupled. The strength-0 control ($0.994$, sd $0.009$, lower exactly $0.000$) is what makes it readable. **Run 1 failed for want of a treatment, not an effect** — strengths $(0,0.25,0.5,1.0)$ deliver doses $(0.00,0.31,0.31,0.35)$, two levels not four, because latents contract toward the origin where tanh is near-linear: **always read `obs-nl`, never `strength`**. **Scope, and it is the whole point: this bounds LEARNING, not identifiability** — a fitted $h$ need not satisfy Lemma C/D's hypotheses (exact conjugacy, additive form, gap in the *learned* spectra). With task 31 proved, the honest reading is a **learning** gap. Successor is task 32 |
| **31** | **NEW: Lemma D — behaviour kills $M_{BA}$** (`identifiability.md` §4.5) | **PROVED** for additive $h_B$ + linear modules; witness `systems.lemma_d_witness`, 5 tests. **STRENGTHENED 2026-08-12 → Lemma D′ (§4.5a): (D1) drops to $1\notin\operatorname{spec}(\tilde f_B)$.** The one-sided gap was doing one job — excluding a degree-0 scale-invariant $\psi$ — and doing it with far more than needed; Step 3 ($\lvert m\rvert\ge2$) turns out not to be load-bearing, since Step 4's iteration $r=(\sigma_2/\sigma_1)^p\ne1$ needs only $p\ge1$. **Consequence: the behavioural kill no longer needs *any* spectral hypothesis**, so it applies where Lemma C, Theorem F and (D1) are all dead — two modules with identical spectra, i.e. task 23's two-oscillator case in linear form. Witness `systems.gapless_resonant_coupling` (degree-1 resonance, exact conjugacy $6.7\times10^{-16}$, $\|M_{BA}\|=0.99$); measured $u$-dependence $0.0074$ (control) vs $0.0883/0.2880/0.4476$ at $c=0.25/0.5/0.7$, control falling like $n^{-1/2}$ so the floor is checked not assumed. **Also found: Step 4 assumes a single homogeneous degree while Step 2 allows several — pre-existing (true with (D1) too), unflagged until then. CLOSED 2026-08-14 as Lemma D″ (§4.5b), and my $k+1$-by-Vandermonde guess was wrong: the argument has to run on second moments, and the count is the sumset $\lvert P+P\rvert+1$, not $\lvert P\rvert+1$** | Replaces the partial-iVAE obligation of task 27 with a dynamical argument. **Reconciliation with task 29 superseded**: the apparent conflict was not a learning gap at all — `exp12` never imposed Lemma D's behavioural hypothesis (§3.12, task 32), so there was nothing to reconcile. **Open:** (a) non-additive $h_B$ — the graded reduction survives but Step 4's characteristic-function factorisation needs independence; (b) anisotropic variance modulation; (c) nonlinear modules — **CLOSED 2026-08-14 (§4.5c), and not via §5.3**: additive $h_B$ forces $\tilde f_B$ affine, so that half is vacuous, and Step 2 restated as "$\psi$ is a Koopman eigenfunction of $f_A$" holds for arbitrary $f_A$. Cost is level richness (a limit point in the level set), not regularity. **(a) sharpened 2026-08-09:** the missing piece **cannot come from behaviour** — `systems.nonadditive_behavioural_escape` is a non-additive $h_B$ satisfying **(D1)–(D4) exactly** with $M_{BA}\neq0$ ($h_B = R(\gamma z_{A,1})z_B$, which preserves a spherical $p_B$ for every $u$), so no sharpening of (D4) closes it. What excludes it is **Step 1**: it is not a modular conjugacy, since $\theta\circ f_A-\theta$ must be constant and the fixed point forces $\theta$ constant. Needs $\dim z_B\ge2$ (at $1$ the transports are the isolated $\pm\mathrm{id}$). `identifiability.md` §4.5, 3 tests |
| **32** | **close the gap between Lemma D and `exp12`** | **RESOLVED — and not by any of the three candidates I listed.** The answer is **(iv): the experiment never imposed the behavioural hypothesis.** `models._behavioural_penalty` scored the pinned block's conditional moments on the *raw* block, so it falls like $\varepsilon^2/\varepsilon^4$ when the block shrinks; the optimiser satisfied it by making the block **21× smaller** than its partner, while that block still carried the $u$-varying latent at **dCor 0.99** (scale-normalised $u$-dependence **1.07**, against $0.15$ for a genuinely invariant block and $1.09$ for the true *varying* one). So the discrepancy was never a hypothesis the fit *breaks* — it is one the experiment never *applied*. Full write-up §3.12; fix is `TrainConfig.behavior_whiten=True` (default), which restores the constraint (u-dep $\to 0.037$ at matched weight, scale ratio $\to 1.4\times$). **Consequence: exp11 and exp12's behavioural conclusions are void**, and `approaches.md` §B.1's B column is *untested*, not refuted. Successor is task 33 |
| **33** | **does B∘C survive a nonlinear observation map once behaviour is *actually* imposed?** | **ANSWERED, PARTLY YES** (`exp13`, 8 restarts × 4 doses × 2 penalties). **(a) The forbidden direction is now uniformly killed.** `upper` $\le 0.081$ over *every* dose and *every* restart, against $0.316$ raw — Lemma C's half is clean once behaviour stops being paid off with scale. **(b) Block-diagonality improves sharply at the high doses**: `jac_diag` $0.702\to0.893$ at dose $0.43$ and $0.567\to0.815$ at $0.60$. **exp12's "degrades to a filtration" is refuted.** **(c) But it is not uniform, and the exception is sharp** — at dose $0.31$ the whitened fit lands *triangular in all 8 restarts* (`jac_diag` $0.546$, `lower` $0.416$, **sd $0.029$**). Deterministic, so not underpowering. So the defensible claim is: behaviour genuinely supplies its kill, and B∘C reaches block-diagonal at 3 of 4 doses — not at all of them. Successor is task 34 |
| **38** | **NEW: the "partial iVAE lemma" is PUBLISHED — reposition the behavioural half** | open, and it **changes what Route B can claim** (`literature.md` §1.3). The obligation task 27 called the B∘C front line — "$u$-invariant complement identified as a subspace, not a corollary of Khemakhem et al." — is the literature's **block-identifiability** (von Kügelgen et al. 2021, Def. 4.1: $\hat c = h(c)$ for invertible $h$ — verbatim our target and verbatim §7). Two theorems in *our* setting, an auxiliary variable indexing a law rather than paired views: **Kong et al. arXiv 2306.06510 Thm 4.2** and **Sun et al. arXiv 2208.14161 Prop. 4.2**, both giving the invariant block up to invertible transformation, both allowing invariant/varying dependence, and Kong's componentwise-monotonic domain action $z_s=f_u(\tilde z_s)$ **exactly matches our variance modulation** $z_A=s(u)\tilde z_A$. **Lemma D is not redundant** — it needs **two** behaviour levels against Kong's $2n_s+1$ and Sun's $2\ell+1$, and buys that economy with the dynamics (the gap forces coupling degree $\ge2$). That economy is now the defensible novelty of the behavioural half, and it should be stated that way rather than as "we proved a lemma nobody had". **Blocking check first:** Kong's A2 is *componentwise* conditional independence, which our within-module coordinates violate — read from summaries, A2/A3 look like they serve the finer $z_s$ conclusion while the block conclusion rests on A1 + A4 (a cylinder-set condition, not a factorisation), but **this was not read from the proof.** Verify before citing |
| **35** | **Route B′ — auxiliary variable = the time index $t$** | open, **promoted: it matches the target model class and it is free.** Autonomous LFADS is random $g_0$ + deterministic generator, so all trial-to-trial randomness is in the initial condition; at fixed $t$ across trials that is an **ensemble**, exactly what TCL/iVAE consume. Path-conditioning (PCL/SNICA) is excluded by the target itself, not by a modelling choice. Available because the process is non-stationary (task 8 correction). **Not a cheaper $u$ for Lemma D** — Lemma D needs one block *flat* in $u$, and time moves every block ($t$-dep $1.32$/$4.20$ vs behaviour's $1.14$/$\mathbf{0.14}$), so there is no $t$-invariant subspace. What it powers is the TCL/iVAE **variability** condition: per-module natural parameters go like $s_i^{-2t}$, linearly independent across modules **iff the contraction rates differ** — i.e. satisfied exactly when Lemma C's gap holds. Same hypothesis, second extraction. **It may answer task 23**, the route's sharpest limitation: $u=t$ sees the **rotation number**, which Lyapunov exponents provably cannot. Measured — two limit cycles, $\omega=0.5$ vs $1.3$, *identical* spectra $\{0,-0.9163\}$ so Lemma C is dead, yet phase signatures separate by $1.571$ rad; at $\omega_1=\omega_2=0.9$ separation collapses to $0.006$ (**equal rotation numbers are B′'s resonance analogue**). Three costs: **(i)** at a fixed point only $\approx9$–$10$ usable $t$ levels before the fast block underflows, against Khemakhem's $nk+1=9$ — right at the line; **(ii)** on a *cycle* the variability is **persistent**, not transient ($t$-dep $1.489$ early, $1.493$ late, coherence held $0.881$; uniform initial phase → $0.057$, correctly dead), so B′ is **stronger off the fixed point than on it** — I first claimed the opposite and was corrected by measurement. Caveat: coherence held exactly because the model is noiseless with $\beta=0$; real dephasing sets a finite horizon, longer than (i)'s but not infinite; **(iii)** cost 3 — if TCL-style identification works, modularity may do no work. Defence: within-module coordinates are dependent, so plain TCL does not apply and a block version is needed. **Demoted 2026-08-04 by §1.3:** dropping the VAE commitment retires the *reason* to want an auxiliary variable at all, so B′'s TCL/iVAE half is no longer wanted; what survives is only its **rotation-number** sensitivity, which task 37 does need. **Zero repo footprint** — prose in this row plus `literature.md` §1.2's correction box, no module, no experiment, and `identifiability.md` §4.4's task-23 gap still reads `TODO(gap)` as though nothing had been measured |
| **36** | **add process noise to the modules — but it is a CHANGE OF TARGET, not a fidelity fix** | open, and **demoted 2026-08-04 after checking LFADS.** I originally justified this as "determinism was chosen for convenience, not the science — LFADS latents are stochastic." **That justification is wrong.** In an autonomous LFADS model one samples $g_0$ from the prior and then simulates a **deterministic** RNN forward; the only per-timestep stochasticity is the *inferred inputs* $u_t$, which §1.1 scopes out. So under this project's own scope, LFADS is exactly **random initial condition + deterministic flow** — which is precisely what `make_dataset` builds. Determinism here is *faithful to the target*. What remains true: determinism is what makes PCL/SNICA inapplicable (`literature.md` §1.2 point 1 — the pair $(z_t,z_{t-1})$ lives on the graph of $f_i$), and adding noise would open that family, close §3.8's distributional-equivalence `TODO(gap)`, and make the sibling SNICA implementation reusable. But that is **moving to a model class where identifiability is easier**, not modelling LFADS more accurately, and it should be argued on those terms. Note the link to §3.8: if input drive $u_t$ is ever brought back in scope, LFADS's stochastic inputs *would* supply a per-timestep stochastic drive, and this task merges with that one |
| **37** | **NEW: re-target from block-diagonality to dynamical invariants** | **specified, partly delivered.** `identifiability.md` §6.4 is now the spec: a six-row table (K, level dims, ordering, per-level Lyapunov spectrum, top-level rotation number, attractor topology) with the estimator and the hypothesis identifying each. Rows 3–5 are `metrics.dynamical_fingerprint` and are built; rows 1–2 rest on (F4)/nonlinear (B2); row 6 is **not built**. §6.3 is the sharp caveat and it was not obvious: $f_1$ is identified *outright* ($h_1=h_1(z_1)$ is a genuine conjugacy, so its rotation number is identified), but for $i\ge2$ what is identified is $f_i$ as **fibre dynamics of a skew product** — a conjugacy along the orbit of the base point, genuine only over a fixed or periodic point. Unconditionally only $\Lambda(f_i)$ survives at depth. That is the *theoretical* counterpart of §3.13(b)'s measured $100\times$ asymmetry between dominant and dominated modules, which is a good sign for both. Open: row 6, and the empirical half (39/40). Original entry: open. §7 already concedes that what is identified is the **partition plus each $f_i$'s conjugacy class**, never coordinates. The applied goal may need strictly less than block-diagonality of $h$: the number of modules, their dimensions, and per-module invariants (Lyapunov spectrum, rotation number, attractor topology). That reads as *"this population carries a slow 2-D rotation at 8 Hz and a fast 3-D decaying component"* — usable by a neuroscientist, and testable. Lemma C (filtration) plus task 35 (rotation number) may deliver it **without** ever proving $h$ block-diagonal, i.e. without needing tasks 33/34 to close |
| **34** | **NEW (ACTIVE): why does the dose-$0.31$ arm land triangular in every restart?** | open — **the current front line.** It is the one arm where imposing behaviour *lowers* `jac_diag` (whitened $0.546$ vs raw $0.730$), and the tightness (sd $0.025$/$0.029$ over 8 restarts) rules out optimiser noise: it is a property of the objective at that dose, not of the seed. Leading hypothesis, and it is my error to check first: **`W_WHITENED = 1.0` was calibrated on the endpoint doses only** ($0.00$ and $0.60$), so the interior was never tuned — sweep $w$ at `strength=0.5` before looking for anything deeper. Note the decoders form a *one-parameter family* (same rng draw, scaled), so this is not a decoder-draw artifact. Second hypothesis: `obs-nl` is non-monotone in `strength` in a way that makes $0.31$ special beyond its dose |
| **30** | **NEW: `exp11` methodology — report distributions, not best-of-$N$** | open, unambiguous (§3.11). Selection by `fit_quality` carries no structural information ($\mathrm{corr}=-0.044$ / $+0.279$); with sd $0.104$ the current point estimate is near a coin flip. Also raise `STEPS` — 1200 was tuned for the linear decoder and undertrains the nonlinear one into a *reversed* result. Do this before re-running exp11 as a gate |
| **39** | **NEW: co-smoothing adequacy gate on real data** | open — §6 "empirical program". Nested ladder unconstrained ⊃ triangular ⊃ block-diagonal, scored by held-out-**neuron** co-smoothing (fit ~80% of the population, refit a *fresh* decoder to the remaining ~20%). Target: Neural Latents Benchmark maze/RTT — standardised, and carries LFADS baselines. **This would be the first real data in the repo**; everything so far is `make_dataset`. Answers *is the structure there*, i.e. it settles diagonal-vs-triangular empirically instead of by §7's genericity argument. **It cannot answer identifiability** — the metric is constant on gauge orbits, by construction |
| **40** | **invariant agreement across disjoint neuron splits** | **METRIC BUILT AND VALIDATED (`exp14`); works under a linear observation map, open under a strong nonlinear one.** The test: fit independently on disjoint neuron subsets, then compare the *fits to each other* — identifiable dynamics give different coordinates and the same invariants. Varies the **data**, not the seed, so unlike restarts it excludes "artifact of this sample of neurons". Machinery: `metrics.dynamical_fingerprint` + `invariant_agreement`, plus `spectra.rotation_number` for the invariant spectra provably cannot see. **Validated on exact systems** — blind to a within-module gauge change *and* to the §3.7 triangular conjugacy, not blind to the §3.1 regrouping ($0.223$) or a frequency change; both directions are required and passing only one is easy. **Linear decoder: works** — 16/16 cross-split comparisons agree, rotation error median $2.5\times10^{-4}$ (max $5.1\times10^{-4}$), with signs *and* module order differing freely between fits, i.e. the gauge correctly quotiented out. **Negative control: works** — a frequency change is detected at $0.0638$ against a true separation of $0.06366$, 0/16 below threshold. **Transverse Lyapunov exponent does NOT agree** (median $0.052$, $208\times$ worse than $\rho$ on the same fits) — §3.13(b). **Nonlinear decoder (dose $0.574$, 8 neurons/side): fails, and not from undertraining** — $3\times$ the budget improves `fit_quality` $2.1\times$ and makes recovery *worse* ($0.080\to0.125$); it is **population size** up to $\approx32$ neurons/side, after which it plateaus. **What then remains is per-restart mode collapse** — 2 of 12 fits put both modules on one factor — and neither `coherence` ($r=-0.48$) nor `fit_quality` ($r=+0.24$, wrong sign) flags it. `DynamicalFingerprint.duplicate_modules` does, with no ground truth — screening on it takes the nonlinear arm's median $\rho$ error $0.0633 \to 0.0032$ ($20\times$) while leaving the negative control at $0.0636$, which is what makes it a fix rather than a filter. **Protocol for real data: many restarts, screen on duplicate invariants, report the *fraction* of agreeing splits plus the median, never a max.** `exp14` is **11/11** with that protocol. Full numbers in §3.13(e). **Still open: the residual 33% (a fit that misses a factor rather than duplicating one — undetectable without truth), and any real data at all.** **NEW 2026-08-12 — this validation is *outside* Theorem F, now quantified:** re-scoring the 24 saved fingerprints against the new (F3) gate gives `is_filtration` in **0 of 24**, median chain gap $-0.65$. Correct and by design (two limit cycles = rider 2), but it means the agreement is evidence that **(F3) is sufficient, not necessary**, and that the rotation number does work no theorem here licenses — i.e. support for the task-23 conjecture rather than for Theorem F. `identifiability.md` §6.5. **RETRACTED the same day — the task-23 conjecture is false** (§7 of `counterexamples.md`, task 23). Only the $GL(2,\mathbb{Z})$ orbit of the rotation vector is identified, so the per-module $\rho$ these fits agree on are **not** invariants of the data; both land on the same lattice basis for reasons of parameterisation, not because the observations pin it. **And the negative control's margin is 4× smaller than reported:** quotienting by $GL(2,\mathbb{Z})$ it sits $0.0159$ from an image of the true system, not $0.0637$. It still rejects, with a quarter of the headroom — use `spectra.rotation_lattice_margin`, which `invariant_agreement` now reports as a note (never scored: whether the ambiguity bites depends on shear, which a fingerprint does not carry) |

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

**Empirical half added 2026-08-04.** The theory half is C + task 37; the data
half is tasks 39/40, specified under "The empirical program on real data" below.
They are independent — 39/40 need a fitted model and no theorem — so they run in
parallel, and 39 is what converts §7's genericity argument into a measurement.

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

### The empirical program on real data (2026-08-04)

**Status: planned, nothing built. The repo is 100% synthetic** (`make_dataset`
only). §1.0 requires hypotheses checkable on real recordings, so this is the gap
between the current state and the applied claim. Two metrics, **neither needing
ground truth**, answering the two tiers of §1.2 respectively.

**1. Adequacy gate — co-smoothing over a nested ladder (tasks 39).**
Fit on ~80% of the population, infer latents, fit a *fresh* decoder from those
latents to the held-out ~20% of neurons, score prediction there. This is the
Neural Latents Benchmark's primary metric, so results land on the same footing
as published LFADS baselines. It cannot be won by memorising single-neuron
noise, because the scored neurons were never seen. Run it over

$$\text{unconstrained} \supset \text{filtration (triangular)} \supset \text{modular (block-diagonal)}$$

Nested, so held-out performance directly tests the structural hypotheses that
§3.7 and §7 argue from theory. **If triangular matches unconstrained while
block-diagonal loses, the filtration reading is confirmed on data** — much
stronger than the genericity argument in §7.

> **Co-smoothing is gauge-invariant, and that is exactly why it cannot answer
> identifiability.** If $\hat z = h(z)$, refitting the held-out decoder gives
> $D\circ h$ — same predictions, same score. The metric is constant on gauge
> orbits, so competing decompositions score identically **by construction**, not
> by coincidence. Same mechanism as §3.5: the decoder absorbs $h$. A clean
> adequacy gate; a useless identifiability test.
>
> **Design choice, not a neutral measurement:** how gauge-invariant it is depends
> on the held-out decoder class. A *linear* held-out decoder is invariant only
> under linear $h$, so it implicitly rewards latents from which the population is
> linearly readable. Defensible — close to what the field means by a good latent
> space — but it is a thumb on the scale. Choose deliberately and say which.

**2. Identifiability test — invariant agreement across disjoint neuron splits
(task 40).** Fit independently on two *disjoint* neuron subsets: two models of
the same circuit from different samples. If the dynamics are identifiable they
recover different coordinates but the **same invariants** — filtration order,
per-module Lyapunov spectra, rotation numbers. Agreement confirms; disagreement
falsifies.

Stronger than restart-to-restart agreement because it varies the **data**, not
just the seed, so it rules out "the structure is an artifact of this sample of
neurons" — which restarts cannot. Across sessions or animals on one task,
stronger still. **This is the direct empirical statement of what the project has
been proving, and nobody runs it.**

**Two caveats, real but not fatal.** (a) If the constraint is only
*approximately* true — likely for any recurrent circuit — then at modern data
volumes the bias term dominates and the unconstrained model wins even where the
structure is a good description; **"fits worse" does not cleanly mean "structure
absent"**. (b) Constrained models are harder to optimise (`exp13` is sharply
weight-sensitive), so a worse fit can be an optimisation artifact. Both argue for
reporting the ladder as a **gap with spread over restarts**, never a single
winner (§3.11).

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

**The filtration is the generic object, not a fallback — do not read §3.7 as a
defeat.** A filtration needs only invariant *subspaces*. A direct-sum
decomposition needs an invariant subspace **plus an invariant complement**, and
the complement is the fragile half: every linear operator has invariant
subspaces, not every one splits into a direct sum of them — that is what a
Jordan block is. Triangular is generic; block-diagonal is special. The circuit
reading agrees. One-way influence is common in cortex (slow contextual or
preparatory signals shaping faster movement dynamics; a condition-invariant
component not driven by the condition-dependent one), whereas two factors
coexisting in one population with *neither* influencing the other is the
strictly stronger and less likely claim.

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
