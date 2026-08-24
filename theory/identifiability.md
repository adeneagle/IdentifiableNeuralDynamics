# Identifiability of modular latent dynamics — corrected statement

Replaces the conjecture and proof sketch of the original draft
(`docs/brief_v0.md`), which are false and defective respectively; see CLAUDE.md
§2–3 and `counterexamples.md`.

Claims that depend on an unproved lemma are flagged `TODO(gap)` inline, per
CLAUDE.md §8. §7 lists the open ones together; §6 is the theorem that has none.

---

## 0. Scope

The governing constraint (CLAUDE.md §1.1): the target is an **autonomous,
single-area, nonlinear** latent dynamical system that is identifiable. Modules
are **dynamical factors within one population** — separate timescales, distinct
oscillatory components, subspaces evolving without reference to each other —
**not brain regions**. Multi-region and inter-area communication are out of
scope; input drive is a nice-to-have, not a requirement.

Two things follow for what is written below. First, the noise and input
caveats in §1 are *deferred by decision*, not open threats. Second, the
triangular/filtration conclusion of §4.2 stops being a shortfall and becomes the
natural target: within a single population, "a slow component evolving
autonomously and a faster one driven by it" is the expected structure, whereas a
symmetric partition into interchangeable modules is not.

### 0.1 What is proved — the short version

Added 2026-08-24 because the file has outgrown a linear read. Each row links to
the section that states it; the ledger is what a write-up should be assembled
from.

| result | says | status | key hypothesis |
|---|---|---|---|
| **Prop. T1** (§6.0) | the fitted transition is *automatically* conjugate to the true one, so every conjugacy invariant is identified | **free** — no theorem | injective decoders |
| **Theorem A** (§3) | linear decoder ⟹ $h$ is a block permutation | **proved, sharp** | (A1) indecomposable, (A2) disjoint spectra — both necessary |
| **Theorem F** (§6.2) | $h$ is block *lower-triangular*; the flag, every head system and the level spectra are identified | **proved** modulo (F4) | (F3) ordered separation — **holds 2/36 on real data** |
| **Theorem B** (§5.3) | analytic decoder ⟹ $h$ block-diagonal | **proved at a fixed point** | (B4′) non-resonance — **identically false if any module rotates** (Prop. N, §5.3) |
| **Lemma C/C′** (§4.4) | a one-sided gap kills one cross-derivative | **proved**, reaches periodic attractors | oriented gap; §3.7 proves both orientations can never hold |
| **Lemma D–D‴** (§4.5–4.5c) | behaviour kills the cross-derivative the gap cannot | **proved** | additive $h_B$, two behaviour levels |
| **Prop. S** (§14.1) | Route B kills every coupling valued in a **non-compact** group; the residue is compact | **proved** | $p_B$ of finite nonsingular second moment |
| **Prop. C** (§15.1) | a rotational coupling **strictly decreases** the phase resultant | **proved, exact** | $T\perp z_B$ |
| **Theorem D** (§15.12) | independence of module marginals ⟹ block-diagonal | **proved at a contracting fixed point** | $\log\rho_2+\sigma_1<0$ — $f_2$'s rate against $f_1$'s *own spread*; **free for a conformal module** |
| **Theorem D′** (§15.13) | the same on a limit cycle | **sketch only** — see the caveat there | trivial $\operatorname{Stab}(p_1)$ |

**Known FALSE, not merely unproved.** Block-diagonality under (B1)–(B4) at
$C^1$: there is an explicit $C^\infty$ counterexample, so triangular is *sharp*
(§4.3). And for two oscillatory modules only the $GL(2,\mathbb{Z})$ **orbit** of
the rotation vector is identified (§7 of `counterexamples.md`).

**The one condition that keeps recurring.** §14 and §15 close on the same
object: no module's law may have a nontrivial orthogonal symmetry. That is a
property of the *recording* — whether trials are phase-aligned — not of the
spectra, and it is checkable with no ground truth.

**Open:** the nonlinear matching lemma (B3), nonlinear indecomposability (B2),
distributional equivalence with noise, nonlinear perturbation, and Theorem D′.

---

## 1. Setting

Latent state $z_t \in \mathbb{R}^d$ partitioned into $K$ modules evolving
autonomously,

$$z_{t+1}^{(i)} = f_i\!\left(z_t^{(i)}\right), \qquad F = f_1 \oplus \cdots \oplus f_K,$$

observed through $x_t = g(z_t) + \epsilon_t$. Two representations
$(F, g)$ and $(\tilde F, \tilde g)$ are **equivalent** when they induce the same
distribution over observation sequences from the same initial-condition
distribution.

Write $\Omega$ for the closure of the region the trajectories actually visit.

> **Support (CLAUDE.md §3.6).** Every statement below is a statement about
> $\Omega$ only. Off $\Omega$ nothing constrains $h$ at all. If trajectories
> collapse onto a low-dimensional attractor, $\Omega$ is that attractor and the
> conclusions are correspondingly weak. This is the regime real recordings are
> in, and §4 of `counterexamples.md` shows it is not a footnote: whether the
> spectral hypotheses hold *depends on how large $\Omega$ is*.

> **Noise.** The draft defined equivalence pathwise as $Wz_t = \tilde W\tilde z_t$,
> which drops $\epsilon_t$ entirely. Distributional equivalence is the right
> notion and it is not free: with $\epsilon_t \sim N(0, \sigma^2 I)$ and a linear
> decoder, $(W, \sigma)$ and $(cW, \sigma)$ differ, so the latent scale becomes
> identifiable while $\sigma$ trades off against decoder gain. Everything below
> is stated in the noiseless limit; the noisy statement is **not** done.
> `TODO(gap)`

---

## 2. Why this splits into two theorems

CLAUDE.md §3.5. If $g(z) = Wz$ with $W, \tilde W$ of full column rank $d$, then
$W z_t = \tilde W \tilde z_t$ gives $\tilde z_t = (\tilde W^{+} W) z_t$
immediately, so

$$h = \tilde W^{+} W \in GL(d)$$

is **forced before any dynamics are used**. The nonlinear conjugacy machinery of
the draft is therefore irrelevant to the model as specified, and the motivating
reparameterisation ambiguity does not arise. Verified in
`tests/test_counterexamples.py::test_full_column_rank_decoder_forces_h_linear`.

So: **Theorem A** is the linear-decoder statement — complete, proved.
**Theorem B** is the nonlinear-decoder statement — the setting the LFADS
motivation actually requires, and still open.

---

## 3. Theorem A — linear decoder

> **Theorem A.** Let $x_t = W z_t$ with $W$ of full column rank, $F = \bigoplus_i f_i$
> linear, and let $(\tilde F, \tilde W)$ be equivalent with $\tilde F$ modular and
> every block indecomposable. Assume
>
> * **(A1)** each $f_i$ is indecomposable;
> * **(A2)** the spectra $\mathrm{spec}(f_i) \subset \mathbb{C}$ are pairwise disjoint.
>
> Then $K = \tilde K$, the block dimensions agree as multisets, and there is a
> permutation $\sigma$ with
> $$h = P_\sigma (h_1 \oplus \cdots \oplus h_K), \qquad h_i \in GL(d_i),$$
> and $\tilde f_{\sigma(i)}$ similar to $f_i$.

*Proof.* $h \in GL(d)$ by §2; apply `linear_case.md` Theorem L to $S = h$. $\square$

Sharp in both hypotheses: dropping (A1) gives the §3.1 regrouping counterexample,
dropping (A2) gives the $J_2(\lambda) \oplus J_2(\lambda)$ counterexample
(`counterexamples.md` §1–2). Certified numerically in `exp01`; the matching
lemma §3.2 asks for is a corollary here, established *before* the spectral
hypothesis is used (`linear_case.md` §5).

**Perturbation (CLAUDE.md §4 step 7).** For $F_\epsilon = F + \epsilon C$ with
$C$ supported off the diagonal blocks, the recovered partition *is* the pair of
invariant subspaces, and standard perturbation theory gives

$$\angle(U_i^\epsilon, U_i) \;=\; O\!\left(\frac{\epsilon \|C\|}{\mathrm{sep}}\right).$$

Measured in `exp04`: log-log slope in $\epsilon$ of $1.0000$ across three gaps,
the constant gap-independent to within a factor $1.85$, and breakdown at
$\epsilon \approx \mathrm{sep}$ ($19.6°$ at $\epsilon = \mathrm{sep}$, $55.6°$ at
$10\,\mathrm{sep}$). So the target claim "within $O(\epsilon)$ provided the gap
exceeds $C\epsilon$" holds and is sharp — **in the linear case**.

---

## 4. Theorem B — nonlinear decoder (conditional, incomplete)

Now $x_t = g(z_t)$ with $g$ an injective immersion. Equivalence gives
$h = \tilde g^{-1} \circ g$, a diffeomorphism $\Omega \to \tilde\Omega$ with

$$h \circ F = \tilde F \circ h \quad \text{on } \Omega.$$

$h$ is now genuinely nonlinear and the question is whether modularity forces it
to factor.

**Hypotheses.**

* **(B1) Regularity and support.** $\Omega$ compact and $F$-invariant with
  $\mathrm{int}\,\Omega \neq \emptyset$; $h$ is $C^1$ with $\sup_\Omega \|Dh\| < \infty$
  and $\sup \|Dh^{-1}\| < \infty$. The bounded-derivative part is what the cocycle
  bound is measured against. The **nonempty-interior** part is not optional: the
  jet-extraction step of the normal-form route (Route A, `route_a_assessment.md`
  §2.1) needs $\Omega$ to contain an open set, and if $\Omega$ is thin — a single
  orbit — the conjugacy equation holds only on isolated points, constrains no
  cross-derivatives, and the conclusion is **false** (explicit counterexample,
  `route_a_assessment.md` §3.5). This is CLAUDE.md §3.6's support caveat promoted
  to a hypothesis; `systems.sample_initial_conditions` spreads trajectories over
  an annulus precisely to make it hold.
* **(B2) Indecomposability.** Each $f_i$ admits no further splitting into
  independently evolving factors on $\Omega_i$. The nonlinear analogue of (A1);
  making this precise is itself open. `TODO(gap)`
* **(B3) Matching.** A bijection $\sigma$ pairing the modules of $F$ with those
  of $\tilde F$, with $\tilde f_{\sigma(i)}$ conjugate to $f_i$. `TODO(gap)` — see
  §7, and §6.7 for how far Theorem F reduces it (the ordering half is free there,
  and is *not* free here).
* **(B4) Dichotomy separation.** The Lyapunov (Sacker–Sell) spectra
  $\Lambda(f_i) \subset \mathbb{R}$ are pairwise disjoint. This replaces the
  draft's Assumption 4, which was stated on pointwise Jacobian spectra and is
  unusable — see `counterexamples.md` §4.

### 4.1 The cross-derivative argument, corrected

Write $h = (h_1, \dots, h_K)$ and $M_{ij} := \partial h_i / \partial z_j$.
Differentiating $h_i(F(z)) = \tilde f_i(h_i(z))$ with respect to $z_j$:

$$\boxed{\;M_{ij}(F z)\, Df_j(z_j) \;=\; D\tilde f_i(h_i(z))\, M_{ij}(z)\;}$$

**The left $M_{ij}$ sits at $Fz$, not $z$.** The draft wrote both at the same
point, turning a cocycle relation into a pointwise Sylvester equation — valid
only at fixed points of $F$, i.e. a linearisation result at equilibria
(CLAUDE.md §3.3). Iterating,

$$M_{ij}(F^n z)\, Df_j^{(n)}(z_j) = D\tilde f_i^{(n)}(h_i(z))\, M_{ij}(z),$$

so with (B1),

$$\|M_{ij}(z)\| \le \left\|\left[D\tilde f_i^{(n)}\right]^{-1}\right\| \cdot \sup_\Omega\|M_{ij}\| \cdot \left\|Df_j^{(n)}\right\|,$$

whose exponential rate is $\lambda_{\max}(f_j) - \lambda_{\min}(\tilde f_i)$.
Hence

> **Lemma C (cocycle).** If $\lambda_{\max}(f_j) < \lambda_{\min}(\tilde f_i)$
> then $M_{ij} \equiv 0$ on $\Omega$.

Verified numerically to 12 digits: measured rate $-0.30538$ against predicted
$-0.30538$, error $1.5\times10^{-12}$, with $\log\|M_n\|$ falling to $-119$ over
400 steps. With the gap closed the rate is $0.00000$ and $M$ is not driven to
zero — so the gap, not the algebra, is doing the work (`exp05` part 2b).

### 4.2 Where it stops

Lemma C gives one cross-derivative block. Block-diagonality of $h$ needs *both*,
and the two requirements are mutually exclusive:

$$\lambda_{\max}(f_2) < \lambda_{\min}(f_1) \le \lambda_{\max}(f_1) < \lambda_{\min}(f_2) \le \lambda_{\max}(f_2)$$

is a contradiction (`counterexamples.md` §3, measured in `exp05` part 2c: the
two rates are exact negatives, summing to $8.7\times10^{-10}$).

> **What Theorem B currently delivers.** Under (B1)–(B4) with the gap oriented so
> that module 1 dominates, $M_{12} \equiv 0$, so $h_1 = h_1(z_1)$ and
> $$h(z_1, z_2) = \big(h_1(z_1),\; h_2(z_1, z_2)\big)$$
> — $h$ is **triangular**, a skew product. The foliation by module-2 fibres is
> preserved and $h_1$ conjugates $f_1$ to $\tilde f_1$.

This is weaker than the target statement but not vacuous: a triangular $h$ still
identifies module 1 as a dynamical *factor* (a quotient system), which is enough
to pin down $f_1$'s conjugacy class — fixed-point structure, attractor topology,
Lyapunov spectrum.

### 4.3 And the triangular conclusion is sharp

Block-diagonality does not merely fail to follow — **it is false**. With
$f_i(z_i) = \mu_i z_i$, $0 < \mu_1 < \mu_2 < 1$, the map

$$h(z_1, z_2) = \left(z_1 + c\,\mathrm{sgn}(z_2)|z_2|^{p},\; z_2\right), \qquad p = \frac{\log\mu_1}{\log\mu_2} > 1,$$

is an exact conjugacy satisfying (B1)–(B4) as stated, and is triangular but not
block-diagonal. Full discussion in `counterexamples.md` §5; constructed as
`systems.triangular_conjugacy_counterexample()`.

Two things follow. First, **(B4) as stated is too weak** — it asks for disjoint
spectra, whereas Lemma C needs the *oriented* gap, and §4.2 shows the oriented
gap cannot hold in both directions at once. Second, **regularity alone does not
help**: the resonant case $\mu_1 = \mu_2^m$ makes $h$ a polynomial, so even
$C^\infty$ admits the counterexample. Cross-module **non-resonance** is a
necessary hypothesis, not a convenience.

So no proof of block-diagonality under (B1)–(B4) can exist. The hypotheses must
change. `TODO(gap)`

### 4.4 Lemma C does not need the fixed point

Everything above was *measured* at an attracting fixed point, and the
block-diagonality route of §5.3 genuinely requires one — Poincaré–Dulac is a
Poincaré-domain phenomenon. Lemma C is not. Its derivation quantifies over
Lyapunov exponents and never mentions an equilibrium, so it should hold on any
compact invariant set on which those exponents exist. It does.

> **Lemma C′ (attractors).** Let $\Lambda \subset \Omega$ be a compact invariant
> attracting set whose Lyapunov exponents are **uniform over its basin** — every
> orbit in the basin realises the same spectrum. Then Lemma C holds verbatim on
> the whole basin: $\lambda_{\max}(f_j) < \lambda_{\min}(\tilde f_i)$ forces
> $M_{ij}\equiv 0$ there.
>
> Only the **bounded-derivative** clause of (B1) is used —
> $\sup_\Omega\|M_{ij}\| < \infty$, which a $C^1$ conjugacy on a compact
> invariant set supplies. The **open-interior** clause
> $\mathrm{int}\,\Omega\neq\emptyset$ is *not* used, and neither is (B2) or (B3).
> That matters: $\mathrm{int}\,\Omega\neq\emptyset$ is exactly what the
> normal-form route consumes (§5.3 step 1), and an attractor is precisely where
> it is most likely to fail.

*Proof.* Unchanged from §4.1. The iterated relation and the bound
$\|M_{ij}(z)\| \le \|[D\tilde f_i^{(n)}]^{-1}\|\,\sup_\Omega\|M_{ij}\|\,\|Df_j^{(n)}\|$
are pointwise identities on any invariant set. Uniformity supplies
$\tfrac1n\log$ of each factor converging to the stated exponent *at every* $z$,
so $B_n(z)\to0$ for every $z$ and $M_{ij}(z)=0$ for every $z$. $\square$

The uniformity clause is what makes this stronger than a bare appeal to
Oseledets, and it is not an idle hypothesis. Oseledets gives the rates only
$\mu$-a.e. for an ergodic $\mu$; on a limit cycle $\operatorname{supp}\mu$ is the
cycle itself, so that route would conclude $M_{ij}\equiv0$ *on the cycle* and say
nothing about the basin — exactly the ω-limit-set weakness that makes Route 2 of
§5.1 useless. A **normally hyperbolic** attracting periodic orbit or invariant
circle has uniform exponents (every basin orbit converges to it, so its
finite-time exponents converge to the Floquet exponents), and there the
conclusion is recovered on all of $\Omega$.

Verified in `exp08`: on an attracting invariant circle the predicted rate holds
to $2.4\times10^{-14}$ across a sweep of the dominated module, and $M$ is forced
to zero for exactly those partners below the threshold and no others — the
crossing sitting at the radial multiplier $|1-2a|$ to the digit.

The uniformity clause is measured too, not just assumed: the rate is constant to
$2.8\times10^{-16}$ across starting radii from $0.02$ to $2.05$, a hundredfold range
covering essentially the whole basin (which is bounded — $r < \rho\sqrt{(1+a)/a}$
$= 2.0817$ at the defaults, outside which the discrete radial map escapes
immediately). So the decay really does hold at every point of $\Omega$ and not
merely on the attractor, which is the whole difference between Lemma C′ and the
useless a.e. version.

**A new structural constraint falls out.** An attracting cycle carries a
**neutral** exponent $0$ along the phase direction. Lemma C needs
$\lambda_{\max}(f_j) < \lambda_{\min}(\tilde f_i)$, and $\lambda_{\max} = 0$ for
an oscillatory module while $\lambda_{\min} < 0$ for any contracting one. Hence:

1. **An oscillatory module can never be dominated.** It sits at the *top* of the
   filtration or is not separated at all. This is not a limitation of the
   argument — it is a fact about the ordering, and it says the phase of an
   autonomous oscillation is the slowest thing in the system.
2. **At most one oscillatory module can appear in an identified filtration.**
   Two cycles both contribute $0$, so they *share* an exponent, (B4) fails
   outright, and neither cross-derivative is forced. Two independent limit
   cycles in one population are **not** separable by this route, for any
   frequencies and any contraction rates.

> **This collides with the stated scope, and the collision is real.** CLAUDE.md
> §1.1 defines a module as "separate timescales, **distinct oscillatory
> components**, subspaces that evolve without reference to each other" — and
> point 2 says the two-oscillator case, the most natural reading of "distinct
> oscillatory components", is exactly the one the spectral route cannot do.
> Frequency does not help: Lyapunov exponents are blind to rotation number, so
> $\omega_1 \neq \omega_2$ buys nothing (verified with $\omega = 0.50$ vs $0.90$).
> Two co-existing rhythms in one population is not an exotic hypothesis, so this
> is a live limitation on the applied claim, not a corner case.
>
> It is a limitation of *this* route, not necessarily of the problem. Separating
> two cycles needs an invariant finer than the exponent multiset — the rotation
> number is the obvious candidate, and it is a genuine conjugacy invariant that
> the Lyapunov spectrum discards. That is a different argument, not a
> strengthening of this one.
>
> **UPDATE (2026-08-04).** The rotation number is now *built and measured* —
> `spectra.rotation_number`, certified in `exp14` part 1 to machine precision on
> known-answer blocks and invariant under a nonlinear change of coordinates. Two
> limit cycles with $\omega = 0.5$ and $1.3$ have Lyapunov spectra agreeing to
> $3.2\times10^{-18}$ (so `spectral_gap` is exactly $0$ and this lemma has
> nothing to work with) while their rotation numbers differ by $0.0637$
> turns/step. So the *invariant* that separates them exists and is estimable
> from data, including from a fitted model.
>
> `TODO(gap)` — what is still missing is the **theorem**: an argument that a
> conjugacy must preserve the per-module rotation numbers *and* that they pin
> down the decomposition. Rotation number being a conjugacy invariant of a
> single circle map is classical; using it to force a *splitting* is not, and
> nothing here proves it. The measurement is a prerequisite, not the result.
>
> **RESOLVED NEGATIVELY (2026-08-12) — the missing theorem is false.**
> `counterexamples.md` §7: for two shear-free attracting cycles,
> $h(z_1,z_2) = (z_1 z_2/|z_2|,\, z_2)$ is an exact modular conjugacy (residual
> $6.7\times10^{-16}$, (F1) satisfied on an annulus with
> $\sup\|Dh\|=2.72$) carrying $(\omega_1,\omega_2)$ to
> $(\omega_1+\omega_2,\omega_2)$. Two cycles span an invariant torus, and a
> conjugacy acts on $H_1(T^2)=\mathbb{Z}^2$, so **what is identified is the
> rotation *vector* up to $GL(2,\mathbb{Z})$ — not the per-module rotation
> numbers.** This is §1's regrouping counterexample in oscillatory form.
>
> **No caveat survives.** Shear in the donor block breaks the *naive-angle*
> construction ($7.8\times10^{-2}$ at $\beta_2=0.3$), which briefly looked like
> an escape. It is not: the object that advances rigidly is the **asymptotic
> phase** $\Theta = \theta + \beta\sum_k(g^k(r)-\rho)$, and rebuilding $h$ with
> it gives an exact conjugacy at every shear ($\le1.8\times10^{-15}$ over
> $\beta_1\in\{0,0.5\}$, $\beta_2\in\{0,0.3,0.8\}$). In hindsight forced: shear
> is not a conjugacy invariant of a single cycle — $(r,\Theta)$ removes it — so
> it could never have protected one. `counterexamples.md` §7.1.

Point 2 is the resolution of a case that previously produced numbers that could
not be reconciled with the rate formula: it is outside Lemma C's hypotheses, not
a counterexample to it. `spectra.spectral_gap` returns exactly $0$ there.

> **Numerical note, load-bearing.** $\sigma_{\min}$ of an accumulated Jacobian
> product is not measurable past $n \approx 36/(\lambda_{\max}-\lambda_{\min})$:
> the ratio $\sigma_{\min}/\sigma_{\max}$ falls under machine epsilon and the
> SVD returns noise, whose *slope* reads $\lambda_{\max}$. Every fixed-point
> measurement in `exp05` is safe because a `TwistBlock` has spectrum
> $\{\log s, \log s\}$ — spread $0$, no horizon — but a limit cycle has spread
> $|\log|1-2a||$ and a horizon near $n=39$. Fitted over $n\in[200,400)$ the naive
> bound wanders by $2.7$ across `n_max` and by $0.7$ across initial conditions on
> the *same* cycle, and in one no-gap case it reports a decisively negative rate,
> i.e. **a false certification of Lemma C's conclusion where its hypothesis
> fails**. Use `spectra.inverse_jacobian_product_logs` (σ_max of the inverse
> cocycle, stable at every $n$); `cocycle_bound` does, and reports the discarded
> value as `naive_rate` alongside `n_resolvable`. The shear $\beta$, initially
> suspected, is innocent: it shifts the intercept by exactly
> $\sqrt{(\beta/2a)^2+1}$ and leaves the rate invariant.

**What this buys.** Theorem F (the filtration, §6) is the route standing on
proved ground, and it now extends past the fixed-point regime to periodic
attractors — the case §5.3's caveat (b) had to exclude for block-diagonality.
**Still open:** attractors with *non-uniform* exponents (chaotic, non-uniformly
hyperbolic). There Oseledets returns only the a.e. statement and the conclusion
retreats to $\operatorname{supp}\mu$; whether it extends to the basin is
unresolved. `TODO(gap)`

### 4.5 Lemma D — behaviour kills the cross-derivative the gap provably cannot

**New (2026-08-03).** §4.3 shows the spectral gap yields a *triangular* $h$ and
that this is sharp: the orientation that would kill the surviving block is
self-contradictory (§3.7), so the dynamics are genuinely exhausted. Lemma D
closes the remaining direction using the behavioural auxiliary instead. It is the
theorem Route B needed, and it is obtained **dynamically** — not by patching
Khemakhem et al., whose assumption (iv) fails globally for $u$-invariant
components (`route_a_assessment.md` §6.1).

**Setting.** $F = f_A \oplus f_B$, $\tilde F = \tilde f_A \oplus \tilde f_B$, both
modular; $f_A, \tilde f_B$ linear and semisimple (or read as linear parts, with
the expansion below formal). $h \circ F = \tilde F \circ h$.

- **(D1) One-sided gap.** $\rho(\tilde f_B) < \rho_{\min}(f_A) < 1$, where
  $\rho_{\min}$ is the least eigenvalue modulus. *This is Lemma C's own
  hypothesis* — D asks for nothing extra from the dynamics.
- **(D2) Variance-modulated behaviour.** Conditional on $u$, the law of $z_A$ is
  $\sigma_u\,\mu_A$ for a fixed $\mu_A$ of full-dimensional support, with **at
  least two distinct** $\sigma_u$. (This is exactly `behavior.py`'s
  `mode="variance"`: $z^A \sim N(0, s(u)^2 I)$.)
- **(D3)** $z_B \perp z_A$ with $u$-invariant law.
- **(D4)** The law of $h_B(z_A, z_B)$ is $u$-invariant.

> **Lemma D.** Under (D1)–(D4), with $h_B$ of additive form
> $h_B(z_A,z_B) = z_B + \psi(z_A)$, we have $\psi \equiv 0$: $h_B$ does not
> depend on $z_A$. With §4.2's $M_{AB}\equiv 0$ this gives **block-diagonal $h$**.

**Proof.**

*Step 1 — the conjugacy makes $\psi$ a semiconjugacy.* The $B$-component of
$h\circ F = \tilde F\circ h$ reads
$f_B z_B + \psi(f_A z_A) = \tilde f_B(z_B + \psi(z_A))$. Taking $\tilde f_B$
linear and matching the $z_B$-free part, $\psi \circ f_A = \tilde f_B \circ \psi$.
So a nonzero $\psi$ is precisely a **shared factor** between the modules — the
§3.8/§7.1 failure mode, here made quantitative.

*Step 2 — only resonant degrees survive.* Expand $\psi = \sum_{m}\psi_m$ into
parts homogeneous of multi-degree $m$ in the eigencoordinates of $f_A$. Then
$\psi_m(f_A z) = \lambda_A^{m}\,\psi_m(z)$, so Step 1 forces
$\lambda_A^{m}\psi_m = \tilde f_B \psi_m$: either $\psi_m \equiv 0$, or
$\lambda_A^{m} \in \operatorname{spec}(\tilde f_B)$ — a **cross-module
resonance**. (If cross-module non-resonance is assumed, this alone gives
$\psi\equiv0$ and behaviour is not needed. Lemma D's content is the resonant
case, which §3.7 proves is *not* vacuous and *cannot* be removed by regularity.)

*Step 3 — the gap forces degree $\ge 2$.* Suppose $\lambda_A^m = \tilde\lambda_B
\in \operatorname{spec}(\tilde f_B)$. Taking moduli,
$$\rho_{\min}(f_A)^{|m|} \;\le\; |\lambda_A^{m}| \;=\; |\tilde\lambda_B| \;\le\; \rho(\tilde f_B) \;<\; \rho_{\min}(f_A).$$
Since $\rho_{\min}(f_A) < 1$, the outer inequality forces $|m| > 1$, hence
$|m| \ge 2$. **The gap that makes Lemma C work is the same fact that gives
behaviour its grip** — every surviving coupling is at least quadratic, so it
*must* respond to a change of scale.

*Step 4 — behaviour detects every such degree.* By (D2), $z_A = \sigma_u\zeta$,
and homogeneity gives $\psi(\sigma\zeta) = \sigma^{p}\psi(\zeta)$ with
$p = |m| \ge 2$. By (D3) $\zeta \perp z_B$, so the characteristic function of
$h_B$ factorises: $\varphi_{h_B}(t) = \varphi_{z_B}(t)\,\varphi_{\psi}(\sigma^{p}t)$.
$\varphi_{z_B}$ is continuous with $\varphi_{z_B}(0)=1$, hence nonzero on a
neighbourhood $U$ of the origin. (D4) with two levels $\sigma_1 \neq \sigma_2$
then gives $\varphi_\psi(\sigma_1^{p}t) = \varphi_\psi(\sigma_2^{p}t)$ on $U$.
Put $r = (\sigma_2/\sigma_1)^{p}$; since $p \ge 2$ and $\sigma_1\neq\sigma_2$,
$r \neq 1$, and WLOG $r<1$. Iterating, $\varphi_\psi(t) = \varphi_\psi(r^k t) \to
\varphi_\psi(0) = 1$, so $\varphi_\psi \equiv 1$ on $U$ and $\psi(\zeta) = 0$
a.s. As $\psi$ is polynomial and $\mu_A$ has full-dimensional support,
$\psi \equiv 0$. $\blacksquare$

**Two levels suffice.** Step 4 uses only $\sigma_1 \ne \sigma_2$ — sharply weaker
than iVAE's $nk+1$ points with an invertible matrix of natural-parameter
differences, and it is *why* the assumption-(iv) obstruction is avoided rather
than confronted.

**Why degree $0$ is the unique escape, and why (D1) closes it.** A coupling that
hides from behaviour must have $\sigma$-invariant law; for homogeneous $\psi$
that means $p = 0$, i.e. $\psi$ scale-invariant (e.g. $z_A/\|z_A\|$, whose law is
$\sigma$-free for isotropic $\mu_A$ — verified: variance $1.2457/1.2491/1.2447$
across $\sigma = 0.6/1.0/1.6$). But Step 2 then requires
$\lambda_A^{0} = 1 \in \operatorname{spec}(\tilde f_B)$, i.e. $\rho(\tilde f_B)\ge1$
— no contraction in $B$, contradicting (D1). **The scale-invariant escape and the
spectral gap are mutually exclusive.**

### 4.5a Lemma D′ — (D1) is far more than the proof needs

**New (2026-08-12).** The paragraph above is the key to a much stronger
statement, and it was hiding in plain sight. It says (D1)'s job is to exclude
**degree $0$**. But excluding degree $0$ needs only $1 \notin
\operatorname{spec}(\tilde f_B)$ — not a gap, not an ordering, not even a
contraction. And Step 3, the only other place (D1) appears, is **not load-bearing
for the conclusion**: it establishes $|m|\ge2$, whereas Step 4's iteration
$r = (\sigma_2/\sigma_1)^{p} \neq 1$ needs only $p \ge 1$.

> **Lemma D′.** Replace (D1) by
>
> * **(D1$'$)** $1 \notin \operatorname{spec}(\tilde f_B)$.
>
> Then, under (D1$'$), (D2)–(D4) with $h_B$ additive and $\psi$ homogeneous,
> the conclusion of Lemma D holds: $\psi \equiv 0$, so $M_{BA}\equiv0$.

*Proof.* Steps 1–2 are unchanged (pure algebra; no spectral hypothesis was ever
used there). **Step 3 is deleted.** Degree $0$ is excluded by (D1$'$) via Step 2,
exactly as the paragraph above argues. For $p \ge 1$, Step 4 runs verbatim:
$\sigma_1 \neq \sigma_2$ and $p\ge1$ give $r \neq 1$, and the iteration
$\varphi_\psi(t) = \varphi_\psi(r^k t)\to1$ forces $\psi(\zeta)=0$ a.s. $\square$

(D1) $\Rightarrow$ (D1$'$), since (D1) makes $\tilde f_B$ a strict contraction.
So Lemma D′ strictly generalises Lemma D, and **Step 3 is demoted from a step of
the proof to an observation about what the coupling looks like when a gap
happens to be present.**

**Why this is worth having: it reaches the case Theorem F cannot.** Two modules
with *identical* spectra are the linear form of task 23's two-oscillator
problem — (B4) is exactly $0$, (F3) is not ordered, Lemma C has no gap, and
(D1) fails outright. (D1$'$) holds there for free.

> **Witness** (`systems.gapless_resonant_coupling`, asserted in
> `tests/test_behavior.py`). Both modules are $s\,R(\omega)$ with $s=0.85$,
> $\omega=0.70$; $\psi = cI$ is **degree 1**, resonant because $f_A$ and
> $\tilde f_B$ share their spectrum (resonance residual exactly $0$). Then
> $h(z_A,z_B) = (z_A,\, z_B + c\,z_A)$ is an exact conjugacy (residual
> $6.7\times10^{-16}$) with $\|M_{BA}\| = 0.99$. Spectra identical:
> `spectral_gap` $=0$, `filtration_gap` $=0$ and **not ordered**; `gap_holds`
> is `False`; $\min|\lambda_B - 1| = 0.65$, so (D1$'$) holds.
>
> Behaviour still kills it. Normalised $u$-dependence of $h_B$ at two levels
> $\sigma\in\{0.6,1.6\}$, against the $c=0$ control, with $n$ per level:
>
> | $n$ | $c=0$ | $c=0.25$ | $c=0.5$ | $c=0.7$ |
> |---|---|---|---|---|
> | $2\times10^3$ | 0.0336 | 0.1478 | 0.2915 | 0.4677 |
> | $1.28\times10^5$ | **0.0074** | 0.0883 | 0.2880 | 0.4476 |
>
> The control falls like $n^{-1/2}$ while every treated column converges to a
> nonzero limit — so the signal is real and the floor is sampling noise, checked
> rather than assumed (§3.9). At $n=1.28\times10^5$ even $c=0.25$ clears the
> floor $12\times$.

**And the degree-1 resonance is exactly "equal frequencies".** For two
rotation-scalings at the same rate, $\lambda_A^m$ has modulus $s^{|m|}$, so only
$|m|=1$ can resonate, and it does iff $\omega_A = \pm\omega_B$. Measured with
`systems.sylvester_kernel_dim`: kernel dimension $2$ at $\omega_B = \pm0.70$ and
$\mathbf{0}$ at $\omega_B \in \{1.30, 2.10\}$. **So for two oscillatory modules
of equal contraction rate, either the frequencies differ and no coupling exists
at all — behaviour is not even needed — or they agree and two behaviour levels
remove it.** That is a complete answer for this class, and it is the first
statement in this repo that separates two oscillators with a theorem rather than
a measurement (cf. §6.5, where the rotation-number route is still `TODO(gap)`).

> **A gap found while checking this, and it is pre-existing — now CLOSED in
> §4.5b.** Step 4 writes $\psi(\sigma\zeta) = \sigma^p\psi(\zeta)$ — it treats
> $\psi$ as homogeneous of a *single* degree. Step 2 permits several degrees to
> survive at once (different $m$ resonating with different eigenvalues of
> $\tilde f_B$), and then $\Psi_\sigma = \sum_j \sigma^{p_j}\psi_{p_j}$ is not
> homogeneous and the iteration does not run. **This was not a cost of dropping
> (D1)** — with (D1) the surviving set is $\{|m|\ge2\}$, still not a singleton —
> so Lemma D as previously stated had the same hole.
>
> **My conjectured repair was the right shape and the wrong count.** I guessed
> "$k$ degrees, $k+1$ levels, by a Vandermonde in $\sigma^{p_j}$". A Vandermonde
> in $\sigma^{p_j}$ does appear, but at $k+1$ levels it only forces
> $\mathbb{E}[\Psi_p(\zeta)] = 0$ for each $p$ — first moments, which do not give
> $\Psi_p \equiv 0$. The argument has to run on **second** moments, and then the
> relevant count is the size of the **sumset** $P+P$, not of $P$. §4.5b.

> **Witness** (`systems.lemma_d_witness`, asserted in `tests/test_behavior.py`).
> In complex coordinates $f_A(z) = s e^{i\alpha}z$, $\psi(z)=z^2$,
> $\tilde f_B(w) = s^2 e^{2i\alpha}w$: then $h(z_A,z_B) = (z_A,\,z_B + c\,z_A^2)$
> is an **exact** conjugacy of a modular $F$ to itself (residual
> $2.7\times10^{-15}$), is **not** block-diagonal ($\|\partial h_B/\partial z_A\|
> = 1.75$), and **satisfies the one-sided gap** ($\rho(f_B)=s^2 < s$) — so it is a
> live obstruction that §4.2 provably cannot remove. The resonance is exact
> ($|\lambda_A^2 - \lambda_B| = 1.1\times10^{-16}$), and behaviour resolves it:
> $\operatorname{var}(h_B) = 1.25,\,2.96,\,13.85$ at $\sigma = 0.6,\,1.0,\,1.6$
> (exactly $1 + 4c^2\sigma^4$).

**Open.** (a) *Non-additive $h_B$.* The graded reduction still applies —
writing $h_B = \sum_m z_A^m c_m(z_B)$, the conjugacy gives
$\lambda_A^m c_m(f_B z_B) = \tilde f_B c_m(z_B)$ and evaluating at the fixed
point $z_B = 0$ reproduces Steps 2–3 — but Step 4's characteristic-function
factorisation uses independence, which fails when $c_m$ depends on $z_B$.
`TODO(gap)`.

> **Sharpening (a): the missing piece cannot come from behaviour.** Step 4's
> entire behavioural input is (D4). There is a non-additive $h_B$ satisfying
> **(D1)–(D4) exactly** with $M_{BA}\neq0$: with $p(z_B\mid u) = N(0,I_2)$,
> $$h(z_A,z_B) \;=\; \big(z_A,\; R(\gamma z_{A,1})\,z_B\big),\qquad R=\text{rotation}.$$
> For every $u$ and every fixed $z_A$, $R(\cdot)z_B \sim N(0,I)$ independent of
> $z_A$, so the law of $h_B$ is exactly $u$-invariant — (D4) holds — while
> $\|M_{BA}\| = |\gamma|\,\|z_B\|$. Measured: normalised $u$-dependence at the
> sampling floor for $\gamma$ up to $2$. **So no strengthening of (D4) closes (a);
> the work must come from Steps 1–3.** Here $c_m$ depends on $z_B$ linearly, which
> is exactly the independence failure flagged above — the witness shows that
> failure is not a technical artefact of the proof but a real obstruction.
>
> Lemma D's *conclusion* is untouched, and the reason says how (a) must be proved:
> this $h$ is **not** a modular conjugacy. Since $f_B$ is a scaled rotation it
> commutes with $R$, so the $B$-component of $h\circ F = \tilde F\circ h$ needs
> $\theta\circ f_A - \theta$ constant; at the fixed point of a contracting $f_A$
> that constant is $0$, forcing $\theta$ constant. **Step 1 does here what Step 4
> does in the additive case.** Measured: defect $0$ at $\gamma=0$ (exactly) and
> growing monotonically otherwise.
>
> $\dim z_B \ge 2$ is necessary — at $\dim z_B = 1$ the $p_B$-preserving
> transports are the two isolated points $\pm\mathrm{id}$, so a family continuous
> in $z_A$ is constant. The escape is the positive-dimensional transport group,
> which exists for every $p_B$ from dimension 2 up.
>
> `systems.nonadditive_behavioural_escape()`, three tests in `tests/test_behavior.py`. (b) *Anisotropic modulation.* (D2) assumes $u$ scales $z_A$
isotropically; a general covariance modulation makes $\psi(\sigma_u \zeta)$
non-homogeneous in a single scalar and Step 4 needs replacing. `TODO(gap)`
(c) *Nonlinear $f_A$, $\tilde f_B$*: **CLOSED 2026-08-14 in §4.5b–c**, and not via
§5.3. The $\tilde f_B$ half is vacuous — additive $h_B$ forces $\tilde f_B$
affine — and the $f_A$ half needs no hypothesis on the dynamics at all once
Step 2 is read as "$\psi$ is a Koopman eigenfunction of $f_A$". The cost is
level richness, not regularity.

### 4.5b Lemma D″ — several surviving degrees at once

**New (2026-08-14).** This closes the `TODO(gap)` flagged in §4.5a. Steps 1–3
never assumed $\psi$ homogeneous; only Step 4 did, when it wrote
$\psi(\sigma\zeta) = \sigma^p\psi(\zeta)$. Replacing the characteristic-function
iteration with a **second-moment** argument removes the assumption outright.

**Setting.** As in Lemma D′, with $\psi = \sum_{p\in P}\Psi_p$ graded by total
degree, $\Psi_p$ homogeneous of degree $p$, and $P \subset \mathbb{Z}_{\ge1}$
(degree $0$ is excluded by (D1′), which is the one thing that hypothesis is for).
Add:

- **(D5) Finite second moments.** $\mathbb{E}\lVert\Psi_p(\zeta)\rVert^2 < \infty$
  for each $p$. Automatic for polynomial $\psi$ and any $\mu_A$ with all moments
  — in particular for `behavior.py`'s Gaussian modulation.

**$P$ is automatically finite, from the spectra alone.** Step 2 needs
$\lambda_A^{m} \in \operatorname{spec}(\tilde f_B)$. If $f_A$ contracts and
$\tilde f_B$ is invertible then $\rho(f_A)^{|m|} \ge |\lambda_A^m| \ge
\rho_{\min}(\tilde f_B) > 0$, so

$$p \;\le\; \frac{\log \rho_{\min}(\tilde f_B)}{\log \rho(f_A)}.$$

No gap and no ordering is used. A bound below $1$ means **no** degree can
resonate and $\psi\equiv0$ with no behaviour at all — e.g. $\rho(f_A)=0.5$
against $\rho_{\min}(\tilde f_B)=0.9$ gives $0$. (`systems.surviving_degree_bound`.)

> **Lemma D″.** Assume (D1′), (D3), (D4), (D5), $h_B$ additive, and (D2) with
> $L$ distinct levels. If
> $$L \;\ge\; \lvert P + P\rvert + 1,$$
> $P+P = \{p+p' : p,p'\in P\}$ the sumset, then $\psi \equiv 0$.

**Proof.** Fix $t \in \mathbb{R}^{d_B}$ and put $A_p := \langle t, \Psi_p(\zeta)\rangle$,
with Gram matrix $C_{pp'} := \operatorname{Cov}(A_p, A_{p'})$ — positive
semidefinite. Under (D2), $z_A = \sigma_u\zeta$, so
$\langle t,\psi(z_A)\rangle = \sum_p \sigma_u^{\,p}A_p$. By (D3) $z_B$ is
independent of $\zeta$ with $u$-invariant law, so variances add:

$$\operatorname{Var}\langle t, h_B\rangle \;=\; \operatorname{Var}\langle t, z_B\rangle \;+\; V_t(\sigma_u), \qquad V_t(\sigma) \;=\; \vec s(\sigma)^{\!\top} C\,\vec s(\sigma), \quad \vec s(\sigma)_p = \sigma^{p}.$$

(D4) makes the left side $u$-invariant and (D3) makes the first right-hand term
$u$-invariant, so $V_t$ takes the **same value $c$ at all $L$ levels**.

$V_t$ is a polynomial whose exponents lie in $P+P \subset \mathbb{Z}_{\ge2}$, so
it has at most $\lvert P+P\rvert$ terms and $V_t(0) = 0$. Then $V_t - c$ has at
most $\lvert P+P\rvert + 1$ nonzero terms, hence by Descartes' rule at most
$\lvert P+P\rvert$ positive roots. It has $L \ge \lvert P+P\rvert + 1$ of them,
so $V_t \equiv c$ identically; evaluating at $0$ gives $c = 0$, so $V_t \equiv 0$
as a polynomial, i.e. every coefficient
$\gamma_q = \sum_{p+p'=q}C_{pp'}$ vanishes.

*That forces $C = 0$, using positive semidefiniteness.* Induct up the degrees.
The lowest exponent $q = 2p_1$ is attained only by the pair $(p_1,p_1)$, so
$\gamma_{2p_1} = C_{p_1p_1} = 0$; a PSD matrix with a zero diagonal entry has
that whole row and column zero. Having killed rows $p_1,\dots,p_{j-1}$, the
exponent $q = 2p_j$ receives contributions only from $(p_j,p_j)$ and from pairs
involving some $p_i$, $i<j$, which are now zero — so $C_{p_jp_j} = 0$ and row
$p_j$ vanishes too.

In particular $\operatorname{Var}(A_p) = 0$, so $\langle t,\Psi_p(\zeta)\rangle$
is a.s. constant. It is a homogeneous polynomial of degree $p \ge 1$ and $\mu_A$
has full-dimensional support, so it is constant on an open set, hence constant,
hence $0$. This holds for every $t$, so $\Psi_p \equiv 0$ for every $p$. $\blacksquare$

**Why second moments and not the characteristic function.** Step 4's iteration
needs a *scaling* action on the argument, and with several degrees the map
$\vec s \mapsto (r^{p}s_p)$ does not preserve the curve $\{\vec s(\sigma)\}$ —
that is exactly the obstruction. Variances collapse the whole problem onto a
single scalar polynomial in $\sigma$, where the level count is elementary.

#### The two-level economy mostly survives

The bound above is worst-case. Two refinements recover $L = 2$ in the cases that
actually arise, and the second is sharp.

**(R1) Symmetric $\mu_A$ kills the odd coefficients.** The coefficient of
$\sigma^{q}$ is $\sum_{p+p'=q}\operatorname{Cov}(A_p,A_{p'})$, a moment of a
homogeneous polynomial of degree $q$; for $\mu_A$ symmetric and $q$ odd it
vanishes exactly. Only $(P+P)\cap 2\mathbb{Z}$ counts. *Verified*: for
$P=\{1,2\}$ under $N(0,I)$ the entry $\operatorname{Cov}(A_1,A_2)$ is a pure
sampling floor, RMS over 64 draws falling $4.19\times10^{-2} \to 2.07\times10^{-2}
\to 9.17\times10^{-3} \to 4.48\times10^{-3}$ as $n$ goes $5\text{k}\to320\text{k}$
— ratios $2.02, 2.26, 2.05$, i.e. $n^{-1/2}$, so the floor is checked and not
assumed (§3.9).

**(R2) Opposite parities give two levels.** If no two distinct elements of $P$
share a parity, (R1) kills every off-diagonal term and
$V_t(\sigma) = \sum_p \sigma^{2p}\operatorname{Var}(A_p)$ has non-negative
coefficients and no constant term — strictly increasing on $(0,\infty)$. **Two
levels suffice.** $\lvert P\rvert = 1$ is the special case, so Lemma D′ is
recovered exactly.

**(R3) For $\lvert P\rvert = 2$ the criterion is sharp, and it is a correlation
threshold.** With $D_j := \sigma_1^{\,j} - \sigma_2^{\,j}$, a two-level tie
exists for some relative scaling of the two degrees **iff**

$$\operatorname{corr}(A_p, A_q)^2 \;\ge\; \frac{D_{2p}\,D_{2q}}{D_{p+q}^{\,2}}.$$

The threshold is close to $1$, so hiding from two levels requires the two degrees
to be **nearly perfectly anticorrelated** under $\mu_A$
(`systems.two_level_tie_threshold`). Measured:

| $P$ | levels | threshold on $\lvert\operatorname{corr}\rvert$ | realised under $\mu_A$ | tie? |
|---|---|---|---|---|
| $\{1,2\}$ | $0.6, 1.6$ | $0.9689$ | $-0.0004$ — Gaussian, opposite parity | no |
| $\{1,3\}$ | $0.6, 1.6$ | $0.9444$ | $+0.7749 = 3/\sqrt{15}$ — Gaussian | **no** |
| $\{2,3\}$ | $0.6, 1.6$ | $0.9961$ | $+0.0003$ — Gaussian | no |
| $\{1,2\}$ | $0.6, 1.6$ | $0.9689$ | $+0.9782$ — skewed $N(1,0.3^2)$ | **yes** |

Row 2 is the informative one: $P=\{1,3\}$ has *equal* parities, so (R2) does not
apply and the counting bound asks for $4$ levels — but the realised correlation
$3/\sqrt{15}$ falls well short of the threshold, so two levels do suffice.
**The counting bound is conservative; (R3) is what is true.**

**And (R3) says how to choose the levels.** By AM–GM,
$\sigma_2^{2p}\sigma_1^{2q} + \sigma_1^{2p}\sigma_2^{2q} \ge
2(\sigma_1\sigma_2)^{p+q}$, so $D_{2p}D_{2q}\le D_{p+q}^2$ and the threshold
always lies in

$$\left[\ \frac{2\sqrt{pq}}{p+q},\ 1\ \right) ,$$

the lower end — the geometric-to-arithmetic mean ratio of the two degrees —
attained as the levels coincide, and rising to $1$ as they separate:
$\sigma_2/\sigma_1 = 1.5, 5, 20$ give $0.9488, 0.9869, 0.9989$. A tie is
therefore never free, and **spreading the two behaviour levels strictly shrinks
the escape**. That is a design instruction for any experiment imposing (D2), and
it costs nothing.

> **Witness** (`systems.multidegree_resonant_coupling`, asserted in
> `tests/test_behavior.py`). $f_A = \mu I_2$ and $\tilde f_B =
> \operatorname{diag}(\mu,\mu^2)$ with $\mu = 0.8$; then
> $$\psi(z) = \big(c_1\langle a,z\rangle,\ c_2\langle a,z\rangle^2\big)$$
> satisfies $\psi\circ f_A = \tilde f_B\circ\psi$ exactly with **two** surviving
> degrees, $P = \{1,2\}$ (residuals $0$ and $0$ — both resonances are structural,
> not numerical). $h(z_A,z_B) = (z_A, z_B + \psi(z_A))$ is an exact modular
> conjugacy (residual $3.6\times10^{-15}$) with mean $\lVert M_{BA}\rVert_F = 1.84$,
> and $\min\lvert\lambda_B - 1\rvert = 0.20$ so (D1′) holds. Measured
> $V_t(\sigma)$ matches the Gram prediction to $2\times10^{-15}$ relative at
> $\sigma \in \{0.6, 1.0, 1.6, 2.2\}$.

**The honest boundary: a tie defeats the argument, not the lemma.** At the
constructed two-level tie (skewed $\mu_A$, ratio $k = -0.1583$) the variance of
$h_B$ agrees to $3\times10^{-17}$ — and the rest of the law does not:

| $\sigma$ | mean | variance | skewness | kurtosis |
|---|---|---|---|---|
| $0.6$ | $+0.6888$ | $0.016842$ | $-0.267$ | $3.09$ |
| $1.6$ | $+1.2983$ | $0.016842$ | $-1.770$ | $7.31$ |

So (D4) — equality of *laws* — still fails there, and the coupling is still
detectable at two levels by any statistic other than the variance. What Lemma D″
establishes is the level count **for the second-moment argument**; whether two
levels always suffice for (D4) itself is not settled here. `TODO(gap)`

**Net effect on the economy claim.** Against Kong's $2n_s+1$ and Sun's $2\ell+1$
(task 38), the behavioural half now costs: two levels whenever the degrees have
distinct parities under a symmetric $\mu_A$ (R2), two levels for the two-degree
cases that arise in practice (R3), and $\lvert P+P\rvert + 1$ in the worst case —
with $\lvert P\rvert$ itself bounded by the spectra. The two-level economy is
**not** lost to the multi-degree case.

### 4.5c Lemma D‴ — nonlinear modules, via Koopman eigenfunctions

**New (2026-08-14).** This is open item (c). It turns out to be **half vacuous
and half closed**, and neither half needs the normal-form machinery of §5.3 —
which matters, because §5.3 is Poincaré-domain (fixed-point) only, and routing
(c) through it would forfeit exactly the reach Lemma D′ was proved to gain.

**The $\tilde f_B$ half is vacuous: additive $h_B$ *forces* $\tilde f_B$ affine.**
Step 1 matches the $z_B$-free part of
$f_B(z_B) + \psi(f_A(z_A)) = \tilde f_B\big(z_B + \psi(z_A)\big)$. That split is
available only if $\tilde f_B$ is additive. So a nonlinear $\tilde f_B$ is not a
gap in Lemma D — it is **outside the additive class by construction**. Measured:
with $\tilde f_B(w) = \mu w + 0.3w^2$ the two sides differ by $1.17$, against
$2.2\times10^{-16}$ for the affine one. (Non-additive $h_B$ is open item (a),
which is separately obstructed — `nonadditive_behavioural_escape`.)

**The $f_A$ half: Step 2 was never about linearity.** With $\tilde f_B$ affine,
$\tilde f_B(w) = \tilde Bw + b$, Step 1 reads $\psi\circ f_A = \tilde B\,\psi$.
In the eigenbasis of $\tilde B$ that says, componentwise,

$$\psi_i \circ f_A \;=\; \tilde\lambda_{B,i}\,\psi_i,$$

i.e. **each component of $\psi$ is a Koopman eigenfunction of $f_A$**, with
eigenvalue the corresponding eigenvalue of $\tilde B$. For linear $f_A$ the
Koopman eigenfunctions are the monomials $z^m$ with eigenvalue $\lambda_A^m$,
which is Step 2 verbatim. **So Step 2 holds for arbitrary $f_A$** — no
linearity, no hyperbolicity, no normal form, no Poincaré domain.

> **The surviving coupling between two modules is a Koopman eigenfunction of the
> driving module whose eigenvalue is an eigenvalue of the driven one.** That is
> the statement to carry forward; the monomial/resonance form is the linear
> special case.

This also connects two results that looked unrelated. For an attracting limit
cycle the asymptotic phase $\Theta$ of §7.1 satisfies $\Theta\circ f = \Theta +
\omega$, so $e^{i\Theta}$ is a **unimodular Koopman eigenfunction** with
eigenvalue $e^{i\omega}$ — verified to $2.0\times10^{-15}$ at shears
$\beta = 0, 0.5, 1.2$. The torus regrouping of §7 was built out of Koopman
eigenfunctions all along, which is why it was insensitive to shear.

> **Lemma D‴.** Let $h_B(z_A,z_B) = z_B + \psi(z_A)$ with $\psi$ real-analytic,
> $\tilde f_B$ affine with linear part $\tilde B$, and $f_A$ **arbitrary**.
> Assume (D3), (D4), (D5),
>
> * **(D1″)** $0$ is the only fixed point of $\tilde f_B$ — equivalently, for
>   affine $\tilde f_B$, $1 \notin \operatorname{spec}(\tilde B)$;
> * **(D2″)** the level set $\{\sigma_u\}$ has a limit point in $(0,\infty)$.
>
> Then $\psi \equiv 0$, hence $M_{BA} \equiv 0$.

**Proof.** Fix $t$ and put $W_t(\sigma) := \operatorname{Var}\langle t,
\psi(\sigma\zeta)\rangle$. Exactly as in §4.5b, (D3) makes variances add and
(D4) forces $W_t$ to take one value across the levels. Under (D5) and
analyticity of $\psi$, $W_t$ is analytic in $\sigma$; being constant on a set
with a limit point it is constant, and $W_t(0) = 0$ because $\psi(0)$ is
deterministic. So $W_t \equiv 0$, hence $\langle t,\psi(\sigma\zeta)\rangle$ is
a.s. constant for every $t$, hence $\psi$ is constant on $\operatorname{supp}
\mu_A$ and therefore — analytic, full-dimensional support — constant everywhere,
$\psi \equiv c$. Step 1 then gives $c = \tilde f_B(c)$, so $c$ is a fixed point
of $\tilde f_B$, and (D1″) gives $c = 0$. $\blacksquare$

Note the proof never touches $f_A$. **All the dynamics does is force $\psi$ to
be a Koopman eigenfunction** — the killing is entirely distributional.

#### The price is level richness, and it is genuinely necessary

Lemma D″'s finite counts do **not** transfer, and the reason is structural: a
Koopman eigenfunction of a nonlinear map is not homogeneous, so
$W_t(\sigma)$ stops being a polynomial in $\sigma$ and there is nothing to
count. In the witness $\psi$ involves $\tanh$, whose Taylor expansion has
infinitely many degrees — $19$ above $10^{-9}$ by order $25$ — so
`surviving_degree_bound` does not apply.

**And no finite count can work.** With $\operatorname{spec}(\tilde B) =
\{\mu,\dots,\mu^{k}\}$ there are $k$ eigenfunctions $\phi,\dots,\phi^{k}$, hence
$k-1$ free ratios, and they can be solved to tie $k$ levels exactly:

| eigenfunctions $k$ | levels tied | spread of $W_t$ across them | $\lVert c\rVert$ |
|---|---|---|---|
| $3$ | $0.6, 1.0, 1.6$ | $5.6\times10^{-14}$ | $1.14$ |
| $4$ | $0.5, 0.9, 1.3, 1.8$ | $2.8\times10^{-15}$ | $2.58$ |

$\lVert c\rVert \ne 0$, so these are genuine nonzero $\psi$. Since $k$ is bounded
by the number of distinct eigenvalues of $\tilde B$, **the number of levels the
second-moment argument needs grows with $\dim z_B$** — which is why (D2″) asks
for a limit point rather than a count.

**Same honest boundary as §4.5b.** These ties defeat the *argument*, not (D4):
the variance agrees while the rest of the law does not. What is established is
that the second-moment method cannot be pushed below $k+1$ levels, so (D2″) is
load-bearing rather than an artefact of how the proof is written. `TODO(gap)`

> **Witness** (`systems.koopman_coupling_witness`, asserted in
> `tests/test_behavior.py`). $f_A(z) = \operatorname{artanh}(\mu\tanh z)$ with
> $\mu = 0.7$ — analytic, contracting, and $16.75\%$ nonlinear against its best
> linear fit — with $\tilde B = \operatorname{diag}(\mu,\mu^2)$ and
> $\psi = (c_1\tanh, c_2\tanh^2)$. Koopman residuals
> $1.7\times10^{-16}$ and $2.2\times10^{-16}$; semiconjugacy $2.2\times10^{-16}$;
> full conjugacy $4.4\times10^{-16}$. Behaviour still kills it at two levels here
> ($\operatorname{Var}\psi_1 = 0.181, 0.320, 0.457$ at $\sigma = 0.6, 1.0, 1.6$),
> because $\tanh$ is odd and so (R1)'s parity argument survives verbatim —
> $\operatorname{Cov}(\phi,\phi^2) = 2.2\times10^{-5}$, a sampling floor.

**The trade, stated plainly.** Structure in the dynamics and richness in the
behaviour are interchangeable. Linear modules impose a grading on $\psi$ and buy
the two-level economy; nonlinear modules impose nothing and pay with a
continuum of levels. For the applied claim this is a mild price — a continuously
varying behavioural covariate (reach speed, angle) is *more* natural than two
discrete conditions, not less.

**What (c) leaves open.** Only the interaction with open item (a): both here and
in §4.5b, $h_B$ is additive. Nonlinear *modules* are done; non-additive
*couplings* are not, and §4.5a's escape witness shows that half cannot be closed
from (D4) at all.

---

## 5. Routes past the obstruction

### 5.1 Two dead ends — do not re-derive these

Both were proposed here before the §4.3 counterexample was found. **Both are
dead at $C^1$ regularity**, because §4.3 exhibits a $C^1$ conjugacy satisfying
every hypothesis they would use. Recorded so the next session does not spend
time on them.

**Route 1 — apply Lemma C on the inverse.** $h^{-1}$ conjugates $\tilde F$ to
$F$, and its cross-derivative obeys the same relation with $f$ and $\tilde f$
swapped — so the useful direction is the *same* one, not the complementary one.
No new information. In the §4.3 example $h^{-1}$ is triangular in the same
orientation as $h$, which settles it.

**Route 2 — run the cocycle backwards.** Rearranging the boxed relation,
$$M_{21}(F^n z) = D\tilde f_2^{(n)}(h z)\, M_{21}(z)\, \left[Df_1^{(n)}(z_1)\right]^{-1},$$
whose rate is negative under the *same* one-sided gap, giving
$\|M_{21}(F^n z)\| \to 0$ — i.e. $M_{21} \equiv 0$ on the $\omega$-limit set but
not on $\Omega$. This is **optimal, not merely partial**: in the §4.3 example
the $\omega$-limit set is the origin, $M_{21}$ does vanish there, and it is
nonzero elsewhere. So the conclusion cannot be strengthened.

### 5.2 The live route — strengthen the hypotheses, not the argument

Since no proof exists under (B1)–(B4), the hypotheses have to change. Two
additions, both cheap in our setting:

1. **$h$ real-analytic.** This is *free*, and more than free: with `tanh` (or
   any analytic-activation) MLP decoders, $g, \tilde g$ are real-analytic, and
   $h = \tilde g^{-1}\circ g$ is real-analytic on the visited region (analytic
   inverse function theorem on an immersion). Real-analyticity is strictly
   stronger than the $C^\infty$ we relied on before, and it is exactly what makes
   the formal-to-smooth step (§5.3, step 4) collapse — see §5.4. It is a genuine
   (mild) modelling commitment: **analytic activations, not ReLU** — ReLU nets are
   only piecewise-linear and the argument below fails for them.
2. **Cross-module non-resonance**, i.e. excluding
   $\lambda_i = \sum_j m_j \lambda_j$ relations across modules. §4.3 shows some
   such condition is *necessary*: the resonant case admits a *polynomial* (hence
   analytic) counterexample, so analyticity does **not** remove it. But it is a
   measure-zero condition — a learned model does not sit on it.

### 5.3 The assembled statement (Theorem B, fixed-point regime)

> **Theorem B (block-diagonal, analytic + non-resonant).** Let $x_t = g(z_t)$
> with $g$ a real-analytic injective immersion (e.g. a `tanh`-MLP decoder),
> $F = \bigoplus_i f_i$ real-analytic with $0$ an attracting fixed point of each
> $f_i$, and $(\tilde F, \tilde g)$ equivalent with $\tilde F$ modular and
> real-analytic. Assume
>
> * **(B1$''$)** $\Omega$ compact, $F$-invariant, with a limit point in the
>   domain (a single contracting orbit suffices — see §5.4; this replaces the
>   $C^\infty$ route's $\mathrm{int}\,\Omega\neq\emptyset$);
> * **(B4$'$)** the module linear-part spectra are pairwise disjoint **and**
>   cross-module non-resonant in the full multi-index sense
>   ($\lambda_{i,a}\neq\sum_j m_j\nu_j$ for every multi-index $|m|\ge2$ drawn
>   with multiplicity from the full spectrum with support meeting a module
>   $\neq i$; `spectra.cross_module_nonresonant`).
>
> Then $h = \tilde g^{-1}\circ g$ (automatically real-analytic) satisfies
> $h = P_\sigma\circ(h_1\oplus\cdots\oplus h_K)$ near $0$ and on all of $\Omega$,
> with $\sigma$ and the equal block dimensions a *conclusion*, and each
> $\tilde f_{\sigma(i)}$ analytically conjugate to $f_i$.
>
> *(A $C^\infty$ variant holds with $\mathrm{int}\,\Omega\neq\emptyset$ and
> `tanh`-MLPs relaxed to any smooth decoder, at the cost of the (FLAT-D) $C^k$
> distortion bound in step 4; §5.4 and `route_a_assessment.md` §2.4.)*

*Proof (assembled; `route_a_assessment.md` §2 is the audit).* Five steps.

1. **Jets at $0$** — the conjugacy relation's jets hold at $0$ because
   $\bigcup_n F^n(\mathrm{int}\,\Omega)$ accumulates there (§2.1). *This is where
   (B1$'$)'s open-interior clause is consumed; on a thin $\Omega$ the conclusion
   is false, `counterexamples.md`.*
2. **Per-module normal forms** — Sternberg (contraction case, **verified from
   the primary texts**, §2.2) conjugates each $f_i$ to its normal form $N_i$;
   the product $\psi=\bigoplus\psi_i$ is block-diagonal by construction.
3. **Formal lemma** (§2.3, proved inline): the formal series of the conjugacy
   between $\bigoplus N_i$ and $\bigoplus\tilde N_i$ is $P_\sigma\circ$
   block-diagonal — degree 1 is `linear_case.md` Theorem L (this yields $\sigma$
   and block dimensions, i.e. **(B3) is a conclusion**), and cross-module
   components at higher degree vanish by (B4$'$) non-resonance.
4. **Formal $\Rightarrow$ analytic** — the identity theorem (§5.4). With analytic
   decoders, $\psi_i$ (analytic normal-form conjugacies, Poincaré–Dulac in the
   Poincaré domain) and $h$ are real-analytic, so $k=\tilde\psi\circ h\circ\psi^{-1}$
   is real-analytic; its Taylor series at $0$ is $P_\sigma\circ$ block-diagonal
   (step 3), and a real-analytic map whose Taylor series has no cross-block terms
   *has none* on the connected component. So $k$ is block-diagonal — no distortion
   estimate, no (FLAT-D).
5. **Matching / dimensions** (§2.5) — already delivered by step 3's degree-1
   part. $\square$

**Dependency ledger (analytic route).** Poincaré–Dulac analytic normal form in
the Poincaré domain (classical; contractions are in it, no small divisors);
Theorem L (proved in repo); the formal lemma (inline); the identity theorem for
real-analytic functions (classical). **Nothing is left unwritten** — the
$C^\infty$ route's residual, (FLAT-D)'s $C^k$ bound, is not on this path at all.

**Two caveats kept honest.** (a) Under *full*-spectrum non-resonance Poincaré
linearises $F$ outright, so that tier is **robustness of Theorem A** against
decoder ambiguity; the nonlinear content needs a *within-module* resonance kept,
which Poincaré–Dulac retains as finitely many analytic normal-form terms (Tier 2
of `approaches.md` §A.2). **That tier is now certified non-empty**
(`approaches.md` §A.2.1, `exp09`): $f(z_a,z_b) = (\mu z_a,\ \mu^2 z_b + c z_a^2)$
satisfies cross-module non-resonance while its resonance
$\lambda_b - \lambda_a^2 = \mu^2-\mu^2 = 0$ — which vanishes identically in $\mu$,
so it is structural — obstructs linearisation, leaving $c$ as a normal-form
invariant. So Theorem B's nonlinear claim has actual content and does not collapse
onto Theorem A. The same witness shows the *linearised* (B2) test is a false
negative in exactly this regime (§A.2.2), which is a gap in the learning
machinery, not in the theorem. (b) **Non-fixed-point attractors are not covered *by
this theorem*** — limit cycles push into the *Siegel* domain, where small
divisors return and analyticity alone no longer suffices (Bruno-type arithmetic
conditions enter). `TODO(gap)` This caveat is specific to the **normal-form**
route to block-diagonality; it does **not** apply to Lemma C, which needs no
normal form and does extend to periodic attractors — see §4.4. So off the fixed
point the filtration survives and block-diagonality does not, which widens the
gap between Theorem F and Theorem B rather than closing it.

> **Sharpened 2026-08-25 — caveat (b) is a structural exclusion, not a technical
> gap, and it is one line.** I had been reading it as "limit cycles are Siegel,
> small divisors return, hard". The hypothesis is worse off than that: it is
> *identically false* the moment any rotation is present anywhere in the system.
>
> **Proposition N.** If some module has a Lyapunov exponent $\lambda_{i,0}=0$
> — equivalently a Floquet multiplier of modulus $1$: any invariant circle,
> limit cycle, or neutral rotation — then cross-module non-resonance (B4$'$)
> fails for **every** other module and at **every** order $\ge2$.
>
> *Proof.* Let $\nu$ be any exponent of a module $j\neq i$. Then
> $\nu = \lambda_{i,0} + \nu$ is a resonance of order $2$ whose multi-index has
> support meeting module $i\neq j$. Multiplicatively: multiplier $1$ means
> $\mu = 1\cdot\mu$, so the monomial $z_i^{(0)}z_j$ is resonant. Appending the
> neutral direction $k$ more times gives order $k+2$. $\square$
>
> Verified with `spectra.is_cross_module_nonresonant`: two limit cycles at
> $a=(0.3,0.4)$ give $30$ resonances at order $\le4$; **one** limit cycle
> against a plain contracting spiral already gives $12$, the first being
> $\lambda_{2,b}=\lambda_{1,\text{neutral}}+\lambda_{2,b}$. Two contracting
> spirals at $(0.92,0.55)$ are non-resonant, as expected.
>
> **Consequence, and it is the reason to stop looking for a fix.** Combined with
> (F3) failing for two cycles (both hulls reach $0$, §6.5 rider 2) and Route B
> being blind to the lattice regrouping (§13.3), **all three routes are dead in
> the oscillatory regime, each for a different and now-understood reason.** That
> is a complete negative rather than three open problems, and it is what
> redirects the effort onto §14–15: characterise the ambiguity exactly instead
> of trying to remove it.

### 5.4 Why analyticity closes step 4 (and weakens (B1$'$))

Two consequences of $h$ being real-analytic rather than merely $C^\infty$.

**Step 4 is the identity theorem.** In the $C^\infty$ category, a block-diagonal
$\infty$-jet does *not* force a block-diagonal map — the flat remainder is exactly
the §4.3 counterexample, and killing it needed (FLAT-D) plus a $C^k$ distortion
bound (the $C^\infty$ route, still valid, `route_a_assessment.md` §2.4 and
`exp07`). In the *analytic* category there is no flat remainder: an analytic
function equals its Taylor series on the connected component, so a block-diagonal
jet **is** a block-diagonal map. The hardest step of the smooth proof becomes a
one-line classical fact. The whole fixed-point case then rests on Poincaré–Dulac
(analytic normal form, no small divisors in the Poincaré domain) + the identity
theorem + the formal lemma — all textbook or already proved here.

**(B1$'$) weakens.** The open-interior hypothesis $\mathrm{int}\,\Omega\neq\emptyset$
was needed because a $C^\infty$ conjugacy is unconstrained off a thin $\Omega$
(§3.5 counterexample). An analytic conjugacy is determined by its values on any
set with a limit point, and a contracting trajectory accumulates at the fixed
point — so a single convergent orbit pins $h$ down. Under analyticity, replace
$\mathrm{int}\,\Omega\neq\emptyset$ by "$\Omega$ has a limit point in the domain",
which the dynamics supply for free.

**What analyticity does not do.** It does not remove cross-module non-resonance:
the §4.3 counterexample is polynomial. And it does not reach non-fixed-point
attractors — the clean analytic theory is a Poincaré-domain (contraction)
phenomenon; the Siegel domain (rotations, limit cycles) reintroduces small
divisors. The finite-dimensionality of the MLP class *beyond* analyticity is the
natural lever for that open case — it is how nonlinear-ICA results obtain
identifiability at all (restrict the function class) — but it is a theorem to
prove, not a corollary here.

---

## 6. Theorem F — the filtration, and what it identifies

*CLAUDE.md task 11 (Route C) and task 37. This is the statement that stands
entirely on proved ground: no `TODO(gap)` is consumed by its dynamics, and the
one hypothesis it cannot yet discharge is named in §6.7.*

Theorem B asks whether $h$ is block-diagonal. §4.3 answers no — the target is
false, not merely unproved. Theorem F asks the question that survives: **what
does an equivalence identify, given that it is only triangular?** The answer is
an ordered filtration plus a per-level invariant vector, and that is enough for
the applied claim (CLAUDE.md §1.0).

### 6.0 Tier 1 — the global conjugacy class costs nothing

Before any modularity is used at all:

> **Proposition T1.** Let $(F,g)$ and $(\tilde F,\tilde g)$ be equivalent with
> $g,\tilde g$ injective on $\Omega$ and $\tilde\Omega$. Then
> $h := \tilde g^{-1}\!\circ g$ is a bijection $\Omega\to\tilde\Omega$ and
> $$\tilde F = h\circ F\circ h^{-1} \quad\text{on } \tilde\Omega .$$

*Proof.* $g(z_t)=\tilde g(\tilde z_t)$ and $\tilde g$ injective give
$\tilde z_t = h(z_t)$; substituting into $\tilde z_{t+1}=\tilde F(\tilde z_t)$
gives $h(F(z_t))=\tilde F(h(z_t))$ at every visited point, hence on $\Omega$.
$\square$

So **the fitted transition is automatically conjugate to the true one**, and
every conjugacy invariant of the whole system is identified with no theorem, no
auxiliary variable, and no spectral hypothesis. What that buys, and at what
regularity — the distinction matters, because a fitted $h$ is a diffeomorphism
but its derivative bounds are not free:

| invariant | regularity of $h$ needed |
|---|---|
| latent dimension $d$ | $h$ a homeomorphism |
| number of fixed points / periodic orbits, and their periods | topological |
| stability type (attracting / repelling / saddle) | topological |
| attractor topology; topological entropy | topological |
| rotation number on an invariant circle | topological |
| **Lyapunov spectrum** (global, as a multiset) | $C^1$ with $\sup\|Dh\|,\sup\|Dh^{-1}\|<\infty$ |

Only the last row needs (F1) below; the rest are free. This is CLAUDE.md §1.2
Tier 1, and it is worth stating first because it is *already true of everything
built in this repo* and it frames what the rest adds. Its caveats are the
ordinary ones: correct latent dimension, exact fit, and $\Omega$ only.

**What Tier 1 does not give is the decomposition.** $h$ may mix factors —
§3.1's regrouping and §4.3's triangular conjugacy both act inside the Tier 1
conclusion. Everything below is about recovering the *parts*.

### 6.1 The hypothesis (B4) was never the right one

Lemma C needs the **oriented** gap $\lambda_{\max}(f_j) < \lambda_{\min}(\tilde f_i)$.
(B4) asks only that the module spectra be *disjoint*, and §4.3 already records
that this is strictly weaker. Theorem F therefore replaces it with ordered
separation, which is the honest hypothesis and is cheap to check:

> **(F3) Ordered separation.** After indexing modules so that
> $\lambda_{\max}(f_1) > \lambda_{\max}(f_2) > \cdots$, the *convex hulls* of the
> module spectra are pairwise disjoint:
> $$\lambda_{\min}(f_i) \;>\; \lambda_{\max}(f_{i+1}) \qquad (1 \le i < K).$$
> The weakest link, $\min_i[\lambda_{\min}(f_i)-\lambda_{\max}(f_{i+1})]$, is
> `spectra.filtration_gap`.

The gap between (B4) and (F3) is not a technicality — **it is exactly the §3.1
counterexample.** With $\lambda=(0.90,0.75,0.60,0.45)$, the true grouping has
`spectral_gap` $=+0.2231$ and chain gap $=+0.2231$; the regrouping that swaps
coordinates 2 and 3 still has `spectral_gap` $=+0.1823$ — comfortably
"disjoint" — while its chain gap is $\mathbf{-0.2231}$, because the two hulls
$[-0.5108,-0.1054]$ and $[-0.7985,-0.2877]$ interleave. **(F3) rejects the
regrouping counterexample outright, and (B4) does not.** That is the single
sharpest reason to state Theorem F with (F3).

(F3) also reproduces a measured threshold with no free parameter. In `exp08` an
attracting invariant circle (spectrum $[-0.9163,\,0]$) is paired with a
contracting module swept across the crossing; `filtration_gap` $>0$ agrees with
the measured `forces_M_zero` at **every** sweep point on both sides
($s = 0.20\ldots0.38$ positive and forcing, $s = 0.42, 0.50$ negative and not),
the crossing sitting at $\log|1-2a|$ to the digit. Note what this rules in as
well as out: a module with a *wide* hull — and a limit cycle's is
$[\lambda_{\text{transverse}},\,0]$, as wide as it gets — can fail (F3) against
a module nested strictly inside it, even though it is "on top". Ordered
separation is a statement about intervals, not about which module is fastest.

### 6.2 The theorem

> **Theorem F (filtration identifiability).** Let $(F,g)$ and $(\tilde F,\tilde g)$
> be equivalent, $F=\bigoplus_{i=1}^K f_i$ and $\tilde F=\bigoplus_{i=1}^{K}\tilde f_i$
> modular, $h=\tilde g^{-1}\!\circ g$. Assume
>
> * **(F1) Regularity.** $\Omega$ compact and $F$-invariant; $h$ is $C^1$ with
>   $\sup_\Omega\|Dh\|<\infty$ and $\sup\|Dh^{-1}\|<\infty$.
>   *No open-interior clause — see the note below.*
> * **(F2) Uniform exponents.** The Lyapunov exponents of each $f_i$ and
>   $\tilde f_i$ are realised by **every** orbit in $\Omega$, not merely a.e.
>   (Lemma C′; holds at attracting fixed points and at normally hyperbolic
>   attracting periodic orbits / invariant circles.)
> * **(F3) Ordered separation**, as above, for both $F$ and $\tilde F$.
> * **(F4) Matching.** $\tilde f_i$ is conjugate to $f_i$ for each $i$, in the
>   common (F3) ordering. `TODO(gap)` — reduced but not discharged; §6.7.
>
> Then $h$ is **block lower-triangular** with respect to that ordering,
> $$M_{ij} := \partial h_i/\partial z_j \equiv 0 \quad\text{for all } j>i,
> \qquad\text{i.e.}\qquad h_i = h_i(z_1,\dots,z_i),$$
> and consequently, for every $i$:
>
> 1. **the flag is preserved** — $h$ carries the foliation
>    $\mathcal{F}_i = \{z_{\le i} = \text{const}\}$ (leaves of dimension
>    $d_{i+1}+\cdots+d_K$) to $\tilde{\mathcal{F}}_i$, and
>    $\mathcal{F}_1 \supset \mathcal{F}_2 \supset \cdots \supset \mathcal{F}_{K-1}$
>    is a canonical nested family;
> 2. **each head system is identified** — $h_{\le i} := (h_1,\dots,h_i)$ is a
>    diffeomorphism conjugating $F_{\le i} := f_1\oplus\cdots\oplus f_i$ to
>    $\tilde F_{\le i}$, so the whole chain of quotient systems
>    $$F \twoheadrightarrow F_{\le K-1} \twoheadrightarrow \cdots
>      \twoheadrightarrow F_{\le 1} = f_1$$
>    is identified up to conjugacy;
> 3. **the level spectra are identified** — $\Lambda(f_i) = \Lambda(F_{\le i})
>    \setminus \Lambda(F_{\le i-1})$ as multisets, each side computed from its
>    own representation.

*Proof.* Tier 1 (Prop. T1) gives $h\circ F = \tilde F\circ h$ on $\Omega$.
Differentiating in $z_j$ gives the cocycle relation of §4.1; (F1)'s bounded
derivatives and (F2)'s uniformity put Lemma C′ (§4.4) in force at **every** point
of $\Omega$. For $j>i$, (F3) plus (F4) give
$\lambda_{\max}(f_j) \le \lambda_{\max}(f_{i+1}) < \lambda_{\min}(f_i) = \lambda_{\min}(\tilde f_i)$,
so Lemma C′ forces $M_{ij}\equiv0$. That is the displayed triangularity.

(1) $h_i$ depending only on $z_{\le i}$ is exactly the statement that $h$ maps
fibres of $\pi_{\le i}$ into fibres of $\tilde\pi_{\le i}$.

(2) The same computation applied to $h^{-1}$ — which is a conjugacy in the
reverse direction between systems with the *same* spectra, hence the same
orientation — makes $h^{-1}$ triangular too, so $h_{\le i}$ is a bijection onto
$\tilde\Omega_{\le i}$; and $h_{\le i}\circ F_{\le i} = \tilde F_{\le i}\circ h_{\le i}$
holds because $\pi_{\le i}$ semiconjugates $F$ to $F_{\le i}$.

(3) Conjugate systems with (F1) bounds have equal Lyapunov spectra, applied to
$F_{\le i}$ and $F_{\le i-1}$ separately. $\square$

> **What is *not* used, and why it matters.** $\mathrm{int}\,\Omega\neq\emptyset$
> is not used, nor is (B2) indecomposability, nor any normal form, nor
> analyticity. Those are precisely what Theorem B consumes, and precisely what
> fails on an attractor. **Theorem F therefore holds where Theorem B cannot go**
> — at attracting periodic orbits and invariant circles (§4.4), which is the
> regime a neural recording is actually in. That, not its strength, is why it is
> the priority.

### 6.3 The successive factors, and where their conjugacy is genuine

Conclusion 2 identifies the *head* systems $F_{\le i}$, not the individual $f_i$
for $i\ge2$. The distinction is real and should not be blurred:

- **$f_1$ is identified outright.** $h_1 = h_1(z_1)$ is a genuine conjugacy
  $f_1 \to \tilde f_1$. Every conjugacy invariant of the slowest module — fixed
  points, attractor topology, Lyapunov spectrum, **rotation number** — is
  identified in the strong sense.
- **For $i\ge2$, what is identified is $f_i$ as the fibre dynamics of a skew
  product.** Restricting the conjugacy to a fibre gives
  $$h_i\big(F_{\le i-1}(c),\, f_i(z_i)\big) = \tilde f_i\big(h_i(c, z_i)\big),$$
  a conjugacy *along the orbit of $c$*, not at fixed $c$. It becomes a genuine
  conjugacy $f_i\to\tilde f_i$ when $c$ is a fixed point of $F_{\le i-1}$, and a
  conjugacy of $f_i^{\,p}$ to $\tilde f_i^{\,p}$ when $c$ is $p$-periodic.
  Unconditionally, conclusion 3 still identifies $\Lambda(f_i)$.

So the invariant that survives at every level with no side conditions is the
**spectrum**; the finer invariants are unconditional only at the top of the
filtration. That is not a defect of the write-up — §3.13(b) measures the same
asymmetry in fitted models, where the dominated module's invariants come back
$100\times$ worse than the dominant one's.

### 6.4 What a filtration claim consists of (task 37)

The deliverable is not "$h$ is block-diagonal". It is this list, every entry of
which is estimable from a fitted model with no ground truth:

| # | quantity | estimator | identified by |
|---|---|---|---|
| 1 | number of levels $K$ | partition search | (F4) / §6.7 |
| 2 | level dimensions $d_1,\dots,d_K$ | `selection` lattice | (F4) / §6.7 |
| 3 | the **ordering** | `spectra.filtration_gap` | (F3), free once spectra are known |
| 4 | per-level Lyapunov spectrum | `spectra.module_lyapunov_spectra` | Thm F(3) |
| 5 | rotation number of the top level | `spectra.rotation_number` | Thm F(2) + §6.3 |
| 6 | attractor topology per level | — (not built) | Tier 1 + Thm F(2) |

Items 3–5 are `metrics.dynamical_fingerprint`, and `metrics.invariant_agreement`
compares two such fingerprints without ground truth — the empirical form of this
theorem (task 40, `exp14`). Read as a sentence, a positive result is *"this
population carries a slow 2-D rotation at 8 Hz, and a faster 3-D decaying
component driven by it"* — which is what §0 says the object of study is, and is
strictly less than block-diagonality.

**Three measurement caveats, all load-bearing and all already paid for
elsewhere.** (a) Read items 4–6 *inside the data horizon*: past it a fitted map
invents an attractor, and the rotation number reads a confident, coherent $0$
(§3.13(a)). (b) Recoverability is **per-invariant**, so report agreement
per-invariant, never as one boolean (§3.13(b)). (c) Screen restarts on
**duplicate invariants**, not on fit quality or coherence, both of which are
uninformative and one of which has the wrong sign (§3.13(e)).

### 6.5 Two structural riders

Carried from §4.4, and they belong in the theorem statement because they are
testable claims about a real population rather than proof hygiene.

1. **An oscillatory module is the top of the filtration or is not separated at
   all.** Its $\lambda_{\max}=0$, and (F3) requires everything below it to have
   $\lambda_{\max}<0$. The phase of an autonomous oscillation is the slowest
   thing in the system.
2. **At most one oscillatory module can appear.** Two cycles both contribute
   $\lambda_{\max}=0$, so their hulls overlap, (F3) fails, and neither
   cross-derivative is forced. §6.1's remark sharpens this: it is not only two
   cycles — *any* module whose hull nests inside another's is unseparable, so
   rider 1 does not make oscillations safe.

Rider 2 collides with §0's own definition of a module ("distinct oscillatory
components"), and the collision is the sharpest limitation on the applied claim.
The rotation number is the invariant that distinguishes two cycles and it is
built and validated (`spectra.rotation_number`, `exp14`) — but **using it to
force a splitting is not proved**, and nothing here proves it. `TODO(gap)`

> **(F3) is sufficient, not necessary — and the repo's own best empirical result
> is on the wrong side of it.** `exp14` part 4, the task-40 validation, fits two
> limit cycles at $\omega = 0.5$ and $1.3$; the experiment chose that system
> deliberately, as "exactly where Lemma C has no gap to use". Evaluating (F3) on
> its **24 saved fingerprints** — no refitting needed, §3.13(d) saved them for
> exactly this — gives `is_filtration` in **0 of 24**, median chain gap
> $\mathbf{-0.65}$, uniformly across the linear, nonlinear and negative-control
> arms. And yet in the linear arm all 16 cross-split comparisons agree, at a
> median rotation error of $2.5\times10^{-4}$, while the negative control
> correctly rejects.
>
> So invariant recovery happens in a regime Theorem F provably does not reach.
> That is not a contradiction — (F3) is a sufficient condition — but it is the
> honest statement of where the theory and the measurement stand relative to
> each other.
>
> **RETRACTION (same day).** This box originally continued: "*and it is evidence
> for the conjecture this rider flags as `TODO(gap)`: the rotation number is
> doing work that no theorem here licenses*." **That reading is wrong, and the
> conjecture it pointed at is false** — `counterexamples.md` §7 exhibits an
> exact modular conjugacy of two shear-free cycles carrying
> $(\omega_1,\omega_2)$ to $(\omega_1+\omega_2,\omega_2)$. Only the
> $GL(2,\mathbb{Z})$ orbit of the rotation vector is identified, so the
> per-module rotation numbers `exp14` agrees on are **not** invariants of the
> data; both fits land on the same lattice basis for reasons of parameterisation
> and initialisation, not because the observations determine it. The agreement
> is real and reproducible; what it is evidence *of* is weaker than claimed.
>
> **Two caveats that must travel with it.** (i) The fitted model class *imposes*
> modularity, so producing two modules is automatic and proves nothing. What is
> not automatic is that two fits on **disjoint neuron subsets** agree on *which*
> two — 2 of 12 restarts fail exactly there, by mode collapse (§3.13(e)) — and
> that a frequency change is rejected. (ii) The negative control's margin is
> smaller than it looks: quotienting by $GL(2,\mathbb{Z})$, the control
> $(\omega_1,0.90)$ sits $0.0159$ from an image of the true system, not the
> $0.0637$ the coordinatewise comparison reports. It still rejects, with a
> quarter of the headroom. Use `spectra.rotation_lattice_margin`.
>
> Note also what this says about the diagnostics. `order_margin` reads $0.0011$
> here, which §3.13(d) correctly treats as an *undetermined* ordering; the chain
> gap reads $-0.65$, a *definite* failure. Same verdict, different strength, and
> the difference matters: a tie invites a tie-break, a definite failure does not.

### 6.6 What Theorem F does not claim

- **Not coordinates.** Within a level, $h_i$ is an arbitrary diffeomorphism.
- **Not block-diagonality.** §4.3 is a counterexample, not a gap; the surviving
  cross-block $M_{ij}$, $j<i$, is genuinely nonzero in general. Killing it needs
  behaviour (Lemma D, §4.5) or analyticity + non-resonance (Theorem B, §5.3).
- **Nothing off $\Omega$.**
- **Not a claim about anatomy.** Levels are timescale-separated dynamical
  factors within one population.

### 6.7 The one open hypothesis, and how far it has been reduced

(F4) is the matching lemma of CLAUDE.md §3.2, and Theorem F needs much less of
it than Theorem B does. The general problem — pair blocks across representations
by conjugacy invariants, *before* using any spectral hypothesis — is open
nonlinearly. Here it collapses in two steps:

1. **The ordering is free.** (F3) holding on both sides means each
   representation's modules occupy disjoint ordered intervals of a spectrum
   that Tier 1 already identifies as a multiset. So $\sigma$ cannot permute:
   any admissible correspondence is order-preserving. This is where the §3.1
   regrouping dies (§6.1), and it is the part of §3.2 that Theorem B has to
   assume outright.
2. **What remains is the coarsening.** Both sides partition the *same* ordered
   multiset of exponents into consecutive groups; the residual freedom is
   whether one side splits a level the other keeps whole. That is exactly
   nonlinear indecomposability, (B2) / open problem 3 below — and it is the
   filtration-side reading of §3.1: **the regrouping ambiguity is a coarsening
   ambiguity, nothing more.**

**Reconciliation with `approaches.md` §C, which says "(B2) not needed".** Both
are right, and the difference is what is being claimed:

- **Report a filtration.** Take the levels as given by whichever decomposition
  is in hand. Then the coarsening ambiguity is *tolerated* — the flag one
  reports may be coarser than the finest one — and (B2) is genuinely not needed.
  This is `approaches.md` §C's reading, and it is the honest default on data.
- **Claim *the* finest filtration.** Then the coarsening must be pinned, and
  that is (B2). This is `approaches.md` §C's own cost 1 ("identifies how any two
  modular explanations relate, not that the decomposition is unique"), stated
  from the other side.

So Theorem F's dynamics are unconditional; its one gap is a single, named,
already-tracked lemma, and that gap is only consumed by the *stronger* of the
two readings. Note the empirical program does not wait on either:
`selection.py` + `exp06` recover the finest partition from data by fit and
uniqueness, and `exp03` confirms the (F3) chain on a case with
$\Lambda(f_1)=\{-0.0513\}^2$, $\Lambda(f_2)=\{-0.3567\}^2$, chain gap $0.3054$,
partition recovered in 5/5 converged restarts.

---

## 7. Open problems, in priority order

1. ~~**Theorem F — filtration identifiability.**~~ **Written up — §6.** One
   hypothesis changed in the writing and it is worth flagging: the theorem needs
   **ordered separation (F3)**, not (B4) disjointness, and the difference is
   exactly the §3.1 regrouping counterexample (§6.1). Two things came out as
   *conclusions* rather than hypotheses — the ordering half of the matching
   lemma, and the identification of every head system $F_{\le i}$, not merely of
   $f_1$. What remains open under it is item 3 below, which §6.7 shows is the
   only residual freedom.
2. **The matching lemma (B3)**, CLAUDE.md §3.2. Pair indecomposable blocks
   across representations by conjugacy invariants — Lyapunov spectrum,
   fixed-point structure, entropy — *before* any spectral hypothesis is used.
   `linear_case.md` §5 proves the linear instance and gives the template.
   `TODO(gap)`
3. **Nonlinear indecomposability (B2).** Define it so that Lemma 2 of
   `linear_case.md` has an analogue. The linear proof uses that the primary
   projections are polynomials in $F$; there is no nonlinear substitute, so this
   needs a genuinely dynamical argument. `TODO(gap)`
4. **Distributional equivalence with noise** (§1). `TODO(gap)`
5. **Nonlinear perturbation.** `exp04` settles the linear $O(\epsilon/\mathrm{gap})$
   claim. The nonlinear version needs invariant *manifolds* in place of invariant
   subspaces and the dichotomy gap in place of `sep`. `TODO(gap)`

---

## 8. Scope of interpretation claims

Carried over from CLAUDE.md §5, and now with a sharper upper bound on what can
be claimed.

Even on full success, the within-module $h_i$ is an arbitrary diffeomorphism.
What is identified is the **partition** plus each $f_i$'s **conjugacy class** —
fixed-point structure, attractor topology, Lyapunov spectrum. Not coordinates.
Nothing here licenses reading "motor primitive" off a latent axis.

Given §4.2–4.3, the honest current claim is weaker still: a **triangular** $h$,
so what is identified is an ordered *factor* structure, not a symmetric
partition. Under the §0 scope this is the intended claim rather than a
concession — the factors are timescale-separated components of one population,
and their ordering is part of the content. It remains **not** a claim that the
factors correspond to anatomically or functionally labelled subpopulations.

This is why `metrics.py` reports partition-level quantities first and MCC second.
A high MCC with the wrong partition is a failure — see
`tests/test_metrics_and_models.py::test_high_mcc_with_a_wrong_partition_is_not_recovery`,
where the §3.1 swap scores $\mathrm{MCC} = 1.0$ and on-block fraction $0.5$.

---

## 9. Position against the literature

`TODO(gap)` — CLAUDE.md §4 step 8, not yet done. The comparison to make:

- **Hyvärinen & Morioka** (time-contrastive, permutation-contrastive) and
  **Hälvä & Hyvärinen** (HMM) obtain identifiability from *temporal dependence
  plus non-stationarity*, recovering latents up to componentwise transformation
  — i.e. they identify **coordinates**, which is more than we claim.
- **Khemakhem et al.** (iVAE) needs an observed auxiliary variable; we have none.
- The distinguishing claim is that modular *dynamics* identifies structure with
  **no auxiliary variable and no distributional prior on the latent**. PCL is the
  nearest neighbour (no auxiliary) but needs mutually independent *scalar
  stochastic* sources with a nonvanishing cross-derivative of the joint
  log-density, which is structurally false for deterministic dynamics.

  > **CORRECTION (2026-08-12).** This bullet previously read "autonomous,
  > **stationary**, no-auxiliary-variable", and stationarity was offered as the
  > thing that puts us outside the existing results. **That is wrong, and it
  > conflates an autonomous *system* with a stationary *process*.** The dynamics
  > are time-invariant, but $\{z_t\}$ is stationary only if $z_0$ is drawn from
  > the invariant measure, and `make_dataset` deliberately spreads initial
  > conditions over an annulus — so every dataset here is non-stationary *by
  > construction*, and measurably so (with $u=t$ the two blocks'
  > scale-normalised $t$-dependence is $1.32$ and $4.20$; CLAUDE.md task 8).
  > The positioning cannot rest on absence of non-stationarity. What it rests on
  > instead is §1.3: those theorems constrain $p(z\mid u)$, and a deterministic
  > latent map with an injective decoder has no such prior to constrain — its
  > identifiability content is conjugacy plus spectra, which is what §6 states.
- What we claim is also *weaker in kind*: those results recover **coordinates**
  up to componentwise transformation; we claim a **filtration of multidimensional
  factors**. The two are not competing statements about the same object.
- Autonomy was previously flagged as the main disanalogy with LFADS. Under §0 it
  is **out of scope by decision**, not an open threat. If input drive is added
  later, Vahidi et al. 2024 is the entry point (CLAUDE.md §4.3).

---

## 10. What real data says — MC_Maze

**New (2026-08-14), `exp15_nlb.py` / `exp15b_linearity.py`.** The first
contact between this theory and a recording. Neural Latents Benchmark
**MC_Maze** (DANDI 000128): macaque M1+PMd, 1721 training trials of a delayed
reach through a maze, 182 sorted units, 108 reach conditions.

Preprocessing, and each choice is a hypothesis rather than a default:

| choice | value | why |
|---|---|---|
| alignment | movement onset, $[-250, +450]$ ms | the epoch in which motor cortex is standardly modelled as autonomous — §0's scope, imposed by the window rather than argued for |
| bins | 20 ms | $T = 35$, comparable to the synthetic experiments |
| observable | condition-averaged PSTHs, $\sqrt{\text{rate}}$ | §1 fixes the target as a *deterministic* flow from a random initial condition, and a **condition is an initial condition**. Averaging its repeats removes the Poisson noise the target class does not model. Single trials would force the distributional equivalence CLAUDE.md §3.8 puts out of scope |
| smoothing | 40 ms Gaussian | split-half PSTH reliability $r = 0.658$. A wide kernel could in principle manufacture smooth low-dimensional dynamics, so it is controlled twice: §10.1 sweeps it, and §10.3's neuron halves are **disjoint**, so their PSTH noise is independent and smoothing cannot create agreement between them |

### 10.1 The latent flow is linear, and that governs everything below

Measured before anything was interpreted, because it decides what a modularity
result on this data is *allowed* to claim. Fit the best one-step linear map on
the top-$k$ principal components, then ask what a full quadratic expansion adds,
as a share of total latent variance:

| swept | range | nonlinear share |
|---|---|---|
| smoothing $0/20/40/80$ ms | — | $0.38\%,\ 0.21\%,\ 0.21\%,\ 0.20\%$ |
| window (4 settings) | $[-700,900]$ ms to $[0,300]$ ms | $0.12\%$–$0.50\%$ |
| dimension $d = 2 \dots 10$ | — | $0.09\%$–$0.53\%$ |
| dataset | MC_Maze, MC_Maze_Small | $0.21\%,\ 0.26\%$ |
| **single trial**, no averaging | MC_Maze | $0.15\%$–$0.23\%$ |
| **single trial, second task** | MC_RTT | $0.04\%$–$0.12\%$ |

**Smoothing is not manufacturing it, and the unsmoothed row is what shows that.**
At $0$ ms the linear $R^2$ falls to $0.838$ — but the quadratic expansion
recovers only $0.38$ of the missing $16$ points, so the shortfall there is PSTH
sampling noise, not curvature. Wherever the noise is controlled the flow is
$\ge 98.5\%$ linear.

**The two escapes a sceptic would reach for are closed, and both fail in the
same direction.** (i) *Condition averaging might be smoothing curvature away
across repeats* — but single trials give $0.15\%$–$0.23\%$, indistinguishable
from the averaged $0.21\%$. (ii) *A stereotyped reach might simply not visit the
nonlinear part of state space* — but MC_RTT, continuous random target pursuit
with no trial structure at all, is **less** nonlinear ($0.04\%$–$0.12\%$), not
more.

> **Consequence, and it is a hard limit on the empirical claim.** For a real
> matrix whose eigenvalues are distinct complex-conjugate pairs, the real Jordan
> form is *already block diagonal* — and at $d=4$ the fitted map's eigenvalues
> are exactly such pairs. So on this data **modularity is generic, not a
> restriction**. What a MC_Maze result can validate is **Theorem A**, the linear
> case proved in `linear_case.md`, and *not* Theorem B, whose entire difficulty
> (§4.3's triangular counterexample) is nonlinear.
>
> Stated plainly: *the regime where this project's hard theorems bite is not the
> regime these motor cortex recordings are in* — and after the single-trial and
> MC_RTT arms, that is a statement about the **data**, not about the
> preprocessing. **The nonlinear identifiability theory cannot be tested on
> these benchmarks, because the phenomenon it is about is not present in them.**
>
> That is a useful negative. It says the next real-data step is not a better
> estimator but a *different regime* — a longer horizon, an area with genuine
> multistability, or a task that leaves the near-linear operating point — and
> that for motor cortex in the movement epoch, the already-proved linear theory
> is not a stepping stone to the interesting case but very nearly the whole
> story.

A methodological note of the CLAUDE.md §3.9 family, found here. Reporting the quadratic
gain *relative to the residual* is unreliable: on an exactly linear system the
linear residual is already at the numerical floor and a quadratic term still
removes $\approx 60\%$ of it. Sixty per cent of nothing. Only the share of total
variance separates "curved" from "clean", and the known-answer test asserts on
that.

### 10.2 Adequacy — the constraint is free (task 39)

Nested ladder, scored by held-out-**neuron** co-smoothing: 126 units fitted, 32
held out, a fresh linear readout refitted from the latents to the held-out units
and scored on unseen $(\text{condition}, \text{time})$ points.

| rung | co-smoothing $R^2$, median | range over 5 restarts |
|---|---|---|
| unconstrained | $0.2979$ | $[0.2977, 0.2989]$ |
| triangular | $0.2981$ | $[0.2978, 0.2991]$ |
| modular | $0.2983$ | $[0.2979, 0.2997]$ |

The three are indistinguishable, with the *most* constrained marginally ahead —
consistent with equal bias and lower variance. **Imposing block-diagonal
autonomous dynamics costs nothing in population reconstruction.**

**And the metric is not dead**, which CLAUDE.md §3.11 insists on checking before
reading a null: the same score moves cleanly with latent dimension, $0.159 \to 0.292 \to
0.341 \to 0.371 \to 0.420$ for $d = 2,4,6,8,10$ — a span two orders of magnitude
larger than the restart spread that separates the rungs.

Two caveats, and the first is fatal to over-reading this:

1. **§10.1 makes it nearly a statement of linear algebra.** A flat ladder is
   what a linear flow with paired complex eigenvalues *must* produce. The
   constraint is not falsified; it is also barely tested.
2. Co-smoothing is **gauge-invariant by construction** — if $\hat z = h(z)$ the
   refitted readout is $D \circ h$ and the score is unchanged. It is an adequacy
   gate and can never be an identifiability test. Same mechanism as §2: a
   full-column-rank decoder absorbs $h$.

### 10.3 Identifiability — invariant agreement across disjoint neurons (task 40)

Fit independently on **disjoint** neuron halves (79/79, rate-stratified so the
halves are comparable populations), 3 independent splits $\times$ 6 restarts a
side, then compare the fits *to each other* — no ground truth anywhere. Three
negative controls, each altering only half B:

| control | what it preserves | what it destroys |
|---|---|---|
| **per-neuron circular shift** | every neuron's own time course, smoothness, autocorrelation | the cross-neuron alignment that makes a *shared* latent exist |
| time reversal | everything | the arrow of time |
| within-condition time shuffle | marginal rates | all temporal structure |

Only the circular shift is load-bearing, and this was decided before scoring.
The other two are too easy: near-neutral dynamics ($\lvert\lambda\rvert \approx
0.99$) is nearly time-*reversible*, so reversal moves the exponents by about the
fit noise; and shuffling is detectable by anything. **Asserting against a control
the metric cannot fail is self-congratulation.**

**The result is per-invariant, and the two halves of the fingerprint disagree.**

| invariant | treatment | circshift null | ratio |
|---|---|---|---|
| rotation number | $0.00208$ | $0.02173$ | $\mathbf{10.4\times}$ |
| rotation, relative to module separation | $0.112$ | $36.9$ | $\mathbf{330\times}$ |
| GL(2,$\mathbb{Z}$) lattice margin | $0.00151$ | $0.01553$ | $\mathbf{10.3\times}$ |
| global Lyapunov spectrum | $0.02645$ | $0.02536$ | **none** |
| per-module Lyapunov spectra | $0.02880$ | $0.02710$ | **none** |

So: **the rotation number is recovered; the Lyapunov spectrum is not.** Two
disjoint neuron halves agree on rotation to $11\%$ of the gap between the two
modules' own rotation numbers, and the lattice margin also clears the
random-rotation-vector null ($0.00151$ against $0.00228$). On the spectra they
agree no better than a half compared against circularly-shifted junk.

> **This inverts the expectation, and the inversion is the interesting part.**
> The spectrum is the *theoretically* cleaner invariant — it is Tier 1, it costs
> no theorem, and §6.5's $GL(K,\mathbb{Z})$ ambiguity cannot touch it. The
> rotation number is the one carrying a caveat. Empirically it is the other way
> round.
>
> The mechanism is CLAUDE.md §3.13(b) — recoverability tracks where the orbits
> spend time — arriving in a new form. Over 35 bins at $\lvert\lambda\rvert
> \approx 0.99$ the system contracts by $30\%$ in total, so there is almost **no
> contraction to measure** and the exponents are weakly determined. Phase, by
> contrast, advances all the way through the window. A quantity is identifiable
> only if the data moves along it.

**A ground-truth-free validity flag, found by accident.**
`DynamicalFingerprint.duplicate_modules` flags **18 of 18** circshift-control
fits and **1 of 36** treatment fits. §3.13(e) introduced it as a mode-collapse
detector for individual fits; it turns out to separate real population dynamics
from the null outright, using only the fitted model. That is worth more on real
data than any tolerance, because nothing else here is checkable without truth.

**(F3) holds in 2 of 36 fits**, chain gap median $-0.042$. So the best empirical
result again sits **outside** the best theorem — the same pattern `exp14` found
on synthetic cycles (0 of 24), and for the same reason: the module spectra
overlap, so Theorem F never applies. (F3) is sufficient, not necessary.

> **A bug in my own scoring, corrected and recorded.** The checks first read
> $6/6$. Three of them were vacuous: the circular-shift control screens out
> *completely* (all 18 fits duplicate-flagged), leaving its screened arm empty,
> and the helper mapped `NaN` to $+\infty$ so "treatment $<$ control" passed
> against nothing. `NaN` now fails, and the two arms are compared like-for-like.
> The corrected score is $7/9$, with the two spectrum comparisons failing — as
> they should. Same family as CLAUDE.md §3.9: **a comparison that cannot fail is
> not a test.** The checks were re-derived offline from the dumped fingerprints
> without refitting, which is what §3.13 keeps them for.

### 10.4 What this licenses, and what it does not

**Licensed.** On MC_Maze, in the movement epoch, at $d = 4$: two disjoint halves
of the population independently recover the *same pair of rotation numbers*,
$10\times$ better than a null that preserves every single-neuron statistic. The
modular constraint costs nothing in held-out-neuron prediction.

> **WEAKENED by §11.3(a) (2026-08-15).** This paragraph originally called that
> "a real identifiability result". It is not, quite. `exp16` runs the same
> protocol on two limit cycles — where non-identifiability is *proved*, only the
> $GL(2,\mathbb{Z})$ orbit being pinned — and the protocol still returns
> "identified", because both fits happen to land on the same lattice
> representative. So what is demonstrated here is **estimator reproducibility
> across disjoint neuron samples**: real, non-trivial, and decisively failed by
> the shuffle null, but strictly weaker than identifiability. Read every claim
> below in those terms until §11.5's adversarial-initialisation test is run.

**Not licensed, and the gap is not small:**

1. **It validates Theorem A, not Theorem B.** §10.1: the flow is $\ge 99\%$
   linear, so modularity is generic rather than restrictive here.
2. **The rotation numbers are pinned only up to $GL(2,\mathbb{Z})$** (§6.5, task
   23). The margin clears its null, so the agreement is not an artefact of the
   lattice — but what is identified is the *orbit*, not the pair.
3. **The Lyapunov spectrum is not recovered at all**, so the filtration half of
   Theorem F is untested here rather than confirmed. §11.3(c) shows this
   reproduces under *known* ground truth with a nonlinear decoder, so it is an
   estimator property rather than a fact about MC_Maze.
4. **Agreement is reproducibility, not identifiability** — §11.3(a).
5. One dataset, one area, one epoch, one monkey.

The honest one-line summary: *the structure is there and one of its two
invariants is identifiable, in a regime where the theory that applies is the
one that was already proved.*

---

## 11. Calibrating the Route C instrument — `exp16`

**New (2026-08-15).** §10 ran the split-and-compare protocol on data where
nobody knows the right answer. This runs it where the answer is **certain, in
both directions**, on the three systems this repo owns whose non-identifiability
is *proved*. Two stages, and separating them is the point: an **analytic** stage
with exact systems and no fitting, so any failure is the metric's; then the
**fitted** protocol, so any additional failure is the estimator's.

### 11.1 The metric is correctly calibrated (analytic, exact)

| check | measured | required |
|---|---|---|
| radial shear gauge (§7.1) | agree; $\rho$ err $1.0\times10^{-10}$, spec err $1.3\times10^{-10}$ | must **agree** |
| §3.1 regrouping | disagree; spec err $0.2231$ | must **disagree** |
| (F3) true vs regrouped | $+0.2231 \to -0.2231$ | must flip sign |
| torus rotation (§7) | moves by $0.2069$ | must move |
| torus after $GL(2,\mathbb{Z})$ | margin $0.00\times10^{0}$ | must vanish |
| (F3) on two neutral cycles | $-0.9163$ | must reject |

Six for six. The fingerprint is blind to what §8 declines to identify and
sensitive to what the counterexamples change — which is the minimum an
instrument must satisfy before it is pointed at data.

### 11.2 The fitted protocol, crossed with the observation model

$d=4$, partition $[2,2]$, 160 neurons split 80/80, 4 restarts a side, each arm
run under a linear and an `MLPDecoder` observation map. "Identified" means the
cross-split error beats the circular-shift null by $3\times$.

| arm | decoder | obs-nl | (F3) | $\rho$ err | $\rho$ null | Tier-1 | T1 null | verdict |
|---|---|---|---|---|---|---|---|---|
| A filtration | linear | $0.000$ | $+0.500$ | $0.0103$ | $0.0553$ | $0.0195$ | $0.4465$ | ROT+SPEC |
| A filtration | mlp | $0.423$ | $+0.134$ | $0.0030$ | $0.0550$ | $0.1687$ | $0.1897$ | ROT only |
| B regrouping | linear | $0.000$ | $-0.521$ | $0.0080$ | $0.0102$ | $0.1085$ | $0.5255$ | SPEC only |
| B regrouping | mlp | $0.637$ | $-0.395$ | $0.0081$ | $0.0122$ | $0.2567$ | $0.5789$ | **neither** ✓ |
| C torus | linear | $0.000$ | $-0.676$ | $0.0004$ | $0.2068$ | $0.0149$ | $0.2074$ | ROT+SPEC ✗ |
| C torus | mlp | $0.506$ | $-0.598$ | $0.0171$ | $0.2011$ | $0.3383$ | $1.1538$ | ROT+SPEC ✗ |
| D gauge | linear | $0.000$ | $+0.488$ | $0.0076$ | $0.0639$ | $0.0111$ | $0.2681$ | ROT+SPEC |
| D gauge | mlp | $0.383$ | $-0.038$ | $0.0413$ | $0.0681$ | $0.0957$ | $0.2253$ | **neither** ✗ |

### 11.3 Three findings, two of them limits on the method

**(a) Arm C is a false positive, and it is the important result.** Two limit
cycles are provably identified only up to $GL(2,\mathbb{Z})$ — §11.1 confirms the
regrouping moves the rotation vector by $0.2069$ with lattice margin exactly
zero. Yet the two independently fitted halves agree to $0.0004$, under *both*
decoders. The reason is not subtle: **both fits land on the same lattice
representative.** An ambiguity being available does not mean the optimiser
explores it.

> **Cross-split agreement is necessary for identifiability, not sufficient.**
> What the protocol of §10.3 measures is *estimator reproducibility across
> disjoint samples of neurons*. That is a real and non-trivial property — the
> circular-shift null fails it decisively — but it is strictly weaker than
> identifiability, and on a system where non-identifiability is proved the
> protocol returns "identified".
>
> This confirms experimentally the retraction already recorded against task 40:
> two fits "land on the same lattice basis for reasons of parameterisation, not
> because the observations pin it". §10.4's licensed claim must be read with
> this attached.

**(b) Gauge-blindness of the metric is not gauge-robustness of the estimator.**
Arm D is arm A seen through a radial shear. Analytically the fingerprint is
blind to it to $10^{-10}$ (§11.1). Under a *linear* decoder the fitted protocol
recovers it cleanly. Under the MLP decoder it fails — $\rho$ err $0.0413$
against a null of $0.0681$, a ratio of $1.65$ — and the fitted (F3) gap collapses
from $+0.488$ to $-0.038$. The invariant is untouched; the *estimate* of it is
not. Only running both stages separates these.

**(c) §10.3's real-data result reproduces under known ground truth.** Arm A
under the MLP decoder recovers the rotation number superbly ($0.0030$ against a
null of $0.0550$) and the Lyapunov spectrum not at all ($0.1687$ against
$0.1897$) — with the ground truth known and (F3) *holding* at $+0.134$. So "the
rotation number is recoverable, the spectrum is not" is a property of the
estimator under a nonlinear observation map, **not** a peculiarity of MC_Maze.
That is the strongest support §10.3 has, and it arrives from a completely
different direction.

### 11.4 What (F3) is actually good for

The pre-registered prediction was that $\operatorname{sign}(\texttt{filtration\_gap})$
tracks recoverability. It holds **one way only**:

- $\text{(F3)} > 0$: identified in $3/3$ cells.
- $\text{(F3)} \le 0$: identified in $2/5$ — and both exceptions are arm C.

> **Usable rule.** A positive fitted (F3) is a *positive* indicator: the modules
> are spectrally separated, so the filtration is meaningful and the protocol
> recovers it. A non-positive (F3) is **not** a reliable negative — it warns that
> the modules are not spectrally separable, but it does not follow that the
> protocol will notice the resulting ambiguity.

**A caveat on the verdict rule, of the CLAUDE.md §3.9 family.** Arm C is scored
as recovering the *spectrum* because both its modules have identical spectra
$\{0, \log\lvert 1-2a\rvert\}$ — so agreeing on them is free, while the shuffled
null destroys them entirely and scores badly. The comparison cannot fail, which
is precisely why it passes. (F3) $< 0$ is the flag for exactly this degeneracy:
when the hulls coincide there is nothing for spectral agreement to mean.

### 11.5 Consequence: what needs building

The arm-C failure is specific and fixable. The protocol currently *hopes* the
optimiser explores the ambiguity. It should instead be made to:

> **Adversarial initialisation.** Start the two fits at deliberately *different*
> representatives — the alternative grouping for §3.1, the lattice image for §7 —
> and refit. If the data pins the representation, they converge back to the same
> invariants; if it does not, they stay apart. That converts agreement from a
> necessary condition into something much closer to sufficient.

Both alternative representatives already exist in code
(`regrouping_counterexample` returns `system_tilde`,
`torus_regrouping_counterexample` returns the lattice image), and arm C is the
known-answer case that says whether the fix works.

> **BUILT AND RUN — see §12 (`exp17`, 2026-08-16). It works.** Under adversarial
> initialisation arm C's cross-split agreement goes $0.00018 \to 0.1271$ on the
> *same* metric, i.e. the false positive above becomes a correct rejection, and
> the escape control still returns to the truth at $0.0001$. Verdict correct in
> all four arms.
>
> **§10.3's MC_Maze conclusion is still estimator reproducibility, not
> identifiability**, because the adversarial test has not been run on that data —
> §12.6 says how it can be, using the lattice image of the *fitted* latents.

### 11.6 What actually protects a contracting module: (F1), not (F3)

**Found while designing the adversarial-initialisation test (2026-08-15), and it
refines §7/task 23.** The natural reading of §7 is that the $GL(K,\mathbb{Z})$
ambiguity is a pathology of *limit cycles*. It is not.

Take arm A — two contracting spirals, $s = 0.92$ and $0.55$, spectra
comfortably separated so **(F3) holds at $+0.50$**. The lattice map
$h(z_1,z_2) = (z_1z_2/\lvert z_2\rvert,\ z_2)$ is an **exact conjugacy** there,
carrying $\omega_1 \mapsto \omega_1+\omega_2$: residual $6.7\times10^{-16}$,
measured at donor radii $1$, $10^{-2}$ and $10^{-6}$ alike. So spectral
separation does **not** protect the rotation numbers.

What protects them is **(F1)**. The conjugacy has
$\lVert Dh\rVert \sim 1/\lvert z_2\rvert$, and the two arms differ entirely in
how small the donor gets on the visited region:

| system | $\min\lvert z_2\rvert$ over the trial | $\sup\lVert Dh\rVert$ | (F1)? |
|---|---|---|---|
| two contracting spirals ($s_2 = 0.55$) | $1.6\times10^{-8}$ | $6\times10^{7}$ | **fails** |
| two limit cycles | $0.80$ | $1.2$ | holds |

> **The lattice ambiguity bites exactly when the donor module does not decay.**
> A contracting module is protected because the regrouping map blows up where
> its orbits actually go — §0's support caveat doing real work rather than
> hedging. Two neutral or oscillatory modules have no such protection, which is
> why §7's counterexample is built from limit cycles and why `exp15`'s MC_Maze
> fits ($\lvert\lambda\rvert \approx 0.99$, contracting only $30\%$ over the
> window) sit on the wrong side of it.

**Checkable, and it should be reported alongside any rotation claim:** compute
$\min\lvert z_i\rvert$ over the visited region for each module. Bounded away from
zero means the lattice ambiguity is live for that module; decaying to zero means
(F1) excludes it. This is a *different* diagnostic from `filtration_gap`, and
§11.3(a) shows (F3) alone is not enough.

### 11.7 Why §11.5's fix needs a nonlinear encoder

A correction to §11.3(a)'s diagnosis. I attributed arm C's false positive to
"both fits landing on the same lattice representative" — the optimiser failing
to explore. That is at most half of it.

The fitted model in `exp16` has a **linear encoder**, so its latents are
$\hat z = L\,g(z)$. Under a linear generating decoder that is *linear in $z$*,
and the lattice map is not — so the alternative representative is **outside the
fitted model class**, and the protocol could not have detected it whatever the
optimiser did. Under the MLP generating decoder it becomes partially reachable,
and the fits still agreed; there the "optimiser did not explore it" reading does
apply.

Consequence for the adversarial test: it is only meaningful when the fitted
class can *represent* the alternative. Warm-starting a linear encoder at a
nonlinear representative measures how fast the projection back onto the linear
class happens, not whether the data pins the representation. **`encoder="mlp"`
is therefore a requirement of the test, not a variant of it.**

---

## 12. Adversarial initialisation — `exp17`, and task 41 closed

**NEW (2026-08-16).** §11.3(a) left the method with a hole: the split-and-compare
protocol returned *identified* on arm C, two limit cycles, where
non-identifiability is **proved**. Both fits happened to land on the same
lattice representative, so what §10.3 measured was estimator reproducibility
across neuron samples. This section closes that.

### 12.1 The fix, and why the readout had to change too

Stop letting the optimiser choose the representative:

> warm-start the two halves at **deliberately different** representatives, then
> train normally and see whether the data pulls them back together.

`train.warm_start_to_latents` drives encoder, decoder and transition onto a
designated latent representative; ordinary training then runs unchanged from a
fresh optimiser.

The scored quantity is **not** fit-to-fit agreement. Reusing it would inherit
the defect being repaired, so the primary readout asks the direct question —
*is this fit nearer R1 or nearer R2* — against the two analytic targets, on the
invariant and the modules the construction says separate them. Those are fixed
before any fit runs, because $h$ is written down.

**Restricting to the informative modules is required, not a convenience.** A
module where R1 and R2 agree cannot say which representative a fit picked, and
on a contracting system it is exactly the module the data does not constrain.
Measured: the module the lattice map moves comes back to $3\times10^{-4}$ at
every donor rate, while the donor's own rotation number is never recovered
($0.0039$ against a true $0.1751$ at $s=0.55$, still $0.1101$ at $s=0.80$).
`invariant_agreement` reports a **max** over modules, so an unrestricted
comparison would have decided arm A entirely on an invariant nobody measured —
§3.13(b), and the mirror image of §11.3(f)'s comparison that cannot fail.
`AgreementReport` now also carries `per_module_spectrum` / `per_module_rotation`.

### 12.2 The arms, and the analytic pre-flight

| arm | $h$ | conjugacy defect | $\sup\lVert Dh\rVert$ | truth |
|---|---|---|---|---|
| A spirals | lattice $z_1\mapsto z_1z_2/\lvert z_2\rvert$ | $6.1\times10^{-16}$ | $1.3\times10^{6}$ | (F1) fails |
| C cycles | same | $1.1\times10^{-15}$ | $1.9$ | (F1) holds; **not identifiable** |
| B regroup | coordinate permutation | $0$ | $1.0000$ | (F1) holds; **not identifiable** |
| E escape | $(z_1+cz_2,\ z_2)$ | — | $1.477$ | **not a conjugacy** |

Arm E is the load-bearing control (§3.11): $HFH^{-1}$ carries **0.4305** of its
mass off-block, so no modular $\tilde F$ conjugates through that $h$. Without
it, "the fit stayed at R2" would be unattributable — it could be inertia.

> $\sup\lVert Dh\rVert$ is now **measured**, not inferred. $1/\min\lvert z_2\rvert$
> is the right estimate for the lattice map and for nothing else: arm B's blocks
> decay to $1.2\times10^{-5}$ and its permutation still has $\lVert Dh\rVert=1$
> exactly. Arm E's $1.477$ matches $\lVert[[I,0.8I],[0,I]]\rVert$ analytically,
> which is what makes the measurement a measurement.

### 12.3 Reachability: (F1) as an empirical statement

Warm-start residual against warm-start budget, no main training — and, at the
largest budget, what the warm-started model's **fingerprint** reads:

| arm | at R1 | at R2 (200 / 800 / 3200) | ratio | fingerprint reads |
|---|---|---|---|---|
| A | $0.0013$ | $0.1922 \to 0.0912 \to 0.0589$ | $45\times$ | **R1** ($0.0556$ vs $0.0881$) |
| C | $0.0001$ | $0.0012 \to 0.0004 \to 0.0003$ | $3\times$ | **R2** ($0.0043$ vs $0.1313$) |
| B | $0.0039$ | $0.0144 \to 0.0243 \to 0.0033$ | $0.85\times$ | **R2** ($0.0743$ vs $0.1208$) |

**My pre-registered prediction here was wrong and the check fails** (1 of 19):
I predicted arm A's alternative would stay unreachable at $>0.25$, and it
reaches $0.0589$. The correct statement is sharper. Arm A's alternative is
reachable *in latent values* to $6\%$ and is still not reachable **as a model** —
the warm-started dynamics read R1. So (F1) does not make the alternative
unrepresentable pointwise; it makes it unrepresentable as a *conjugacy*, which
is where (F1) actually lives.

Checking the fingerprint and not only the encoder residual is what made that
visible. An encoder sitting on R2 with a transition still at its initialisation
would have made every downstream verdict meaningless.

### 12.4 The result

Four arms, 4 restarts each, 3 conditions (half 1 at R1; half 2 at R1 matched;
half 2 at R2 adversarial). 63 minutes, 48 fits.

| arm | verdict | at R2 | $a\to$R1 | $a\to$R2 | sep | resolving | correct? |
|---|---|---|---|---|---|---|---|
| A spirals | returned | 0/4 | $0.0207$ | $0.1652$ | $0.1194$ | $81\times$ | yes |
| C cycles | **STAYED** | **4/4** | $0.1271$ | $0.0001$ | $0.1273$ | $917\times$ | yes |
| B regroup | **STAYED** | **4/4** | $0.1249$ | $0.0372$ | $0.1079$ | $3.1\times$ | yes |
| E escape | returned | 0/4 | $0.0001$ | — | — | — | yes |

**The verdict is right in all four arms**, and the matched control lands nearer
R1 in 4/4 in every arm, so the readout can tell the two apart at all.

**The cleanest way to see the repair is on `exp16`'s own metric.** Same data,
same fits, same measurement — only the initialisation differs:

| arm | fit-to-fit, matched init | fit-to-fit, adversarial init |
|---|---|---|
| A spirals | $0.0049$ | $0.0207$ |
| **C cycles** | $\mathbf{0.00018}$ | $\mathbf{0.1271}$ |
| B regroup (spectra) | $0.0211$ | $0.1165$ |
| E escape | $0.0020$ | $0.0033$ |

Under matched initialisation every arm agrees — **reproducing §11.3(b)'s false
positive for C**. Under adversarial initialisation the two arms where
non-identifiability is *proved* disagree, and the two that should agree still
do. That is the whole of task 41 in one table.

### 12.5 Neither (F3) nor min|z_i| is a universal predictor

| arm | survived | $\min\lvert z_{\text{donor}}\rvert$ | (F3) gap |
|---|---|---|---|
| A | no | $8.1\times10^{-9}$ | $\mathbf{+0.5146}$ |
| C | **yes** | $8.1\times10^{-1}$ | $\mathbf{-0.1453}$ |
| B | **yes** | $1.2\times10^{-5}$ | $+0.1011$ |
| E | no | $8.2\times10^{-9}$ | $+0.5159$ |

Two readings, and both matter.

**For the lattice family — arms A and C, the *same* $h$ — §11.6 is confirmed
exactly.** $\min\lvert z_{\text{donor}}\rvert$ separates them by eight orders of
magnitude and predicts the outcome; **(F3) orders them backwards**, positive
where the alternative dies and negative where it survives.

**But arm B breaks both as universal rules.** It has (F3) $=+0.10$ *and* a donor
radius of $10^{-5}$, and its alternative survives anyway — because its $h$ is a
permutation, whose derivative is $1$ however small the blocks get. So
$\min\lvert z_i\rvert$ is the checkable diagnostic **for the lattice family**,
not a map-independent one, and **(F3) $>0$ does not imply the representation is
pinned.** Theorem F's hypothesis buys the filtration; it does not buy the
representative.

### 12.6 What this licenses, and what it does not

**Licensed.** Adversarial initialisation converts cross-split agreement from a
necessary condition into something much closer to sufficient: it returns the
right answer on three systems where the answer is proved and on one control that
must be rejected, and it flips exactly the cases §11.3(b) got wrong.

**Not licensed, and these are real.**

1. **Arm A's negative is weaker than arm C's positive.** The warm start never
   put arm A *at* R2 (§12.3), so for that arm the test partly did not apply. The
   two readings — "the data pulled it back" and "the class cannot hold it" —
   coexist, and only the second is established.
2. **Arm B resolves its own question by only $3.1\times$.** Its separation is
   $0.108$ against an estimator error of $0.035$. The verdict is unanimous
   across restarts, but the margin is thin, and it is scored on spectra, the
   invariant §10.3 found least recoverable.
3. **Four restarts.** Every arm came out $4/4$ or $0/4$, so restart noise is not
   what is at issue, but §3.11 asks for more before a *continuous* readout.
4. **This is synthetic.** It has not been run on MC_Maze.

**Consequence for §10.3, and it is not yet discharged.** The MC_Maze conclusion
must still be read as estimator reproducibility, because the adversarial test
has not been run there. It now *can* be: the fitted model supplies modules, so
the lattice image of the **fitted** latents is constructible without knowing the
true $h$ — apply $z_1 \mapsto z_1z_2/\lvert z_2\rvert$ to the fit's own latents,
warm-start a second fit there, and refit. §12.5 says what to report beside it:
$\min\lvert z_i\rvert$ per module, **not** `filtration_gap`. On MC_Maze that
number is the concerning one — $\lvert\lambda\rvert \approx 0.99$ over 35 bins
means nothing decays, which is precisely the regime where the lattice ambiguity
is live.

---

## 13. Can Route B kill what Route C cannot? — `exp18`

Route C is dead in the oscillatory regime and not for want of effort: two limit
cycles have identical spectral hulls, so (F3) fails outright, and §7 of
`counterexamples.md` exhibits the $GL(2,\mathbb{Z})$ lattice regrouping as an
*exact* modular conjugacy. §12 confirmed the fitted consequence — warm-started at
that alternative, the fit stays there.

Route B is the natural candidate to finish the job, and on paper it should.
Lemma D′ (§4.5a) needs **no spectral hypothesis at all**, only
$1\notin\operatorname{spec}(\tilde f_B)$, so it applies exactly where Lemma C,
Theorem F and (D1) are all dead; its own witness is a pair of modules with
identical spectra. So the question has a clean shape:

> does a behavioural auxiliary variable reject the lattice representative that
> the dynamics alone cannot?

**The answer is no, and the obstruction is a symmetry rather than an estimator
failure.** `exp18` is 9/12, with the analytic pre-flight 3/3 and the
pre-registered headline **refuted**.

### 13.1 Lemma D's own modulation hypothesis is unavailable here

(D3) is variance modulation, $z_A = s(u)\,\tilde z_A$. A limit cycle attracts
every radius to $\rho$, so a radial conditioning of the initial law is
*forgotten*: the modulated block's $u$-dependence falls $0.799 \to 0.034$ and the
between-level mean-radius gap falls $6.0\times10^{-1} \to 1.8\times10^{-12}$ over
thirty steps. What persists on a cycle is the **phase**, which is not (D3).

So in the regime where Route C has no hypothesis, Lemma D has none either. What
`exp18` tests is therefore the Route B **mechanism** — the $u$-invariant subspace
is canonical, so $h$ must map it into itself — and not Lemma D as proved. §3.13(b)
in a third form: a hypothesis is usable only where the data still moves along it.

### 13.2 The non-additive escape becomes a genuine conjugacy on a cycle

Write the regrouping donor-first, in complex coordinates:

$$h(z_A, z_B) = \bigl(z_A,\; z_B\, z_A/|z_A|\bigr).$$

The orientation is forced: Lemma D kills $M_{BA} = \partial h_B/\partial z_A$, so
the $u$-varying block must be the **donor**. With the roles swapped the coupling
sits in the block Lemma D does not touch and every arm returns a null for a
trivial reason.

In this form the invariant block is **rotated** by the donor's phase — which is
exactly `systems.nonadditive_behavioural_escape`, recorded in §4.5 as satisfying
(D1)–(D4) with $M_{BA}\neq0$. What excluded it there was **Step 1**: conjugating
modular dynamics to modular dynamics needs $\theta\circ f_A-\theta$ constant, and
at the *fixed point* of a contracting $f_A$ that constant is forced to $0$, hence
$\theta\equiv0$. **Replace the fixed point with an attracting limit cycle and the
obstruction evaporates** — with $\theta=\arg z_A$ the increment is exactly
$\omega_A$. Verified: conjugacy defect $7.5\times10^{-16}$, carrying
$\omega_B\mapsto\omega_B+\omega_A$. Built as
`systems.rotational_behavioural_escape`.

So the escape §4.5 excluded is available in precisely the regime Route B was
being asked to cover.

### 13.3 Behaviour is blind to a coupling that acts by a symmetry of $p_B$

The reparameterised invariant block is $z_B$ rotated by an angle independent of
$z_B$. If $p_B$ is rotationally symmetric — which it is whenever the cycle's
phase law is uniform — then

$$p(h_B \mid u) \;=\; \int (T_\theta)_*p_B \; dp(\theta\mid u) \;=\; p_B
\qquad\text{for every } u,$$

whatever the donor's phase does. Measured: the detector reads $0.016$ for the
regrouped block against its own $0.044$ finite-sample floor — *below* the noise.
The floor is checked, not assumed: it falls with slope $-0.44$ in $n$ against the
expected $-1/2$.

**The general statement, and it explains Lemma D's hypothesis retroactively.**
Let $h_B(z_A,z_B) = T(z_A)\cdot z_B$ with $T$ valued in a group $G$ acting on the
$z_B$ space, and let $z_A \perp z_B$ given $u$. If $p_B$ is $G$-invariant then
(D4) holds exactly while $M_{BA}\neq0$. **Route B's kill therefore requires $p_B$
to have trivial symmetry under the group in which the coupling takes values.**
Additive $h_B$ — Lemma D's case — is $G=$ translations, and *no* probability law
on $\mathbb{R}^{d_B}$ is translation-invariant. That is why additivity was the
right hypothesis rather than a convenience, and it also explains §4.5's remark
that the escape needs $d_B\ge2$: at $d_B=1$ the orthogonal group is the discrete
$\{\pm1\}$, so no family continuous in $z_A$ exists.

### 13.4 Breaking the symmetry works in the data — and does not survive learning

Concentrating the recipient's phase *without* making it $u$-dependent keeps it a
legitimate invariant block and turns the detector back on, monotonically:

| $\kappa_B$ | $0.0$ | $0.5$ | $1.0$ | $2.0$ | $4.0$ |
|---|---|---|---|---|---|
| block under R2 | $0.016$ | $0.205$ | $0.441$ | $0.747$ | $0.956$ |
| ratio to floor | $0.4\times$ | $6.1\times$ | $13.8\times$ | $9.4\times$ | $16.7\times$ |

That looked like a design rule. **It is not, and this is `exp18`'s sharpest
finding.** The $2\times2$ — recipient symmetric/asymmetric $\times$ penalty
off/on, every fit warm-started at R2 — returns:

| cell | verdict | adv $\to$R1 | adv $\to$R2 | matched `fitq` | adv $u$-dep (true) |
|---|---|---|---|---|---|
| sym, $w=0$ | SURVIVED | $0.0795$ | $0.0001$ | $1.24\times10^{-4}$ | $0.203$ ($0.127$) |
| sym, $w=0.1$ | SURVIVED | $0.0797$ | $0.0001$ | $1.16\times10^{-4}$ | $0.091$ ($0.065$) |
| asym, $w=0$ | SURVIVED | $0.0795$ | $0.0001$ | $6.91\times10^{-5}$ | $0.961$ ($0.217$) |
| **asym, $w=0.1$** | **SURVIVED** | $0.1101$ | $0.0314$ | $2.13\times10^{-3}$ | $0.310$ ($0.152$) |

The pre-registered prediction was that the last row would **reject**. It did not,
in 4/4 restarts, and the check is committed failing (§8 of CLAUDE.md).

What happened instead is specific. The penalty did bite — it pulled the invariant
block $0.961\to0.310$ and moved the fit off R2 ($0.0001\to0.0314$) — but the fit
moved *away* from R1 ($0.0795\to0.1101$, i.e. past R2, since the whole R1–R2
separation is $0.0796$), at a $30\times$ fit-quality cost. And the matched arm at
the same weight sits at R1 with **both** a lower penalty and a better fit. So R1
is preferred and not reached.

**Why: the encoder restores the symmetry.** The discriminating measurement is the
circular concentration $|\mathbb{E}e^{i\phi}|$ of the fitted invariant block
(`exp18_mechanism`; no second-moment quantity can see this, since whitening makes
an isotropic and a phase-concentrated block identical):

| fit | concentration | by $u$ | $u$-dep |
|---|---|---|---|
| matched (at R1) | $0.809$ | $[0.811, 0.808]$ | $0.101$ |
| adversarial (at R2) | $\mathbf{0.270}$ | $[0.313, 0.429]$ | $0.436$ |
| *data*, R1 block | $0.854$ | | |
| *data*, R2 block | $0.392$ | | |

The matched fit reproduces the data's own concentration and is flat across $u$.
The adversarial fit's is $0.270$ — **below even the R2 representation's own
$0.392$**: the encoder has smeared the phase *further* than the regrouping
already did, which is exactly the direction that makes the block rotationally
symmetric and the coupling invisible. The escape is not fully reached (the per-$u$
concentrations still differ), and that partial hiding is what the fit-quality
cost buys. Unanimous over 3 restarts — adversarial $0.270/0.264/0.211$ against
matched $0.809/0.818/0.807$, a $3.1\times$ separation with no overlap.

**And the escape shows up *within* the treatment arm, which is stronger than the
contrast with the matched one.** `exp18_mechanism` reuses `exp18`'s seeds, so the
two runs join per restart:

| restart | fitted concentration | $u$-dep | distance to R2 |
|---|---|---|---|
| r0 | $0.2703$ | $0.4359$ | $0.0495$ |
| r1 | $0.2640$ | $0.4313$ | $0.0496$ |
| r2 | $\mathbf{0.2106}$ | $\mathbf{0.1886}$ | $\mathbf{0.0130}$ |

Monotone in all three columns: **the more the encoder symmetrises the block, the
better it satisfies the behavioural penalty, and the more firmly it holds R2.**
The restart that hides best is the one that scores best — which is the signature
of an evaded constraint rather than an imposed one, and is why §13.5 asks for the
concentration to be reported beside any $u$-dependence.

So the design rule of §13.4's first paragraph is a statement about the **data**,
and the model class can undo it. The $u$-invariant subspace is canonical only if
$p_B$ has trivial symmetry **in every representation the model can reach** — a
far stronger requirement than trivial symmetry in the data. Third instance of the
§3.12 pattern: a structural constraint satisfied by moving in the gauge rather
than by acquiring the structure.

### 13.5 What this licenses, and what it does not

**Licensed.** Route B does not rescue the $GL(K,\mathbb{Z})$ ambiguity for
oscillatory modules. It is blind when $p_B$ is rotationally symmetric (§13.3,
analytic and exact), and where it can see, imposing it at a weight that preserves
the fit does not restore R1 (§13.4, 4/4 restarts). Combined with §11.6 — (F1),
not (F3), is what excludes the lattice regrouping — the honest position is that
**for two non-decaying oscillatory modules the rotation vector is identified only
up to the lattice, and neither route removes it.**

**Not licensed.** (i) This is the Route B *mechanism*, not Lemma D, which §13.1
shows has no applicable hypothesis here — Lemma D and D′–D‴ stand unchanged in
their own settings. (ii) The $2\times2$'s negative is about the *adversarial*
protocol; it does not show that a random-init fit with behaviour on would fail to
prefer R1, though §11.3 is the reason not to read a random-init agreement as
evidence either. (iii) Four restarts, one system, one $(\omega_A,\omega_B)$ pair.
(iv) `duplicate_modules` flags 4/4 in **every** cell including the $w=0$ controls,
because two limit cycles have identical spectra by construction — so check 9 is
vacuous on this system rather than failing, the `exp15` trap in a new place, and
it is kept and marked rather than deleted.

**The one checkable thing that survives, and it is for real data.** The condition
"$p_B$ has no symmetry that the coupling lives in" is a property of the recorded
distribution, testable with no fitting at all: compute the circular concentration
of a candidate module's phase. Trial-aligned neural data plausibly satisfies it in
the *data* — trials do not begin at uniformly distributed phases — but §13.4 says
that is not sufficient, since the encoder can flatten it. Any Route B claim on
real data should report the fitted block's concentration alongside its
$u$-dependence, or it cannot tell an imposed constraint from an evaded one.

---

## 14. Route B's exact reach — the compact-stabiliser dichotomy (`exp20`)

**New (2026-08-24).** §13.3 records that behaviour is blind to a coupling acting
by a symmetry of $p_B$ and stops at the slogan. It completes into a dichotomy,
and the completion is worth having because it turns `exp18`'s negative into a
statement with a boundary: Route B's kill is **not** restricted to additive
$h_B$, and the residual ambiguity is always a *compact* group.

### 14.1 The proposition

> **Proposition S.** Let $h_B(z_A,z_B) = T(z_A)\cdot z_B$ with $T$ valued in a
> group $G$ acting affinely on $\mathbb{R}^{d_B}$, let $z_A \perp z_B$ given $u$,
> and let $p_B$ have finite nonsingular second moment. Then **(D4)** holds with
> $M_{BA}\neq0$ **iff** $T$ takes values in a single coset of
> $\mathrm{Stab}_G(p_B)$; and
> $$\mathrm{Stab}_G(p_B) \;\subseteq\; \Sigma^{1/2}\,O(d_B)\,\Sigma^{-1/2},$$
> a **compact** group, where $\Sigma = \mathrm{Cov}(p_B)$.

*Proof.* $(A,b)_*p_B = p_B$ preserves the first two moments, so
$A\Sigma A^\top = \Sigma$ and $A\mu + b = \mu$. The first gives
$\Sigma^{-1/2}A\Sigma^{1/2} \in O(d_B)$; the second determines $b$ from $A$. A
closed subgroup of a conjugate of $O(d_B)$ is compact. Conversely, if
$T(z_A) \in T_0\,\mathrm{Stab}$ for all $z_A$ then $T(z_A)_*p_B = (T_0)_*p_B$
for every $z_A$, so the law of $h_B$ is $u$-free while $\partial h_B/\partial z_A
\neq 0$. $\square$

**Two consequences, and they run in opposite directions.**

1. **Lemma D generalises well past additivity.** Any coupling valued in a group
   that is non-compact modulo the stabiliser — translations, scalings, shears —
   is killed. "No probability law is translation-invariant" (§13.3) is the
   $d_B$-dimensional shadow of compactness, so additivity was never the load-
   bearing hypothesis; *non-compactness* was.
2. **The escape is compact, hence small and measurable.** It is at most
   $\Sigma^{1/2}O(d_B)\Sigma^{-1/2}$ — and note *conjugate to*, not *equal to*.
   An anisotropic $p_B$ does not remove the escape, it moves it.

### 14.2 What `exp20` certifies — analytic, 15/15

Exact distributions, no fitting, so any failure is the criterion's or the
detector's. Every coupling is a one-parameter subgroup $T = \exp(\theta X)$ with
$\theta = c\,z_{A,0}$, and $c$ is **bisected to a common displacement**
$\mathbb{E}\lVert h_B-z_B\rVert/\mathbb{E}\lVert z_B\rVert = 0.300$ in every arm,
so "invisible" can never be "weak". The floor is measured at the same $n$ and
falls with log-log slope $-0.509$ against the expected $-1/2$.

| arm | $p_B$ | generator | $u$-dep | $\times$ floor | verdict |
|---|---|---|---|---|---|
| 1a | isotropic | translation | $0.1242$ | $10.4$ | killed |
| 1b | isotropic | scaling | $0.2449$ | $20.5$ | killed |
| 1c | isotropic | shear | $0.1753$ | $14.7$ | killed |
| 1d | isotropic | **rotation** | $0.0077$ | $\mathbf{0.64}$ | **invisible** |
| 2a | anisotropic | rotation | $0.1732$ | $12.9$ | killed |
| 2b | anisotropic | $\Sigma^{1/2}R\Sigma^{-1/2}$ | $0.0128$ | $\mathbf{0.95}$ | **invisible** |

Arms 2a/2b are the pair that distinguishes "conjugate to $O(d_B)$" from "equal
to $O(d_B)$", and they separate by $13.5\times$ at matched displacement. Making
$p_B$ anisotropic does not buy immunity — it relocates the blind group, and the
relocated one is exactly as blind.

### 14.3 The detector is blinder than the criterion, and structurally so

Arm 3 was pre-registered as a *negative prediction* and it landed. Take a
non-Gaussian $p_B$, **centred and whitened exactly** ($\lVert\mu\rVert =
1.4\times10^{-15}$, $\lVert\Sigma - I\rVert = 2.5\times10^{-15}$) and asymmetric
enough that $\mathrm{Stab}_{O(2)}(p_B)$ is trivial (4th circular harmonic
$0.165$). Proposition S says (D4) **fails** — and it does, grossly:

| statistic | coupled | uncoupled | ratio |
|---|---|---|---|
| 3rd circular harmonic gap across $u$ | $0.2134$ | $0.0030$ | $\mathbf{71\times}$ |
| `block_u_dependence` (whitened) | $0.0054$ | $0.0072$ (floor) | $0.75\times$ |

So the conditional laws are wildly different and **the detector reads below its
own noise floor**. Separating those two was the point of the harmonic test:
without it "at floor" would have been unattributable, equally consistent with
Proposition S saying there is nothing to detect.

**The sharpening this forces on §13.3, and it is not a detail.** §13.3 blames
$p_B$'s rotational symmetry. For a *whitened second-moment* detector the blind
group is $O(d_B)$ for **every** $p_B$, because whitening makes the second moments
isotropic by construction. So CLAUDE.md §3.12's fix — whiten the block so the
penalty is $GL(d_B)$-invariant — and the rotational blindness are **the same
fact**, not an unrelated repair. The gauge-invariance we require is precisely
what forfeits the orthogonal direction. That is a genuine tension in the
objective, and it was not visible until the criterion and the detector were
measured apart.

### 14.4 What a whitened second-moment detector *can* still see

Only a $u$-dependent **mean direction**. Arm 4 sweeps $\lVert\mathbb{E}z_B\rVert$
under a rotation coupling:

| $\lVert m\rVert$ | $0$ | $0.25$ | $0.5$ | $1.0$ | $2.0$ |
|---|---|---|---|---|---|
| circular concentration | $0.001$ | $0.155$ | $0.303$ | $0.557$ | $0.844$ |
| $\times$ floor | $0.77$ | $0.90$ | $\mathbf{11.9}$ | $25.3$ | $27.4$ |

Monotone, and a **threshold** rather than a ramp: detection turns on between
concentration $0.155$ and $0.303$.

**That bracket is where `exp18`'s mechanism measurement sits.** Its fitted
adversarial block had concentration $\mathbf{0.270}$ — inside the bracket, on the
undetectable side — against a matched arm at $0.809$, far above it, and the
data's own R2 value $0.392$, just above. `exp18` §13.4's "the encoder restores
the symmetry" is therefore quantitatively predicted here from an experiment with
no fitting in it at all: the encoder is **centring the block**, and it only has
to move the concentration below $\approx0.3$ to become invisible.

### 14.5 What this licenses, and what it does not

**Licensed.** (i) Lemma D's additivity can be replaced by *non-compactness of
the coupling group modulo $\mathrm{Stab}(p_B)$*, which is strictly weaker and
covers scalings and shears. (ii) The residual ambiguity after a Route B kill is
at most a compact group conjugate to a subgroup of $O(d_B)$ — finite-dimensional,
and measurable without fitting. (iii) The design rule for the fitted model is
**concentration $\gtrsim 0.3$ on the pinned block**, not "break the rotational
symmetry of $p_B$", and it is a mean-direction condition.

**Not licensed.** (a) This is stage 1: it measures **(D4) and its detector**, not
whether a given $T$ arises from a genuine modular conjugacy — §13.2 already shows
those differ, the rotational escape being excluded at a fixed point by Step 1 and
real on a cycle. (b) Proposition S assumes $p_B$ has finite nonsingular second
moment; heavy-tailed laws are not covered, and the stabiliser argument would need
Cartan's theorem instead. (c) $d_B = 2$ throughout; the compactness argument is
dimension-free but nothing here checks $d_B \ge 3$, where $O(d_B)$ has a much
richer subgroup lattice. (d) Nothing here says the concentration condition
survives learning — that is stage 2, and `exp18` is the reason to expect it needs
to be *imposed* rather than inherited.

---

## 15. Route D — independence of the module marginals (`exp21`)

**New (2026-08-25).** §14 characterised what Route B can and cannot reach: it
kills every coupling valued in a non-compact group, and the residue is compact —
for the lattice regrouping, a rotation of the invariant block. Stage 2 was meant
to *impose* the missing precondition as an objective term. Building it produced
two facts that redirect the whole question, and a third that is better than what
was being aimed at.

### 15.1 Proposition C — the escape is strictly resultant-decreasing

Write $R(w) := \lVert\mathbb{E}[w/\lVert w\rVert]\rVert$ for the directional
resultant of a block, computed after whitening by its **uncentred** second
moment (see §15.3 for why uncentred).

> **Proposition C.** Let $h_B = T(z_A)\,z_B$ with $T$ valued in $O(d_B)$ and
> independent of $z_B$. Then
> $$\mathbb{E}\!\left[\tfrac{h_B}{\lVert h_B\rVert}\right]
>   = \mathbb{E}[T]\;\mathbb{E}\!\left[\tfrac{z_B}{\lVert z_B\rVert}\right],
>   \qquad\text{so}\qquad R(h_B)\le R(z_B),$$
> with equality iff $T\,m$ is a.s. constant, where $m=\mathbb{E}[z_B/\lVert z_B\rVert]$.
> For $d_B=2$ and $m\neq0$ that forces $T$ a.s. constant.

*Proof.* $\lVert Tz\rVert=\lVert z\rVert$ for $T$ orthogonal, and independence
factorises the expectation. $\mathbb{E}[T]$ is an average of orthogonal matrices,
so $\lVert\mathbb{E}[T]\rVert_{\mathrm{op}}\le1$ by convexity; equality on $m$
needs $Tm$ a.s. equal by strict convexity of the ball. In $SO(2)$ a rotation
fixing a nonzero vector is the identity. $\square$

In $d_B=2$ this is the circular convolution theorem: the phase of $h_B$ is
$\phi_B+\theta$, so the resultants **multiply**, $R_{h_B}=R_\theta R_{z_B}$.

**Why it matters.** The compact residue Proposition S leaves behind does not
merely fail to be invisible — it moves a scalar functional *monotonically*. So
the true representative is that functional's maximiser, and **no behavioural
variable is involved**.

### 15.2 But the objective form is unavailable

The obvious move — add $R$ to the loss and maximise — does not survive contact.
Optimising the whitened resultant over an unconstrained 2-D point cloud reaches
$0.9996$ by giving the radius a heavy tail: whitening fixes the second moment,
and a heavy-tailed radius then lets the *direction* concentrate almost
arbitrarily (spread $524\times$; §15.8). So a term rewarding concentration is
payable without changing the representation — CLAUDE.md §3.12, §3.15 and §13.4 a fourth time, caught here
before it was run rather than after.

What survives is a **comparison across fits of the same data**, with the radial
tail reported beside it so a degenerate fit is visible.

### 15.3 The statistic, and the near-miss inside it

`exp18_mechanism` used the raw circular concentration $\lvert\mathbb{E}e^{i\phi}\rvert$.
That is invariant under rescaling and rotating the block but **not** under
shearing it, and §7 grants the whole of $GL(d_b)$ inside a module — so the raw
statistic is a gauge quantity in the §3.12 sense.

Whitening by the **uncentred** second moment $S=\mathbb{E}[ww^\top]$ fixes that
and is exactly $GL(d_b)$-invariant: under $w\mapsto Aw$ the Cholesky factor goes
to $ALQ$ with $Q$ orthogonal, so the whitened points rotate and $R$ does not move.

**Uncentred is load-bearing.** Whitening by the *covariance* — centring first, as
every other detector in this repo does — makes a strongly concentrated block
score $0.0177$ and a phase-randomised one $0.0196$, i.e. indistinguishable:
centring removes the mean direction, and by §14.4 the mean direction is the
entire signal. The obvious construction is exactly blind to the quantity it
exists to hold. What it does *not* quotient out is a translation, and that is
correct rather than a gap — $h_B=T(z_A)z_B$ acts linearly, which distinguishes
the origin.

### 15.4 Proposition R′ — independence is the criterion; the resultant proxies it

The reason the true representative maximises $R$ is not really about
concentration:

> In the true representation the modules' phases are **independent**. Under a
> lattice image $\phi_B\mapsto\phi_B+\phi_A$ the donor's phase appears in *both*
> coordinates, so the modules become **dependent**.

> **Proposition R′.** The true representative is the unique element of the
> $GL(K,\mathbb{Z})$ orbit, up to signed permutation, in which the modules'
> phase laws are statistically independent — provided **no** module has a
> uniform phase law.

**The proviso is exactly right and the obvious version of it is wrong**, which is
worth spelling out because I got it backwards first. The relevant fact is

$$\phi_i + \phi_j \perp \phi_j \iff \phi_i \text{ is uniform},$$

*not* $\phi_j$. Adding an independent uniform angle makes the **marginal**
uniform, but independence of the pair needs the angle being *added* to be the
uniform one: conditioning on $\phi_j$ shifts $\phi_i$'s law, and a shifted
non-uniform law still depends on the shift.

So a single uniform module already opens an escape, and it is a specific one:
elements that donate the **uniform** module's phase into another coordinate,
leaving the uniform module alone. `exp21` part 7 shows both halves at
$\kappa=(4.0,0.0)$ — module 2 uniform. The element $(1,0;1,1)$, which pours
$\phi_2$ into module 1's slot, **ties** with the identity; the element
$(1,1;0,1)$, which pours the *concentrated* $\phi_1$ into module 2's, is rejected
at $0.86$. With **both** uniform the orbit is entirely undetermined.

This reframes the criterion as the standard nonlinear-ICA assumption rather than
a new aesthetic preference, and it is exactly the distinction §1.3 drew: iVAE
needs conditionally independent **scalar** components, which excludes rotation
outright, whereas what is wanted here is independence of **modules**, which a
2-D rotation satisfies perfectly well. **Modularity of the dynamics plus
independence of the initial conditions is what pins the representative.**

### 15.5 Route D, and the two escapes that bound it

Applied beyond the lattice, independence does considerably more, and the pattern
is the right one — it rejects the two escapes that block Theorems B and F, and
is blind to the one a different hypothesis already covers:

| counterexample | what it blocks | independence |
|---|---|---|
| §4.3 triangular conjugacy | makes block-diagonality **false** under (B1)–(B4) | **rejects** |
| §7 lattice regrouping | makes rotation numbers non-identifiable | **rejects** |
| §3.1 regrouping | (B2) indecomposability's job | correctly blind |

§4.3 is the important one: it is polynomial, hence $C^\infty$, so no regularity
hypothesis removes it — and it makes the modules dependent.

**Two escapes bound the route, and they are different in kind.**

1. **Hyvärinen–Pajunen.** Independence alone is insufficient for nonlinear ICA,
   and the counterexample lives inside our own setting: with both phase laws
   *uniform*, $(\phi_A,\phi_B)\mapsto(\phi_A,\phi_A+\phi_B)$ preserves
   independence exactly. So Route D is a **conjecture** — what is claimed is that
   independence *plus being a modular conjugacy* is rigid. `TODO(gap)`
2. **The Gaussian degeneracy, and it costs nothing.** iid Gaussian modules are
   rotation-invariant, so a rotation preserves independence and Route D is blind.
   But that rotation is a *modular conjugacy* only when the two modules have the
   same map — with distinct eigenvalues $RFR^{-1}$ acquires off-diagonal mass. So
   **(A2), which Theorem A already assumes, excludes it**, and Route D composes
   with (A2) without a Gaussian hole.

### 15.6 Why this is the route to prefer

Not on strength — on the *kind* of hypothesis, which is §1.0's criterion.

| | (B4$'$) non-resonance | (F3) ordered separation | Route D independence |
|---|---|---|---|
| about | the spectra | the spectra | the initial-condition law |
| in the oscillatory regime | **identically false** (Prop. N) | fails (both hulls reach $0$) | available |
| checkable on data | yes | yes, and it mostly fails | yes, model-free (dCor) |
| can the experimenter arrange it? | no | no | **partly** — it is a fact about how trials are seeded |

Route D constrains the latent *distribution* rather than the map, so it is
orthogonal to A, B and C rather than a weakening of any of them.

### 15.7 Where Route D sits in the literature

The static half of Route D is **independent subspace analysis** — independence of
*blocks* rather than of scalar components, which is exactly what a 2-D
oscillatory module needs and exactly what iVAE-style componentwise independence
forbids (§1.3). The shape of the positioning is therefore §1.0's preferred one:

* independence of blocks under **linear** mixing is classical, and is Theorem A's
  territory anyway;
* under **nonlinear** mixing, independence alone is *not* identifiable
  (Hyvärinen–Pajunen), which §15.5 reproduces inside our own setting;
* the contribution is that **modular dynamics** supplies what independence lacks.

> **Provenance, unverified.** The ISA attribution above is from recollection
> (Hyvärinen & Hoyer, independent subspace analysis, ~2000) and has **not** been
> checked against primary text. §4.3 records that this repo has already
> mis-attributed one arXiv ID from memory; treat the citation as a lead for
> `literature.md`, not as established, and read the paper before it appears in a
> write-up. What does **not** depend on the citation is anything measured in
> `exp21` — those are properties of constructed distributions.

### 15.8 What `exp21` measures — 25/26

Parts 1–3 and 5–8 are analytic (exact distributions, no fitting), so any failure
there is the criterion's or the instrument's. Part 4 refits `exp18`'s own arms.

**Proposition C is exact** (part 1). Over a grid of concentrations the product
law holds to $1.8\times10^{-3}$, and the equality case lands: at $\kappa_\theta=0$
(constant $\theta$) $R_{h_B}=0.6982$ against $R_{z_B}=0.6982$.

**The objective form is unavailable** (part 3). Free maximisation reaches
$0.9996$ with a radial spread of $524\times$ — a configuration nothing in the
data resembles, reached because whitening pins the second moment while leaving
the radius free.

**The resultant separates R1 from R2 on `exp18`'s own system** (part 2), and
fails to at $\kappa_B=0$ exactly as it must:

| $\kappa_B$ | $0$ | $0.5$ | $1$ | $2$ | $4$ |
|---|---|---|---|---|---|
| $R_1/R_2$ | $1.07$ | $1.72$ | $2.09$ | $2.10$ | $1.97$ |

**Sensitivity: the resultant dominates where behaviour is blind** (part 5). By
Proposition C the *relative* deficit is $1-R_\theta$, a property of the coupling
alone — so it does not degrade as the block flattens, while the behavioural
detector's signal-to-floor scales with $R_{z_B}$:

| $R_{z_B}$ | $0.002$ | $0.126$ | $0.239$ | $0.434$ | $0.647$ | $0.748$ |
|---|---|---|---|---|---|---|
| behaviour ($\times$ floor) | $0.66$ | $4.2$ | $4.4$ | $15.4$ | $40.9$ | $58.3$ |
| relative deficit | — | $15.6\%$ | $16.2\%$ | $15.2\%$ | $13.3\%$ | $10.6\%$ |

At $R_{z_B}=0.126$ behaviour is **below** its detection threshold while the
deficit is undiminished. **Both die at $R_{z_B}=0$**, so the resultant is not an
independent escape-closer — it is a uniformly more sensitive instrument for the
same condition.

**Independence is sharper by one to two orders of magnitude** (part 7), against a
measured floor of $0.0023$:

| $\kappa$ | identity | best non-permutation | margin |
|---|---|---|---|
| $(4,4)$ | $0.0017$ | $0.1681$ | $98\times$ |
| $(4,1)$ | $0.0027$ | $0.0821$ | $31\times$ |
| $(1,0.5)$ | $0.0035$ | $0.0129$ | $3.7\times$ |
| $(4,0)$ | $0.0035$ | $0.0036$ | $1.0\times$ |
| $(0,0)$ | $0.0042$ | $0.0041$ | $1.0\times$ |

The identity sits *at the floor*, i.e. it is genuinely independent rather than
merely least dependent — and the last two rows are §15.4's proviso, measured.

**Route D against the three escapes** (part 8), dCor baseline $0.065$: §4.3
triangular $0.531$ ($8.2\times$), §7 lattice $0.623$ ($9.6\times$), §3.1
regrouping $0.077$ ($1.2\times$, correctly blind), Hyvärinen–Pajunen uniform-phase
lattice $0.054$ ($0.8\times$, the honest hole). The instrument also sees §3.10's
blind spot: on the *even* coupling $h_B=z_B+cz_A^2$, dCor rises monotonically
$0.106\to0.366$ while the linear probe stays at $0.007$–$0.057$.

### 15.9 Under learning — and the one check that fails

Part 4 refits `exp18`'s adversarial/matched arms, scoring both criteria. Data
targets: $R_1=0.7449$, $R_2=0.3912$.

| restart | matched $R$ | adversarial $R$ | matched dCor | adversarial dCor |
|---|---|---|---|---|
| 0 | $0.7154$ | $0.2833$ | $0.273$ | $0.469$ |
| 1 | $0.7130$ | $0.2828$ | $0.278$ | $0.461$ |
| 2 | $0.7116$ | $0.2055$ | $0.278$ | $\mathbf{0.272}$ |

**The resultant is correct in 3/3.** Radial tails are $1.5$–$1.7$ in every arm,
so no fit reached its score by part 3's degeneracy. This also confirms `exp18`'s
conclusion survives the move to a gauge-invariant statistic — its raw numbers
were $0.809$ vs $0.270$, the whitened ones $0.713$ vs $0.283$.

**The independence criterion is 2/3, and the check is committed failing.** On
restart 2 the adversarial fit's modules are as independent as the matched fit's
($0.272$ against $0.278$). That is not noise, and it is not a defect in the
criterion — **it is the Hyvärinen–Pajunen escape of §15.5, found by gradient
descent.** Restart 2 is also the arm with the lowest resultant ($0.2055$, against
$0.283$ for the other two) and the lowest $u$-dependence ($0.189$ against
$0.43$): the encoder flattened the recipient's phase furthest toward uniform, and
at a uniform phase the lattice map *is* independence-preserving. The evasion and
the theoretical hole are the same object.

**So the two criteria are complementary, and neither alone suffices.**
Independence is far sharper when it applies; the resultant still separates
restart 2 by $3.5\times$ precisely because flattening the phase is what it
measures. Report both.

### 15.10 What this licenses, and what it does not

**Licensed.** (i) Proposition C, proved and exact: the compact residue of
Proposition S moves a scalar functional strictly, so it is canonicalisable
without any behavioural variable. (ii) Proposition R′ as a *criterion* — the
independent representative is unique up to signed permutation when no module's
phase law is uniform. (iii) Route D rejects the two counterexamples that block
Theorems B and F while leaving (B2)'s alone. (iv) Both statistics are
$GL(d_b)$-invariant and computable on a fit with no ground truth.

**Not licensed.** (a) **Route D is a conjecture, not a theorem** — independence
alone is insufficient for nonlinear ICA, §15.5 exhibits the counterexample inside
this setting, and §15.9 shows an optimiser finding it. What needs proving is that
independence *plus modular conjugacy* is rigid. `TODO(gap)` (b) Nothing here is a
proof that the *fitted* model attains the independent representative; part 4 is
three restarts of one system. (c) $d_B=2$ throughout, and Proposition C's
equality case is only sharp there — in higher dimensions $T$ is pinned only on
the mean direction. (d) The whole approach needs non-uniform phase laws, which is
a property of how trials are seeded, and on a continuous recording with no
alignment it may simply be absent.

### 15.11 Toward a proof — the escape group, and why B and D have the same hole

Task 47 asks whether independence *plus modular conjugacy* is rigid. The
following reduces the question and, in doing so, unifies it with §14.

**The escape group, exactly.** Theorem F gives $h$ block lower-triangular, so
write $h=(h_1(z_1,z_2),\,h_2(z_2))$ and put $g_s:=h_1(\cdot,s)$. Because $h_2$ is
a bijection in $z_2$, independence of the images is equivalent to
$h_1(z_1,z_2)\perp z_2$. With $z_1\perp z_2$ that says: **$g_s$ pushes $p_1$ to
the same law $\nu$ for every $s$**. Hence for any $s,s'$,

$$g_{s'}^{-1}\circ g_s \ \text{ preserves } p_1 .$$

So the independence-preserving triangular maps are exactly $g_s = T_0\circ u_s$
with $u_s$ in the $p_1$-preserving group — which for diffeomorphisms is
infinite-dimensional, so independence *alone* cannot be enough. That is §15.5's
Hyvärinen–Pajunen limit, derived rather than quoted.

**What the conjugacy adds.** The $1$-component of $h\circ F=\tilde F\circ h$ reads

$$g_{f_2(s)}\circ f_1 \;=\; \tilde f_1 \circ g_s ,$$

so the family $\{g_s\}$ is not free: it is determined along each $f_2$-orbit by
its value at one point, and over a **fixed or periodic** point $s^*$ of $f_2$ the
map $g_{s^*}$ is a genuine conjugacy $f_1\to\tilde f_1$. Combining the two
displays, the transports $u_s$ must be $p_1$-preserving **and** compatible with
$f_1$'s own dynamics. The natural conjecture is therefore that the escape is
valued in

$$\operatorname{Stab}(p_1)\ \cap\ C(f_1),$$

the measure-preserving maps that also commute with the module's own map.

**Both counterexamples are of exactly this form**, which is the evidence for the
shape:

| escape | $\operatorname{Stab}(p_1)$ | in $C(f_1)$? | regularity |
|---|---|---|---|
| $h_1=z_1\operatorname{sgn}(z_2)$, $p_1$ symmetric | $\{\pm1\}=O(1)$ | yes ($f_1$ odd) | **discontinuous at $z_2=0$** |
| $h_1=R(\theta(z_2))z_1$, $p_1$ rotationally symmetric | $SO(2)$ | yes ($f_1$ a rotation-scaling) | smooth; a real conjugacy on a cycle (§13.2) |

Measured: the first is an exact conjugacy (defect $0$) with dCor $0.0248$ against
a baseline of $0.0247$ while changing $50.3\%$ of samples; the second reads
$0.0465$ against $0.0401$, and **concentrating $p_1$'s phase sends it to
$0.8512$**. The $d_1=1$ case is discrete ($\pm1$), so no *continuous* family
exists there — §4.5's remark that the escape needs $d_B\ge2$, recovered here from
the group rather than from the construction.

> **This is the unification with §14.** Proposition S says Route B's irreducible
> residue is a coset of $\operatorname{Stab}_G(p_B)$. The display above says Route
> D's hole is $\operatorname{Stab}(p_1)\cap C(f_1)$. **They are the same object**,
> so the two routes do not fail independently — and neither is a fallback for the
> other. What closes *both* is a single checkable condition on the recording:
> **every module's law has trivial stabiliser**, i.e. no module is distributionally
> invariant under a nontrivial symmetry.

> **Sharpened conjecture (task 47).** Let $h$ be a modular conjugacy with both
> representations having independent module marginals. If
> $\operatorname{Stab}(p_i)$ is trivial for every $i$, then $h$ is block-diagonal
> up to permutation. `TODO(gap)`
>
> What is proved above: the reduction to $g_{s'}^{-1}g_s\in\operatorname{Stab}(p_1)$,
> and the orbit constraint $g_{f_2(s)}\circ f_1=\tilde f_1\circ g_s$. What is not:
> that the two together force $u_s$ constant. The missing step is a rigidity
> statement for a $p_1$-preserving family compatible with $f_1$ — plausibly the
> same $C(f_1)$-centraliser argument that makes §4.5's Step 1 work, but it has not
> been written.

### 15.12 Theorem D — Route D closes at a contracting fixed point

§15.11's reduction closes, and the resulting hypothesis is *not* a spectral
separation between the modules.

> **Theorem D.** Let $h$ be a modular conjugacy, block lower-triangular
> (Theorem F, or assumed), $h=(h_1(z_1,z_2),h_2(z_2))$, and write
> $g_s:=h_1(\cdot,s)$. Assume
>
> * **(D-a) Independence.** $z_1\perp z_2$, and the images $h_1\perp h_2$.
> * **(D-b) A contracting base.** $f_2$ has an attracting fixed point $s_0$ in
>   $\Omega_2$ with rate $\rho_2<1$, and $s\mapsto g_s$ is Lipschitz into $C^1$.
> * **(D-c) Conformality budget.** $f_1$ (or its linear part) has Lyapunov spread
>   $\sigma_1=\lambda_{\max}(f_1)-\lambda_{\min}(f_1)$ with
>   $$\log\rho_2+\sigma_1<0 .$$
>
> Then $h_1$ does not depend on $z_2$, so $h$ is **block-diagonal**.

*Proof.* $h_2$ is a bijection in $z_2$, so (D-a) is equivalent to
$h_1(z_1,z_2)\perp z_2$, i.e. $(g_s)_*p_1=\nu$ for a single $\nu$ and every $s$.
Hence $u_s:=g_{s_0}^{-1}\circ g_s$ preserves $p_1$, with $u_{s_0}=\mathrm{id}$.

The $1$-component of $h\circ F=\tilde F\circ h$ is
$g_{f_2(s)}\circ f_1=\tilde f_1\circ g_s$. Evaluating at the fixed point gives
$\tilde f_1=g_{s_0}f_1g_{s_0}^{-1}$; substituting and cancelling $g_{s_0}$ on the
left yields the **cocycle relation**

$$u_{f_2(s)}\circ f_1=f_1\circ u_s,\qquad\text{i.e.}\qquad
u_s=f_1^{-1}\circ u_{f_2(s)}\circ f_1 ,$$

and iterating, $u_s=f_1^{-n}\circ u_{f_2^{n}(s)}\circ f_1^{n}$ for every $n$.

*The origin is fixed, and this step is not free.* The Lipschitz bound below needs
$u_s(0)=0$, which nothing so far supplies. It comes from using (D-a) at **every
time**, which is legitimate because $F$ is modular: if the module marginals are
independent at $t=0$ they are independent at every $t$, so $(g_s)_*p_1^{(t)}$ is
$s$-free for all $t$ and hence $u_s$ preserves $p_1^{(t)}:=(f_1^{t})_*p_1$ for all
$t$. Since $f_1$ contracts, $p_1^{(t)}\Rightarrow\delta_0$, and a continuous $u_s$
preserving the limit gives $\delta_{u_s(0)}=\delta_0$, i.e. $u_s(0)=0$. *(This is
the one place the whole time-family of independence constraints is used rather
than a single one — and it is why independence at a single snapshot would not
suffice.)*

Write $u_t=\mathrm{id}+v_t$, so $v_t(0)=0$. By (D-b), $f_2^n(s)\to s_0$ at rate
$\rho_2$ and $u_{s_0}=\mathrm{id}$, so $\operatorname{Lip}(v_{f_2^n(s)})=O(\rho_2^{\,n})$. And
$\lVert f_1^{-n}v(f_1^{n}x)\rVert\le\operatorname{cond}(f_1^{n})\operatorname{Lip}(v)\lVert x\rVert$
with $\operatorname{cond}(f_1^n)=e^{n\sigma_1+o(n)}$. Hence
$\lVert u_s-\mathrm{id}\rVert\le C\,\rho_2^{\,n}e^{n\sigma_1+o(n)}\to0$ by (D-c).
So $u_s=\mathrm{id}$ for every $s$, i.e. $g_s=g_{s_0}$. $\square$

**(D-c) is not Lemma C's gap, and the difference is the point.** Lemma C needs a
separation *between* the modules, $\lambda_{\max}(f_2)<\lambda_{\min}(f_1)$, which
§3.7 shows can never hold in both directions and which (F3) then fails on real
data. (D-c) compares $f_2$'s rate to $f_1$'s **own spread** — an *internal*
property of one module. In particular:

> For a **conformal** $f_1$ — a rotation-scaling, i.e. any `TwistBlock`, whose
> spectrum $\{\log s,\log s\}$ is repeated — $\sigma_1=0$ and **(D-c) is free**:
> any contracting $f_2$ will do.

So the oscillatory module that (B4$'$) excludes identically (Prop. N) and that
(F3) cannot order is exactly the case where Theorem D costs nothing.

Verified numerically at the level of the mechanism: iterating
$A_s=f_1^{-n}A_{f_2^n(s)}f_1^{n}$ with $A_{f_2^n s}\to I$ at rate $\rho_2$, the
residual $\sup_{n>40}\lVert A_s-I\rVert$ is $2.2\times10^{-3}$ for conformal $f_1$
at $\rho_2=0.9$, and for a non-conformal $f_1$ with $\sigma_1=1.386$ it is
$4.0\times10^{-5}$ when $\rho_2e^{\sigma_1}=0.8<1$ but **$1.2\times10^{32}$** when
$\rho_2e^{\sigma_1}=3.6>1$ — the condition is sharp, not conservative.

**Every known escape fails exactly one hypothesis**, which is the check that the
hypotheses are load-bearing rather than decorative:

| escape | fails |
|---|---|
| §4.3 triangular, $g_s = $ translation by $c\operatorname{sgn}(s)\lvert s\rvert^{p}$ | **(D-a)** — the translation moves the law, so independence is lost (measured $8.2\times$ baseline) |
| $h_1=z_1\operatorname{sgn}(z_2)$ | **(D-b)** — $s\mapsto g_s$ jumps at $s=0$ |
| $h_1=R(\theta(z_2))z_1$, $p_1$ symmetric | **(D-b)** — it lives on a *cycle*; at a fixed point $\theta\circ f_2-\theta$ constant forces $\theta$ constant (§4.5 Step 1) |

**What remains open is the non-fixed-point case, and it is exactly where §15.11's
stabiliser condition is needed.** On a limit cycle there is no $s_0$ to contract
to, the cocycle iteration has nothing to converge against, and the rotational
escape is a genuine conjugacy (§13.2, defect $7.5\times10^{-16}$). So the honest
division is: **at a fixed point, Theorem D — no stabiliser hypothesis required;
off it, the conjecture of §15.11 with trivial $\operatorname{Stab}(p_i)$.**
`TODO(gap)`

**Theorem D and Theorem F are complementary, and the cleanest case shows it.**
Take $f_1$ conformal at $0.8$ and $f_2$ slower at $0.9$. Then (F3) **holds**, but
in the order $[1,0]$ — $f_2$ leads — so Theorem F gives $h_2=h_2(z_2)$ and
explicitly *permits* $h_1$ to depend on $z_2$. Killing that surviving block by
Lemma C would need $\lambda_{\max}(f_2)<\lambda_{\min}(f_1)$, i.e.
$\log0.9<\log0.8$, which is false — and by §3.7 the two orientations can never
both hold anyway. Theorem D's condition is $\log0.9+0<0$, which does hold. **So
Theorem D removes exactly the cross-block Theorem F is forced to leave in**, and
it does so with a hypothesis §3.7 does not obstruct. Pinned in
`tests/test_spectra_and_cocycle.py::test_theorem_D_kills_the_cross_block_theorem_F_is_forced_to_permit`.

### 15.13 Theorem D′ — the limit-cycle case, and where the stabiliser enters

Theorem D used the fixed point twice: to get $\tilde f_1=g_{s_0}f_1g_{s_0}^{-1}$,
and to make $u_{f_2^n(s)}\to\mathrm{id}$. On an attracting **limit cycle** there is
no $s_0$ to contract to, and §13.2's rotational escape is a genuine conjugacy —
so something must be added. It turns out to be exactly Proposition S's object.

> **Theorem D′.** As Theorem D, but with $f_2$ attracting a limit cycle with
> transverse rate $\rho_2^{\perp}$. Assume
>
> * **(D-b′)** $\log\rho_2^{\perp}+\sigma_1<0$;
> * **(D-d)** $\operatorname{Stab}_{GL(d_1)}(p_1)$ is **trivial** — the whitened
>   directional law of module 1 has no nontrivial orthogonal symmetry.
>
> Then $h$ is block-diagonal.

*Proof sketch.* The transverse contraction plays the role the fixed point played:
$f_2^{n}(s)$ approaches the cycle at rate $\rho_2^{\perp}$, so by the same
distortion bound $u_s$ is determined by its restriction to the cycle, and the
nonlinear part of $u$ dies under (D-b′). Once $u_s$ is linear, independence
applies **pointwise in $s$**: §15.11 gives $u_{s\#}p_1=p_1$ for each $s$
*separately*, so $u_s\in\operatorname{Stab}_{GL}(p_1)$, which is compact by
Proposition S, and (D-d) makes it $\{I\}$. Hence $g_s=g_{s_0}$ for all $s$, i.e.
$h_1$ does not depend on $z_2$. $\square$ `TODO(gap)` — the linearisation step
only; see the caveat below.

> **Simplification, found on re-reading (2026-08-24), and it removes one of the
> two flagged gaps.** An earlier version of this sketch took linear parts of
> $u_s=f_1^{-n}u_{f_2^{n}(s)}f_1^{n}$, argued that for conformal $f_1$ the
> conjugation is trivial so $A$ is constant along $f_2$-orbits, and then used
> **density** of those orbits to make $A$ constant before applying the stabiliser
> once. That route carried a `TODO(gap)` for $\omega_2$ rational, where the orbits
> are finite and the density step is unavailable.
>
> **The density step was never needed.** Independence is a statement about every
> $g_s$ individually — §15.11 derives $u_{s\#}p_1=p_1$ for each $s$ — so the
> stabiliser applies pointwise and there is nothing to propagate along an orbit.
> Consequently **(D-d) alone finishes the linear case, at every rotation number**,
> rational or irrational, and no periodic-orbit variant is required.
>
> Note the ordering constraint below is what makes this available at all: the
> pointwise argument needs $u_s$ *linear* first, because a nonlinear $u_s$ can
> preserve $p_1$ without its linear part doing so. So the two observations are
> complementary — one removes a gap, the other says why the remaining gap is the
> only one.

**Why (D-d) is the right hypothesis and not an extra assumption.** It is
Proposition S's stabiliser, so §14 and §15 close on the same condition. It is
exactly what the two known smooth escapes need in order to exist: the rotational
escape requires $p_1$ rotationally symmetric, and §15.9's restart-2 evasion is an
optimiser *manufacturing* that symmetry by flattening the phase. And it is
checkable with no ground truth — circular concentration, or the harmonic content
of the whitened directional law.

**Measured, and it is the same number both ways.** With $p_1$ rotationally
symmetric the rotational escape reads dCor $0.0465$ against a baseline of
$0.0401$ (invisible); concentrating $p_1$'s phase sends it to $\mathbf{0.8512}$.
The hypothesis is not decorative — it is the difference between the escape
existing and not.

> **The whole picture, in one line.** At a **fixed point** Route D needs no
> symmetry hypothesis (Theorem D); on a **cycle** it needs exactly the trivial
> stabiliser that Proposition S already identified as Route B's irreducible
> residue (Theorem D′). So the two routes that looked independent fail and close
> on the *same* condition, and that condition is a property of the recording —
> whether trials are phase-aligned — rather than of the spectra.

> **Caveat on §15.13's sketch, added on re-reading it.** Theorem D′ is billed as
> a sketch and it should be read as a weaker claim than Theorem D, which is
> written out in full. Two steps needed care that the sketch glossed; **the
> simplification above closes the second one and leaves the first as the sole
> obligation**, so read point 2 as the reason point 1 is load-bearing rather than
> as an independent gap:
>
> 1. **Killing the nonlinear part.** At a fixed point this works because
>    $u_{f_2^n(s)}\to\mathrm{id}$, so the residual is $O(\rho_2^{\,n})$. On a cycle
>    $u_{f_2^n(s)}$ converges to $u$ *restricted to the cycle*, which need not be
>    the identity, so the argument has to be rerun with the transverse coordinate
>    alone and it is not obvious it survives unchanged.
> 2. **Ordering.** The sketch kills the nonlinear part first and then applies the
>    stabiliser. The reverse order does not work: $u_s$ preserving $p_1$ does
>    **not** imply its *linear part* does, so "independence puts $A$ in
>    $\operatorname{Stab}_{GL}(p_1)$" is only available once $u_s$ is known to be
>    linear.
>
> Also, the $t\to\infty$ trick that pins $u_s(0)=0$ in Theorem D is unavailable
> when module 1 is itself oscillatory, since $p_1^{(t)}$ then does not converge
> to $\delta_0$. **What is solid is the fixed-point theorem and the empirical
> claim** — the escape exists iff $\operatorname{Stab}(p_1)$ is nontrivial, measured
> at $0.0465$ versus $0.8512$. The cycle theorem is a conjecture with a plausible
> route, not a result. `TODO(gap)`
>
> **Sole remaining obligation, stated precisely so it can be attacked or
> refuted:** *on an attracting limit cycle, under (D-b′), is every $u_s$ linear?*
> Everything else in Theorem D′ now follows — the stabiliser step is pointwise
> and needs no density, no conformality of $f_1$, and no condition on $\omega_2$.
> If the answer is no, the counterexample is a nonlinear $p_1$-preserving $u_s$
> compatible with the cocycle, which would be a new escape and worth more than
> the theorem.

### 15.13a The linearisation step closes when $f_1$ has a fixed point — and the condition is again a *spread*

The obligation §15.13 is left with is answerable by the same device as
Theorem D, applied to the Taylor jet instead of to the map. Expand $u_s$ at
module 1's fixed point (take it to be $0$; §15.12's $t\to\infty$ argument gives
$u_s(0)=0$, and it is available here because $f_1$ contracts, so
$p_1^{(t)}\to\delta_0$). Write $A_1=Df_1(0)$. In
$u_s=f_1^{-n}u_{f_2^{n}(s)}f_1^{n}$ a homogeneous degree-$p$ term $Q$ is carried
to $A_1^{-n}Q(A_1^{n}\,\cdot\,)$, whose norm obeys

$$\bigl\lVert A_1^{-n}Q(A_1^{n}\,\cdot\,)\bigr\rVert \;\le\; \frac{(\rho_1^{\max})^{np}}{(\rho_1^{\min})^{n}}\,\lVert Q\rVert .$$

The exponent is $n\bigl[p\log\rho_1^{\max}-\log\rho_1^{\min}\bigr]$, so the
degree-$p$ term decays iff $p\log\rho_1^{\max}<\log\rho_1^{\min}$. Substituting
$\log\rho_1^{\min}=\log\rho_1^{\max}-\sigma_1$ turns this into
$(p-1)\log\rho_1^{\max}+\sigma_1<0$, and since $\log\rho_1^{\max}<0$ increasing
$p$ only helps — the binding case is $p=2$:

> **(D-e)** $\log\rho_1+\sigma_1<0$, with $\rho_1$ module 1's top rate and
> $\sigma_1$ its Lyapunov **spread**.

Given $\lVert u_{f_2^{n}(s)}\rVert_{C^2}$ bounded uniformly in $s$ — the same
regularity Theorem D already assumes — every Taylor coefficient of degree $\ge2$
is $0$, so $u_s$ is **formally linear**, and §15.13's pointwise stabiliser step
finishes it.

**Three things worth separating out.**

1. **(D-e) is a third condition of the same shape, and that is the pattern.**
   Theorem D needs $\log\rho_2+\sigma_1<0$, (D-b′) needs
   $\log\rho_2^{\perp}+\sigma_1<0$, and (D-e) needs $\log\rho_1+\sigma_1<0$.
   All three compare a rate to **module 1's own spread**, never to a separation
   *between* modules — which is exactly why §3.7's two-sided obstruction does not
   apply to any of them. **All three are free for a conformal $f_1$.**
2. **Anisotropy breaks it, not slowness.** $\operatorname{diag}(0.6,0.3)$
   contracts *harder* than a $0.95$ spiral ($\log\rho_1=-0.51$ against $-0.05$)
   and its quadratic jet **grows** $6.2\times$ per ten steps, because its spread
   is $0.69$. Same shape as §3.14's "width, not speed, is what disqualifies".
   Certified sharp: the measured decay ratio matches $e^{10(\log\rho_1+\sigma_1)}$
   to $<10^{-3}$ relative on both sides of the boundary, over five cases
   (`tests/test_spectra_and_cocycle.py`, §"Theorem D′ … linearisation").
3. **Formal $\Rightarrow$ actual needs analyticity, and it is the familiar
   price.** The argument kills the jet, not a flat remainder. For analytic $h$ —
   free with `tanh`-MLP decoders, §3.7 — the identity theorem finishes it. In
   $C^\infty$ a flat $u_s$ survives the jet argument, which is (FLAT-D) again,
   the same residual Route A carries.

**Scope, and it must not be overread: this does *not* close the two-oscillator
case.** (D-e) and the $u_s(0)=0$ step both need $f_1$ to have a contracting fixed
point. In task 23's lattice regrouping $h(z_1,z_2)=(z_1z_2/\lvert z_2\rvert,z_2)$
the *recipient* module is itself a limit cycle, so there is no fixed point to
expand at and this argument is silent. What is closed is the configuration
**one oscillatory module driving one contracting module** — where Theorem F is
either unavailable ((F3) cannot order a cycle against a module whose hull it
contains) or permits the very cross-block Theorem D removes. Two oscillators
remains the open case, as it is for every other route in this document.
