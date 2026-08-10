# Identifiability of modular latent dynamics — corrected statement

Replaces the conjecture and proof sketch of the original draft
(`docs/brief_v0.md`), which are false and defective respectively; see CLAUDE.md
§2–3 and `counterexamples.md`.

Claims that depend on an unproved lemma are flagged `TODO(gap)` inline, per
CLAUDE.md §8. There are three, and §6 lists them together.

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
  of $\tilde F$, with $\tilde f_{\sigma(i)}$ conjugate to $f_i$. `TODO(gap)` — see §6.
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

**What this buys.** Theorem F (the filtration, §6.1) is the route standing on
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
(c) *Nonlinear $f_A$, $\tilde f_B$*: read the above as the statement for the
linear parts; the full nonlinear case needs the normal-form machinery of §5.3.

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

## 6. Open problems, in priority order

1. **Theorem F — filtration identifiability.** *Now the priority.* §4.3 shows the
   symmetric-partition target is false, but the triangular structure of §4.2 **is**
   forced and is already proved. Restate it as the headline theorem: modules are
   identified as an *ordered filtration* by Lyapunov spectrum. No new hypotheses
   are needed, and under §0 the filtration is the more apt object anyway. The
   work is assembling §4.2 into a standalone statement, pinning down what
   "identified" means for a filtration (the flag of invariant foliations, and
   each factor's conjugacy class), and checking it against `exp03`.
   **§4.4 widens its scope before it is written:** the statement holds at
   attracting periodic orbits too, not only at fixed points, and it carries two
   structural riders worth stating in the theorem — an oscillatory module is
   always the top element of the filtration, and there can be at most one of
   them. Both are testable claims about timescale structure in a real
   population, which is what §0 says the object of study is.
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

## 7. Scope of interpretation claims

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

## 8. Position against the literature

`TODO(gap)` — CLAUDE.md §4 step 8, not yet done. The comparison to make:

- **Hyvärinen & Morioka** (time-contrastive, permutation-contrastive) and
  **Hälvä & Hyvärinen** (HMM) obtain identifiability from *temporal dependence
  plus non-stationarity*, recovering latents up to componentwise transformation
  — i.e. they identify **coordinates**, which is more than we claim.
- **Khemakhem et al.** (iVAE) needs an observed auxiliary variable; we have none.
- The distinguishing claim is that modular *dynamics* identifies structure in the
  **autonomous, stationary, no-auxiliary-variable** setting where none of those
  results apply. Under the §0 scope this is not a hedge — it is precisely the
  regime we have restricted to, and `literature.md` §1.2 finds no existing result
  covering it. PCL is the nearest neighbour (stationary, no auxiliary) but needs
  mutually independent *scalar stochastic* sources with a nonvanishing
  cross-derivative of the joint log-density, which is structurally false for
  deterministic dynamics.
- What we claim is also *weaker in kind*: those results recover **coordinates**
  up to componentwise transformation; we claim a **filtration of multidimensional
  factors**. The two are not competing statements about the same object.
- Autonomy was previously flagged as the main disanalogy with LFADS. Under §0 it
  is **out of scope by decision**, not an open threat. If input drive is added
  later, Vahidi et al. 2024 is the entry point (CLAUDE.md §4.3).
