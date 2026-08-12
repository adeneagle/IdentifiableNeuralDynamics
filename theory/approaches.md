# Candidate routes to an identifiable nonlinear model

Three live avenues, plus one cross-cutting gap that blocks two of them. Written
as a decision document: what each one claims, what it costs, what is already
proved, and what single fact would settle it.

Context: the original target — $h = h_1 \oplus \cdots \oplus h_K$ under
(B1)–(B4) — is **false**, see `counterexamples.md` §5. All three routes below are
responses to that. Scope throughout is CLAUDE.md §1.1: autonomous, single-area.

---

## Comparison

| | **A. Diagonality** | **B. Hybrid** | **C. Filtration** |
|---|---|---|---|
| **Claim** | $h$ block-diagonal up to permutation | block-diagonal between behaviour-varying and invariant parts | $h$ triangular; modules ordered by Lyapunov spectrum |
| **Strength** | strongest — the original goal | strong on the A/B split | weakest — a relation, not a uniqueness claim |
| **Extra hypotheses** | $C^\infty$ + multi-index non-resonance + $\mathrm{int}\,\Omega \neq \emptyset$ | an observed behavioural variable $u$ (variance-modulating) + one-sided gap, **varying block dominant** (§B.1) | (F1) bounded derivatives + (F2) uniform exponents + **(F3) ordered separation**. No interior clause, no analyticity. (F3) is strictly stronger than (B4) — see the box in §C |
| **Needs (B2) indecomposability** | **derivable** in Tier 1; jet-algebraic in Tier 2 (§A.2) | **yes**, within the invariant block | **no** to report *a* filtration; **yes** to claim *the finest* one (§6.7) |
| **Needs (B3) matching** | **yes** — open | partly | **no** for the ordering, which (F3) forces; the residue is the coarsening, i.e. (B2) again |
| **Proved?** | no; proof plan only | **both halves proved** in a restricted setting: Lemma C (dynamics) + Lemma D (behaviour, additive $h_B$ + linear modules, §4.5). Partial-iVAE lemma no longer needed — Lemma D discharges it dynamically | **yes, and written up** — Theorem F, `identifiability.md` §6 |
| **Nonlinear?** | yes, near a fixed point | yes | yes, unconditionally |
| **Off the fixed point?** | **no** — normal forms need the Poincaré domain; limit cycles are Siegel | **yes** — the dynamics half is Lemma C, which reaches attractors (§4.4) | **yes** for periodic attractors (Lemma C′, §4.4) |
| **Blocked by the learning gap?** | yes | yes | **no** |

The short version: **A** is the prize, and after the §A.2 assessment it splits —
its easy tier is really *robustness of Theorem A* against decoder ambiguity,
while its hard tier carries the nonlinear content and has one gap left. **B**
introduces a genuinely new mechanism but changes the model. **C** is the only
thing currently standing on proved ground, and as of 2026-08-12 the only one
written as a theorem (§6).

**The new row is the sharpest separation between A and C.** Task 22 established
that Lemma C survives at attracting periodic orbits (`identifiability.md` §4.4),
which is exactly where A's machinery cannot follow — an analytic normal form
needs the Poincaré domain, and a limit cycle's neutral exponent puts it in the
Siegel domain where small divisors return. So the two routes *diverge* off the
fixed point rather than A subsuming C there. Since an autonomous oscillation is
the least exotic structure a neural population can have, this is a substantive
argument for C beyond "it is the one we can prove". It comes with two structural
riders: an oscillatory module is always the **top** of the filtration, and a
filtration can contain **at most one** of them.

---

## A. Block-diagonality under $C^\infty$ + non-resonance

**Claim.** The original target, with (B4) strengthened.

**Why it is not dead.** The §5 counterexample uses
$h_1 = z_1 + c\,\mathrm{sgn}(z_2)|z_2|^{p}$, $p = \log\mu_1/\log\mu_2$. Smoothness
at the origin requires $p \in \mathbb{Z}$, i.e. $\mu_1 = \mu_2^m$ — a *countable*,
hence measure-zero, set of parameters. Measured: the $C^1$ counterexample exists
for 1841/2000 random $(\mu_1,\mu_2)$ pairs, but for non-integer $p$ the
$\lceil p \rceil$-th derivative blows up ($4.3 \to 14.4 \to 48.4$ as
$z \to 0$). So $C^\infty$ + non-resonance excludes this entire family.

**Both extra hypotheses are cheap.** $C^\infty$ is *free* in the model class:
$h = \tilde g^{-1} \circ g$ with both decoders smooth MLPs. Non-resonance is
generic.

**Proof route.** `literature.md` §2.2: per-module Sternberg linearisation →
homological-equation centralizer argument → `linear_case.md` Theorem L. Valid
near an attracting hyperbolic fixed point.

### A.1 The hypothesis has to be stated on multi-indices

Two $C^\infty$ counterexamples (`counterexamples.md` §6, verified here) kill the
obvious phrasings:

- **Pairwise non-resonance is not sufficient.** $\mu_1 = \mu_2\mu_3$ with
  $(0.15, 0.50, 0.30)$: every pairwise log-ratio is $\ge 0.263$ from an integer,
  yet $h = (z_1 + c z_2 z_3, z_2, z_3)$ is an exact polynomial conjugacy.
- **Rotation angles give no protection.** A 2-D scaled rotation plus a 1-D module
  at rate $\rho^2$ admits $h = (x, y, z_2 + c(x^2+y^2))$ for *every* $\theta$ —
  phases cancel in $|w|^2$.

So the hypothesis is: no $\lambda \in \Lambda_i$ equals $\sum_k m_k \nu_k$ with
$|m| \ge 2$, the $\nu_k$ drawn from the full exponent multiset **with
multiplicity**, at least one from a module $\neq i$. Implemented as
`spectra.cross_module_resonances`. Note this bites our own systems: $s =
(0.95, 0.9025)$ is resonant because $0.9025 = 0.95^2$.

### A.2 The route splits into two tiers, and only one is hollow

The hollowness worry was **confirmed for one tier and refuted for the other**.

| | **Tier 1** — non-resonance on the *full* spectrum | **Tier 2** — cross-module resonances only |
|---|---|---|
| Sternberg | linearises $F$ outright | within-module resonant terms survive |
| Content | rigidity of the *finest spectral clustering* against an unknown smooth decoder | genuine nonlinear invariants (resonant coefficients) |
| (B2) | **derivable** — nonlinear indecomposability $\iff$ linear-part indecomposability at jet level, so Theorem L(i) transfers | equivalence *fails*; becomes decidable jet algebra |
| Proof status (analytic decoders) | closes; Poincaré linearises $F$ | **closes — Poincaré–Dulac + identity theorem, nothing unwritten** |
| Proof status ($C^\infty$ fallback) | closes; Sternberg primary-verified | closes *modulo* (FLAT-D)'s $C^k$ bound |
| Non-empty? | trivially | **yes** — witness in §A.2.1, certified in `exp09` |
| Honest billing | robustness of **Theorem A**, not identification of nonlinear dynamics | identification of nonlinear dynamics |

Tier 1 is not vacuous — $h$ is a priori an arbitrary diffeomorphism, and per §3.5
that decoder ambiguity is the model's actual problem, so rigidity against it is a
real result. But it must be **advertised as robustness of Theorem A**, not as a
nonlinear identifiability theorem. Under full non-resonance (B2) additionally
forces every module to be a single-exponent block, since a non-resonant module
with two distinct exponents linearises to something decomposable.

### A.2.1 Tier 2 is non-empty — the witness

The tier split only buys something if Tier 2 can actually be inhabited: some
system must satisfy cross-module non-resonance *while* carrying a live
within-module resonance. Otherwise every admissible system linearises, Route A
collapses onto Tier 1, and "identification of nonlinear dynamics" is hollow.
**It is inhabited.** Certified in `exp09`; built as `systems.tier2_witness()`.

$$f(z_a, z_b) = \big(\mu z_a,\; \mu^2 z_b + c\, z_a^2\big), \qquad 0<\mu<1,\; c \neq 0.$$

The linear part is $\mathrm{diag}(\mu, \mu^2)$, and the monomial $z_a^2$ in the
$z_b$ slot has homological eigenvalue

$$\lambda_b - \lambda_a^2 = \mu^2 - \mu^2 = 0,$$

which vanishes **identically in $\mu$** — a structural resonance, not a tuned
coincidence. So $c$ cannot be conjugated away: the homological equation
$LQ - Q\circ L = P$ is diagonal in the monomial basis and its only zero eigenvalue
sits exactly where $P$ lives. Measured: one zero eigenvalue at degree 2, the
nearest non-resonant one $0.147$ away, obstruction $=c$; setting $c=0$ makes it
removable, so the obstruction tracks $c$ rather than being an artefact
(`normalform.linearization_obstruction`).

Three things make it the right witness.

- **It is exactly what Tier 1 forbids.** Full-spectrum non-resonance excludes
  $\mu^2 = \mu\cdot\mu$, so under Tier 1 this module linearises and there is no
  nonlinear invariant left. Tier 2 is defined by keeping it.
- **Cross-module non-resonance survives**, because the resonance is *within* a
  module — `spectra.cross_module_resonances` already skips relations supported on
  a single module. The witness pairs $\mu=0.7$ with $\nu=0.5$ and passes;
  $\nu = \mu^2$ (exponent collision) and $\nu = \mu^3$ (genuine cross resonance)
  are correctly rejected. So Tier 2's hypotheses are **satisfiable**, not merely
  consistent-looking.
- **The resonance is visible in the dynamics, not just in the algebra.** The
  closed form is
  $$b_n = \mu^{2n} z_b + n\,c\,\mu^{2(n-1)} z_a^2,$$
  so $b_n/\mu^{2n}$ drifts *linearly* in $n$ at rate $c z_a^2/\mu^2$ — a **secular
  term**, which no linear map produces. Measured slope matches the exact value to
  $0$ (err $0.0$); with $c=0$ the same quantity is constant to $0.0$.

What is identified is honest to state: diagonal rescaling sends
$c \mapsto c\beta/\alpha^2$, so $c$ normalises to $1$ and the invariant is the
*binary* one — linearisable or not. That is enough for non-emptiness (the
conjugacy class of $f$ is not that of any linear map), but a continuous modulus
would need two independent resonant coefficients, which this 2-D example does not
have. `TODO(gap)` if a continuous invariant is wanted.

### A.2.2 A consequence for the learning machinery, not for the theory

The same witness exposes a **false negative in (B2) certification**. Its linear
part $\mathrm{diag}(\mu,\mu^2)$ has two distinct real eigenvalues, so
`linear.n_indecomposable_summands` returns $2$ and
`selection.certify_fitted_model` — which linearises the fitted
`ModularTransition` at its fixed point — reports the module as **decomposable**.

But the map does not decompose. A complementary invariant factor would be a curve
$z_b = \varphi(z_a)$ with $\varphi(0)=\varphi'(0)=0$, i.e. $z_b/z_a^2$ constant
along orbits; the closed form gives
$b_n/a_n^2 = z_b/z_a^2 + n c/\mu^2$, **unbounded**, drifting by exactly $c/\mu^2$
per step (max deviation $2.7\times10^{-14}$ over 40 steps). Invariance at
quadratic order requires $k\mu^2 = \mu^2 k + c$, i.e. $c=0$. So the only invariant
line through the origin is $\{z_a = 0\}$ and the map is dynamically
indecomposable.

This matters because it sits **precisely in Tier 2** — the regime carrying Route
A's nonlinear content is the regime where the linearised indecomposability test
is unreliable, and it fails in the dangerous direction: it would accept an
over-split partition. CLAUDE.md step 13 records the learning gap as closed "for
the linearisable regime", and this is the sharp statement of why that
qualification is load-bearing.

**The nonlinear test is now built** (`selection.block_nonlinear_certificate`,
certified in `exp09` parts 5–6). It reads the quadratic jet at the fixed point in
the eigenbasis of the linear part and asks whether a **resonant** monomial couples
two eigendirections: a resonance $\lambda_i = \lambda^m$ with
$\mathrm{supp}(m)\cup\{i\}$ straddling the linear splitting cannot be conjugated
away, so it obstructs the direct-product structure. The delicacy — and the reason
the linear test was not merely lazy — is that a *non-resonant* cross term is
removable and must **not** flag: $f=(\mu z_a,\ 0.6\,z_b + c z_a^2)$ has a literal
$z_a^2$ term but $\lambda_b-\lambda_a^2 = 0.6-0.49\neq0$, so it is decomposable,
and the test correctly clears it. The verdict is coordinate-invariant (it survives
conjugating the witness by a shear). It is exposed as
`certify_fitted_model(nonlinear=True)`, opt-in so the linear-only callers are
unchanged.

**The right verdict is graph connectedness, and getting that right fixed a real
defect.** The criterion (proved at degree 2 in `route_a_assessment.md` §4.1a) is:
the module is indecomposable iff the *resonance-coupling graph* over its
eigendirections is connected. The first version flagged "a resonant coupling
exists", which agrees with connectedness for a two-sub-block module (the witness)
but **over-reports at three**: $(\mu z_0,\ \mu^2 z_1 + c z_0^2,\ \nu z_2)$ with
$\nu$ non-resonant couples only $\{0,1\}$ and splits off $z_2$ — decomposable as
$\{0,1\}\oplus\{2\}$ — yet "any coupling" calls it indecomposable. Over-reporting
is the direction the fit cannot catch (a decomposable module fits its split
exactly), so the partition search would keep a coarser-than-finest partition. Now
corrected via `normalform.resonance_coupling_components`.

Two honest limits remain. (i) The guarantee is stated at **degree 2**: there the
eigenbasis jet coefficient of a resonant monomial *is* the normal-form invariant
(no lower-degree terms feed in), so the test is exact and both directions are
proved; higher-degree resonances would need the coefficient after full
normalisation, and there $G_2$-disconnected becomes only suggestive (a
higher-degree edge can reconnect). (ii) For a **fitted** model the eigenvalues and
jet carry fit error, so an exact resonance $\mu^2=\mu\cdot\mu$ is only approximate
once learned and a near-resonance is genuinely ambiguous from finite data — hence
the exposed `res_tol`/`coeff_tol` knobs. `TODO(gap)` for the higher-degree and the
near-resonance cases; the degree-2 exact case, which is where the Tier 2 witness
lives, is closed.

Tier 2 is where the nonlinear content lives. Its one delicate step is
formal $\Rightarrow$ smooth — upgrading the block-diagonal $\infty$-jet (from the
formal lemma) to a block-diagonal map. **Real-analyticity of $h$ makes this
free** (`identifiability.md` §5.4): a real-analytic map equals its Taylor series,
so a block-diagonal jet is a block-diagonal map (identity theorem), and with
`tanh`-MLP decoders $h$ *is* analytic. In the analytic category the fixed-point
case is Poincaré–Dulac (analytic normal form, no small divisors in the Poincaré
domain) + the identity theorem + the formal lemma — all classical or proved here,
nothing unwritten.

The **$C^\infty$ fallback** (no analyticity assumed) instead needs the
jet-realisation lemma **(FLAT-D)**, which does not close by citing Chen 1963
(saddle-only, p. 697) but is reduced to a classical estimate via the
wave-operator construction — kept in `route_a_assessment.md` §2.4 and `exp07`:

> **(FLAT-D).** Two $C^\infty$ contraction diffeomorphism germs with the same
> $\infty$-jet at the fixed point are $C^\infty$-conjugate by a diffeomorphism
> tangent to the identity to infinite order.

**Its existence half is now LOCATED and source-verified** (`route_a_assessment.md`
§2.4 follow-up). Chaperon, *Géométrie différentielle et singularités de systèmes
dynamiques*, Astérisque **138–139** (1986), Théorème 2(i) p. 107 (open access on
Numdam; source existence confirmed independently) states, for $C^\infty$
contraction germs of $\mathbb{Z}$-actions, that equal $\infty$-jets imply
$C^\infty$ conjugacy, with no semisimplicity and no non-resonance assumed, and its
result is stated to remain valid when the unstable factor is trivial — i.e. the
pure-contraction case Chen excluded is explicitly covered. So the existence of the
smooth conjugacy is settled for exactly our setting.

**The flat-tangency clause is now reduced to a classical estimate** — no longer a
research gap. It is what the Tier-2 telescoping endgame consumes (the conjugacy
tangent to identity to infinite order). Rather than proof-mine Chaperon's scanned
construction, it follows from the wave-operator limit $h=\lim_n\Psi^{-n}\Phi^n$
directly: (i) it is a conjugacy and (ii) has identity $\infty$-jet at $0$ — both
exact and elementary — and (iii) it converges in every $C^k$ by the standard
Sternberg/Nelson distortion bound, whose key inequality $S^N/s^{k+1}<1$ (for
flatness order $N$ large per $k$) is isolated and whose $C^0/C^1$ convergence,
exactness, and flat-tangency are verified numerically (`exp07`,
`tests/test_flat_tangency.py`, across linear/nonlinear $\Phi$ and two flatness
rates). See `route_a_assessment.md` §2.4 "self-contained closure". The only
unwritten step is the textbook $C^k$-for-all-$k$ bookkeeping (Nelson proves the
flow version in full; this is its discrete transposition) — a referee-level
check, not a flagged unknown. Sternberg 1959 III and Belitskii could not be
accessed; Banyaga–de la Llave–Wayne states only a finite-order version.

### A.3 A previously unstated hypothesis

The jets-at-0 step needs $\mathrm{int}\,\Omega \neq \emptyset$. If $\Omega$ is
thin — a single orbit — the conjugacy equation holds only on isolated points and
constrains no cross-derivatives, so the conclusion is **false**. This is
CLAUDE.md §3.6 turning into a hypothesis rather than a caveat, and it must be
added to (B1).

**Status of the sources.** Sternberg's statements are now **verified from the
primary texts** (`route_a_assessment.md` §1.1): 1957 Theorem 2 (contraction,
finite $C^k$, $k > \log s/\log S$) and 1958 Theorem 1 (full multi-index,
$l = \infty \Rightarrow C^\infty$). The Tier 1 / Tier 2 boundary is confirmed and
does not move. The only remaining work on Route A is Tier 2's (FLAT-D).

---

## B. Hybrid: behavioural auxiliary + one-sided gap

**Claim.** Split latents into $z^A$ (conditional law varies with behaviour $u$)
and $z^B$ (invariant). Then $h$ is block-diagonal across the $A/B$ split.

**The mechanism, and why it is not just "route A on a subset".** The
$u$-invariant subspace is **canonical** — a direction lies in it iff its
conditional distribution does not move with $u$ — so any valid $h$ must map it
into itself. Hence

$$M_{BA} = \partial h_B / \partial z^A \equiv 0$$

**with no spectral, regularity, or resonance hypothesis at all.** Measured: leaking
$z^A$ into $h_B$ at $\varepsilon = 0.01$ already makes $h_B$ $u$-dependent
(0.007 → 0.033; 0.55 at $\varepsilon = 0.05$), contradicting the definition of the
subspace it must land in.

This matters because of `counterexamples.md` §3: the cocycle argument supplies
one cross-derivative but *provably not both*. **Behaviour supplies the other.**
Two-sided-from-dynamics is impossible; one-from-behaviour plus one-from-dynamics
is not. Neither ingredient suffices alone.

The asymmetry is real: the $u$-invariant subspace is canonical but its complement
is not (adding invariant directions to varying ones keeps them varying). So
behaviour alone gives *triangular*, and the cocycle closes it to block-diagonal.

### B.1 The B∘C composition, explicit (and the alignment condition)

Numerically demonstrated in `exp10`; the behavioural half is `src/idyn/behavior.py`.

**Model.** Observations $(x_t, u_t)_{t=0}^T$, latent $z_t = (z^A_t, z^B_t)$ with
*autonomous modular* dynamics $z^A_{t+1} = f_A(z^A_t)$, $z^B_{t+1} = f_B(z^B_t)$
and decoder $x_t = g(z_t)$. The behaviour label $u$ conditions the latent law:
$p(z^A_0 \mid u)$ varies with $u$ (assumption of variability), $p(z^B_0 \mid u)$
does not. Two representations are equivalent when the joint law of $(x_{0:T}, u)$
agrees. $h = \tilde g^{-1}\circ g = (h_A, h_B)$, cross-derivatives
$M_{BA} = \partial h_B/\partial z^A$, $M_{AB} = \partial h_A/\partial z^B$.

Two lemmas kill the two cross-derivatives, from *disjoint* hypotheses:

- **(behaviour) $M_{BA}\equiv 0$.** The $u$-invariant subspace is canonical, so
  $h_B$ must land in it and hence be $u$-invariant; as $z^A$ carries $u$-variation,
  $h_B$ cannot depend on it. No spectral/regularity/resonance hypothesis. Measured
  (`exp10` part 1): the net $u$-dependence of $h_B = z^B + \varepsilon z^A$ is $0$
  at $\varepsilon=0$ and rises monotonically to $0.29$ at $\varepsilon=0.5$.
- **(dynamics = Lemma C) $M_{AB}\equiv 0$** under the one-sided gap
  $\lambda_{\max}(f_B) < \lambda_{\min}(f_A)$ (A dominant). Proved. Measured: rate
  $-0.588$ against predicted $-0.588$; and the *reverse* rate is $+0.588$, so the
  cocycle **cannot** kill $M_{BA}$ — it is genuinely the behavioural direction.

> **Theorem (B∘C, informal).** With behavioural variability on $z^A$ and a
> one-sided module Lyapunov gap with $z^A$ dominant, $h$ is block-diagonal across
> the $A/B$ split — each block identified up to its within-block transformation
> ($z^A$ up to the iVAE $\sim_A/\sim_P$, $z^B$ up to conjugacy class).
>
> **Status (2026-08-04).** The behavioural half is **proved** in a restricted
> setting — additive $h_B$, linear modules — as `identifiability.md` §4.5
> (Lemma D). The apparent empirical contradiction from `exp11`/`exp12` has been
> **withdrawn**: those runs never imposed the behavioural hypothesis, because the
> penalty they optimised was gauge-dependent and the optimiser satisfied it by
> shrinking the pinned block (CLAUDE.md §3.12). There was no learning gap to
> reconcile — there was an experiment that omitted a hypothesis. `exp13` is the
> re-run; until it lands, this statement is **untested under learning**, not
> contradicted by it.

**The alignment condition (new).** The two kills close the problem *only when they
land on different cross-derivatives*, which requires **the behaviour-varying block
to be the spectrally dominant one**. If instead the *invariant* block dominates,
the cocycle kills $M_{BA}$ — the same one behaviour already kills — and $M_{AB}$
survives, so the result is merely **triangular** (`exp10` part 4: $M_{AB}$ rate
$+0.588$, unconstrained; $M_{BA}$ killed twice over). So B∘C $\ge$ C always, and
$=$ block-diagonal exactly under alignment. This is a genuine, checkable
precondition, not folklore: behaviour and the spectral gap must point *opposite*
ways across the partition.

**Why modularity stays load-bearing here** (answering cost 3 below in the
composition regime). In the composed argument, behaviour supplies *one* zero and
the *dynamics* supply the other — and neither can supply both: behaviour alone is
triangular (its complement is not canonical), dynamics alone is triangular (the
two-sided gap is a contradiction, §3.7). So the block-diagonal conclusion rests on
the modular dynamics for a cross-derivative behaviour provably cannot reach. That
is exactly the regime where the auxiliary is *not* strong enough to dissolve the
claim (cost 3): $u$ modulates a strict subset, and modularity does the rest.
`exp10` part 3 is the truth table — only the block-diagonal $h$ passes both tests,
the $M_{BA}$ leak is caught by behaviour alone, the $M_{AB}$ leak by dynamics
alone.

**What is proved vs open.** Lemma C (the dynamics half) is proved and holds on
attractors, not just fixed points (§4.4) — so B∘C inherits C's reach past the
Poincaré domain, which is the whole appeal over Route A. The behavioural half is
the canonical-subspace argument, currently at the *mechanism-verified* level; the
theorem it needs is the **partial-iVAE** statement of cost 1 (the $u$-invariant
complement identified only as a subspace), which is not a corollary of Khemakhem
et al. — verified.

> **RESOLVED (2026-08-03) — see `identifiability.md` §4.5, "Lemma D".** The
> behavioural half is now **proved** for additive $h_B$ with linear modules, and
> *not* by proving a partial iVAE theorem. The argument is dynamical: the
> conjugacy forces the coupling to be a semiconjugacy $\psi\circ f_A =
> \tilde f_B\circ\psi$; only cross-module *resonant* degrees survive; the
> one-sided gap **itself** forces every surviving degree to be $\ge 2$; and a
> homogeneous $\psi$ of degree $p\ge2$ scales as $\sigma^p$ under variance
> modulation, so two behaviour levels detect it by a characteristic-function
> argument. The unique escape is a scale-invariant (degree-0) coupling, which
> requires $1 \in \operatorname{spec}(\tilde f_B)$ — exactly what the gap forbids.
>
> This **avoids** Khemakhem assumption (iv) rather than repairing it, and needs
> two levels where iVAE needs $nk+1$. Cost 1 below is therefore discharged in
> this setting. Remaining: non-additive $h_B$, anisotropic modulation, nonlinear
> modules. `TODO(gap)`

> **Empirical caveat, WITHDRAWN 2026-08-04 (CLAUDE.md §3.12).** An earlier
> revision of this note reported that `exp11`/`exp12` weaken the conclusion to
> triangular under a nonlinear observation map — the allowed $M_{BA}$ mass
> running $0.17$–$0.41$ instead of vanishing — and concluded that "behaviour
> supplies its zero only when $h$ is already linear, and B∘C collapses to C".
>
> **That conclusion was not supported by those experiments.** The behavioural
> penalty they trained against scored the pinned block's conditional moments on
> the raw block, so it falls like $\varepsilon^2$/$\varepsilon^4$ as the block
> shrinks. The optimiser paid it with scale rather than invariance: the pinned
> block came out $21\times$ smaller than its partner, scored a raw
> $u$-dependence of $0.0015$, and still carried the $u$-varying latent at
> distance correlation $0.99$ (scale-normalised $u$-dependence $1.07$, against
> $0.15$ for a genuinely invariant block). **$M_{BA}$ was never constrained**, so
> its failure to vanish is not evidence about behaviour.
>
> The lesson worth keeping: a *penalty* can measure the gauge just as a metric
> can (§3.10), and it is worse there, because a loss does not merely misreport —
> it steers. The fix is the same move: whiten the block, making the penalty
> invariant under $GL(d_b)$, which is the freedom §7 grants within a module
> anyway. The weight must then be recalibrated, since the whitened penalty is
> $O(1)$ where the raw one was $O(\text{scale}^{2..4})$.
>
> `exp13` re-runs the sweep under both penalties, 8 restarts each. **The
> retraction is confirmed and the composition partly vindicated:** the forbidden
> $M_{AB}$ mass is now suppressed *uniformly* ($\le 0.081$ at every dose and
> restart, against $0.316$), and block-diagonality recovers at the two highest
> doses ($0.702\to0.893$, $0.567\to0.815$). **But not uniformly** — at dose
> $0.31$ the whitened fit is triangular in all 8 restarts (`jac_diag` $0.546$,
> allowed-lower $0.416$, sd $0.029$), which is too tight to be seed noise. So
> §B.1's claim survives as "behaviour does supply its zero", not yet as
> "block-diagonal at every dose". See CLAUDE.md task 34.

**Compatible with the scope.** Random initial conditions — inherent to LFADS —
make $p(z_t|u)$ a genuine full-rank density at every $t$ (verified: `cov(z_t)`
stays rank 4 out to $t=30$). No process noise needed. The degeneracy that kills
PCL is in the *joint* law of $(z_t, z_{t+1})$, which lies on the graph of $F$;
iVAE needs only the marginal. Modularity also *preserves* conditional
factorisation for all $t$ (cross-module correlation 0.009 at $t=30$).

**Three costs.**

1. **Standard iVAE does not apply off the shelf.** *Verified from the paper*
   (`route_a_assessment.md` §6.1): the assumption of variability (Khemakhem
   Thm 1(iv)) needs $nk+1$ points of $u$ with the $nk \times nk$ matrix of
   **natural-parameter** differences invertible. For $u$-invariant components
   those rows vanish for every $u$, so by the paper's own characterisation no
   choice of points satisfies (iv) — the theorem fails **globally**, delivering
   nothing even for the $u$-varying block, and no partial version exists in the
   paper. A **partial/block version** must be proved fresh: "the $u$-varying
   subspace is identified up to permutation and componentwise transformation; the
   invariant complement only as a subspace." Not a corollary. `TODO(gap)`
1b. **The behavioural signal must modulate variances, not just means.** *Verified
   from the paper* (their Proposition 1): mean-only modulation of Gaussian latents
   caps identifiability at $\sim_A$ (identifiability up to an affine map)
   regardless of any partial theorem. So $u$ must move second-or-higher sufficient
   statistics ($k \ge 2$) — a constraint on the experiment, not just the proof.
2. **Contraction erodes the signal.** Module variance under $s = 0.70$: $2.02 \to
   0.057\ (t{=}5) \to 0.0016\ (t{=}10) \to \sim 0\ (t{=}20)$. Behaviour separates
   conditions in slow factors across the trial, in fast factors only early.
   Condition number of $\mathrm{cov}(z_t)$ degrades $0.98 \to 1.1\times10^{-8}$.
3. **It can dissolve its own claim.** Reframing $X = G(z_0)$ with
   $G(z_0) = (g(F^t z_0))_t$ injective is *plain iVAE with $G$ as the mixing
   function* — and that argument never uses modularity. A sufficiently strong
   auxiliary identifies everything and makes the modular constraint
   non-load-bearing. The claim survives **only** when $u$ modulates a strict
   subset, which is exactly the regime that makes it interesting.

**Within $z^B$** — several invariant modules among themselves — the full
two-sided problem persists unchanged. Behaviour shrinks the hard problem; it does
not alter its character there.

---

## C. Filtration

> **WRITTEN UP (2026-08-12) — `identifiability.md` §6, "Theorem F".** This
> section is now the *rationale*; the statement, hypotheses and proof live there.
> Two things changed in the writing and both matter here:
>
> 1. **The hypothesis is (F3) ordered separation, not (B4) disjointness.** The
>   gap is not cosmetic — the §3.1 regrouping counterexample *passes* (B4) at
>   $+0.1823$ and fails (F3) at $-0.2231$. `spectra.filtration_gap` computes it;
>   `spectra.spectral_gap` computes the wrong one (CLAUDE.md §3.14).
> 2. **"(B3) nearly free" below is right, and §6.7 says exactly how far.** The
>   *ordering* half is free — with (F3) on both sides, any correspondence is
>   forced to be order-preserving. What remains is the **coarsening**, which is
>   the same thing as cost 1 below; see the reconciliation in §6.7.

**Claim.** $h$ is triangular: modules are identified as an **ordered flag** by
Lyapunov spectrum, not as an unordered partition.

**Already proved.** Lemma C: a one-sided gap
$\lambda_{\max}(f_2) < \lambda_{\min}(\tilde f_1)$ forces $M_{12} \equiv 0$, so
$h_1 = h_1(z_1)$ and $h(z_1,z_2) = (h_1(z_1), h_2(z_1,z_2))$.
`identifiability.md` §4.2. Verified nonlinearly in `exp05` on `TwistBlock`s
($\beta \neq 0$): measured rate $-0.30538$ against predicted $-0.30538$, error
$1.5\times10^{-12}$.

**Genuinely nonlinear, and free of both open lemmas.**

- **(B2) not needed** — no "finest" decomposition is claimed. If a module turns
  out to be decomposable, the flag simply refines.
- **(B3) nearly free** — the Lyapunov ordering *is* the matching; the slow factor
  corresponds to the slow factor. `literature.md` §3.3: orbit-separation rates
  are preserved by bi-Lipschitz conjugacies, giving the ordered correspondence
  *before* any Sylvester-type hypothesis.

So it runs on (B1) + a one-sided gap, both checkable. Everything else in this
project is conditional on unproved lemmas.

**Consistent with the counterexample.** The §5 map is *itself* triangular. It
refutes block-diagonality while leaving this untouched — which is why §4.2 is
**sharp** rather than merely unrefuted.

**Two costs.**

1. It identifies how any two modular explanations *relate*, not that the
   decomposition is **unique**. "This population has these factors" still needs
   (B2). It identifies the relationship without discovering the structure.
2. It commits the science to asserting a **hierarchy**. Under §1.1 that reads as
   timescale structure — a slow component evolving autonomously, a faster one
   driven by it — which is defensible for a single population but is a narrower
   scientific claim than a symmetric decomposition.

---

## Cross-cutting: learning an indecomposable model

Blocks A and B. C is immune. **Largely closed** (`src/idyn/selection.py`, `exp06`);
one piece remains open.

The problem was that every indecomposability check ran on a **ground-truth**
object (`ce["F"]`, `F_t`, `exp03`'s `J0`); nothing touched the *fitted* model, so
a `[2,2]` fit could silently converge to two secretly-1-D modules (the §3.1
non-identifiable regime) unnoticed.

Three steps, increasing cost:

1. **Certify the fitted model — done.** `selection.certify_fitted_model`
   linearises the learned `ModularTransition` at its fixed point and runs
   `is_indecomposable` per block. First check in the repo on a fit, not on ground
   truth.
2. **Search the partition lattice — done.** `selection.select_finest_partition`
   fits every integer partition of $d$ and takes the finest that survives. `exp06`
   runs it on two datasets with opposite correct answers, and the result is
   sharper than expected — **neither fit nor uniqueness suffices alone**:
   - On two genuinely-2-D oscillators, over-splitting an indecomposable rotation
     costs $180\times$–$320\times$ in fit, so **fit** cleanly selects $[2,2]$;
     uniqueness alone would wrongly accept the over-split $[2,1,1]$ (also
     "unique").
   - On four independent 1-D factors, the true finest $[1,1,1,1]$ and the
     decomposable regroupings $[2,2]$, $[3,1]$ fit within $1.1\times$ of each other
     (exactly the `exp02` degeneracy — regroupings fit equally), so **fit is
     blind** and **uniqueness** is what selects $[1,1,1,1]$.

   Together they recover the true finest on both, and each selected model
   certifies as indecomposable. So the operational recipe is: reject fit-blowups
   (splitting an indecomposable block), then break ties among equal-fit partitions
   by uniqueness.
3. **Nonlinear indecomposability test — still open.** The certification in step 1
   is *local* (the linearisation at the fixed point). Decomposability is up to
   conjugacy (the $\varphi$ of `identifiability.md`), so a global test means
   searching over coordinate changes, and off the fixed point we have no handle.
   `TODO(gap)`

**Consequence for existing results, updated.** `exp03`'s 5/5 recovery was
conditional on knowing the true system satisfies (A1)+(A2). `exp06` removes that
dependence for the linearisable regime: the finest partition is now selected
*from data* without being given the answer, and the fit is certified. What
remains conditional is the nonlinear (non-linearisable-attractor) case, via
step 3.

---

## Reading

If the goal is the **strongest theorem**: A. Its remaining cost is now small and
named — the flat-tangency half of (FLAT-D) (a one-page check against Chaperon;
existence half located) and (B3) — with (B2) derivable in Tier 1 and the
learning gap closed for the linearisable regime (`exp06`).

If the goal is a **working method on real recordings**: B, accepting that
behaviour does much of the identifying and that the modular contribution is
confined to the behaviour-invariant remainder.

If the goal is **something provable now**: C, accepting a weaker and more
hierarchical claim.

These are not mutually exclusive. C is a strict weakening of A, so proving A
subsumes it; and B composes with either. The one thing that must be decided
before writing anything up is which claim the paper is *about*, because the
metric, the experiments, and the positioning against nonlinear ICA all differ.
