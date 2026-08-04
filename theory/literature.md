# Literature: positioning (step 8) and recon for the two blocking gaps (steps 9, 4)

Written for CLAUDE.md §6 steps 8 and 9 and open problems 1–4 of
`identifiability.md` §6. One section per question. Every external claim carries
one of four provenance tags, used literally:

- **verified from the paper** — we read the relevant text (local PDF or full-text
  rendering) and the statement reported here paraphrases it;
- **from the abstract only** — verified against the abstract or a secondary
  source (search result, citing paper), full text not read;
- **recollection, unverified** — standard-looking background we did not confirm
  this session; treat citations so tagged as pointers, not references;
- **could not find** — searched for and not found; absence is the reported fact.

Derivations done inline are tagged **checked by direct computation here**; they
are self-contained and do not depend on any citation.

> **Inventory correction.** `C:\Users\alexa\Downloads\1603.06277v5.pdf` is *not*
> the TCL paper as CLAUDE.md §4.3 states. arXiv 1603.06277 is Johnson, Duvenaud,
> Wiltschko, Datta, Adams, *Composing graphical models with neural networks for
> structured representations and fast inference* (SVAE, NeurIPS 2016) — verified
> from the paper (title page of the local file and the arXiv abstract both
> checked). TCL is arXiv **1605.06336**; we read it via its full-text rendering.
> The SVAE paper is still relevant — it is the inference machinery SNICA builds
> on — but it contains no identifiability theory.

---

## 1. Step 8 — nonlinear ICA with temporal structure

What each result assumes and concludes, at theorem level. Throughout, "sources"
means the latent components $s_i(t)$ and "mixing" the observation map $f$
(their notation; our $g$).

### 1.1 The five results

**TCL** — Hyvärinen & Morioka, *Unsupervised feature extraction by
time-contrastive learning and nonlinear ICA*, NeurIPS 2016, arXiv 1605.06336.
Verified from the paper (full-text rendering). Model: $x_t = f(s_t)$, $f$
smooth invertible, sources mutually independent, and **nonstationary** through
an observed segmentation: within segment $\tau$ each $s_i$ is drawn from an
exponential-family density whose parameters $\lambda_{i,v}(\tau)$ vary by
segment. Theorem 1 (with modulation dimension $V=1$): if the modulation
parameter matrix has full rank, TCL recovers $q(s_t)$ — the sufficient
statistic applied coordinatewise — **up to an invertible linear map**;
Corollary: if $q$ is monotone in $|s|$, sources are recovered up to strictly
monotone coordinatewise transforms (after a final linear ICA step). The
segment index is effectively an observed auxiliary variable.

**PCL** — Hyvärinen & Morioka, *Nonlinear ICA of temporally dependent
stationary sources*, AISTATS 2017 (PMLR v54). Verified from the paper (we
extracted the PMLR PDF text). Model: $x(t) = f(s(t))$, $f: \mathbb{R}^n \to
\mathbb{R}^n$ bijective, $C^2$ with $C^2$ inverse. Sources: mutually
independent, **stationary ergodic** scalar stochastic processes, "uniformly
dependent" (Def. 1: the cross-derivative $\partial^2 \log p(s_t,
s_{t-1})/\partial s_t\,\partial s_{t-1}$ exists, is continuous, and vanishes
**nowhere**), and none "quasi-Gaussian" (Def. 2: that cross-derivative does not
factor as $c\,\sigma(s_t)\sigma(s_{t-1})$; excludes Gaussian processes and
monotone transforms of them). Theorem 1: the learned representation equals the
sources **up to coordinatewise (strictly monotone) transforms and permutation**.
No auxiliary variable, no nonstationarity. The quasi-Gaussian case (which
includes Gaussian) is treated separately in their §3.3 with weaker guarantees,
and their discussion recalls the classical linear fact that Gaussian sources
are separable **only if their autocorrelation functions differ** (verified
from the paper; the classical fact is their citation of Tong et al. /
Molgedey–Schuster, not their result).

**HMM-NLICA** — Hälvä & Hyvärinen, *Hidden Markov nonlinear ICA*, UAI 2020,
arXiv 2006.12107. Verified from the paper (full-text rendering). Model: a
global hidden Markov chain $c_t \in \{1,\dots,C\}$; given $c_t$, sources are
independent with factorial exponential-family densities whose natural
parameters depend on $c_t$; $x_t = f(s_t)$, $f$ bijective. Theorem 2: with the
chain irreducible, full-rank transitions, $C \ge NV+1$ states, and a full-rank
parameter-difference matrix, the model is identifiable up to an invertible
linear transform of the sufficient statistics; Theorem 4 (Gaussian case with
distinct state means): up to permutation and coordinatewise linear maps. The
auxiliary variable is **latent** (the regime label is inferred, not observed),
but the mechanism is still nonstationarity — regime switching — now supplied
by the model instead of the experimenter.

**iVAE** — Khemakhem, Kingma, Monti, Hyvärinen, *Variational autoencoders and
nonlinear ICA: a unifying framework*, AISTATS 2020, arXiv 1907.04809. Verified
from the paper (full-text rendering). Model: observed auxiliary $u$;
$p(z\mid u)$ conditionally factorial exponential family with parameters
$\lambda(u)$; $x = f(z) + \varepsilon$, $f$ injective. Theorem 1: with (i)
noise characteristic function a.e. nonzero, (ii) $f$ injective, (iii)
sufficient statistics linearly independent, (iv) $nk+1$ points $u^j$ making the
$\lambda$-difference matrix invertible, parameters are identifiable up to an
invertible linear map of sufficient statistics; Theorems 2–3 sharpen to
permutation + coordinatewise under extra conditions. Proposition 1: in the
Gaussian location-scale case the linear indeterminacy is **not** reducible —
their own within-framework failure mode.

**SNICA** — Hälvä, Le Corff, Lehéricy, So, Zhu, Gassiat, Hyvärinen,
*Disentangling identifiable features from noisy data with structured nonlinear
ICA*, NeurIPS 2021, arXiv 2106.09620. Verified from the paper (local PDF,
text extracted). Framework: sources are **unconditionally independent scalar
stationary processes** with essentially arbitrary temporal/spatial dependency
structure (each may be driven by its own latent chain, e.g. per-component
SLDS in their $\varepsilon$-SNICA instance); $x_t = f(s_t) + \varepsilon_t$
with $f$ injective (dimension reduction allowed, $M \ge N$) and additive noise
of **unknown** distribution. Two-stage identifiability: **Theorem 1** — under
tail (A1), non-degeneracy (A2), and no-Gaussian-direction (A3) conditions, the
distribution of the noise-free process is recovered from the noisy one, up to
translation; notably (their words) it "does not even assume a mixing as in
ICA". **Theorem 2** — under conditions (B1)-(B2) on the dependency (each
component's finite-dimensional joint density $C^2$ on its support; sufficiently
strong dependence between nearby time points; a condition excluding Gaussian
processes and their trivial transforms), $f$ is identified **up to permutation
of the coordinates and a bijective transformation of each coordinate**
(verified verbatim from their proof text).

### 1.2 Does any of them cover the autonomous, stationary, no-auxiliary case?

> ## ⚠ CORRECTION (2026-08-04) — the question is mis-posed, and the answer below is wrong for TCL and iVAE
>
> **Our setting is not stationary.** This section, and the §8 positioning built
> on it, rest on "autonomous = one transition kernel, ever ⟹ no variability of
> conditional distributions" (point 3 below). That conflates the **transition
> kernel** with the **marginal law**. A single fixed kernel applied to a
> non-invariant $p_0$ produces genuinely time-varying marginals
> $p_t = (F^t)_*p_0$ — and `train.make_dataset` deliberately spreads initial
> conditions over an annulus, precisely so trajectories do not collapse onto a
> single orbit. Every dataset in this repo is non-stationary **by construction**.
>
> Measured (CLAUDE.md task 35): conditioning on $u=t$, the blocks' scale-
> normalised $t$-dependence is $1.32$ and $4.20$ in the contracting system. On
> two limit cycles with concentrated initial phase it is $1.489$ early and
> $1.493$ late — *persistent*, not transient. With uniform initial phase (i.e.
> $p_0$ **is** the invariant measure) it collapses to $0.057$, which is the only
> configuration where the "no variability" claim holds.
>
> **What survives and what does not:**
>
> * **Point 1 survives for PCL and SNICA only.** Their hypotheses are on the
>   *joint* density of consecutive samples within a trajectory, which for
>   deterministic $z_{t+1}=f_i(z_t)$ is supported on a graph. That objection is
>   correct and structural. It does **not** touch TCL or iVAE, which condition
>   on the time index across an **ensemble** of trajectories: at fixed $t$,
>   $p(z_t)$ is a pushforward of a smooth $p_0$ by a diffeomorphism, hence a
>   perfectly good smooth density.
> * **Point 3 is false as applied to TCL/HMM-NLICA/iVAE.** The variability they
>   consume is exactly what relaxation toward an attractor supplies for free.
> * **Point 2 survives, and is now the whole gap.** All five conclude
>   coordinatewise identification for *statistically independent scalar*
>   components; our modules are multidimensional with arbitrary internal
>   dependence. The delta between us and the literature is **granularity, not
>   applicability**.
>
> So the honest positioning is *not* "no existing result covers us". It is: TCL
> and iVAE **do** cover a setting we are in, at the wrong granularity, using an
> auxiliary variable ($u=t$) that is free and that we wrongly believed we did
> not have. Note also that point 2 below already flags the "dependent source
> subspaces" extensions as read *from the abstract only and never chased* —
> that is now the most relevant citation in this file.
>
> Do not cite §8 positioning from this section until it is rewritten.

**No.** PCL and SNICA are the only stationary, no-auxiliary results in the
list, and both fail to apply to our setting for the same three structural
reasons, each verified against their stated hypotheses:

1. **They require genuinely stochastic sources with smooth positive densities.**
   PCL's uniform dependence needs $\log p(s_t, s_{t-1})$ to exist with
   everywhere-nonvanishing cross-derivative; SNICA's Theorem 2 needs $C^2$
   joint densities of $(s_{t_1},\dots,s_{t_m})$. For our deterministic modules
   $z_{t+1} = f_i(z_t)$ the pair $(s_t, s_{t-1})$ is supported on the graph of
   $f_i$ — a measure-zero set with no density. The hypotheses are not merely
   unverified in our setting; they are structurally false. Adding dynamics
   noise would change our model class, not reinterpret it.
2. **Their modules are one-dimensional coordinates.** All five conclude
   "permutation + coordinatewise transformation" for scalar components that are
   *statistically independent* of each other. Our modules are multidimensional
   blocks with arbitrary internal structure, and nothing is independent in a
   statistical sense — no invariant measure is even fixed. (A 2023 review by
   Hyvärinen, Khemakhem, Morioka, arXiv 2303.16535, mentions extensions to
   "dependent source subspaces" — from the abstract only; we did not chase
   these, and none surfaced with an autonomous deterministic theorem.)
3. **The identifying information is different in kind.** TCL/HMM-NLICA/iVAE:
   variability of conditional distributions across segments / states / $u$ —
   none exists for us (autonomous = one transition kernel, ever). PCL/SNICA:
   non-Gaussian conditional dependence of consecutive samples — degenerate for
   us. What we use instead is geometric: invariance of the module foliations
   and separation of dichotomy spectra. The closest point of contact is the
   Gaussian boundary case PCL is built to avoid: linearly mixed Gaussian
   sources are separable only when their autocorrelation functions differ —
   the stochastic-linear shadow of our (A2)/(B4) disjoint-spectrum hypotheses.
   In both worlds "same spectrum" is exactly where identifiability dies
   (compare `counterexamples.md` §2).

Two further honest deltas, one in each direction. First, they identify
**coordinates**; we claim only the **partition** and per-module conjugacy
classes — our conclusion is strictly weaker, and after §3.7 the honest current
conclusion (a filtration) is weaker still. Second, their equivalence is
distributional, which is the right notion; ours is currently pathwise
(`identifiability.md` §1, noise `TODO(gap)`). **SNICA's Theorem 1 is directly
reusable for that gap**: it is a pure observation-side result (no ICA
structure) that upgrades "equal noisy observation distributions" to "equal
noise-free process distributions" under tail conditions — exactly the reduction
open problem 4 needs before our geometric machinery takes over. Verified from
the paper that it is stated free of the mixing model.

**What this means for us.** ~~The positioning claim for §8 survives contact with
the actual theorems: no existing temporal-structure result covers autonomous
stationary deterministic dynamics...~~ **Retracted 2026-08-04 — see the
correction box at the head of §1.2.** The premise ("we have no auxiliary
variable and no non-stationarity") is false: $u=t$ is both, and it is free.

**Revised positioning, to be written into §8.** Split the five results by *what
they condition on*, not by whether they use an auxiliary variable:

* **Ensemble-conditioning (TCL, HMM-NLICA, iVAE)** — consume variability of
  $p(z\mid u)$ across values of $u$. With $u=t$ this setting **applies to us**.
  Their conclusion is coordinatewise, ours is block-level, so the gap is
  granularity: we need a *subspace/ISA* version. Chase the "dependent source
  subspaces" line (Hyvärinen–Khemakhem–Morioka 2023 review, arXiv 2303.16535,
  §1.2 point 2) — it may already be written.
* **Path-conditioning (PCL, SNICA)** — consume non-Gaussian dependence between
  consecutive samples. Inapplicable while the dynamics are deterministic,
  because the pair $(z_t, z_{t-1})$ lives on a graph. **This is a property of
  the target, not an incidental modelling choice** (corrected 2026-08-04): in an
  autonomous LFADS model one samples the initial condition $g_0$ from the prior
  and then simulates a *deterministic* generator forward, so within a trial
  consecutive latent states are deterministically related. Adding process noise
  would open this family, but that is *changing the model class*, not modelling
  LFADS more faithfully. See CLAUDE.md task 36.

**Which family the science is in.** Autonomous LFADS puts *all* trial-to-trial
randomness in $g_0$. At fixed $t$, across trials, that is an ensemble with a
genuinely $t$-varying law — the ensemble-conditioning family's exact input, and
what `train.make_dataset` already generates. So the positioning is not "we
happen to violate PCL's assumptions"; it is that **the target model class lives
in the ensemble-conditioning family and structurally cannot live in the
path-conditioning one.** That makes the subspace/ISA gap (point 2) the only real
obstacle between this project and existing theory.

What modular *dynamics* adds is then not "identifiability where none was
available" but **the block granularity and the dynamical invariants** (Lyapunov
spectrum, rotation number) that coordinatewise results do not deliver. That is a
smaller claim, and an honest one. SNICA Theorem 1 remains reusable as the front
end for open problem 4 (it is stated free of the mixing model).

---

## 2. Step 9 — when does a triangular conjugacy upgrade to a product?

The question from §3.7: Lemma C plus the two-sided obstruction leaves $h$
triangular (a skew product), and we asked whether the literature has a theorem
upgrading skew-product conjugacies to product conjugacies. Short answer: **no
such theorem exists at the regularity Theorem B currently assumes, and one
cannot exist — there is an explicit $C^1$ counterexample to the target
conclusion under (B1)–(B4). With more smoothness plus non-resonance, the
upgrade is classical normal-form theory near a hyperbolic fixed point.** Both
halves below.

### 2.1 A counterexample to Theorem B's target under (B1)–(B4)

Checked by direct computation here. Take $d_1 = d_2 = 1$, $f_i(z_i) = \mu_i
z_i$ with $0 < \mu_1 < \mu_2 < 1$ and no integer resonance ($\mu_1 \neq
\mu_2^m$), e.g. $\mu_1 = 0.3$, $\mu_2 = 0.5$. Set

$$p = \frac{\ln \mu_1}{\ln \mu_2} > 1 \ (\notin \mathbb{Z}), \qquad
h(z_1, z_2) = \big(z_1 + c\,\mathrm{sgn}(z_2)|z_2|^p,\; z_2\big).$$

Then $h \circ F = F \circ h$ exactly: the first component maps to
$\mu_1 z_1 + c\,\mathrm{sgn}(z_2)\mu_2^p |z_2|^p$ and $\mu_2^p = \mu_1$ by
construction. Every hypothesis of Theorem B holds: $\Omega$ = any closed ball
(compact, forward-invariant); $h$ is $C^1$ with $Dh, Dh^{-1}$ bounded (B1) —
$|z_2|^p$ is $C^{\lfloor p \rfloor}$ but not $C^{\lceil p \rceil}$ at $0$, and
$p \approx 1.737$ here, so $C^1$ not $C^2$; both blocks are 1-D, hence
indecomposable (B2); the matching is the identity, $\tilde f_i = f_i$ (B3);
Lyapunov spectra $\{\ln \mu_1\}, \{\ln \mu_2\}$ are disjoint (B4). Yet
$M_{12} = cp|z_2|^{p-1} \not\equiv 0$: $h$ is triangular and **not**
block-diagonal, and no block-diagonal representative exists in its
"equivalence data" — $h$ is the unique conjugacy given the decoders.
Consistency check against §4: module 1 is the fast one, Lemma C kills $M_{21}$
(and indeed $h_2 = z_2$), and the surviving block is exactly fast-receives-slow.
Measured this session ($\mu_1 = 0.3$, $\mu_2 = 0.5$, $c = 0.7$, $p = 1.736966$):
$\max |h(Fz) - F(hz)| = 1.1 \times 10^{-16}$ over $10^4$ points of
$[-1,1]^2$, and $2.8 \times 10^{-17}$ for the resonant $C^\infty$ variant
below — exact conjugacies to machine precision.

Consequences, all sharp:

- **Theorem B as currently stated (conclusion "block-diagonal") is false.**
  The triangular conclusion of `identifiability.md` §4.2 is not a proof gap; it
  is the true statement at $C^1$ regularity. This should be recorded next to
  the `TODO(gap)` in §4.2.
- **Both §5 routes die at $C^1$.** Route 1: $h^{-1} = (z_1 -
  c\,\mathrm{sgn}(z_2)|z_2|^p,\, z_2)$ is triangular with the *same*
  orientation, so "triangularity of both $h$ and $h^{-1}$" holds here and does
  not force block-diagonality — the compatible-orientation hope is answered
  negatively. Route 2: the $\omega$-limit set is $\{0\}$ and $M_{12}(0) = 0$,
  so Route 2's conclusion is correct and *cannot be improved from the
  $\omega$-limit set to $\Omega$* — this example is the witness.
- **The resonant variant is $C^\infty$.** If $\mu_1 = \mu_2^m$ for an integer
  $m \ge 2$, then $h = (z_1 + c z_2^m,\, z_2)$ is polynomial and commutes with
  $F$. So no amount of smoothness rescues the statement when the spectra are
  multiplicatively related across modules: a **cross-module non-resonance
  hypothesis is necessary**, not a technicality.
- The same map read as a self-conjugacy of $F$ shows the **slow foliation is
  not canonical**: $h$ maps the coordinate foliation $\{z_1 = \mathrm{const}\}$
  to the curved invariant family $\{z_1 + c\,\mathrm{sgn}(z_2)|z_2|^p =
  \mathrm{const}\}$, which is tangent to it along the fast axis $\{z_2 = 0\}$
  and distinct off it. A perfectly linear diagonal map already carries two distinct
  invariant "slow product structures". The fast foliation, by contrast, is
  metrically characterized by forward decay rates and is preserved by any
  bi-Lipschitz conjugacy — which is Lemma C in geometric form. On a
  forward-invariant $\Omega$ near an attractor only forward rates exist
  (backward orbits leave $\Omega$), so the fast/slow asymmetry is fundamental,
  not an artifact of the proof.

The regularity threshold is exactly the spectral spread: the family
$|z_2|^p$ exists whenever $h$ is only required to be $C^k$ with $k \le p =
\chi_{\mathrm{fast}}/\chi_{\mathrm{slow}}$ (ratio of Lyapunov exponents).
Requiring $h \in C^k$ with $k > p$, or $h \in C^\infty$, excludes it — and
then, by §2.2, excludes everything else too, absent resonances.

### 2.2 The positive result: normal forms near a hyperbolic attracting fixed point

The relevant classical machinery, all near a fixed point:

- **Sternberg**, *Local contractions and a theorem of Poincaré*, Amer. J.
  Math. 79 (1957) 809–824 — from the abstract only. A $C^\infty$ contraction
  whose multipliers satisfy the non-resonance conditions $\mu_i \neq \mu^\alpha$
  ($|\alpha| \ge 2$) is $C^\infty$-conjugate to its linear part; with
  resonances, to a polynomial normal form. Finite-smoothness versions with loss
  of derivatives controlled by the spectral spread: **Sell**, *Smooth
  linearization near a fixed point* (1985) — from the abstract only ($C^K$
  linearization under strict hyperbolicity plus non-resonance up to a finite
  order $Q(K)$; venue not confirmed this session).
- **Hartman**, *On local homeomorphisms of Euclidean spaces*, Bol. Soc. Mat.
  Mexicana 5 (1960) 220–241 — from the abstract only. $C^2$ (even $C^{1,1}$)
  contractions are $C^1$-linearizable with **no** non-resonance condition.
  Note what this does and does not give: it makes both $F$ and $\tilde F$
  $C^1$-linearizable, but a $C^1$ conjugacy between the linear parts need not
  be linear — §2.1 is exactly such a map — so Hartman-level regularity cannot
  close the gap. Consistent, again, with sharpness at $C^1$.

Assembled statement (our composition of standard pieces; each ingredient
citable, the assembly checked by direct computation here — we did **not** find
it stated as a single theorem anywhere, and flag it as a proposition to prove
properly, not a citation):

> **Proposition (upgrade near an attracting fixed point; to be written up).**
> Let $F = f_1 \oplus \cdots \oplus f_K$ and $\tilde F$ be $C^\infty$ with
> hyperbolic attracting fixed points, $\Omega$ a compact forward-invariant
> neighborhood in the basin, $h$ a $C^\infty$ conjugacy, and (B4) disjoint
> module spectra. Assume additionally **cross-module non-resonance**: no
> multiplier of $Df(0)$ satisfies $\mu_{i,a} = \prod \mu^{\alpha}$ with the
> multi-index $\alpha$ ($|\alpha| \ge 2$) supported on more than module $i$
> alone (within-module resonances are harmless — they only fatten $h_i$).
> Then $h = P_\sigma (h_1 \oplus \cdots \oplus h_K)$ on $\Omega$, with
> $\tilde f_{\sigma(i)}$ conjugate to $f_i$.
>
> *Proof plan.* (i) Per-module Sternberg/normal-form conjugacies can be chosen
> block-diagonal, $\psi = \psi_1 \oplus \cdots \oplus \psi_K$, taking $F$ to
> its (block-diagonal, polynomial) normal form; likewise $\tilde\psi$. (ii)
> $k := \tilde\psi \circ h \circ \psi^{-1}$ is a smooth conjugacy between the
> normal forms fixing $0$. Expanding $k$ at $0$: the degree-1 part intertwines
> the linear parts; each higher homogeneous term of a cross-module component
> solves the homological equation $\kappa_m(Lx) = L\kappa_m(x)$, whose nonzero
> solutions are exactly resonant monomials — excluded across modules by
> hypothesis. So $k$ = (linear) + (within-module resonant terms) + flat, and
> the flat part dies by iterating $\kappa = L^{-n}\kappa L^n$ against a
> contraction. (iii) The linear part is an intertwiner of block-diagonal linear
> maps with disjoint spectra, so `linear_case.md` Theorem L makes it a block
> permutation; matching $\sigma$ and the multiplier equality come out as
> conclusions, exactly as in Theorem A. (iv) The local statement propagates to
> all of $\Omega$ by $h = \tilde F^{-n} \circ h_{\mathrm{loc}} \circ F^n$,
> since every point of the basin enters the linearization neighborhood. $\square$

Hypotheses our setting plausibly meets: the contracting regime of `exp05`
(TwistBlock modules, spiral contraction to $0$) is squarely inside; $C^\infty$
of $h$ is **free in the model** whenever both decoders are $C^\infty$
immersions, since $h = \tilde g^{-1} \circ g$ — true for the default `tanh`
MLPs in `models.py`, false for `relu`. Cross-module non-resonance is a
countable union of codimension-one conditions on multipliers (generic), and
checkable numerically for any concrete pair of modules. What is *not* covered:
$\Omega$ without a fixed point (limit cycles, chaotic attractors), the
non-invertible-on-$\Omega$ case, and anything global.

### 2.3 Away from fixed points: the cohomological frame, and what exists

For general $\Omega$ the question "is this skew product conjugate to a
product" is a cocycle-triviality (cohomology) problem, and the literature
answers it only with periodic-orbit obstructions:

- **Livšic theory** (1971/72): for real-valued cocycles over a hyperbolic base,
  vanishing of all periodic-orbit obstructions is necessary and sufficient for
  (regular) trivialization — from the abstract only (confirmed via multiple
  secondary sources). Extensions to Lie-group-valued and
  diffeomorphism-group-valued cocycles exist: Niticâ–Török (regularity of
  transfer maps for diffeomorphism- and Lie-group-valued cocycles — from the
  abstract only) and a Livšic theorem for low-dimensional diffeomorphism
  cocycles (title verified in search results; authorship not confirmed). This is the correct generalization of "when can the
  $z_1$-dependence of $h_2(z_1, z_2)$ be removed": the answer is never free,
  and the obstructions live on periodic orbits of the base.
- **Journé**, *A regularity lemma for functions of several variables*, Rev.
  Mat. Iberoam. (1988) — from the abstract only: a function uniformly $C^r$
  along the leaves of two transverse foliations with uniformly smooth leaves is
  $C^r$. This is the standard tool for the endgame Route 1 wants — assembling
  regularity (or product structure) from two transverse invariant foliations —
  but it presupposes *both* foliations are in hand, which is precisely what
  §2.1 shows can fail: the second (slow) foliation need not be canonical.
- Uniqueness of strong (fast) invariant foliations and non-uniqueness /
  bounded regularity of weak (slow) ones is Hirsch–Pugh–Shub, *Invariant
  Manifolds*, Springer LNM 583 (1977), and Fenichel for flows — from the
  abstract only. The KAM reducibility literature (quasi-periodic cocycles,
  Eliasson school) plays the same game over rotations with Diophantine
  small-divisor conditions in place of non-resonance — recollection,
  unverified; noted only as a consistency check that *every* known
  skew-to-product upgrade pays an arithmetic condition somewhere.
- Centralizer results: Kopell's lemma and the Szekeres flow (centralizer of a
  $C^2$ interval contraction inside $C^1$ is a one-parameter flow; *Commuting
  diffeomorphisms*, Proc. Symp. Pure Math. XIV, 1970, 165–184) — from the
  abstract only; and $C^1$-generic diffeomorphisms have trivial centralizer
  (Bonatti–Crovisier–Wilkinson, Publ. IHÉS 2009) — from the abstract only.
  Relevant because the set of obstructions to uniqueness of the product
  structure *is* the centralizer of $F$ modulo block-diagonal elements; §2.1
  exhibits a nontrivial coset at $C^1$, and the normal-form proposition says
  the coset is trivial for smooth non-resonant germs. A "generic products have
  product centralizers" theorem would be the global version; **could not
  find** one.

**Direct answers to the §5 routes.** Route 1 (apply Lemma C to $h^{-1}$): dead
as stated — §2.1's inverse is triangular with the same orientation, so no new
block is killed; revived only inside the normal-form hypotheses of §2.2, where
it is subsumed anyway. Route 2 (backwards cocycle): correct but exactly as
strong as stated — $M_{21} \equiv 0$ on the $\omega$-limit set, and §2.1
witnesses that no argument can extend this to $\Omega$ at $C^1$. The third
option recorded in §6 (restate as a filtration) is now known to be not merely
cautious but **sharp** at $C^1$.

**What this means for us.** Open problem 1 splits cleanly. (a) At $C^1$: adopt
the filtration statement; it is the true theorem, and §2.1 belongs in
`counterexamples.md` as its witness (one-line construction, testable). (b) With
$C^\infty$ decoders (the model's own regime): add cross-module non-resonance to
(B1)–(B4) and prove the §2.2 proposition; near attracting fixed points this
closes Theorem B completely, with matching (B3) falling out as a conclusion.
(c) On non-fixed-point attractors, expect Livšic-type periodic obstructions,
not a free theorem. The honest scope sentence for §7: *block-diagonality is a
smooth, non-resonant phenomenon; at low regularity or at resonance only the
spectral filtration is identified.*

---

## 3. Step 4 — invariants for the matching lemma

Question: are Lyapunov spectrum + fixed/periodic-orbit structure + topological
entropy a complete conjugacy invariant on any reasonable class, so that (B3)
can pair modules by comparing invariants?

### 3.1 They are invariants (of the right conjugacy notion)

Checked by direct computation here, since regularity bookkeeping is the whole
game. If $h$ is $C^1$ with $\sup\|Dh^{\pm 1}\| < \infty$ (our (B1)), then
$D\tilde f^{(n)}(hz) = Dh(F^n z)\, Df^{(n)}(z)\, Dh(z)^{-1}$ gives
$\|D\tilde f^{(n)}\| \asymp \|Df^{(n)}\|$, so Lyapunov and dichotomy
(Sacker–Sell) spectra are preserved, as are multipliers along corresponding
periodic orbits (similarity of the derivative cocycles over one period).
Topological entropy is preserved by any topological conjugacy on compact sets
(standard). But under merely topological conjugacy the differentiable
invariants evaporate: all orientation-preserving linear contractions of
$\mathbb{R}^n$ in one topological class are conjugate (recollection,
unverified), so "conjugacy invariant" without a regularity qualifier is
meaningless. (B1) is the right notion and keeps all three.

### 3.2 They are not complete — three verified obstructions

**(a) Toral automorphisms: all three invariants agree, systems not conjugate.**
Topological conjugacy of hyperbolic toral automorphisms is equivalent to
conjugacy of the matrices in $GL(n,\mathbb{Z})$ (Adler & Palais, *Homeomorphic
conjugacy of automorphisms of the torus*, Proc. AMS 16 (1965) 1222–1225 — from
the abstract only). By the Latimer–MacDuffee(–Taussky) correspondence,
$GL(n,\mathbb{Z})$-classes of matrices with a fixed irreducible characteristic
polynomial biject with ideal classes of $\mathbb{Z}[\beta]$, $\beta$ a root —
from the abstract only. Whenever the class number exceeds one there are matrices with
the **same characteristic polynomial** — hence identical entropy, identical
Lyapunov spectrum, identical periodic-point counts for every period
($|\det(A^n - I)|$ is a function of the eigenvalues alone), identical zeta
function — that are **not conjugate**, even topologically. The invariant that
separates them is arithmetic (an ideal class), invisible to any spectral or
counting data.

**(b) Smooth conjugacy: the complete invariant is infinite-dimensional.**
Within one topological conjugacy class of Anosov diffeomorphisms of
$\mathbb{T}^2$, the complete smooth-conjugacy invariant is the **full periodic
multiplier data** — the eigenvalues at every periodic orbit (de la Llave;
Marco–Moriyón; the four-part series *Invariants for smooth conjugacy of
hyperbolic dynamical systems*, Comm. Math. Phys. (1987), and the companion
Livšic regularity work in Ann. of Math. (1986) — from the abstract only;
parts II and IV of the series sighted directly on the journal site). Same
structure one dimension down: for expanding circle maps, degree classifies
topologically (Shub) and periodic multipliers classify smoothly within a
topological class (Shub–Sullivan, *Expanding endomorphisms of the circle
revisited*, ETDS 5 (1985) 285–289 — from the abstract only). So any *finite*
list — entropy, a Lyapunov exponent, finitely many multipliers — is not
complete for smooth conjugacy on these classes; structural stability lets one
perturb high-period data while fixing any finite list (realization step:
recollection, unverified). Worse, in higher dimension even the full periodic
data is insufficient: de la Llave's $\mathbb{T}^4$ example — two Anosov
systems with identical periodic data that are only Hölder conjugate, with
positive rigidity results requiring irreducibility, simple real spectrum, and
transitivity conditions (Gogolev, arXiv 0804.3901, J. Mod. Dyn. 2008; the
example cited there as [dlL92]) — from the abstract only. **Structural remark
for us**: de la Llave's example lives on $\mathbb{T}^2 \times \mathbb{T}^2$
with $L = A \oplus B$ — it is literally a two-module system in our sense, and
the rigidity failure is a cross-module phenomenon. Worth reading in full
before attempting any nonlinear matching proof; the failure mode it exhibits
is the one (B3) must survive.

**(c) Symbolic dynamics: the classification problem is genuinely hard.**
For subshifts of finite type, conjugacy is strong shift equivalence (Williams,
1973/74), shift equivalence is decidable and captures the zeta function /
periodic data, and Williams' conjecture that the two coincide is **false**:
Kim–Roush, J. AMS 1992 (reducible case) and Ann. of Math. 149 (1999)
(irreducible/mixing case), with a shift-equivalent, non-strong-shift-equivalent
pair of primitive $7\times 7$ matrices — from the abstract only. So even
"all periodic data + entropy + the entire zeta function" fails to classify in
the class where those invariants are most computable. At the opposite extreme,
in the measure category entropy alone **is** complete for Bernoulli shifts
(Ornstein — recollection, unverified), which cuts the other way: measurable
isomorphism erases so much structure that product decompositions become wildly
non-unique. Both extremes are warnings about calibrating the conjugacy notion.

Also: for Morse–Smale systems entropy is identically zero and the
classification is combinatorial (Peixoto's graph invariant for flows on
surfaces; scheme-based classifications for gradient-like diffeomorphisms in
low dimension — recollection, unverified), so on that class two of our three
proposed invariants carry no information and the third (periodic structure +
invariant-manifold linking) does all the work.

Positive islands, for completeness: multiplier at the fixed point for 1-D
hyperbolic germs (Koenigs; Sternberg — recollection, unverified); Diophantine
rotation number for circle diffeomorphisms (Herman–Yoccoz — recollection,
unverified); the 1-D and 2-D hyperbolic results of (b). The pattern: complete
invariants exist only in dimension $\le 2$ per module, or with arithmetic
side conditions, and always as *infinite* data sets once smooth conjugacy is
in play.

### 3.3 The reframe: matching does not need complete invariants

The matching lemma as used by Theorem B is weaker than classification: it must
pair the blocks of two systems *already known to be conjugate via $h$*, not
decide conjugacy from invariants. That is obtainable from (B1) + (B4) alone,
by the mechanism §2.1 leaves intact — the fast side is canonical:

1. Under (B4) read as dichotomy separation, order modules by spectrum. The
   nested fast sets $W^{\ge j}(p) = \{q : d(F^n q, F^n p) = O(r_j^n)\}$, for
   $r_j$ in the spectral gaps, are metrically characterized; a bi-Lipschitz
   conjugacy preserves orbit-separation rates exactly, and by §3.1 the two
   systems have the same spectrum, so $h(W_F^{\ge j}(p)) =
   W_{\tilde F}^{\ge j}(hp)$. (Stable-manifold/graph-transform input: HPS —
   from the abstract only. The rate bookkeeping: checked by direct computation
   here.)
2. Hence $h$ preserves the Lyapunov filtration and induces, level by level,
   conjugacies of the successive quotients: $\sigma$ = the spectral ordering,
   with $\tilde K = K$ and matching block dimensions, established **before**
   any Sylvester-type disjointness of $\mathrm{spec}(D\tilde f_i)$ vs
   $\mathrm{spec}(Df_j)$ is invoked — which is what §3.2 of CLAUDE.md demanded.
3. What this does *not* immediately give: $\tilde f_{\sigma(i)}$ conjugate to
   $f_i$ as autonomous systems for $i \ge 2$, because the leaf conjugacies are
   a priori nonautonomous (they move along orbits of the slower modules:
   $H_{f_1 p_1} \circ f_2 = \tilde f_2 \circ H_{p_1}$). At a fixed or periodic
   point of the slower factor they freeze into genuine conjugacies — free in
   the contracting regime, an extra hypothesis (existence of periodic points
   in each module, or transitivity) in general. Checked by direct computation
   here.

So the honest plan for open problem 2: prove (B3) in the **ordered** form via
the filtration — it is a lemma about preserved fast sets, not about invariants
— and note that the unordered form is exactly as hard as open problem 1, since
both fail together in §2.1's example... more precisely, the *ordering* is not
an artifact: the filtration is canonical, the transverse complement is not.

Separately, for (B2) — nonlinear indecomposability and uniqueness of the
finest decomposition — the analogous algebraic question in topological
dynamics is open: direct-product factorizations of expansive actions are
finite, uniqueness of prime factorization is not established, and there is a
non-expansive $\mathbb{Z}$-action with no finite prime factorization at all
(Meyerovitch, *Direct topological factorization for topological flows*, arXiv
1407.8343, extending Lind's 1980s work on products of shifts — from the
abstract only). We **could not find** a Krull–Schmidt theorem for smooth or
topological dynamical systems. Conclusion: (B2) cannot be outsourced;
uniqueness of the finest splitting will have to come from our spectral
separation, mirroring how `linear_case.md` gets it from primary decomposition
rather than from general module theory.

**What this means for us.** Open problem 2: drop the search for a complete
invariant — none exists at any useful generality, and (a)–(c) say the failure
is structural (arithmetic classes, infinite moduli, wild classification), not
a gap in our reading. Prove the matching lemma via the preserved fast
filtration under (B1)+(B4) (§3.3 sketch), with per-module conjugacy classes
extracted at periodic points of slower factors; this also makes $K$ and the
block dimensions conclusions rather than hypotheses, matching Theorem A's
shape. Open problem 3 (B2): expect no help from the literature; the
Meyerovitch/Lind state of the art says uniqueness of factorization is open
even for shifts. Keep reporting invariants in `metrics.py` per the current
convention — partition first — but when a per-module fingerprint is needed for
*diagnostics*, use periodic multiplier data (the provably right invariant in
low dimension) rather than entropy, which is both non-separating (a) and
uninformative on contracting modules (entropy $0$).

---

## 4. Reference list with provenance

| Ref | Where used | Provenance |
|---|---|---|
| Hyvärinen & Morioka, TCL, NeurIPS 2016, arXiv 1605.06336 | §1.1 | verified from the paper (full-text rendering) |
| Hyvärinen & Morioka, PCL, AISTATS 2017, PMLR v54 | §1.1–1.2 | verified from the paper (extracted PDF) |
| Hälvä & Hyvärinen, HMM-NLICA, UAI 2020, arXiv 2006.12107 | §1.1 | verified from the paper (full-text rendering) |
| Khemakhem et al., iVAE, AISTATS 2020, arXiv 1907.04809 | §1.1 | verified from the paper (full-text rendering) |
| Hälvä et al., SNICA, NeurIPS 2021, arXiv 2106.09620 | §1.1–1.2 | verified from the paper (local PDF) |
| Hyvärinen, Khemakhem, Morioka, review, arXiv 2303.16535 | §1.2 | from the abstract only |
| Johnson et al., SVAE, NeurIPS 2016, arXiv 1603.06277 | header | verified from the paper (identity only) |
| Sternberg, Amer. J. Math. 79 (1957) 809–824 | §2.2 | from the abstract only |
| Sell, *Smooth linearization near a fixed point* (1985) | §2.2 | from the abstract only (venue unconfirmed) |
| Hartman, Bol. Soc. Mat. Mexicana 5 (1960) 220–241 | §2.2 | from the abstract only |
| Hirsch–Pugh–Shub, *Invariant Manifolds*, LNM 583 (1977); Fenichel | §2.3, §3.3 | from the abstract only |
| Livšic (1971/72); Niticâ–Török; low-dim diffeo-cocycle Livšic | §2.3 | from the abstract only (last: title only) |
| Journé, Rev. Mat. Iberoam. (1988) | §2.3 | from the abstract only |
| Kopell, Proc. Symp. Pure Math. XIV (1970) 165–184 | §2.3 | from the abstract only |
| Bonatti–Crovisier–Wilkinson, Publ. IHÉS (2009) | §2.3 | from the abstract only |
| Adler & Palais, Proc. AMS 16 (1965) 1222–1225 | §3.2a | from the abstract only |
| Latimer–MacDuffee(–Taussky) correspondence | §3.2a | from the abstract only |
| de la Llave; Marco–Moriyón, series in CMP (1987); Ann. of Math. (1986) | §3.2b | from the abstract only |
| Shub–Sullivan, ETDS 5 (1985) 285–289 | §3.2b | from the abstract only |
| de la Llave $\mathbb{T}^4$ example [dlL92]; Gogolev, arXiv 0804.3901 / JMD 2008 | §3.2b | from the abstract only |
| Williams (1973/74); Kim–Roush, J. AMS 1992 and Ann. of Math. 149 (1999) | §3.2c | from the abstract only |
| Ornstein isomorphism theorem | §3.2c | recollection, unverified |
| Peixoto; Morse–Smale classifications | §3.2c | recollection, unverified |
| Koenigs; Herman–Yoccoz | §3.2 | recollection, unverified |
| KAM reducibility (Eliasson school) | §2.3 | recollection, unverified |
| Meyerovitch, arXiv 1407.8343; Lind (1980s) | §3.3 | from the abstract only |
| Skew-to-product upgrade theorem at $C^1$; Krull–Schmidt for dynamical systems; product-centralizer genericity | §2, §3 | could not find |

The $C^1$ and $C^\infty$-resonant counterexamples in §2.1, the upgrade
proposition and proof plan in §2.2, and the filtration-matching sketch in §3.3
are ours (checked by direct computation here); they should graduate into
`counterexamples.md` / `identifiability.md` only after being written up and
asserted in tests, per repo convention.
