# Route A assessment: $C^\infty$ + cross-module non-resonance

Answers the four questions posed for Route A (`approaches.md` §A,
`identifiability.md` §5.2, `literature.md` §2.2). Provenance tags as in
`literature.md`: **verified from the paper** / **from the abstract only** /
**recollection, unverified** / **could not find**; inline mathematics is tagged
**checked by direct computation here** and is self-contained. Numerical checks
were run this session with the conda interpreter; commutation errors quoted are
$\max|h(Fz) - \tilde F(hz)|$ over $2\times10^4$ uniform points in the unit
ball.

---

## Primary-source verification (2026-07-22)

The papers this file previously cited through secondary sources or
recollection are now held locally
(`C:\Users\alexa\Downloads\RequestedPapers\`) and have been read directly.
Checked this pass, with the headline outcome of each verification:

| Paper | Read | Headline outcome |
|---|---|---|
| Sternberg 1957, *Local contractions and a theorem of Poincaré*, Amer. J. Math. **79**, 809–824 | full text | **V2**: contraction linearization is Theorem 2 (finite smoothness $C^k$, $k > \log s/\log S$; standing semisimplicity assumption pp. 814–815); resonant polynomial normal form is Lemma 5 + Theorem 6; §1.1 below restated from the paper |
| Sternberg 1958, *On the structure of local homeomorphisms of Euclidean n-space, II*, Amer. J. Math. **80**, 623–631 | full text | **V2**: Theorem 1 (pp. 623–624) is the definitive map-level statement — full multi-index condition, $C^l \Rightarrow C^{\lambda(s;l)}$ with $\lambda \to \infty$ as $l \to \infty$, and $l=\infty \Rightarrow C^\infty$; hyperbolicity is *implied* by non-resonance; Borel's theorem is its Lemma 5. **The §A.2 Tier 1 / Tier 2 boundary is confirmed and does not move** |
| Chen 1963, *Equivalence and decomposition of vector fields about an elementary critical point*, Amer. J. Math. **85**, 693–722 | full text | **V1**: the Theorem of Equivalence (formal $\iff$ smooth) is **vector fields only** (p. 693); the paper's diffeomorphism lemma (Lemma 3.1 + Corollary, pp. 700–703) covers **saddles only** — the wedge construction assumes a nonempty unstable part (p. 697). **It does not close Tier 2 by citation in our discrete-time attracting setting.** The gap narrows to one named lemma, (FLAT-D) — see §2.4 |
| Nelson 1969, *Topics in Dynamics I: Flows*, §3 | image-only scan; §3 read page-by-page from rendered pages | cross-check: formal theory including the Jordan case and uniqueness (Thm 3, pp. 36–37) — clears this file's "recollection, unverified" ledger item; wave-operator flat linearization for attracting flows (Thm 6, pp. 42–45); $C^\infty$ Sternberg for flows (Thms 7 and 9, pp. 45, 50) |
| Khemakhem et al. 2020 (iVAE), AISTATS | full text | **V3**: assumption of variability pinned exactly ($nk+1$ points; $nk\times nk$ matrix of **natural-parameter** differences); with $u$-invariant components it fails **globally**, in both formulations (Thm 1 (iv) and Supp. E Thm 6 (iv)); no partial version exists in the paper → Route B needs a new theorem — confirms `approaches.md` §B cost 1 as written. One new warning (their Prop. 1) to propagate — see §6.1 |
| Qiu 2026, arXiv 2604.26950v2, *Weighted linearization of vector fields via a formal Moser trick* | full text | **V4**: weighted-linear normal form is triangular in the weight order (Def. 4.1 + Lemma 4.6(ii)) and by construction **retains** classically-resonant weighted-degree-0 terms (Def. 4.8) — canonical, **not** block-diagonal; the smooth half is literally "apply … Chen's theorem of equivalence" (Cor. 4.12), so vector-fields-only with the same discrete-time caveat as §2.4. Correct billing: canonical-form tool for the filtration (Route C), not a route to block-diagonality — see §6.2 |

Net effect on the verdicts: Q1's hollowness verdict and the §A.2 tier split
are **confirmed unchanged**. Q2's Tier 1 is upgraded — every external citation
is now primary-verified — at the cost of two new footnote-caveats recorded in
§1.1 (the contraction papers' standing semisimplicity assumption; the split
write-up of the $C^\infty$ contraction case). Q2's Tier 2 stays `TODO(gap)`,
with the gap **narrowed and named** (§2.4). All tags below that read "from the
abstract only" / "recollection, unverified" / "secondary" for these six
sources have been replaced with verified statements carrying theorem numbers
and pages.

---

Three findings up front, because they change the shape of the route:

1. Route A splits into **two tiers**. Tier 1 (full non-resonance, within and
   across modules): the proof closes with elementary inline arguments plus one
   classical citation (Sternberg), and (B2)/(B3) become *conclusions* — but the
   dynamics is then smoothly linearizable on the whole basin, so the coordinator's
   hollowness worry is **confirmed in a precise form** for this tier. Tier 2
   (cross-only non-resonance, within-module resonances allowed): genuinely
   nonlinear modules survive, (B2) becomes a decidable jet condition, but the
   proof needs a Chen-type jet-realization lemma. **Update (primary-source
   pass):** Chen 1963 has now been read in full and does *not* state the
   needed discrete-time contraction case — the debt is real but is now
   narrowed to one named lemma, (FLAT-D) in §2.4.
2. The non-resonance hypothesis **must be the full multi-index condition**.
   Two new $C^\infty$ counterexamples, both satisfying (B1)–(B4) and both
   verified to machine precision this session, defeat weaker readings: a $K=3$
   sum-resonance that is *pairwise* non-resonant, and a TwistBlock-relevant
   modulus-resonance where rotation phases cancel identically.
3. Route A as stated has an **unstated hypothesis**: $\Omega$ must contain an
   open set. Without it there is a cheap counterexample (§3.5), because the
   conjugacy equation on a thin $\Omega$ does not pin the cross-derivatives at
   all.

Throughout, $\mu$ denotes multipliers (eigenvalues of $Df(0)$), $\log\mu$
Lyapunov exponents, matching `counterexamples.md` §5.

---

## 1. Q1 — Is Route A hollow?

### 1.1 Sternberg's theorem, from the primary texts

An earlier revision of this section assembled the statement from three
agreeing secondary sources because the primary papers were paywalled. Both
Sternberg papers, Chen 1963, and Nelson's textbook treatment are now held
locally and have been read (Nelson from a page-rendered scan); everything
below is **verified from the paper** cited, with theorem numbers and pages.
Where the classical statements are *weaker or more hedged* than the
secondary-source composite previously recorded here, that is flagged — there
are two such places (semisimplicity; the split $C^\infty$ contraction
write-up), both footnote-level, neither moving the §A.2 tier boundary.

> **Sternberg 1957, Theorem 2 (p. 815) — contraction linearization, finite
> smoothness.** Let $T$ be a local $C^k$ transformation of $\mathbb{R}^n$
> fixing $0$ whose Jacobian eigenvalues $s_1, \dots, s_n$ satisfy (I)
> $|s_i| < 1$ (contraction) and (II) $s_i \neq s_1^{m_1}\cdots s_n^{m_n}$ for
> any non-negative integers $m_j$ with $\sum m_j > 1$, and let
> $k > \log s/\log S$ where $S = \max|s_i|$, $s = \min|s_i|$ (his condition
> (18)). Then some $R \in T^k$ linearizes $T$. Standing assumption for the
> analytic sections (pp. 814–815): $J(T)$ has **no multiple elementary
> divisors**, i.e. is diagonalizable — introduced "for simplicity" but used
> (the estimates run in an eigenvector coordinate system). Two remarks on
> p. 816 are now load-bearing for us: the conjugacy extends to any region
> that eventually enters the local chart via $R \mapsto L^{-r}RT^{r}$ —
> exactly the basin-extension pushforward §1.2 uses — and the same proof
> yields an **analytic** $R$ for analytic $T$.
>
> **Sternberg 1958, Theorem 1 (pp. 623–624) — the definitive hyperbolic map
> statement.** Let $T$ be a $C^l$ local homeomorphism of $E^n$ fixing the
> origin, $s_1, \dots, s_n$ the "(possibly complex or multiple)" eigenvalues
> of $J(T)$. If
> $$(*)\qquad s_i \neq s_1^{m_1}\cdots s_n^{m_n} \ \text{ for all non-negative integers } m_j \text{ with } \textstyle\sum m_j > 1,$$
> then there exist a neighborhood $N$ of the origin and a function
> $\lambda(s_1,\dots,s_n;\,l)$ such that a change of coordinates of class
> $C^{\lambda}$ defined in $N$ linearizes $T$; for fixed $s_i$,
> $\lambda(s_1,\dots,s_n;\,l) \to \infty$ as $l \to \infty$; and "in
> particular, for $l = \infty$, $R$ can be chosen to be of class
> $C^\infty$." Three further verified points. (i) Hyperbolicity is
> **implied, not assumed**: (*) forces $|s_i| \neq 0, 1$ (p. 624; a modulus-1
> eigenvalue resonates with its own conjugate pair). (ii) The paper proves
> the saddle case $0 < k < n$ and **delegates the pure contraction and
> expansion cases to the 1957 paper** (p. 624: "if $k = n$, the mapping $T$
> is a contraction, and if $k = 0$, $T^{-1}$ is. Both of these cases have
> been treated in [6]"). (iii) Its Lemma 5 (p. 627) is **Borel's theorem,
> with an explicit proof** — the Borel citation Tier 2 needs is therefore
> also primary-verified.
>
> **Two footnote-caveats (new; found only by reading the primary texts).**
> (a) *Semisimplicity.* 1957 assumes no multiple elementary divisors
> throughout its analytic sections; 1958's Theorem 1 *statement* admits
> multiple eigenvalues, but its §3 apparatus is again built for $L$ "without
> multiple elementary divisors" (p. 625). So for **maps** whose linear part
> has a nontrivial Jordan block, the letter of the proved statements does not
> cover the smooth step; Chen 1963 removes diagonalizability **for vector
> fields** (stated as the paper's contribution, p. 694), and Nelson's formal
> Theorem 3 covers the Jordan case at jet level (below). Consequence for us:
> Tier-1 "single-exponent" modules of TwistBlock / complex-pair type
> (semisimple) are fully covered; a genuine Jordan-block module carries a
> citation asterisk at the smooth step. (b) *The $C^\infty$ contraction
> case.* For an attracting map with $C^\infty$ data, 1958 Theorem 1's
> $l = \infty$ clause is the natural citation, but the contraction case of
> its proof lives in 1957, which is stated per finite $k$; a single
> $C^\infty$ conjugacy needs the standard bootstrap — Borel-realize the
> unique formal linearization (uniqueness: Nelson Thm 3), then run 1957's
> $L^{-n}R_0T^n$ scheme once, $k$-uniformly. Assembly-level and low-risk,
> but it is an assembly, not a quotation.
>
> **Resonant variant (verified).** Sternberg 1957 Lemma 5 (p. 819) is the
> formal resonant normal form for maps: after a formal change of coordinates
> only resonant monomials $A_{i,m}\,x^m$ with $s_i = s^m$ survive. Theorem 6
> (p. 820): a $C^r$ contraction, $r \ge k > \log s/\log S$ (semisimple
> standing assumption), is $C^r$-conjugate to **its own degree-$k$ Taylor
> polynomial** ("we can assume that $P$ is given by the image of $T$ in
> $F^k$"), and thence to the Lemma-5 normal form. For contractions, (I)
> forces every resonance order $\sum m_i < k$ (p. 819), so the normal form
> is polynomial and **non-resonance is a finite, numerically checkable list
> of inequalities** — confirming the inline bound previously recorded here
> ($\mu_{\min} \le |\mu^\alpha| \le \mu_{\max}^{|\alpha|}$). The saddle-map
> analogue is 1958 Theorem 3 (p. 630), where the surviving nonlinearity is a
> function of finitely many basic resonance monomials (his example:
> $x_1 = sx + xf(xy)$, $y_1 = s^{-1}y + yg(xy)$) — for saddles the resonance
> set is infinite, which is one more reason the attracting scope matters.
>
> **Chen 1963 (verified; the discrete-time analysis is §2.4).** "Theorem of
> Equivalence" (p. 693): two $C^\infty$ **vector fields** with an elementary
> critical point at $0$ are $C^\infty$-equivalent near $0$ **iff** their
> formal Taylor fields are formally equivalent. "Elementary" is defined on
> p. 693 for fields (every eigenvalue of the linearization has nonvanishing
> real part) and on p. 696 for diffeomorphisms ($J(T)$ has no eigenvalue of
> absolute value 1) — an attracting hyperbolic fixed point qualifies in both
> senses. Chen's Corollary 2 (p. 721) derives the vector-field Sternberg
> theorem under his (11.1) $\sum m_i\lambda_i \neq \lambda_j$ with **no
> diagonalizability assumption** — for fields, caveat (a) is discharged.
> (The printed constraint on the multi-index in (11.1) is partially
> illegible in our scan; the proof via his Proposition 10.1 — resonant basis
> $z^{m}z_k\,\partial/\partial z_k$ with $m_i \ge 0$, degree $\ge 2$ for
> $r > 0$ — pins it to the standard $m_i \ge 0$, $\sum m_i \ge 2$ reading.)
>
> **Nelson 1969 (textbook cross-check; §3 of the scan read page-by-page).**
> Theorem 3 (p. 36): formal linearization of maps under the multiplicative
> condition $\mu_i \neq \mu_1^{m_1}\cdots\mu_s^{m_s}$,
> $2 \le \sum m_j \le k$, eigenvalues "counted with their algebraic
> multiplicities"; the normalized formal linearization ($R_1 = I$) is
> **unique**; and the proof (p. 37) shows the homological operator's
> spectrum is $\{\mu_i - \mu^m\}$ for **all** linear parts — Jordan case
> included — by continuity from the diagonalizable case. This clears the
> "recollection, unverified" item consumed in §2.3. Theorem 6 (pp. 42–45):
> for an attracting flow with $X = X_0 + o(x^\infty)$, the wave operator
> $W_-x = \lim_{t\to\infty} U(-t)U_0(t)x$ exists, is a local $C^\infty$
> diffeomorphism conjugating $X$ to $X_0$, and satisfies
> $W_-x = x + o(x^\infty)$ — the flat-realization primitive, **for flows,
> with linear target**. Theorems 7 and 9 (pp. 45, 50): $C^\infty$ Sternberg
> linearization for flows, attracting and general hyperbolic (the eigenvalue
> condition alone suffices; it implies hyperbolicity, p. 39). Nelson also
> reproduces Sternberg's resonant counterexample (p. 32, citing [9, p. 812])
> and the flat-perturbed *rotation* non-example (pp. 39–40) showing flat
> rigidity **fails without hyperbolicity** — a clean negative control for
> §3.1, whose estimate consumes the contraction gap.

One corroborating detail: in dimension 1 the sharp statement (from the
introduction of arXiv 2212.13646, **verified from the paper**, full-text
rendering) is that a $C^r$ hyperbolic germ is $C^r$-linearizable for any
$r > 1$, and this **fails at $r = 1$** — consistent with our §5 counterexample
living at exactly $C^1$, though in the multi-module cross-derivative rather
than in 1-D regularity.

### 1.2 Does our hypothesis set imply linearization?

Split by tier. Write the hypothesis of `identifiability.md` §5.2 as: no
$\log\mu_{i,a} = \sum m_{j,b} \log\mu_{j,b}$ with integer $m \ge 0$,
$\sum m \ge 2$, and support touching more than module $i$.

**Tier 1 — if non-resonance is imposed on the *full* spectrum (within-module
relations excluded too).** Then Sternberg applies to $F$ directly and $F$ is
$C^\infty$-conjugate to $Df(0)$ near $0$; the conjugacy extends to the entire
basin by the standard pushforward $\Psi = L^{-n} \circ \Psi_{\mathrm{loc}} \circ F^n$
(checked by direct computation here: well-defined, independent of $n$, smooth,
because every basin point enters the local chart; and now also **verified from
the paper** — it is Sternberg's own extension remark, 1957 p. 816). **So yes: on the whole
domain where Tier-1 Route A applies, the dynamics is smoothly linear in
suitable coordinates.** The coordinator's worry is correct for this tier, and
it goes further than stated: under full non-resonance, a module whose spectrum
contains two distinct exponents linearizes to a *decomposable* linear block, so
(B2) forces every module to be a **single-exponent indecomposable block**
(Jordan or complex-pair type; a `TwistBlock` qualifies). Tier-1 Route A is
therefore exactly: *the finest spectral clustering of a smoothly-linearizable
contraction is identifiable through an unknown smooth decoder.*

**Tier 2 — cross-only non-resonance, within-module resonances allowed.** Then
$F$ is conjugate to a direct sum of *polynomial normal forms* $N_i$, not
necessarily linear. A module with multipliers $\{\mu, \mu^2\}$ and normal form
$(\mu z_a,\ \mu^2 z_b + c z_a^2)$, $c \neq 0$, is genuinely nonlinear: $c$ is a
formal conjugacy invariant (up to scale) and obstructs linearization. Here the
route does **not** collapse to the linear case; the identified per-module
content includes resonant coefficients — nonlinear fingerprints. This is
partial linearization only.

### 1.3 So is it hollow?

Not hollow — but its value must be *re-advertised*. Two points.

First, even in Tier 1 the theorem is not implied by `linear_case.md`. Theorem L
takes $h$ **linear** as input. In our model $h = \tilde g^{-1}\circ g$ is an
arbitrary smooth diffeomorphism a priori — the entire point of Theorem B
(§2, §3.5 split) is that the nonlinear *decoder*, not the dynamics, creates the
ambiguity. What Route A adds over Theorem A is precisely a rigidity statement:
*smooth conjugacies between non-resonant modular contractions are linear (hence
block-permutation)*. That kills the motivating LFADS-style reparameterization
ambiguity in this regime, which no linear-decoder theorem does.

Second, what is lost in Tier 1 is any claim about identifying **nonlinear
dynamics**: the within-module conjugacy class collapses to the multiplier
spectrum, and the honest headline is "Theorem A is stable under smooth
nonlinear reparameterization near a non-resonant attracting fixed point" — a
statement about robustness of linear structure, not about nonlinear structure.
Anyone reading `approaches.md` row "Nonlinear? yes, near a fixed point" should
understand it as: nonlinear *decoder* yes; nonlinear *dynamics* only in Tier 2
(resonant modules) or not at all (Tier 1).

On the §3.6 interplay: the "visited region shrinks toward the fixed point where
it is linearizable" concern is actually not the failure mode — the proof
*feeds* on linearizability near $0$, and data accumulating at $0$ is what pins
the jets (§2.1 below). The genuine support danger is different: **thin**
$\Omega$ (measure-zero, e.g. finitely many orbits), where the conjugacy
equation does not determine cross-derivatives at all — see §3.5. Route A needs
"$\Omega \supseteq$ an open set", i.e. full-dimensional initial conditions,
which the model's latent prior supplies but no current hypothesis states.

**Verdict for Route A.** Partially hollow, confirmed and made precise: in its
cleanest (Tier-1) form the dynamics is globally smoothly linearizable and the
theorem is a decoder-rigidity result — valuable, publishable, but to be
advertised as robustness of Theorem A, not identification of nonlinear
dynamics. The version with genuinely nonlinear content is Tier 2, which costs
one named lemma, (FLAT-D) — still unclosed after the primary-source pass
(§2.4).

---

## 2. Q2 — Does the §2.2 proof plan close?

Audit of the chain, restated as five steps. Steps (2) and (5) contain the only
analysis; (1), (3), (4) are bookkeeping, formal algebra, and a proved repo
theorem respectively.

### 2.1 Step 1 — from equivalence on $\Omega$ to jets at $0$

$h$ is smooth on a neighborhood (decoders smooth), but $h\circ F = \tilde F\circ h$
holds only on $\Omega$. If $\Omega$ contains an open set $U$, then it contains
the open forward images $F^n(U)$, which accumulate at $0$ (attracting fixed
point), and every partial-derivative identity obtained by differentiating the
conjugacy relation holds on the open set $\bigcup_n F^n(U)$ and extends to its
closure point $0$ by continuity. So **all jets of the conjugacy relation hold
at $0$**, even when $\Omega$ is a disconnected union of shells that never
covers a neighborhood of $0$. Checked by direct computation here. Two
consequences: (i) the germ machinery below is legitimate under the added
hypothesis $\mathrm{int}\,\Omega \neq \emptyset$; (ii) without that hypothesis
the whole route fails — not at this step but at the conclusion, see §3.5.

### 2.2 Step 2 — per-module normal forms, block-diagonal by construction

Each $f_i$ is separately conjugated to its normal form $N_i$ ($=L_i$ in
Tier 1) by some $\psi_i$; then $\psi = \psi_1\oplus\cdots\oplus\psi_K$
conjugates $F$ to $N = \bigoplus N_i$ and is block-diagonal *by construction* —
there is nothing to prove here beyond the per-module Sternberg statement
itself. The external dependency of the entire route is therefore exactly one
classical theorem, in its contraction case — **now verified from the primary
texts** (Sternberg 1957 Thm 2 / Lemma 5 / Thm 6; Sternberg 1958 Thm 1; §1.1),
subject to §1.1's two footnote-caveats: modules with semisimple linear parts
(all TwistBlock / complex-pair and distinct-eigenvalue modules) are fully
covered; a genuine Jordan-block module carries a citation asterisk at the
smooth step, and the single-$C^\infty$-conjugacy contraction statement is a
standard assembly (Borel + Nelson-Thm-3 uniqueness + the 1957 scheme) rather
than one quotable theorem. The weakest citation in Tier 1 is now that
assembly — demoted from "primary text unread" to "routine but not quotable in
one line".

### 2.3 Step 3 — the formal lemma (pure algebra; proof inline)

> **Formal lemma.** Let $H$ be an invertible formal series with
> $H \circ N = \tilde N \circ H$, where $N = \bigoplus_i N_i$,
> $\tilde N = \bigoplus_i \tilde N_i$ are (normal-form) products whose linear
> parts $L, \tilde L$ have module spectra pairwise disjoint, and each module is
> spectrally coherent as in §1.2 (Tier 1: single-exponent indecomposable
> blocks). Assume cross-module non-resonance in the full multi-index sense.
> Then $H = P_\sigma \circ (\text{block-diagonal formal series})$, with $\sigma$
> matching modules of equal spectrum.

*Proof, checked by direct computation here.* Degree 1: $H_1 L = \tilde L H_1$,
so $H_1$ maps each generalized eigenspace of $L$ to that of $\tilde L$ for the
same eigenvalue; module spectra are disjoint and (Tier 1) each module's
spectrum is a single point or conjugate pair, so eigenvalue clustering on both
sides coincides and $H_1 = P_\sigma(\bigoplus_i H_{1,i})$ — this is
`linear_case.md` Theorem L, already proved in the repo, plus the observation
that spectral subspaces respect the module partition. WLOG $\sigma = \mathrm{id}$
and pass to $\kappa = H_1^{-1}H = \mathrm{id} + \kappa_2 + \cdots$. Induction on
degree $m$: suppose $\kappa_{<m}$ block-diagonal. The degree-$m$ part of the
conjugacy equation reads
$\kappa_m \circ L - L \circ \kappa_m = R_m$, where $R_m$ is assembled from
$N$'s and $\tilde N$'s nonlinear (within-module) terms composed with
$\kappa_{<m}$ — all block-diagonal maps, so $R_m$ has **no cross-module
components**. The homological operator on degree-$m$ homogeneous maps has
spectrum $\{\mu^\alpha - \mu_{j}\}$ (standard; unchanged by Jordan structure,
which only makes the operator non-semisimple with the same spectrum — now
**verified from the paper**: Nelson, Theorem 3 proof, p. 37, establishes
exactly this for all linear parts by continuity from the diagonalizable
case, and adds uniqueness of the normalized solution). Cross-module
components of $\kappa_m$ pair a target $\mu_{i,a}$ with a monomial $\alpha$
supported outside module $i$: the diagonal entries $\mu^\alpha - \mu_{i,a}$ are
nonzero by **cross-module non-resonance**, so those components vanish.
Within-module components solve their equations (possibly non-uniquely at
within-resonances; the freedom stays inside blocks). Induction closes.
$\square$

Two remarks. (a) The lemma consumes non-resonance **only** for multi-indices
with cross support, which is why within-module resonances are harmless here.
(b) Complex eigenvalues: run the argument in the complexification and restrict
to real solutions; conjugate pairs enter $\alpha$ jointly — this is where the
§3.3 TwistBlock example lives, and the multi-index condition handles it
correctly while "angle" reasoning does not.

### 2.4 Step 4 — from formal to smooth: where the two tiers separate

The formal lemma says $h$'s Taylor series at $0$ is block-diagonal up to
$P_\sigma$. It remains to conclude that $h$ itself is.

**Tier 1.** $N = L$, $\tilde N = \tilde L$ linear. The formal series of
$k = \tilde\psi \circ h \circ \psi^{-1}$ is then *linear* (all higher
homological equations are homogeneous with invertible operator — no resonances
at all), i.e. $k = k_1 + (\text{flat})$. Write $\rho = k_1^{-1}k = \mathrm{id} + \text{flat}$,
which commutes with $L$ on the relevant forward orbits. Then, for any $N$ with
$\mu_{\max}^{N} < \mu_{\min}$,

$$\|\rho(x) - x\| = \|L^{-n}\big(\rho(L^n x) - L^n x\big)\|
\le \mu_{\min}^{-n}\, C_N \|L^n x\|^{N}
\le C_N \|x\|^{N} \left(\mu_{\max}^{N}/\mu_{\min}\right)^{n} \to 0,$$

so $\rho = \mathrm{id}$ and $k = k_1$ exactly — **checked by direct computation
here**, three lines, no citation. Hence $h = \tilde\psi^{-1} k_1 \psi$ is
block-diagonal up to $P_\sigma$ near $0$, and on all of $\Omega$ by
$h = \tilde F^{-n}\circ h \circ F^n$ (every point of $\Omega$ enters the local
chart; the propagation formula is an identity on $\Omega$ and preserves block
structure because $F, \tilde F$ are products). A single germ at $0$ fixes a
single $\sigma$, so disconnected $\Omega$ cannot mix permutations. **Tier 1
closes.** The complete dependency list: Sternberg (contraction case, cited),
Theorem L (proved in repo), the formal lemma and the flat estimate (inline),
$\mathrm{int}\,\Omega \neq \emptyset$ (new hypothesis).

**Tier 2.** $N_i$ genuinely polynomial. The same reduction needs one more
ingredient: a smooth **block-diagonal** conjugacy $\kappa_{bd}$ realizing the
block-diagonal formal series **exactly** (so that
$\rho = \kappa \circ \kappa_{bd}^{-1}$ is a *self*-conjugacy of $\tilde N$,
$\mathrm{id}+$flat, and the telescoping estimate — same three lines with
$\mathrm{Lip}$ constants in place of $\mu_{\min}^{-n}$ — finishes; exactness
matters: if $\kappa_{bd}$ conjugates only to infinite order, a drift term of
size $(s-\varepsilon)^{-n}(S+\varepsilon)^{n}$ enters the estimate and
destroys it, checked by direct computation here). Realization per module is
Borel's theorem — now **verified from the paper** (Sternberg 1958 Lemma 5,
p. 627) — **plus** the following, which is the load-bearing statement and
deserves a name:

> **(FLAT-D).** Two $C^\infty$ contraction germs of diffeomorphisms of
> $\mathbb{R}^{d_i}$ with the same $\infty$-jet at $0$ are $C^\infty$-conjugate
> by a diffeomorphism tangent to the identity to infinite order.

**Primary-source status of (FLAT-D) after reading Chen 1963 in full: not
closed by citation.** What the paper actually contains:

- The **Theorem of Equivalence** (p. 693) — formally conjugate iff smoothly
  conjugate — is stated and proved for **vector fields only**. Its proof
  reduces to diffeomorphism lemmas via time-one maps ($T = \exp X$, p. 710),
  not the other way around; invoking it for our maps would require $F$ to be
  the time-one map of a $C^\infty$ field, an unverifiable extra hypothesis.
  Our $F$ is a genuine diffeomorphism, so this route is closed.
- The paper's diffeomorphism-level result, **Lemma 3.1 with its Corollary
  (pp. 700–703)**, has *precisely the shape of (FLAT-D)*: $T, U$ local
  diffeomorphisms with an elementary fixed point and contact of infinite
  order at $0$, $T$ possessing **linear** stable/unstable manifolds
  $\mathbb{R}^n = V^+ \oplus V^-$; conclusion $U = \sigma T \sigma^{-1}$ with
  $\sigma$ smooth (seeded by Whitney extension), and flat-tangent to the
  identity — the part-(b) bounds
  $\|x\circ\sigma_k - x\|_r \le K_s(r)\rho^s$ hold for arbitrarily large $s$,
  and are established for all $r$ by the proof's double induction on $r$ and
  $k$ (the *statement's* printed range of $r$ is illegible in our scan; the
  proof and the $C^\infty$ conclusion are not). **But the underlying §2 wedge
  construction explicitly assumes the unstable index set is nonempty** (p. 697);
  the cone $\|x\|_+ = \|x\|_-$ and the crossing count $k(p)$ presuppose a genuine
  saddle. The pure contraction — **our case** — is exactly the degenerate
  configuration the construction excludes, and the same case Sternberg 1958
  delegates back to the 1957 paper (p. 624). Chen's own introduction describes
  Lemma 3.1 as a modification of Sternberg's wedge method for diffeomorphisms
  (p. 694), consistent with this reading.
- For contractions, the primary sources supply two adjacent results, neither
  quotable as (FLAT-D): Sternberg 1957 **Theorem 6** (p. 820) — a $C^r$
  contraction with semisimple linear part, $r \ge k > \log s/\log S$, is
  $C^r$-conjugate to its own degree-$k$ jet; hence two contractions with
  equal $k$-jets are $C^r$-conjugate — finite order, per-$r$, and **without
  jet control on the conjugacy**; and Nelson **Theorem 6** (pp. 42–45) — the
  exact (FLAT-D) statement *with* the flat-tangent conclusion, proved in
  full via the wave operator $W_- = \lim U(-t)U_0(t)$, but **for flows and
  with linear target**.

So the Tier-2 gap **narrows but does not close**. (FLAT-D) is now bracketed
on three sides by fully verified neighbours — saddle diffeomorphisms (Chen
Lemma 3.1), attracting flows (Nelson Thm 6), attracting diffeomorphisms at
finite regularity (Sternberg 1957 Thm 6) — and its proof is a routine
transposition of verified arguments: run Sternberg's
$\lim U^{-n}\circ R_0\circ T^{n}$ scheme against the nonlinear target $U$
with Nelson's derivative-convergence bookkeeping, or Chen's Lemma 3.1 scheme
with the cone deleted and $k(p) \equiv \infty$. But **as of the four primary
sources read, no quotable statement covers the discrete-time attracting
case**, and this project's rules rank a named gap above a strained reading.
Tier 2 therefore stays `TODO(gap)`, re-scoped: *prove (FLAT-D) in-house
(short, low-risk), or locate a source that states it* — candidates, all
still unread here: Sternberg 1959 ("structure of local homeomorphisms III",
Amer. J. Math. 81), Banyaga–de la Llave–Wayne, Belitskii's surveys. One
consolation is unconditional: a vector-field/flow variant of Theorem B
**is** closed by citation today, via Chen's Theorem of Equivalence directly.

**[The location attempt and the subsequent closure of the tangency clause
were carried out the same day — see the subsection below, which supersedes
the two sentences above: the existence half of (FLAT-D) is located and
verified, and the flat-tangency clause is closed from Chaperon's pages
(point 3 below), with a named elementary assembly layer for the
coordinator to check.]**

#### (FLAT-D): location attempt and closure (2026-07-22, follow-up passes)

First assignment: close (FLAT-D) by location, not by proof. Second
assignment (same day): settle the remaining flat-tangency clause, either
from Chaperon's construction (Path 1) or by restructuring Tier 2 to need
only existence (Path 2). Outcome: **Path 2 fails provably (point 2), Path 1
succeeds (point 3)**. Sources actually accessed this session: Chaperon's Astérisque volume (open access, Numdam;
PDF downloaded, statements read from **page images** — the scan's OCR is too
noisy for verbatim quoting); Banyaga–de la Llave–Wayne (BdlLW) via the
mp_arc 94-135 TeX source (downloaded, read directly). Sternberg 1959 III and
Belitskii: **could not access** (JSTOR / Russian Math. Surveys paywalls; no
open copy found) — nothing about their contents is asserted here.

**1. Located — the existence half of (FLAT-D), full strength, attracting
diffeomorphism germs included. Verified from the paper.**
M. Chaperon, *Géométrie différentielle et singularités de systèmes
dynamiques*, Astérisque **138–139** (1986), Soc. Math. France (open access
at Numdam).

- **(4.1), Théorème (Sternberg), p. 86** (proved in (4.2.3)). Paraphrase: for
  $C^\infty$ germs of $G$-actions ($G = \mathbb{R}$ or $\mathbb{Z}$) at fixed
  points of finite-dimensional manifolds, if the time-one map is hyperbolic then
  the two actions are $C^\infty$-conjugate iff they are formally conjugate. For
  $G = \mathbb{Z}$ an action germ *is* a diffeomorphism germ, so this is the
  diffeomorphism analogue of Chen's Theorem of Equivalence: **hyperbolic
  $C^\infty$ germs are $C^\infty$-conjugate iff formally conjugate.**
  (Transcription note: the book's hyperbolicity definition (3.1.2), p. 59, has a
  misprint — it reads "disc" where "circle" is meant — but the intended meaning
  is fixed by the stable/unstable-subspace framework and Hartman–Grobman on the
  same page. Contractions are hyperbolic.)
- **The workhorse — (4.2.3), Théorème 2, p. 107** (part (ii) on p. 108).
  Paraphrase of the setup: $Q = M \times E^+ \times E^-$ ($M$ compact Riemannian,
  $E^\pm$ Euclidean), with $W^\pm$ the stable/unstable factors and
  $\Sigma = W^+ \cap W^-$; $\varphi$ a $C^\infty$ germ along $\Sigma$ preserving
  $W^\pm$ and contracting in the stated norm sense (top rate $c_0^+ < 1$ on $E^+$,
  and $< 1$ for $\varphi^{-1}$ on $E^-$) — no semisimplicity, no non-resonance,
  and an adapted Euclidean norm turns any spectral radius $< 1$ into such a bound.
  Conclusion (i): there is a threshold map $t$ on $\mathbb{N}^*\cup\{\infty\}$
  such that for each $k$, any germ agreeing with $\varphi$ to order $t(k)$ along
  $\Sigma$ is $C^k$-conjugate to it — and the theorem states this **remains valid
  when $E^+$ or $E^-$ is trivial**.
- **Specialization to (FLAT-D)-existence:** take $M$ a point
  ($\Sigma = \{0\}$), $E^- = 0$ (trivial — the case the theorem explicitly
  includes), $k = \infty$. Then $t(\infty) = \infty$ — forced, not assumed:
  $t(\infty) < \infty$ would make finite-order tangency imply
  $C^\infty$-conjugacy, contradicting the fact that a resonant coefficient
  above the tangency order is a formal (hence $C^\infty$-) conjugacy
  invariant (§4.1), and contradicting the (4.1) iff-formal statement. Read:
  **two $C^\infty$ contraction germs of diffeomorphisms with the same
  $\infty$-jet at $0$ are $C^\infty$-conjugate.** Every (FLAT-D) hypothesis
  is met: pure contraction (trivial unstable factor, in print), equal
  $\infty$-jets, germ at a fixed point, no semisimplicity assumption.
  **Verified from the paper** (pp. 86, 105–108 read as images).

**2. Not located: the flat-tangency clause.** (FLAT-D) as stated here also
asks that the conjugacy be tangent to the identity to infinite order — that
is what the §2.4 telescoping endgame consumes (jet-controlled realization;
existence alone provably re-enters the centralizer circle described above).
Chaperon's jet-control statement is Théorème 2\,(ii) (pp. 107–108): the
conjugacy $\bar h$ can be built with $(j^k \bar h)|_{W^-}$ prescribed by a
seed $h_0$. But (ii)'s hypotheses require the jets of $\Psi$ and $\varphi$
to agree **along $W^+$ and $W^-$** (not merely at $\Sigma$), and in the
pure-contraction specialization $W^+ = Q$, so the hypothesis degenerates to
$\Psi = \varphi$; the "reste vrai si trivial" sentence is attached to (i)
only. So, as *statements*: existence located; tangency clause not located.

**3. Bridge note — candidate for the coordinator to check, not settled.**
The tangency clause appears recoverable from Chaperon's own displayed
construction in one page: his conjugacy is the $C^\infty$-limit
$H = \lim_n H_n$, $H_n = R(n)\circ\rho'(-n)$ ((4.1) Lemme 3, p. 87; proved
in (4.2.3), pp. 105–106), where Lemme 1/Lemme 2 build $R$ with
$j^\infty R = j^\infty(\psi_*\rho)$ along $E^+ \cup E^-$ and — when the
input germs already have equal $\infty$-jets at $0$ — with
$j^\infty_0\psi = \mathrm{id}$ (the displayed jets
$z^\pm = \lim j^\infty(\rho'(-n)\rho(n))$ evaluate to $\mathrm{id}$ at the
point $0$, since jets at a fixed point compose functorially and the two jets
coincide). Then $j^\infty_0 H_n = (\hat R)^n(\hat\rho')^{-n} = \mathrm{id}$
for every $n$, and $C^\infty$ convergence on compacts sends each fixed
derivative $D^\alpha H_n(0)$ — constant in $n$ — to $D^\alpha H(0)$, so
$j^\infty_0 H = \mathrm{id}$: the constructed conjugacy is flat-tangent to
the identity. **Single weakest step:** this reads the jet property off the
*construction* (Lemme 1's $z^\pm(0)$, Lemme 2(i)'s jet property at
$\Sigma$), not off a stated theorem — proof-mining of pp. 86–88 and
105–106, to be checked against the images before use.

**4. Other candidates accessed this pass.**
- **BdlLW** (Banyaga–de la Llave–Wayne, *Cohomology equations near
  hyperbolic points and geometric versions of Sternberg linearization
  theorem*; mp_arc 94-135 TeX read in full; published J. Geom. Anal. 6
  (1996) 613–649 — page numbers below refer to the preprint's theorem
  labels). Theorem (maindiff): $f, N$ $C^r$ diffeomorphisms of
  $\mathbb{R}^n$ fixing $0$, $D^if(0) = D^iN(0)$ for $i \le k < r-1$,
  $\mathrm{spec}\,Df(0)$ contained in an inner annulus
  $\{\lambda_-^{-1} \le |z| \le \lambda_+\}$, $\lambda_+ < 1$, union an
  outer annulus (a containment — satisfiable by pure contractions; no
  semisimplicity); conclusion: for every integer $\ell < kA - B$ (explicit
  $A, B$ from the annulus bounds) a $C^\ell$ conjugacy $h$ with $h(0)=0$,
  $Dh(0) = \mathrm{Id}$. **Finite regularity per $k$; no
  $C^\infty$/infinite-order corollary anywhere in the paper; no
  higher-order tangency of $h$ stated.** So BdlLW alone does not state
  (FLAT-D) — it strengthens the Sternberg-1957-Thm-6 shape (drops
  semisimplicity, adds $Dh(0)=\mathrm{Id}$ and explicit constants), and its
  introduction supplies the community provenance: it attributes these theorems to
  Sternberg 1959 III ([St3], as *sketched* there) and to Chen 1963 ([Ch], as a
  proof by other methods) — note "sketched", and note their reading of Chen is
  broader than the letter of Chen's saddle-only Lemma 3.1 as verified above. They also cite [I] =
  Ilyashenko–Yakovenko, *Finitely smooth normal forms...*, Russ. Math.
  Surveys (not accessed; per BdlLW it proves the Sternberg theorem for
  general diffeomorphisms by the deformation method — title says finitely
  smooth).
- **Sternberg 1959 III** (Amer. J. Math. 81, 578–604): **could not
  access.** BdlLW's "sketched in [St3]" is the only thing reported about
  it, and it is reported as *their* characterization.
- **Belitskii** (Russ. Math. Surveys 1978): **could not access.**

**5. Verdict: LOCATED** (existence half — the statement "equal
$\infty$-jets $\Rightarrow$ $C^\infty$-conjugate" for attracting
$C^\infty$ diffeomorphism germs, trivial unstable factor explicitly in
print): Chaperon, Astérisque 138–139 (1986), (4.2.3) Théorème 2(i) p. 107
with the trivial-factor clause, plus the (4.1) Théorème p. 86 (iff-formal,
$G = \mathbb{Z}$) — **verified from the paper**, open access. The
flat-tangency clause of (FLAT-D) is **not located as a statement** in any
source accessed; it is recoverable from the located construction by the
bridge note (point 3 above — proof-mining, flagged, one page to check).
Supersedes, in this file: the "no quotable statement covers the
discrete-time attracting case" sentence above, the (FLAT-D) rows of the §5
table/ledger and top note, and the §2.5 verdict's "not found there" clause
— the coordinator propagates; Tier 2's `TODO(gap)` shrinks to: *check the
point-3 bridge note against Chaperon's pp. 86–88, 105–106 (or prove the
three-line jet remark inline), then cite Chaperon + Borel and close.*

#### (FLAT-D): self-contained closure (2026-07-23, coordinator pass)

The point-3 bridge needs Chaperon's scanned construction, which is not
renderable in this environment. It is not needed: (FLAT-D)'s flat-tangent
conjugacy is produced by the classical telescoping (wave-operator) limit
**directly**, with no dependence on any single source's construction.

> **Construction.** For $C^\infty$ contraction germs $\Phi, \Psi$ at $0$ with
> $j^\infty_0\Phi = j^\infty_0\Psi$, set
> $$h \;=\; \lim_{n\to\infty} \Psi^{-n} \circ \Phi^n .$$

Three claims, the first two rigorous and elementary, the third a *classical
estimate* (verified numerically here, `exp07` and `tests/test_flat_tangency.py`):

1. **Conjugacy.** If the limit exists,
   $\Psi^{-1}\circ h\circ\Phi = \lim_n \Psi^{-(n+1)}\Phi^{n+1} = h$, so
   $h\circ\Phi = \Psi\circ h$. *(Exact; no analysis.)*
2. **Flat-tangent to the identity.** In jet composition at the fixed point,
   $j^\infty_0 h_n = (\widehat\Psi)^{-n}(\widehat\Phi)^{n}$ with
   $h_n := \Psi^{-n}\Phi^n$; since $\widehat\Phi = \widehat\Psi$ this is the
   identity jet for **every** $n$. A $C^\infty$ limit preserves each fixed
   derivative at $0$, so $j^\infty_0 h = j^\infty_0 h_n = \mathrm{id}$. *(Exact;
   the "$C^\infty$ limit of identity-jet maps has identity jet" step is the same
   one-liner the §2.4 endgame already uses.)*
3. **$C^k$ convergence** — the only analytic content. The increment is
   $h_{n+1}-h_n = -\,\Psi^{-(n+1)}\circ r\circ\Phi^{n}$, $r := \Psi-\Phi$ flat at
   $0$. With $S<1$ the top and $s<1$ the bottom contraction rate,
   $\|\Phi^n x\|\lesssim S^n$, $\|D^j\Psi^{-(n+1)}\|\lesssim s^{-(n+1)(j)}$, and
   flatness gives $\|r\|_{C^k,\,|y|\le S^n}\le C_N S^{nN}$ for every $N$. Faà di
   Bruno then bounds $\|h_{n+1}-h_n\|_{C^k}\lesssim (S^N/s^{\,k+1})^{n}$, summable
   once $N > (k+1)\,\log s/\log S$. So for each fixed $k$, a large enough flatness
   order forces geometric $C^k$ convergence — hence $h\in C^\infty$.

Claim 3 is exactly the Sternberg 1957 / Nelson Thm 6 distortion estimate
(Nelson proves it in full for the flow wave operator; the map version is the
same bookkeeping). It is **not re-derived line by line here**; what *is* done is
(i) isolate its single key inequality $S^N/s^{k+1}<1$, and (ii) verify the whole
construction numerically — `exp07` measures $C^0$ and $C^1$ convergence,
exactness of the conjugacy ($<10^{-15}$), and $(h(x)-x)/x^3\to0$, across linear
and nonlinear $\Phi$ and two flatness rates.

**Status of (FLAT-D): reduced to a classical estimate, no longer a research
gap.** Existence is located (Chaperon Thm 2(i), verified); the flat-tangent
strengthening is the telescoping construction above, whose only unwritten step
is a textbook $C^k$ distortion bound, numerically confirmed. The honest residual
is "write out the standard Faà di Bruno estimate", not "find or invent a lemma".
This is a **candidate for a referee to check at the level of a standard
argument**, not a flagged unknown — and it is bracketed by Nelson's fully-written
flow proof, of which it is the discrete transposition.

> **This whole subsection is the $C^\infty$ route, and it is now the *fallback*.**
> If $h$ is taken **real-analytic** — free with `tanh`-MLP decoders — the
> formal-to-smooth step is the identity theorem (a block-diagonal Taylor jet is a
> block-diagonal map), so (FLAT-D) and this $C^k$ estimate are **not needed at
> all**: the fixed-point case closes on Poincaré–Dulac + the identity theorem.
> See `identifiability.md` §5.4. (FLAT-D) and the construction above remain the
> route for a merely-$C^\infty$ decoder.

### 2.5 Step 5 — matching and dimensions

$\sigma$, $\tilde K = K$, and equal block dimensions come out of the formal
lemma's degree-1 step (Theorem L), not in. (B3) is a conclusion on this route,
as `identifiability.md` §5.2 hoped. Nothing further to prove.

**Verdict for Route A (updated after the primary-source pass).** The plan
closes at Tier 1 with one classical citation — **now verified from the
primary texts** (§1.1) — plus two short inline arguments, under one *added*
hypothesis ($\mathrm{int}\,\Omega \neq \emptyset$); the weakest link is no
longer an unread primary but the two §1.1 footnote-caveats (Jordan-block
modules; the assembled single-$C^\infty$-conjugacy contraction statement).
Tier 2 additionally rests on (FLAT-D) — **checked against Chen 1963 and not
found there for the discrete-time attracting case**; real debt, still
flagged `TODO(gap)`, but now named, bracketed by verified neighbours, and
one short lemma wide (§2.4). Nothing in either tier resembles the §3.3
argument-shift error class: every analytic step above is a one-inequality
estimate written out.

---

## 3. Q3 — Attempts to break Route A

Five attacks, in increasing order of success. The instinct behind them: the
hypothesis is stated on the linear part's spectrum, so look for structure the
spectrum does not see (flat terms), and for resonances a loose reading of
"cross-module non-resonance" does not exclude (many-term sums, conjugate
pairs, thin data).

### 3.1 Flat cross-terms — attack fails, provably

Candidate: $h = (z_1 + \eta(z_2),\ z_2)$ with $\eta$ flat at $0$ (all
derivatives vanish), hoping flatness evades a hypothesis stated on
eigenvalues. Commutation with $L$ forces the twisted equation
$\eta(L_2 z) = L_1\,\eta(z)$, and flatness kills it: for $N$ with
$\mu_{2,\max}^N < \mu_{1,\min}$,

$$\|\eta(z)\| = \|L_1^{-n}\eta(L_2^n z)\| \le \mu_{1,\min}^{-n} C_N (\mu_{2,\max}^n\|z\|)^N \to 0,$$

so $\eta \equiv 0$. Checked by direct computation here; note the estimate works
in **both** orientations (any $N$ large enough, regardless of which module is
faster) — flat rigidity, unlike Lemma C, is orientation-free. The same
telescoping closes the general flat case in §2.4. Numerically: the bound factor
at $N=3$, $\mu_1 = 0.3$, $\mu_2 = 0.5$ falls to $3.7\times10^{-23}$ by $n=60$.
Flat perturbations of $F$ itself change nothing either: Sternberg's hypotheses
see only $Df(0)$ and smoothness.

### 3.2 $K = 3$ sum-resonance — breaks every pairwise formulation

Take three 1-D modules, $\mu_1 = 0.15$, $\mu_2 = 0.5$, $\mu_3 = 0.3$, so that
$\mu_1 = \mu_2\mu_3$ exactly, and

$$h(z_1, z_2, z_3) = (z_1 + c\,z_2 z_3,\ z_2,\ z_3).$$

Measured this session: commutation error $5.6\times10^{-17}$; polynomial, hence
$C^\infty$. All of (B1)–(B4) hold: 1-D modules are indecomposable, spectra are
disjoint singletons $\{\log 0.15\}, \{\log 0.5\}, \{\log 0.3\}$, matching is
the identity. And it is **pairwise non-resonant**: all six ratios
$\log\mu_i/\log\mu_j$ are at distance $\ge 0.26$ from the nearest integer
(measured), so no relation $\mu_i = \mu_j^m$ holds for any pair and any $m$.
The resonance is irreducibly three-term: $\log\mu_1 = \log\mu_2 + \log\mu_3$.

Consequences. Any formulation of the hypothesis as pairwise power conditions
("$\mu_i \neq \mu_j^m$", "no integer ratio of exponents") is **false as a
sufficient condition** — this example satisfies it and defeats
block-diagonality. `identifiability.md` §5.2's sum form
"$\lambda_i = \sum_j m_j\lambda_j$" is the right shape **provided** the sum is
read over the full spectrum with multiplicity and arbitrary cross-support $m$,
not over one module at a time. `literature.md` §2.2's multi-index clause
excludes this example (support $\{2,3\} \not\subseteq \{1\}$), as it must.
Also worth porting to `systems.py` next to the §5 counterexample: it is the
$K \ge 3$ witness that the resonance lattice, not pairwise arithmetic, is the
right object.

### 3.3 Conjugate-pair modulus resonance — the TwistBlock trap

Take module 1 a rotation–scaling block with multipliers $\rho e^{\pm i\theta}$
(a `TwistBlock`-type linear part; $\theta = 1.1$, $\rho = 0.7$ measured) and
module 2 one-dimensional with $\mu_2 = \rho^2$. Then

$$h(x, y, z_2) = \big(x,\ y,\ z_2 + c\,(x^2 + y^2)\big)$$

is a $C^\infty$ conjugacy — commutation error $4.4\times10^{-16}$ measured —
because $x^2 + y^2 = |w|^2$ picks up exactly $\rho^2$ under the twist,
**for every $\theta$**: the phases of $\mu$ and $\bar\mu$ cancel in the
monomial $w\bar w$. Every hypothesis of (B1)–(B4) holds, including (B2): a
rotation–scaling block with $\theta \notin \{0,\pi\}$ has no real invariant
line, hence is $\mathbb{R}$-indecomposable. Spectra $\{\log\rho\}$ (double) and
$\{2\log\rho\}$ are disjoint. Lemma C's oriented gap holds toward module 2
($2\log\rho < \log\rho$), kills $M_{12}$, and indeed $h_1 = (x,y)$; the
surviving block $M_{21} = 2c(x, y) \neq 0$ is exactly fast-receives-slow.

Consequences. (i) This is a *third* $C^\infty$ counterexample family under
(B1)–(B4), with a 2-D indecomposable module — the existing §5 family is 1-D —
worth porting. (ii) The practical warning for `exp05`-type systems: **the twist
does not protect against resonance.** The resonance condition must be checked
on radial contraction rates (Lyapunov exponents with multiplicity), because
conjugate phases cancel in exactly the monomials that matter; a TwistBlock
pair with $s_2 = s_1^2$ is resonant no matter the angles. Checking angles or
complex multipliers pairwise would wrongly certify it safe. (iii) The
exponent-form (sum over exponents with multiplicity) hypothesis of §5.2
correctly excludes it: $\log\mu_2 = \log\rho + \log\rho$, cross-support.

### 3.4 Trying to evade the full multi-index condition — no success

Under the full condition (all $\mu_{i,a} \neq \mu^\alpha$, $|\alpha| \ge 2$,
$\alpha$ with cross support; equivalently on exponents with multiplicity,
which is slightly stronger and equally generic), I found no counterexample and
have positive reasons none exists at Tier 1: the formal lemma (§2.3) forces
cross-jets to vanish and flat rigidity (§3.1) kills the remainder — both fully
written out. Mixed resonances $\mu_{1,a} = \mu_{1,b}\,\mu_2$ (own-and-other
support, invisible to naive "$i$ vs $j$" conditions) are excluded by the
cross-support clause; attempts to realize them with an *indecomposable* module
1 force either equal moduli within the pair (conjugate/Jordan, reducing to
§3.3's family, pairwise-visible in exponents) or a decomposable linear module
(violating (B2)) — so within (B1)–(B4) plus full non-resonance I could not
construct anything. Checked by direct computation here at the level of case
analysis; not a theorem, but the failed constructions are exactly the ones the
formal lemma predicts must fail. The honest residual risk is Tier 2's
realization lemma (§2.4), which is a proof gap, not a counterexample
candidate.

### 3.5 Thin $\Omega$ — a genuine counterexample to the route as stated

Nothing in (B1)–(B4) or §5.2 requires $\Omega$ to have interior. Take a single
orbit $\Omega = \overline{\{F^n z^\ast\}} = \{F^n z^\ast\} \cup \{0\}$ (compact,
invariant, inside the basin — (B1) holds). The conjugacy equation constrains
$h$ **only at the orbit points**; its derivatives there are free in all
transverse directions. Concretely, prescribe $h(F^n z^\ast) = F^n z^\ast$ and
extend smoothly with $Dh$ at each orbit point chosen non-block-diagonal
(e.g. $Dh = \mathrm{id} + \varepsilon E_{12}$ along the orbit, mollified);
$h\circ F = F\circ h$ holds on $\Omega$ exactly, all regularity and spectral
hypotheses hold, $F$ can be taken linear diagonal non-resonant — and $h$ is
not block-diagonal at any orbit point. Checked by direct computation here (the
construction is an exercise in extension, not dynamics; a numerical instance
is a few lines if wanted). So with thin data the conclusion is **false**, not
merely unprovable. The fix is the hypothesis already identified in §2.1:
$\Omega \supseteq$ an open set — equivalently, initial conditions with
full-dimensional support, which the model's latent prior provides and which
`exp05` satisfies. This also sharpens `counterexamples.md` §4's theme: the
support caveat decides not only whether hypotheses hold but whether the
conclusion is even determined by the data.

**Verdict for Route A.** The route survives every attack **in its full
multi-index, open-support formulation** — and I could not break that version.
Every weaker formulation in current repo prose is broken explicitly: pairwise
non-resonance (§3.2), angle/complex-pairwise checking (§3.3), no support
hypothesis (§3.5). The two new counterexamples are $C^\infty$, satisfy
(B1)–(B4), verified to $\le 4.4\times10^{-16}$, and are ready to port to
`systems.py` / `tests/test_counterexamples.py`.

---

## 4. Q4 — A narrow (B2) for disjoint-spectra factors

### 4.1 Inside Route A's regime, (B2) is derivable

Define, for germs at an attracting hyperbolic fixed point: $f_i$ is
*indecomposable* if it is not $C^\infty$-conjugate to a nontrivial direct
product. Then, checked by direct computation here at the jet level:

- **Tier 1 (full non-resonance):** $f_i$ is indecomposable **iff** its linear
  part $Df_i(0)$ is indecomposable in the `linear_case.md` sense. ($\Leftarrow$)
  If $f_i \cong g_a \oplus g_b$ then the degree-1 jet splits $Df_i(0)$.
  ($\Rightarrow$) If $Df_i(0) = A \oplus B$, Sternberg linearizes $f_i$ (1957
  Thm 2 / 1958 Thm 1, §1.1 — with §1.1's caveat (a) if the split linear part
  is non-semisimple) and the linear splitting is a smooth splitting. Consequently the **finest**
  decomposition of $F$ is the finest invariant splitting of $Df(0)$, whose
  uniqueness is `linear_case.md` Theorem L(i) — (B2) and its uniqueness
  transfer wholesale, and open problem 3 is *solved in this regime*. As noted
  in §1.2, the answer is restrictive: legal modules are single-exponent
  indecomposable blocks.
- **Tier 2 (within-module resonances):** the equivalence **fails**, and the
  failure is exactly the interesting case. The module
  $f(z_a, z_b) = (\mu z_a,\ \mu^2 z_b + c z_a^2)$ has decomposable linear part
  $\mathrm{diag}(\mu, \mu^2)$, but for $c \neq 0$ it is formally — hence
  smoothly — indecomposable: any product of two 1-D contraction germs is
  formally conjugate to $\mathrm{diag}(\mu, \mu^2)$ (1-D germs have no
  resonances, so each factor linearizes formally), while $c$ is a formal
  invariant of $f$ (the degree-2 homological equation at the resonance
  $\mu^2 = \mu^2$ has vanishing operator on that coefficient; linear changes
  rescale $c$ by $a_b/a_a^2 \neq 0$ but cannot kill it). Four lines, inline.
  So in Tier 2, **nonlinear indecomposability is decided by the resonant
  coefficients of the normal form**: a module is indecomposable iff its
  within-module resonance-coupling graph (nonzero resonant coefficients
  linking sub-blocks) is connected. The lemma below proves this **at degree 2
  for distinct eigenvalues** and states the general conjecture; it is the
  missing "nonlinear indecomposability test" of `approaches.md` (cross-cutting
  §3), at fixed points — certify the fitted module's *normal-form coefficients*,
  not its linearization. Built as `normalform.resonance_coupling_components`,
  wired into `selection.block_nonlinear_certificate`, certified in `exp09`.

#### The resonance-coupling graph (§4.1a)

Let $f$ be a germ at an attracting fixed point with semisimple linear part $L$
whose eigenvalues $\lambda_1,\dots,\lambda_d$ are **distinct**, so each
eigen-coordinate is its own linear sub-block. By Poincaré–Dulac $f$ is
(formally, and — contraction case — smoothly) conjugate to a normal form
$N = L + \sum_{(i,\alpha)\,\mathrm{res}} c_{i,\alpha}\, z^\alpha e_i$ retaining
only resonant monomials ($\lambda_i = \lambda^\alpha := \prod_j \lambda_j^{\alpha_j}$,
$|\alpha|\ge2$). Build the graph $G(N)$: nodes $\{1,\dots,d\}$; join $i,j$ when
some resonant monomial with $c_{i,\alpha}\neq0$ has
$\{i\}\cup\mathrm{supp}(\alpha)\ni i,j$.

> **Lemma (coupling graph).** *(Both directions, all degrees.)* If $G(N)$ has
> components $C_1,\dots,C_r$, then $N=\bigoplus_s N_{C_s}$ is a direct product
> of $r$ factors, one per component — so $r\ge2 \Rightarrow f$ decomposable.
> *(Degree 2, distinct eigenvalues.)* Conversely, if the **degree-2** graph
> $G_2(N)$ is connected then $f$ is indecomposable.

*Proof.* **Disconnected $\Rightarrow$ decomposable** (any degree). If a monomial
$z^\alpha e_i$ had $\{i\}\cup\mathrm{supp}(\alpha)$ meeting two components it
would edge them together, contradiction; so every resonant monomial is supported
within a single component, and grouping coordinates by component writes
$N=\bigoplus_s N_{C_s}$. $\square$

**Connected $\Rightarrow$ indecomposable** (degree 2). The key fact is that the
**degree-2 resonant coefficients are invariant** under the normal-form symmetry
group, so the presence of an edge is coordinate-independent. Two normal forms of
$f$ differ by (i) a linear map in the centraliser $Z(L)$ — for distinct
eigenvalues, diagonal scalings $z_i\mapsto a_i z_i$ — and (ii) a near-identity
$\Phi=\mathrm{id}+\phi_2+\cdots$. Under (ii), the degree-2 part of $\Phi N\Phi^{-1}$
is $N_2 + \mathcal{L}_L\phi_2$ where $\mathcal{L}_L\phi_2=\phi_2\circ L-L\circ\phi_2$
is the homological operator, whose spectrum is $\{\lambda^\beta-\lambda_k\}$ — and
it is **zero on precisely the resonant monomials**, so it cannot alter a resonant
coefficient. Under (i), $c_{i,\alpha}\mapsto c_{i,\alpha}\,a_i/a^\alpha$, a nonzero
rescaling. Hence $c_{i,\alpha}=0$ is coordinate-independent, and $G_2(N)$ is a
genuine invariant of $f$.

Now suppose $f$ is decomposable, $f\cong g\oplus g'$. The splitting induces an
$L$-invariant splitting of the tangent space; as the eigenvalues are distinct,
$L$-invariant subspaces are coordinate subspaces, so the splitting is a partition
$\{1,\dots,d\}=A\sqcup B$. In the split coordinates the normal form is
$N_A\oplus N_B$, which has **no** resonant coefficient coupling $A$ to $B$; by the
invariance just shown, none in *any* normal-form coordinates, so $G_2(N)$ has no
$A$–$B$ edge and is disconnected. Contrapositive: $G_2(N)$ connected $\Rightarrow$
$f$ indecomposable. $\square$

Two things the degree-2 restriction costs, both honest and both open
(`TODO(gap)`). **(a) Higher degrees.** A degree-$m$ resonant coefficient ($m\ge3$)
is changed by lower-degree $\phi$'s feeding in through Faà di Bruno, so its
vanishing is *not* individually invariant; the claim that graph **connectedness**
is still invariant at all degrees — the general conjecture — needs the
normalising group's action on the whole coefficient tuple, not one coefficient at
a time. So $G_2$ connected is conclusive for indecomposability, but $G_2$
*disconnected* is only suggestive: a higher-degree edge can reconnect it. **(b)
Repeated eigenvalues.** The centraliser is then larger (block scalings, and
permutations of equal-eigenvalue coordinates) and the sub-blocks are the linear
indecomposable pieces rather than single coordinates; the argument should carry
over per sub-block but is not written here.

**Consequence for the learning machinery — a fixed applied to a real gap.** The
degree-2 half of this lemma is exactly the criterion
`selection.block_nonlinear_certificate` needs, and it exposed a defect in the
first (task-25) version, which flagged *any* resonant coupling rather than graph
connectedness. Those agree for a two-sub-block module (the `ResonantNodeBlock`
witness) but diverge at three: a module $(\mu z_0,\ \mu^2 z_1+c z_0^2,\ \nu z_2)$
with $\nu$ non-resonant couples only $\{0,1\}$ and **splits off $z_2$** — it is
decomposable as $\{0,1\}\oplus\{2\}$ — yet "any coupling" reports it
indecomposable. That is the *over-reporting* direction, and it is the dangerous
one: a decomposable module fits a split perfectly, so the fit-quality gate of
`exp06` cannot catch it, and the partition search would keep a coarser-than-finest
partition. The certificate now takes connected components and is correct on this
case (`exp09` part 6; `tests/test_selection.py`).

### 4.2 Outside the fixed-point regime

**Could not find** — unchanged from the literature pass: no Krull–Schmidt
theorem for smooth or topological dynamical systems; uniqueness of direct
factorization open even for expansive shifts (Meyerovitch/Lind, from the
abstract only). The disjoint-spectra restriction does not rescue it off fixed
points with the tools in hand: on a forward-invariant $\Omega$ only the fast
side of the spectral splitting is canonical (`literature.md` §2.1's curved
slow-foliation example — a linear diagonal map with two distinct invariant
"slow product structures"), so even *stating* uniqueness of the factorization
requires either two-sided invariance (excluded for attractor basins) or the
germ situation of §4.1. A uniqueness claim "up to filtration-respecting
regrouping" may survive — **recollection-level speculation, unverified**, and
I would not build on it.

**Verdict for Route A.** Within Route A's regime, (B2) stops being an
assumption: derived and characterized in Tier 1, reduced to decidable jet
algebra (one tractable lemma, flagged) in Tier 2. Outside the regime, assume
it; the literature will not supply it.

---

## 5. Summary table for the decision document

Updates `approaches.md` §A's row (this file does not edit it):

| Item | Status after this assessment |
|---|---|
| Hypothesis form | must be **full multi-index** non-resonance over exponents with multiplicity; pairwise forms refuted (§3.2, §3.3); finitely many checkable inequalities for contractions (§1.1) |
| Unstated hypothesis | $\mathrm{int}\,\Omega \neq \emptyset$ (full-dimensional initial conditions); without it, false (§3.5) |
| (B2) | **derived** in Tier 1; decidable jet condition (one tractable lemma) in Tier 2 (§4.1) |
| (B3) | conclusion, not hypothesis (§2.5) — as §5.2 hoped |
| Proof status | Tier 1: **closes; all external citations primary-verified** (Sternberg 1957 Thm 2 / 1958 Thm 1, §1.1), two footnote-caveats (semisimplicity; assembled $C^\infty$ contraction case); formal lemma and flat rigidity inline (§2.3, §2.4). Tier 2: blocked on **(FLAT-D)** — checked against Chen 1963, **not stated there for discrete-time attracting germs** (saddle-only Lemma 3.1); `TODO(gap)`, narrowed to one short lemma (§2.4) |
| Hollowness | Tier 1 is decoder-rigidity over smoothly-linearizable dynamics — real but must be advertised as such; Tier 2 keeps genuinely nonlinear modules (§1.3) |
| New counterexamples to port | $K=3$ sum-resonance; TwistBlock $|w|^2$ resonance; both $C^\infty$, both (B1)–(B4), errors $\le 4.4\times10^{-16}$ (§3.2, §3.3) |
| Practical warning | non-resonance for TwistBlocks must be checked on radial rates; angles cannot help ($w\bar w$ cancels phases) (§3.3) |

Provenance ledger for external claims in this file (rewritten 2026-07-22
after the primary-source pass; see the note at the top):

- Sternberg statement — **verified from the papers**: 1957 Thm 2 (p. 815),
  Lemma 5 (p. 819), Thm 6 (p. 820); 1958 Thm 1 (pp. 623–624), Thm 3
  (p. 630). Two footnote-caveats recorded in §1.1 (semisimplicity; split
  $C^\infty$ contraction write-up).
- Finite-smoothness variant — **verified from the papers**: explicit
  threshold $k > \log s/\log S$ (1957, (18)); loss function
  $\lambda(s_1,\dots,s_n;l) \to \infty$ (1958 Thm 1). The Sell 1985 citation
  is no longer needed for this.
- Chen's theorem — **verified from the paper**: Theorem of Equivalence
  (p. 693, **vector fields only**); elementary critical point defined
  pp. 693 (fields) and 696 (diffeomorphisms), attracting hyperbolic fixed
  points qualify; diffeomorphism flat-conjugacy Lemma 3.1 + Corollary
  (pp. 700–703, **saddles only** — $\Sigma^+ \neq \emptyset$ assumed p. 697);
  nonlinear-target field version Lemma 6.1 (pp. 708–710). The discrete-time
  attracting-germ realization **(FLAT-D) remains unverified — `TODO(gap)`,
  load-bearing for Tier 2 only** (§2.4).
- Borel's theorem — **verified from the paper** (Sternberg 1958 Lemma 5,
  p. 627, with proof).
- Homological-operator spectrum under Jordan structure — **verified from the
  paper** (Nelson Thm 3 proof, p. 37, continuity argument; plus uniqueness
  of the normalized formal linearization).
- Poincaré non-resonance condition — **verified from the papers** (primary:
  Sternberg 1957 (II) / 1958 (*); formal attribution to Poincaré's 1879
  thesis per Qiu §1.1, whose refs. [19], [20] are exactly the two Sternberg
  papers verified here).
- Banyaga–de la Llave–Wayne — still **from the abstract only**; no longer
  load-bearing anywhere in this file (kept only as a candidate source for
  (FLAT-D), §2.4).
- 1-D sharp linearization (Yoccoz attribution) — unchanged: **verified from
  the paper** (arXiv 2212.13646, secondary).
- Khemakhem et al. 2020, Thm 1 / Thms 2–3 / Supp. E Thm 6 / Prop. 1 / proof
  step (31)→(32) / B.2.3 — **verified from the paper** (§6.1).
- Qiu 2026, Thm 1.1 / Def. 3.1 / Def. 4.1 / Lemma 4.6 / Def. 4.8 / Thm 4.9 /
  Lemma 4.11 / Cor. 4.12 / Example 4.10 — **verified from the paper** (§6.2).
- Everything else inline and self-contained.

---

## 6. Annexe: adjacent verifications done with the primary-source pass

Recorded here because the deliverable touches only this file; the propagation
notes at the end state exactly what should change elsewhere, for the
coordinator to apply.

### 6.1 V3 — iVAE's assumption of variability (Khemakhem et al. 2020), for Route B

**The exact statement (verified from the paper).** Model: $x = f(z) + \epsilon$
with $f$ injective, and a conditionally factorial exponential-family prior
(their (7))

$$p_{T,\lambda}(z|u) \;=\; \prod_{i=1}^{n} \frac{Q_i(z_i)}{Z_i(u)}
\exp\Big[\sum_{j=1}^{k} T_{i,j}(z_i)\,\lambda_{i,j}(u)\Big].$$

**Theorem 1** assumes: (i) the noise characteristic function vanishes only on
a measure-zero set; (ii) $f$ injective; (iii) the $T_{i,j}$ differentiable
a.e. with $(T_{i,j})_{j \le k}$ linearly independent on any positive-measure
subset; and (iv) the **assumption of variability** — there exist $nk+1$
distinct points $u^0, \dots, u^{nk}$ such that the $nk \times nk$ matrix

$$L = \big(\lambda(u^1)-\lambda(u^0),\; \dots,\; \lambda(u^{nk})-\lambda(u^0)\big)$$

is invertible. Conclusion: identifiability up to
$T(f^{-1}(x)) = A\,\tilde T(\tilde f^{-1}(x)) + c$ with $A$ invertible
($\sim_A$). Note carefully: $L$ is a matrix of **natural-parameter
differences** $\lambda(u^l)-\lambda(u^0) \in \mathbb{R}^{nk}$ — *not* of
sufficient-statistic differences — and the count is $nk+1$ points ($n$
latents $\times$ $k$ statistics, plus one pivot). The learned side's
$\tilde L$ is *not* assumed invertible; it is derived. Theorems 2 ($k \ge 2$;
$T$ twice differentiable, $f$ with all second-order cross derivatives) and 3
($k = 1$; non-monotonic $T_{i,1}$) upgrade $\sim_A$ to $\sim_P$
(block-permutation). Supp. E, Theorem 6, is an alternative formulation whose
(iv) is: $\lambda$ differentiable and $J_\lambda(u^0)$ invertible at a single
point $u^0$.

**Where invertibility is consumed (verified).** Proof of Theorem 1, Step II:
subtracting the pivot equation gives
$L^T\,T(f^{-1}(x)) = \tilde L^T\,\tilde T(\tilde f^{-1}(x)) + b$ (their
(31)), then both sides are multiplied by $(L^T)^{-1}$ to reach (32). That one
linear solve is the entire role of (iv).

**The Route B question — answered, and it confirms `approaches.md` §B cost 1
as written.** If a latent block $z^B$ is $u$-invariant, the rows
$\lambda_{i,j}(\cdot)$ for $i \in B$ are constant in $u$, so the
corresponding $k\cdot\dim z^B$ rows of $\lambda(u)-\lambda(u^0)$ vanish
**identically for every $u$**. By the paper's own characterization of (iv)
(Supp. B.2.3: it holds iff $h(U)$, $h(u) = \lambda(u)-\lambda(u^0)$, is not
confined to a proper subspace of $\mathbb{R}^{nk}$), $h(U)$ then lies inside
a fixed coordinate subspace of dimension $nk - k\dim z^B$: **no choice of
points can satisfy (iv), and Theorem 1 as stated delivers nothing — not even
about the $u$-varying block.** The Jacobian formulation fails identically
($J_\lambda$ has those rows zero at every $u^0$), and Theorems 2–3 inherit
the failure since they start from (32). **No partial or block version exists
anywhere in the paper** (checked: main text §4, Supp. B, Supp. E). What the
proof structure does offer: Steps I–II are rank-agnostic up to the solve,
and a rank-$r$ variant of the solve yields only "the projection of $T$ onto
$\mathrm{row}(L)$ is an affine function of *all* of $\tilde T$" — a
subspace-level statement about the $u$-varying block whose block-diagonal
refinement is precisely the argument Route B would have to supply.
**Verdict: Route B needs a (partial iVAE) theorem, not a corollary** — the
`TODO(gap)` in `approaches.md` §B cost 1 stands, now grounded in the exact
hypothesis and the exact proof step that fails.

**One new warning to propagate (verified: their Proposition 1, §4.2).** For
$k = 1$, $T_{i,1}(z) = z$, and Gaussian (or exponential) base measure —
i.e. **mean-modulated Gaussian latents** — the linear indeterminacy $A$
*provably cannot* be reduced to a permutation (rotational invariance).
Route B's behavioural modulation must therefore move **variances (or use
$k \ge 2$ statistics)**, not conditional means alone; a mean-only
behavioural signal caps Route B at $\sim_A$-level identifiability no matter
what partial theorem is proved.

### 6.2 V4 — placing arXiv 2604.26950 (Qiu 2026) correctly

The proposed placement — Qiu as a valid modern citation for the standard
multi-index condition, with its own weighted contribution a
descriptive/canonical-form tool for the filtration rather than a new path to
block-diagonality — is **confirmed from the paper** on all three points.

**(a) Triangular structure of the weighted-linear normal form — confirmed.**
Weights are positive integers sorted increasingly (Def. 3.1); the
weighted-linear approximation $X^{[0]}$ is the weighted-degree-0 part of the
field (Def. 4.1), spanned by monomial fields $x^\alpha\,\partial/\partial x_i$
with $\langle w, \alpha\rangle = w_i$ (grading (3.2)–(3.3)). Hence a
component of weight $w_i$ contains only linear terms in same-weight
coordinates or monomials in **strictly lower-weight** coordinates with
weights summing to $w_i$; a lower-weight component can never contain a
higher-weight coordinate. Lemma 4.6(ii) (p. 28) makes the linear shadow
explicit: for admissible $X$, $DX(0)$ is block **upper triangular** in the
weight blocks, and $DX^{[0]}(0)$ is block **diagonal** (the above-diagonal
blocks are deleted). With weights chosen as the integer rate-ratios — legal
exactly when the module log-rates are rationally proportional; the paper's
Example 4.10 ("happy coincidence" $\lambda = w$) is this configuration —
fast-contracting coordinates may carry weighted-linear monomials in slow
coordinates, never conversely: triangular in the rate order, as inferred.

**(b) No block-diagonality in the resonant case — confirmed.** Def. 4.8
declares $(i, \alpha)$ a resonance with respect to $w$ only when
$\langle\lambda,\alpha\rangle = \lambda_i$ **and**
$\langle w,\alpha\rangle > w_i$. A classical cross-module resonance aligned
with the weighting has $\langle w,\alpha\rangle = w_i$ — weighted degree
0 — hence is **exempt from the weighted non-resonance hypothesis and
survives, by construction, inside $X^{[0]}$**. Theorem 4.9 (formal) and
Corollary 4.12 (smooth) conjugate $X$ *to* $X^{[0]}$; when cross-module
resonances are present, the canonical target **contains the resonant
cross-term**. The weighted machinery organizes the surviving triangular
terms into a canonical form; it cannot remove them and offers no route to
block-diagonality.

**(c) Vector-fields-only, with the discrete-time caveat inherited —
confirmed, and sharper than expected.** Theorem 1.1 is stated for formal or
smooth **vector fields**; the smooth half is Corollary 4.12 (p. 31), whose
one-line proof just chains Theorem 4.9, Lemma 4.11, and Chen's theorem of
equivalence. The paper's smooth statement therefore rides on Chen 1963
(its ref. [6]) and carries **exactly the §2.4 transfer caveat** — nothing in
it applies to diffeomorphisms by citation. (Lemma 4.11 is a nice verified
side-fact: weighted non-resonance already implies hyperbolicity.)

**Correct billing (confirmed).** Cite Qiu as a *modern statement* of the
standard multi-index condition, with primary credit to Poincaré (its ref.
[18], the 1879 thesis) and Sternberg (its refs. [19], [20] — exactly the two
papers verified in §1.1); and as a **canonical-form tool for the filtration
reading (Route C)** — its weighted-linear normal form is a canonical
triangular form adapted to the rate order, i.e. a normal form *for the
object Route C identifies* — not as a path to Route A's block-diagonality.
The provenance loop closed cleanly: what this file previously took from
Qiu's preliminaries as a secondary source is now verified against the same
primary texts Qiu cites.

### 6.3 Propagation notes (for the coordinator — nothing outside this file was edited)

1. `approaches.md` §A.2: **no tier-boundary change.** Optionally update the
   "Proof status" row to: Tier 1 "closes; citations primary-verified, two
   footnote caveats (semisimple linear parts; assembled $C^\infty$
   contraction case)"; Tier 2 "one load-bearing gap, now named (FLAT-D)".
   §A.3's closing line ("The Sternberg statement rests on three agreeing
   *secondary* sources…") is now obsolete and can read: "Sternberg verified
   from the primary texts (route_a_assessment §1.1); remaining work is
   Tier 2's (FLAT-D)."
2. `approaches.md` §B cost 1: confirmed as written; add the Proposition-1
   warning — behavioural modulation must affect variances (or use $k \ge 2$
   sufficient statistics); mean-only modulation of Gaussian latents caps
   identifiability at $\sim_A$ regardless of any partial theorem.
3. `CLAUDE.md` §3.7 ("classical Sternberg/normal-form theory is reported to
   close Theorem B near an attracting fixed point"): still fair for Tier 1
   with the §1.1 caveats; for Tier 2 the honest phrasing is "closes modulo
   one named lemma (FLAT-D), whose proof pattern is verified in the adjacent
   flow and saddle cases but which no source read so far states for
   attracting diffeomorphism germs".
4. Anywhere Qiu (arXiv 2604.26950) is cited in support of Route A:
   re-bill per §6.2 — multi-index condition citation and Route C
   canonical-form tool; not evidence toward block-diagonality.
