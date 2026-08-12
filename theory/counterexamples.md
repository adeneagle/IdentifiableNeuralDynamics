# Counterexamples

Every one of these is constructed in code and asserted in
`tests/test_counterexamples.py`, so a future "proof" of the original conjecture
will break the test suite rather than reach the draft.

---

## 1. Regrouping — the conjecture is false without minimality

*This is CLAUDE.md §3.1, reproduced here for completeness.*

**Construction.** $d = 4$, $F = \mathrm{diag}(\lambda_1, \lambda_2, \lambda_3, \lambda_4)$
with distinct $\lambda_i$, $K = 2$, $d_1 = d_2 = 2$. Let $P$ transpose
coordinates 2 and 3, and set $\tilde W = W P^{-1}$.

**Why it kills the conjecture.**

- $\tilde W \tilde z_t = W z_t$ exactly, so the observations are identical;
- $\tilde F = P F P^{-1} = \mathrm{diag}(\lambda_1, \lambda_3) \oplus \mathrm{diag}(\lambda_2, \lambda_4)$
  is modular with the same $K$ and the same block dimensions;
- $P$ is invertible and moves a coordinate across modules, so it is not
  $h_1 \oplus h_2$ up to module permutation.

**Measured.** On-block fraction exactly $0.5$ (chance level for $[2,2]$);
$400/400$ sampled invertible intertwiners mix modules.

**Cause.** $\mathrm{diag}(\lambda_1, \lambda_2)$ is *decomposable*: it splits
into two 1-D invariant subspaces. A decomposition into non-indecomposable blocks
is never unique, because the finer pieces can be regrouped freely.

**Not a linear artifact.** Replace the four eigenvalues by four independent 1-D
nonlinear maps $z \mapsto s_i \tanh(g_i z)/g_i$. The same coordinate
transposition regroups them into a different pair of 2-D modules with bit-for-bit
identical observations (`max |x̃ - x| = 2.2e-16`, `exp02` part (a)).

**Fix.** Require each $f_i$ to be indecomposable and prove uniqueness of the
*finest* decomposition. See `linear_case.md` Theorem L(i).

**Status.** Fixed in the linear case. In `exp02` it is also the negative control:
all three ways of pairing four coordinates reproduce the observations exactly,
and fitting from random restarts finds several of them.

---

## 2. Shared spectra — indecomposability alone is not enough

**This one is not in the original brief.** It matters because the §3.1 fix as
stated there ("require each $f_i$ to be dynamically indecomposable") is not
sufficient, and a proof built on that hypothesis alone would be wrong.

**Construction.** $F = J_2(\lambda) \oplus J_2(\lambda)$ with
$J_2(\lambda) = \begin{pmatrix} \lambda & 1 \\ 0 & \lambda \end{pmatrix}$,
$U_1 = \langle e_1, e_2 \rangle$, $U_2 = \langle e_3, e_4 \rangle$.

**Hypotheses.** Each block is a single Jordan block, hence **indecomposable** —
(A1) holds. But both have minimal polynomial $(t - \lambda)^2$, so they are not
coprime — (A2) fails.

**The alternative decomposition.** Put
$$U_1' = \langle e_1 + e_3,\; e_2 + e_4 \rangle, \qquad U_2' = \langle e_3, e_4 \rangle.$$
$U_1'$ is $F$-invariant: $(F - \lambda)(e_2 + e_4) = e_1 + e_3$ and
$(F - \lambda)(e_1 + e_3) = 0$. It is indecomposable (a $J_2$), $U_1' \oplus U_2' = \mathbb{R}^4$,
and the block dimensions are $[2,2]$ — the same shape. Yet $U_1' \neq U_1$, at a
principal angle of $45°$.

**Measured.** $\dim \mathrm{End}(F) = 8$, where (A1)+(A2) would force $4$;
$>100/200$ sampled invertible intertwiners mix modules.

**Fix.** Add (A2): pairwise coprime minimal polynomials, i.e. disjoint spectra.
`linear_case.md` Theorem L then goes through, and §4 there shows (A1) and (A2)
are independent — neither is removable.

**Nonlinear analogue.** Two modules that are *conjugate to each other* are the
obvious case. CLAUDE.md §3.6 already notes that §7.1 of the draft is too narrow:
failure needs only a shared factor (a common semiconjugate quotient), not full
conjugacy. This counterexample is the linear shadow of that.

---

## 3. The two-sided cocycle obstruction

**Also not in the original brief**, and the most consequential thing found while
building this repo. It says the §3.3 fix, *even executed correctly*, does not
reach the intended conclusion.

**Setup.** With $h = (h_1, \dots, h_K)$ and $M_{ij} := \partial h_i / \partial z_j$,
the corrected (cocycle) relation is

$$M_{ij}(F z)\, Df_j(z_j) = D\tilde f_i(h_i(z))\, M_{ij}(z),$$

and iterating gives $\|M_{ij}\| \to 0$ at rate
$\lambda_{\max}(f_j) - \lambda_{\min}(\tilde f_i)$. So

$$M_{ij} \equiv 0 \iff \lambda_{\max}(f_j) < \lambda_{\min}(\tilde f_i).$$

**The obstruction.** Block-diagonality of $h$ requires *both* $M_{12} \equiv 0$
and $M_{21} \equiv 0$. Once the modules are correctly matched, $\tilde f_i$ is
conjugate to $f_i$, so the two conditions read

$$\lambda_{\max}(f_2) < \lambda_{\min}(f_1) \quad\text{and}\quad \lambda_{\max}(f_1) < \lambda_{\min}(f_2).$$

Chaining them:

$$\lambda_{\max}(f_2) < \lambda_{\min}(f_1) \le \lambda_{\max}(f_1) < \lambda_{\min}(f_2) \le \lambda_{\max}(f_2),$$

a contradiction. **They can never both hold**, for any $K \ge 2$ and any pair
$i \neq j$.

**Measured** (`exp05` part 2c). The two rates are exact negatives of one another:

| $s_2$ | rate($M_{12}$) | rate($M_{21}$) | sum |
|---|---|---|---|
| 0.30 | $-1.15268$ | $+1.15268$ | $8.7\times10^{-10}$ |
| 0.50 | $-0.64185$ | $+0.64185$ | $8.7\times10^{-10}$ |
| 0.70 | $-0.30538$ | $+0.30538$ | $8.7\times10^{-10}$ |
| 0.90 | $-0.05407$ | $+0.05407$ | $8.7\times10^{-10}$ |

**Consequence.** A spectral gap buys a **triangular** $h$ — a skew product, in
which $h_1$ depends on $z_1$ alone while $h_2$ still depends on both — not a
block-diagonal one.

**Status — resolved, and not in our favour.** §5 below exhibits an explicit
conjugacy satisfying (B1)–(B4) that is triangular and not block-diagonal. So the
gap above is not a defect of the argument: **the target conclusion is false**,
and the triangular conclusion is sharp. The two candidate routes once listed in
`identifiability.md` §5 are both dead at $C^1$; see §5.2 there for the
hypotheses that would have to change instead.

---

## 4. Not a counterexample, but a trap: pointwise Jacobian spectra

CLAUDE.md §3.4 says Assumption 4 is unusable because pointwise spectra cross.
Measured, the situation is worse than "they cross somewhere": whether they cross
depends on **how much of state space the data visits**, so the assumption is a
property of the experiment, not of the system.

For the two-oscillator system of `exp05`:

| initial radius | max $\|z\|$ visited | min pointwise spectral distance | Lyapunov gap |
|---|---|---|---|
| 0.6 | 0.80 | $2.5\times10^{-1}$ | 0.3054 |
| 1.0 | 1.31 | $2.5\times10^{-1}$ | 0.3054 |
| 1.4 | 1.89 | $2.5\times10^{-1}$ | 0.3054 |
| 1.8 | 2.36 | $7.4\times10^{-5}$ | 0.3054 |
| 2.2 | 2.87 | $1.2\times10^{-3}$ | 0.3054 |

The pointwise quantity moves by a factor of $3400$; the Lyapunov gap moves by
$1.8\times10^{-10}$. Correspondingly, the pointwise Sylvester operator
$M \mapsto D\tilde f_1 M - M Df_2$ has $\sigma_{\min} = 2.3\times10^{-1}$ on the
small region and $8.3\times10^{-5}$ on the large one — i.e. it becomes singular,
and the draft's pointwise step admits nonzero $M$ exactly there.

This also sharpens CLAUDE.md §3.6: the support caveat is not only about *where*
$h$ is constrained, it decides **whether the hypotheses hold at all**.


---

## 5. Theorem B's conclusion is false under (B1)–(B4)

**Found in the literature pass**, and it settles §3 above: the triangular
conclusion is not an artifact of the proof technique, it is the truth.

**Construction.** $f_i(z_i) = \mu_i z_i$ on $\mathbb{R}$, with $0 < \mu_1 < \mu_2 < 1$,
so $F = \mathrm{diag}(\mu_1, \mu_2)$ and $\tilde F = F$. Set

$$h(z_1, z_2) = \left(z_1 + c\,\mathrm{sgn}(z_2)\,|z_2|^{p},\; z_2\right), \qquad p = \frac{\log \mu_1}{\log \mu_2}.$$

**It is an exact conjugacy.** $h(Fz)$ has first coordinate
$\mu_1 z_1 + c\,\mathrm{sgn}(z_2)\mu_2^{p}|z_2|^{p}$ and $F(h z)$ has
$\mu_1 z_1 + \mu_1 c\,\mathrm{sgn}(z_2)|z_2|^{p}$; these agree precisely because
$\mu_2^{p} = \mu_1$. Measured: $\max|h(Fz) - F(hz)| = 1.3\times10^{-15}$ over
$2\times10^4$ random points.

**Every hypothesis holds.**

| | |
|---|---|
| (B1) regularity | $p \approx 1.737 > 1$, so $\partial h_1/\partial z_2 = cp\,|z_2|^{p-1}$ is continuous, vanishes at $z_2 = 0$, and is bounded on compacts. $Dh$ is unit lower-triangular, $\det Dh \equiv 1$, so $h^{-1}$ is equally regular |
| (B2) indecomposability | blocks are 1-D, trivially indecomposable |
| (B3) matching | $\tilde f_i = f_i$, $\sigma = \mathrm{id}$ |
| (B4) separation | $\Lambda(f_1) = \{\log\mu_1\}$, $\Lambda(f_2) = \{\log\mu_2\}$, disjoint |

**And $h$ is not block-diagonal:** $\partial h_1/\partial z_2 = 1.22$ at $z_2 = 1$.

**No contradiction with Lemma C.** Lemma C needs the *oriented* gap
$\lambda_{\max}(f_2) < \lambda_{\min}(\tilde f_1)$, i.e. $\log\mu_2 < \log\mu_1$,
which is false here. The orientation that does hold, $\log\mu_1 < \log\mu_2$,
forces $M_{21} = 0$ — and indeed $h_2 = z_2$ exactly. The lesson is that **(B4)
as stated (disjoint spectra) is strictly weaker than what Lemma C consumes**,
and §3 above shows the oriented version cannot hold in both directions.

**Smoothness does not rescue it.** Take $\mu_1 = \mu_2^{m}$ for an integer
$m \ge 2$. Then $p = m$ and $h(z_1,z_2) = (z_1 + c z_2^{m},\, z_2)$ is a
polynomial diffeomorphism — $C^\infty$, and still an exact conjugacy (measured
$1.1\times10^{-14}$). This is the classical resonance phenomenon of
Poincaré–Dulac normal form theory, and it shows **cross-module non-resonance is
a necessary hypothesis**, not a technical convenience.

**Consequences.**

1. The triangular conclusion of `identifiability.md` §4.2 is **sharp**.
2. Both routes in the old §5 are dead at $C^1$; Route 2's
   $\omega$-limit-set conclusion is *optimal* — here the $\omega$-limit set is
   the origin, where $M_{21}$ does vanish, and $M_{21} \neq 0$ off it.
3. Any future Theorem B must add hypotheses, not sharpen the argument. The
   live candidate is $C^\infty$ (free in our model class) plus explicit
   cross-module non-resonance — see `identifiability.md` §5.2.

> `systems.triangular_conjugacy_counterexample()`, asserted across six tests in
> `tests/test_counterexamples.py`.

---

## 6. What "non-resonance" has to mean

§5 shows cross-module non-resonance is *necessary*. These two show the obvious
ways of stating it are *wrong*. Both are polynomial, hence $C^\infty$, so they
survive the regularity strengthening that rescues §5. Details and the tier
analysis in `route_a_assessment.md`.

### 6a. Pairwise non-resonance is not sufficient

Three 1-D modules with $\mu_1 = \mu_2\mu_3$ — take $(0.15, 0.50, 0.30)$. Then

$$h(z) = (z_1 + c\,z_2 z_3,\; z_2,\; z_3)$$

is an exact conjugacy: the cross term picks up $\mu_2\mu_3 = \mu_1$. Measured
$4.4\times10^{-16}$.

Every **pairwise** log-ratio is far from an integer — the minimum distance over
all six ordered pairs is $0.263$ — so a pairwise-stated hypothesis is satisfied.
The offending relation is the multi-index one $\log\mu_1 = \log\mu_2 + \log\mu_3$,
which no pairwise test sees.

**Non-resonance must be quantified over multi-indices across all modules:**
there is no $\lambda \in \Lambda_i$ with $\lambda = \sum_k m_k \nu_k$,
$|m| \ge 2$, the $\nu_k$ drawn from the full exponent multiset and at least one
from a module $\neq i$.

### 6b. Rotation angles give no protection

Module 1 a 2-D scaled rotation, spectrum $\{\log\rho, \log\rho\}$; module 2 1-D
at rate $\rho^2$. Then

$$h(x, y, z_2) = (x,\; y,\; z_2 + c(x^2 + y^2))$$

is an exact conjugacy **for every** $\theta$ — verified at
$\theta \in \{0, 0.4, 1.1, 2.7, \pi/2\}$, all $\le 8.9\times10^{-15}$ — because
$x^2 + y^2 = |w|^2$ is rotation invariant. The phases cancel.

So non-resonance must be checked on the **radial rates with multiplicity**. This
is not hypothetical for us: every `TwistBlock` has a *repeated* exponent
$\{\log s, \log s\}$, so it resonates with any module at rate $2\log s$.

**A live trap in our own parameters.** `two_oscillator_system(s=(0.95, 0.70))` is
safe ($\log 0.70/\log 0.95 = 6.95$), but $s = (0.95, 0.9025)$ is resonant —
$0.9025 = 0.95^2$ — and looks entirely innocuous. Guard test systems with
`spectra.is_cross_module_nonresonant`.

> `systems.multiindex_resonance_counterexample()`,
> `systems.repeated_exponent_resonance_counterexample()`, and the checker
> `spectra.cross_module_resonances`; eight tests in `tests/test_counterexamples.py`.

---

## 7. The rotation number does not pin the splitting

**New (2026-08-12).** §1's regrouping showed that a decomposition into
non-indecomposable blocks is not unique. This is its **oscillatory** analogue,
and it settles task 23 negatively: the invariant that was supposed to rescue the
two-oscillator case — where the Lyapunov spectrum provably cannot separate the
modules — is itself not identified.

**Construction.** Let $f_1, f_2$ be attracting limit cycles with frequencies
$\omega_1, \omega_2$ and no shear. In complex coordinates,

$$h(z_1, z_2) \;=\; \left(z_1 \cdot \frac{z_2}{|z_2|},\; z_2\right).$$

Then $h$ conjugates $f_1 \oplus f_2$ to $\tilde f_1 \oplus f_2$, where
$\tilde f_1$ is $f_1$ with $\omega_1 \mapsto \omega_1 + \omega_2$ — **still
modular, still two 2-D blocks, still attracting cycles.** The rotation numbers
move from $(\omega_1,\omega_2)/2\pi$ to $(\omega_1+\omega_2,\omega_2)/2\pi$.

*Verification.* Writing $f_i(z) = g(|z|)\,e^{i\omega_i}\hat z$ with
$\hat z := z/|z|$: since $\widehat{f_2(z_2)} = e^{i\omega_2}\hat z_2$,

$$h(F(z))_1 = g(|z_1|)e^{i\omega_1}\hat z_1 \cdot e^{i\omega_2}\hat z_2
= g(|z_1\hat z_2|)\,e^{i(\omega_1+\omega_2)}\widehat{z_1\hat z_2} = \tilde f_1(h(z)_1),$$

using $|z_1\hat z_2| = |z_1|$. Measured residual $6.7\times10^{-16}$;
$h^{-1}(w_1,w_2) = (w_1\overline{\hat w_2}, w_2)$ inverts to $8.9\times10^{-16}$;
measured rotation numbers $(0.0796, 0.2069) \to (0.2865, 0.2069)$, exactly as
predicted.

**Why this is a genuine counterexample and not a domain artefact.** $h$ is
singular only at $z_2 = 0$, which the basin excludes anyway, and on an annulus
$\Omega$ it is a diffeomorphism with $\sup\|Dh\| = \sup\|Dh^{-1}\| = 2.72$ — so
**(F1) holds**. The coupling is not marginal: $\|\partial h_1/\partial z_2\|$
has mean $1.03$ and minimum $0.55$ over $\Omega$.

**What it does and does not refute.**

- It does **not** touch Theorem F. (F3) fails outright for two cycles — the
  hulls are identical, chain gap $-0.9163$ — so §6 never applied here. The two
  statements are consistent, and that consistency is the point: §6.5's rider 2
  was right to call this case out.
- It **does** refute the conjecture in §4.4 and §6.5 that the rotation number
  could supply what the spectrum cannot. What a conjugacy preserves is the
  rotation *vector up to* $GL(2,\mathbb{Z})$ — the induced automorphism of
  $H_1(T^2)=\mathbb{Z}^2$ — not the individual rotation numbers. The
  construction realises $\left(\begin{smallmatrix}1&1\0&1\end{smallmatrix}\right)$.

**Shear obstructs it, and that is the one piece of good news.** With
$\beta_2 \neq 0$ the angle increment of $f_2$ depends on $r_2$, which the
regrouped module 1 cannot see, so no autonomous $\tilde f_1$ exists: the
residual jumps from $6.7\times10^{-16}$ to $9.3\times10^{-2}$ at $\beta_2=0.3$.
Shear in the *receiving* block is harmless (residual $10^{-15}$ at
$\beta_1 = 0.8$). So the obstruction is specifically **amplitude–phase coupling
in the donor**, which is generic in a real oscillator and absent from the
idealised one. Whether that genuinely restores identifiability, or merely
defeats this particular $h$, is not settled here. `TODO(gap)`

> `systems.torus_regrouping_counterexample()`; the quotient is computable with
> `spectra.rotation_lattice_margin`. Note $\beta = 0$ is `LimitCycleBlock`'s
> **default** and is what `exp14` part 4 uses, so the repo's own two-oscillator
> system is the vulnerable one — see the margin correction in CLAUDE.md task 40.
