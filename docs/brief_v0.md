# CLAUDE.md — Identifiability of Modular Nonlinear Latent Dynamics

Context for AI assistants working in this repo. Read fully before editing theory
files, writing proofs, or running simulations.

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

---

## 2. Status

The draft (`identifiability.md`) contains a conjecture and proof sketch.
**The conjecture as written is false and the proof sketch has a real error.**
Do not build on either without applying the fixes in §3.

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

### 3.2 The spectral assumption is on the wrong pair (BLOCKING)

The Sylvester step needs $\mathrm{spec}(D\tilde f_1) \cap \mathrm{spec}(Df_2) = \emptyset$
— transformed system 1 against original system 2. The draft assumes disjointness
only among the $f_i$. In the §3.1 counterexample $\mathrm{spec}(\tilde f_1) = \{\lambda_1,\lambda_3\}$
overlaps $\mathrm{spec}(f_2) = \{\lambda_3,\lambda_4\}$, which is exactly why it fails.
This cannot be bootstrapped without first knowing the module correspondence.

**Fix:** a matching lemma pairing indecomposable blocks across representations by
conjugacy invariants, established *before* the spectral hypothesis is used.

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

### 3.4 Assumption 4 is not usable as stated

"Jacobian spectra generically distinct" — pointwise Jacobian spectra vary with $z$
in a nonlinear system and will generically cross somewhere.

**Fix:** replace with disjoint **dichotomy (Sacker–Sell) / Lyapunov spectra**.

### 3.5 The linear decoder already collapses $h$ to linear

If $W, \tilde W$ both have full column rank $d$ and $W z_t = \tilde W \tilde z_t$,
then $\tilde z_t = (\tilde W^+ W) z_t$, so $h \in GL(d)$ — forced, before dynamics
enter. All nonlinear conjugacy machinery in the draft is therefore unnecessary for
the model as specified, and the motivating nonlinear reparameterization ambiguity
does not arise at all.

**Fix:** split into two theorems — linear decoder (base case) and nonlinear
decoder $x_t = g(z_t)$ (the setting the motivation actually requires).

### 3.6 Secondary issues

- **§7.1 too narrow.** Failure needs only a *shared factor* (common semiconjugate
  quotient), not $f_1 = f_2$ or full conjugacy.
- **Noise dropped.** $\epsilon_t$ appears in the model but equivalence is defined
  pathwise as $W z_t = \tilde W \tilde z_t$. Should be equality of observation
  *distributions*, with the attendant latent-scale / noise-variance tradeoffs.
- **Support.** All conclusions hold only on the closure of the visited region. If
  trajectories collapse to a low-dimensional attractor, $h$ is unconstrained off it.
  State explicitly — this is the regime real recordings are in.
- **Autonomy is the main disanalogy with LFADS.** LFADS latents are input-driven;
  shared inputs $u_t$ couple modules and destroy block-diagonality.

---

## 4. Next steps, in order

1. **Restate the theorem with indecomposability** (§3.1). Prove uniqueness of the
   finest modular decomposition.
2. **Settle the linear case completely.** With $h \in GL(d)$, determine when
   $A F A^{-1}$ block-diagonal forces $A$ block-permutation. Finishable quickly;
   serves as base case and sanity check.
3. **Swap in dichotomy-spectrum separation** (§3.4) and redo the Section 5 proof
   as a cocycle argument (§3.3).
4. **Prove the matching lemma** closing the $\tilde f$ vs $f$ gap (§3.2).
5. **Move to a nonlinear decoder** $x_t = g(z_t)$, $g$ an injective immersion.
   Steps 1–4 build the machinery this requires.
6. **Numerical falsification before more theory.** Simulate two 2D nonlinear
   oscillators with separated exponents; fit unconstrained vs. modular models;
   check whether the recovered partition matches. **Include the §3.1 regrouping
   counterexample as a negative control** — the fit must be non-unique there. If
   it isn't, the assumptions are hiding something.
7. **Perturbation result.** Add $\epsilon$-coupling between modules; characterize
   how the recovered partition degrades. Target: "recovered partition within
   $O(\epsilon)$ of truth provided spectral gap exceeds $C\epsilon$." Prioritize
   over sharpening the exact conditions in §9 of the draft — an exact theorem that
   evaporates at $\epsilon > 0$ will not support the interpretive claims.
8. **Position against the literature.** Nonlinear ICA with temporal structure
   already gives identifiability from temporal dependence: Hyvärinen & Morioka
   (time-contrastive, permutation-contrastive), Hälvä & Hyvärinen (HMM variant),
   Khemakhem et al. (iVAE). Be explicit about what modular *dynamics* adds beyond
   conditioning on a time index.

---

## 5. Scope note for interpretation claims

Even on success, within-module $h_i$ is an arbitrary diffeomorphism. What is
identified is the **partition** plus each $f_i$'s **conjugacy class** — fixed-point
structure, attractor topology, Lyapunov spectrum. Not coordinates. Nothing here
licenses reading "motor primitive" off a latent axis. Keep claims in the draft
calibrated to this.

---

## 6. Conventions

- $K$ = number of modules; $d_i$ = dimension of module $i$; $d = \sum_i d_i$.
- $F$ = full latent transition; $f_i$ = module transition; $W$ = linear decoder.
- $h$ = candidate reparameterization; $\tilde{\cdot}$ denotes the alternative
  representation.
- $M := \partial h_1 / \partial z_2$ throughout the cross-derivative argument.
- Flag any claim that depends on an unproved lemma inline as `TODO(gap)`.
