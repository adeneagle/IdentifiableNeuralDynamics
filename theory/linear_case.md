# The linear case, settled

CLAUDE.md §4 step 2. This is not a warm-up: by §3.5 a full-column-rank linear
decoder already forces $h \in GL(d)$ before dynamics enter, so for the model as
originally specified the linear case *is* the whole question.

Everything here is proved. The numerical certificate is
`experiments/exp01_linear_base_case.py`; the code is `src/idyn/linear.py`.

---

## 1. Setup

Let $F \in GL(d, \mathbb{R})$. Give $\mathbb{R}^d$ the structure of an
$\mathbb{R}[t]$-module $M_F$ via $t \cdot v = Fv$. Then:

- an $F$-invariant subspace is a submodule;
- a **modular decomposition** is $\mathbb{R}^d = U_1 \oplus \cdots \oplus U_K$
  with each $U_i$ $F$-invariant, and we write $f_i = F|_{U_i}$;
- $U \neq 0$ is **indecomposable** if $U = V \oplus W$ with $V, W$ $F$-invariant
  forces $V = 0$ or $W = 0$.

Write $\chi_F = p_1^{m_1} \cdots p_r^{m_r}$ for the factorisation of the
characteristic polynomial into distinct monic irreducibles over $\mathbb{R}$
(so each $p_j$ is linear or an irreducible quadratic), and

$$P_j := \ker p_j(F)^{m_j}$$

for the **primary components**.

---

## 2. Two lemmas

**Lemma 1 (primary decomposition).** $\mathbb{R}^d = P_1 \oplus \cdots \oplus P_r$,
each $P_j$ is $F$-invariant, and the projections $\pi_j$ onto $P_j$ along the
others are polynomials in $F$.

*Standard.* The $p_j^{m_j}$ are pairwise coprime, so a partial-fraction identity
$\sum_j a_j(t) \prod_{k \neq j} p_k(t)^{m_k} = 1$ exhibits each $\pi_j$ as a
polynomial in $F$. $\square$

The consequence that matters is that $P_j$ is determined by $F$ alone — no
choice is involved. This is the sense in which the primary decomposition is
*canonical*, and it is the property the $U_i$ must be shown to inherit.

**Lemma 2.** Every indecomposable $F$-invariant subspace lies inside a single
primary component.

*Proof.* Let $U$ be $F$-invariant. Each $\pi_j$ is a polynomial in $F$, so
$\pi_j(U) \subseteq U$, and therefore
$U = \bigoplus_j \pi_j(U) = \bigoplus_j (U \cap P_j)$,
a decomposition of $U$ into $F$-invariant subspaces. If $U$ is indecomposable,
exactly one summand is nonzero. $\square$

---

## 3. The theorem

> **Theorem L (linear rigidity).**
> Let $\mathbb{R}^d = U_1 \oplus \cdots \oplus U_K$ be $F$-invariant. Assume
>
> * **(A1)** each $f_i = F|_{U_i}$ is indecomposable;
> * **(A2)** the minimal polynomials of the $f_i$ are pairwise coprime
>   (equivalently: the spectra $\mathrm{spec}(f_i) \subset \mathbb{C}$ are
>   pairwise disjoint).
>
> Then:
>
> **(i)** $\{U_1, \dots, U_K\}$ is exactly the set of primary components of $F$.
> In particular $K = r$ and the decomposition is **unique** — it is the finest
> modular decomposition and there is no other with indecomposable blocks.
>
> **(ii)** Let $S \in GL(d)$ with $\tilde F = S F S^{-1}$, and suppose
> $\mathbb{R}^d = \tilde U_1 \oplus \cdots \oplus \tilde U_L$ is
> $\tilde F$-invariant with every $\tilde F|_{\tilde U_j}$ indecomposable. Then
> $L = K$ and there is a permutation $\sigma$ with $S\, U_i = \tilde U_{\sigma(i)}$.
> Equivalently, in bases adapted to the two decompositions,
> $$S = P_\sigma \, (S_1 \oplus \cdots \oplus S_K).$$

*Proof of (i).* By Lemma 2, $U_i \subseteq P_{j(i)}$ for a unique $j(i)$. If
$j(i) = j(i')$ with $i \neq i'$, then $\min(f_i)$ and $\min(f_{i'})$ are both
powers of $p_{j(i)}$, hence not coprime, contradicting (A2). So $i \mapsto j(i)$
is injective. Since $P_j = \bigoplus_i (P_j \cap U_i) = \bigoplus_{i : j(i) = j} U_i$
and every $P_j \neq 0$, it is also surjective. Hence it is a bijection and
$U_i = P_{j(i)}$. $\square$

*Proof of (ii).* Each $S^{-1}\tilde U_j$ is $F$-invariant, because
$F S^{-1}\tilde U_j = S^{-1}\tilde F \tilde U_j \subseteq S^{-1}\tilde U_j$; and
it is indecomposable, because $S^{-1}$ is an isomorphism of
$\mathbb{R}[t]$-modules from $(\tilde U_j, \tilde F)$ to $(S^{-1}\tilde U_j, F)$.
These subspaces decompose $\mathbb{R}^d$. By Lemma 2 and part (i), each
$S^{-1}\tilde U_j \subseteq P_{k(j)} = U_{k(j)}$. Summing,
$U_i = \bigoplus_{j : k(j) = i} S^{-1}\tilde U_j$, and $U_i$ is indecomposable by
(A1), so exactly one $j$ has $k(j) = i$. Writing $\sigma$ for the inverse
bijection gives $S^{-1}\tilde U_{\sigma(i)} = U_i$. $\square$

**Two things worth noting about (ii).** First, the matching multiset of block
dimensions is a *conclusion*, not a hypothesis — the fix proposed in CLAUDE.md
§3.1 assumed it, and it does not need to be assumed. Second, nothing was
assumed about $\tilde F$ beyond indecomposability of its blocks; in particular
(A2) for $\tilde F$ comes for free.

---

## 4. Sharpness: neither hypothesis can be dropped

**Dropping (A1) — CLAUDE.md §3.1.** $F = \mathrm{diag}(\lambda_1, \dots, \lambda_4)$
with distinct $\lambda_i$, $U_1 = \langle e_1, e_2\rangle$,
$U_2 = \langle e_3, e_4\rangle$. Each block is decomposable. The transposition
$P = (2\,3)$ satisfies $P F P^{-1} = \mathrm{diag}(\lambda_1,\lambda_3) \oplus
\mathrm{diag}(\lambda_2,\lambda_4)$, still modular with the same shape, and $P$
is not a block permutation. Measured: on-block fraction exactly $0.5$.

**Dropping (A2) — new, see `counterexamples.md` §2.** $F = J_2(\lambda) \oplus J_2(\lambda)$.
Both blocks are indecomposable, so (A1) holds, but they share an eigenvalue.
$\langle e_1 + e_3,\, e_2 + e_4 \rangle$ is $F$-invariant, indecomposable, and
complemented by $\langle e_3, e_4\rangle$, giving a genuinely different
decomposition of the same shape. Measured: $\dim \mathrm{End}(F) = 8$ against
the $4$ that (A1)+(A2) would predict.

So (A1) and (A2) are independent and both necessary.

---

## 5. The matching lemma, in the linear case

CLAUDE.md §3.2 objects that the Sylvester step needs
$\mathrm{spec}(D\tilde f_1) \cap \mathrm{spec}(Df_2) = \emptyset$ — the
*transformed* system 1 against the *original* system 2 — and that this cannot be
bootstrapped without already knowing which module corresponds to which.

Theorem L(ii) resolves this in the linear case, and resolves it in the right
order: the correspondence $\sigma$ is established by the module-theoretic
argument, **before** any spectral hypothesis about the tilde system is used.

> **Corollary M (linear matching).** Under (A1)+(A2), $S$ restricts to an
> isomorphism $U_i \to \tilde U_{\sigma(i)}$ intertwining $f_i$ and
> $\tilde f_{\sigma(i)}$. Hence $\tilde f_{\sigma(i)}$ is *similar* to $f_i$, and
> in particular $\mathrm{spec}(\tilde f_{\sigma(i)}) = \mathrm{spec}(f_i)$.
> The hypothesis §3.2 asks for is therefore automatic once $\sigma$ is known.

This is the template the nonlinear matching lemma should follow: get the
correspondence from a decomposition-uniqueness argument, then read off the
spectral hypothesis. The nonlinear version is still open — see
`identifiability.md` §6, `TODO(gap)`.

---

## 6. The computable certificate

$\mathrm{Hom}(F, \tilde F) = \{S : SF = \tilde F S\}$ is the complete solution
set of the linear identifiability problem — the admissible reparameterisations
are exactly its invertible elements. It is the null space of
$S \mapsto SF - \tilde F S$, computable exactly, so nothing needs to be fitted.

Under (A1)+(A2), Theorem L gives $\mathrm{Hom}(f_i, \tilde f_j) = 0$ unless
$j = \sigma(i)$, hence

$$\dim \mathrm{Hom}(F, \tilde F) = \sum_i \dim \mathrm{End}(f_i),$$

and every element — invertible or not — is supported on matched blocks. This
gives a sharp two-part test, implemented as `linear.intertwiner_space` and
checked in `exp01`:

| case | (A1) | (A2) | $\dim\mathrm{Hom}$ | predicted | invertible elements mixing modules |
|---|---|---|---|---|---|
| scaled rotations, distinct moduli | ✓ | ✓ | 4 | 4 | 0 / 400 |
| §3.1 regrouping | ✗ | ✓ | 4 | 4 | 400 / 400 |
| $J_2(0.8) \oplus J_2(0.8)$ | ✓ | ✗ | **8** | 4 | 400 / 400 |
| §3.1 refined to $[1,1,1,1]$ | ✓ | ✓ | 4 | 4 | 0 / 400 |

Note row 2: the dimension alone does not detect the §3.1 failure — the *support
pattern* does. The block-energy matrix is uniform ($0.25$ everywhere) instead of
being confined to matched blocks. Both diagnostics are needed.

---

## 7. What this settles, and what it does not

**Settled.** For a linear decoder of full column rank and linear latent
dynamics, identifiability of the module partition holds if and only if the
claimed decomposition is the finest one and the blocks have disjoint spectra.
The conclusion is the partition plus each $f_i$ up to similarity; the
within-module maps $S_i$ are arbitrary module isomorphisms, i.e. coordinates
inside a module are not identified. That is CLAUDE.md §7, made precise.

**Not settled.** Everything nonlinear. The module-theoretic proof uses that
$\pi_j$ is a *polynomial* in $F$, which has no nonlinear analogue; the
replacement is a dynamical decomposition argument, and it is open. See
`identifiability.md`.
