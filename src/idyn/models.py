"""Torch models: unconstrained vs modular latent dynamics.

Two decoder settings, kept separable in code because CLAUDE.md §3.5 says they
are separate theorems:

* ``decoder="linear"``  -- Theorem A.  A full-column-rank W already forces the
  reparameterisation h into GL(d) before any dynamics are used.
* ``decoder="mlp"``     -- Theorem B.  x = g(z) with g an injective immersion;
  this is the setting the LFADS motivation actually requires, and where h can
  genuinely be nonlinear.

Two transition settings:

* ``UnconstrainedTransition`` -- one MLP on all of R^d.
* ``ModularTransition``      -- K independent MLPs, one per module.  This is the
  structural constraint whose identifying power the project is testing.

### A note on the whitening penalty

The latent scale is unidentifiable (z -> cz with W -> W/c), so unconstrained
fitting is degenerate and collapses.  We pin it with a soft penalty driving
Cov(z) to I.  This is not free: it restricts the recovered h from GL(d) to the
family ``{Q Sigma^{-1/2} : Q orthogonal}``.

That restriction is checked to be harmless for the questions asked here:

* the §3.1 regrouping maps are *permutations*, hence orthogonal, and the true
  latents there are independent (Sigma diagonal), so every regrouping remains
  reachable -- the negative control still can fail, which is the point;
* it removes only scalings and shears, which are within-module changes of
  coordinate that §7 already declines to identify.

If a future experiment needs unrestricted h, drop the penalty and parameterise
the latents directly instead of through an encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "mlp",
    "UnconstrainedTransition",
    "ModularTransition",
    "LatentDynamicsModel",
    "TriangularTransition",
    "LearnedBlock",
    "LearnedSystem",
    "ModelConfig",
]


def mlp(d_in: int, d_out: int, hidden: Sequence[int], act: str = "tanh") -> nn.Sequential:
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
    if act not in acts:
        raise ValueError(f"unknown activation {act!r}; choose from {sorted(acts)}")
    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), acts[act]()]
        prev = h
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


class UnconstrainedTransition(nn.Module):
    """z -> z + net(z).  Residual form so the identity is easy to represent."""

    def __init__(self, d: int, hidden: Sequence[int] = (64, 64), act: str = "tanh"):
        super().__init__()
        self.d = d
        self.partition = [d]
        self.net = mlp(d, d, hidden, act)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


class ModularTransition(nn.Module):
    """F = f_1 (+) ... (+) f_K with one residual MLP per module.

    Block-diagonality is structural, not penalised: module i literally never
    sees the other modules' coordinates.  That is the constraint whose
    identifying power is under test.
    """

    def __init__(
        self, partition: Sequence[int], hidden: Sequence[int] = (32, 32), act: str = "tanh"
    ):
        super().__init__()
        self.partition = list(partition)
        self.d = int(sum(self.partition))
        self.nets = nn.ModuleList([mlp(k, k, hidden, act) for k in self.partition])
        bounds, off = [], 0
        for k in self.partition:
            bounds.append((off, off + k))
            off += k
        self.bounds = bounds

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        parts = []
        for (a, b), net in zip(self.bounds, self.nets):
            zi = z[..., a:b]
            parts.append(zi + net(zi))
        return torch.cat(parts, dim=-1)


class TriangularTransition(nn.Module):
    """The skew product: module i sees modules 1..i, never i+1..K.

    This is the object §3.7 proves the spectral gap actually delivers -- a
    *filtration*, not a direct sum -- and §7 argues is the generic one, since a
    filtration needs only invariant subspaces while a direct sum needs an
    invariant complement too.

    It sits strictly between the other two transitions, so the three form a
    nested ladder

        unconstrained  >  triangular  >  modular,

    which is what the task-39 co-smoothing gate scores.  Nesting is what makes
    held-out performance a legitimate test: same bias ordering, decreasing
    variance, so a *drop* at a rung is evidence that rung's constraint is false.
    """

    def __init__(
        self, partition: Sequence[int], hidden: Sequence[int] = (32, 32), act: str = "tanh"
    ):
        super().__init__()
        self.partition = list(partition)
        self.d = int(sum(self.partition))
        bounds, off = [], 0
        for k in self.partition:
            bounds.append((off, off + k))
            off += k
        self.bounds = bounds
        # module i is driven by everything up to and including itself
        self.nets = nn.ModuleList([mlp(b, k, hidden, act) for (_, b), k in zip(bounds, self.partition)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        parts = []
        for (a, b), net in zip(self.bounds, self.nets):
            parts.append(z[..., a:b] + net(z[..., :b]))
        return torch.cat(parts, dim=-1)


@dataclass
class ModelConfig:
    n_obs: int
    d: int
    partition: list[int] | None = None  # None => unconstrained transition
    decoder: str = "linear"  # "linear" (Theorem A) or "mlp" (Theorem B)
    encoder: str = "linear"
    hidden_dyn: tuple[int, ...] = (32, 32)
    hidden_obs: tuple[int, ...] = (64, 64)
    act: str = "tanh"
    # "modular" (block-diagonal), "triangular" (skew product) or "unconstrained".
    # Only consulted when `partition` is set; the three are the task-39 ladder.
    structure: str = "modular"

    def __post_init__(self) -> None:
        if self.partition is not None and sum(self.partition) != self.d:
            raise ValueError(f"partition {self.partition} does not sum to d={self.d}")
        if self.decoder not in {"linear", "mlp"}:
            raise ValueError("decoder must be 'linear' or 'mlp'")
        if self.encoder not in {"linear", "mlp"}:
            raise ValueError("encoder must be 'linear' or 'mlp'")
        if self.structure not in {"modular", "triangular", "unconstrained"}:
            raise ValueError("structure must be modular, triangular or unconstrained")

    @property
    def modular(self) -> bool:
        return self.partition is not None and self.structure != "unconstrained"


class LatentDynamicsModel(nn.Module):
    """Encoder -> latent dynamics -> decoder, fit by prediction + reconstruction."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, n_obs = cfg.d, cfg.n_obs

        self.enc = (
            nn.Linear(n_obs, d, bias=False)
            if cfg.encoder == "linear"
            else mlp(n_obs, d, cfg.hidden_obs, cfg.act)
        )
        self.dec = (
            nn.Linear(d, n_obs, bias=False)
            if cfg.decoder == "linear"
            else mlp(d, n_obs, cfg.hidden_obs, cfg.act)
        )
        if cfg.modular and cfg.structure == "triangular":
            self.dyn: nn.Module = TriangularTransition(cfg.partition, cfg.hidden_dyn, cfg.act)
        elif cfg.modular:
            self.dyn = ModularTransition(cfg.partition, cfg.hidden_dyn, cfg.act)
        else:
            self.dyn = UnconstrainedTransition(d, cfg.hidden_dyn, cfg.act)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, T+1, n_obs).  Returns latents, reconstruction and prediction."""
        z = self.encode(x)
        return {
            "z": z,
            "x_rec": self.dec(z),
            "z_pred": self.dyn(z[:, :-1]),  # predicts z[:, 1:]
        }

    def losses(
        self,
        x: torch.Tensor,
        w_recon: float = 1.0,
        w_dyn: float = 1.0,
        w_white: float = 1.0,
        u: torch.Tensor | None = None,
        invariant_slice: slice | None = None,
        w_behavior: float = 0.0,
        behavior_whiten: bool = True,
        behavior_per_time: bool = True,
    ) -> dict[str, torch.Tensor]:
        out = self(x)
        z, z_pred = out["z"], out["z_pred"]

        recon = torch.mean((out["x_rec"] - x) ** 2)
        dyn = torch.mean((z_pred - z[:, 1:]) ** 2)

        flat = z.reshape(-1, z.shape[-1])
        centred = flat - flat.mean(0, keepdim=True)
        cov = centred.T @ centred / max(centred.shape[0] - 1, 1)
        white = torch.mean((cov - torch.eye(cov.shape[0], device=cov.device)) ** 2)

        total = w_recon * recon + w_dyn * dyn + w_white * white

        # Route B: penalise u-dependence of the block designated *invariant*. If a
        # fitted invariant coordinate has picked up a u-varying latent, its
        # conditional law moves with u; driving that to zero is how behaviour kills
        # the cross-derivative M_BA (theory/approaches.md §B.1). Differentiable
        # analogue of behavior.block_u_dependence.
        behavior = z.new_zeros(())
        if u is not None and invariant_slice is not None and w_behavior > 0.0:
            behavior = self._behavioural_penalty(z, u, invariant_slice,
                                                 whiten=behavior_whiten,
                                                 per_time=behavior_per_time)
            total = total + w_behavior * behavior

        # `fit_quality` excludes the whitening and behaviour terms: they are
        # well-posedness / structural devices, not part of the reconstruction, and
        # including them would let a restart look worse for a reason unrelated to fit.
        return {
            "total": total,
            "recon": recon,
            "dyn": dyn,
            "white": white,
            "behavior": behavior.detach() if torch.is_tensor(behavior) else behavior,
            "fit_quality": (recon + dyn).detach(),
        }

    @staticmethod
    def _behavioural_penalty(
        z: torch.Tensor, u: torch.Tensor, block: slice, whiten: bool = True,
        per_time: bool = True,
    ) -> torch.Tensor:
        """Between-``u`` spread of the first two conditional moments of ``z[..., block]``.

        ``z`` is ``(B, T+1, d)`` and ``u`` is ``(B,)``; the u-label is broadcast over
        time. Zero exactly when the block's conditional law does not move with ``u``.

        ### Why ``whiten`` defaults to True (CLAUDE.md §3.12)

        On raw ``w`` this penalty is **not scale-invariant**: under ``w -> eps w``
        the mean term falls like ``eps^2`` and the covariance term like ``eps^4``.
        So the cheapest way for the optimiser to satisfy it is not to make the
        block u-invariant but to make it *small* -- and that is what it did.  In
        ``exp11``/``exp12`` at ``w_behavior = 5`` the fitted "invariant" block came
        out 21x smaller than its partner, scored a raw u-dependence of 0.0015,
        and still carried the u-varying latent at distance correlation 0.99.
        Rescaled to unit variance its u-dependence is 1.07, against 0.15 for a
        genuinely invariant block: the constraint was satisfied by gauge, never
        imposed.  Any conclusion drawn from those runs about behaviour supplying a
        cross-derivative zero is therefore void.

        Whitening by the block's own pooled covariance makes the penalty invariant
        under the whole of ``GL(d_b)``, which is exactly the freedom §7 grants
        within a module -- so it can no longer be paid off with a coordinate
        change, only with genuine distributional invariance.  Same reasoning as
        ``metrics.jacobian_block_report(standardize=True)`` (§3.10 trap 1), one
        level down: there it was a readout that measured the gauge, here it was
        the objective.

        ``whiten=False`` restores the old behaviour, for reproducing those runs.

        ### Why ``per_time`` defaults to True (CLAUDE.md §3.15)

        Pooling every timestep into one sample set before scoring destroys any
        u-dependence carried by a **rotating** quantity.  On an oscillatory module
        the block's phase advances by ``omega`` per step, so over a trial it wraps
        several times and the time-pooled law is close to rotationally uniform for
        every ``u`` -- even when the per-timestep laws differ completely.

        Measured on exp18's data, the recipient block under the lattice
        regrouping: per-timestep u-dependence **0.979**, time-pooled **0.017**, a
        59x collapse that leaves it only 3.8x above a genuinely invariant block's
        pooled 0.0044.  The u-varying *donor* itself falls 0.991 -> 0.106.  At
        that contrast no weight can impose the constraint, which is exactly what
        the exp18 calibration found: at ``w`` from 1 upward the fitted block sat
        at u-dependence 1.01 while the penalty reported itself satisfied.

        Scoring each timestep and averaging preserves the structure.  This is not
        a refinement of §3.12 but the same error one axis over: there the penalty
        was invariant under a group it should have seen through (scale), here
        under an average it should not have taken.

        ``per_time=False`` restores the pooled form.  A **radial** modulation does
        not rotate, so pooling only mixes decay stages and both forms see a leak
        there: measured on exp13-style data (contracting twists, variance
        modulation), injecting ``z_B += 0.5 z_A`` moves the pooled score
        0.00085 -> 0.402 (473x) against the per-time 0.00169 -> 1.275 (754x).
        Same order, so `exp13`'s conclusions are untouched -- but note the
        absolute scales differ by ~3x, so a weight still does not transfer across
        the flag any more than it transferred across ``whiten`` (§3.12).
        """
        d_b = block.stop - block.start

        def _score(w: torch.Tensor, lab_of: torch.Tensor) -> torch.Tensor:
            if whiten:
                wc = w - w.mean(0, keepdim=True)
                cov = wc.T @ wc / max(wc.shape[0] - 1, 1)
                # ridge keeps the Cholesky defined while the block is still
                # collapsing early in training; far below any scale the penalty
                # cares about.
                cov = cov + 1e-6 * torch.eye(d_b, device=cov.device, dtype=cov.dtype)
                L = torch.linalg.cholesky(cov)
                w = torch.linalg.solve_triangular(L, wc.T, upper=False).T
            means, covs = [], []
            for lab in torch.unique(lab_of):
                wl = w[lab_of == lab]
                m = wl.mean(0)
                wc_l = wl - m
                means.append(m)
                covs.append(wc_l.T @ wc_l / max(wl.shape[0] - 1, 1))
            means = torch.stack(means)   # (n_u, d_b)
            covs = torch.stack(covs)     # (n_u, d_b, d_b)
            return means.var(0).sum() + covs.var(0).sum()

        if not per_time:
            w = z[:, :, block].reshape(-1, d_b)
            u_rep = u.view(-1, 1).expand(z.shape[0], z.shape[1]).reshape(-1)
            return _score(w, u_rep)

        # Whitening is per-timestep too: a common whitener would reintroduce the
        # pooled geometry through the back door.
        return torch.stack([_score(z[:, t, block], u) for t in range(z.shape[1])]).mean()


class LearnedBlock:
    """One block of a fitted ``ModularTransition``, as a ``spectra.HasJacobian``.

    The block MLP sees only its own coordinates, so it can be iterated alone --
    which is what makes a per-module Lyapunov spectrum and rotation number well
    posed on a *fitted* model at all.

    Cast the model to ``double()`` before wrapping.  The Jacobian is a central
    difference at ``eps=1e-6``; on a float32 model that puts the roundoff floor
    on top of the signal.
    """

    def __init__(self, dyn: ModularTransition, k: int):
        a, b = dyn.bounds[k]
        self.net = dyn.nets[k]
        self.dim = b - a

    def _f(self, Z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(Z, float), dtype=torch.float64)
            return (t + self.net(t)).numpy()

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        return self._f(np.atleast_2d(z)).reshape(z.shape)

    def jacobian(self, z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        z = np.asarray(z, float).reshape(self.dim)
        E = np.eye(self.dim) * eps
        out = self._f(np.vstack([z + E, z - E]))
        return ((out[: self.dim] - out[self.dim :]) / (2.0 * eps)).T


class LearnedSystem:
    """``partition`` + ``blocks`` -- the whole interface a fingerprint needs.

    Lets ``metrics.dynamical_fingerprint`` run on a fitted model exactly as it
    runs on a ground-truth ``ModularSystem``, which is what task 40 requires:
    the comparison is fit-to-fit, with no ground truth anywhere.
    """

    def __init__(self, dyn: ModularTransition, partition: Sequence[int] | None = None):
        self.partition = list(partition if partition is not None else dyn.partition)
        self.blocks = [LearnedBlock(dyn, k) for k in range(len(self.partition))]
