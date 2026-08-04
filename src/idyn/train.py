"""Fitting loop and dataset construction.

Reproducibility is a hard requirement (CLAUDE.md §8): every function takes an
explicit seed or Generator, nothing touches global RNG state, and ``fit_many``
uses a deterministic sequence of restart seeds so a reported non-uniqueness
result can be re-run exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
import torch

from idyn.behavior import conditioned_initial_conditions
from idyn.models import LatentDynamicsModel, ModelConfig
from idyn.systems import LinearDecoder, MLPDecoder, sample_initial_conditions

# Either observation model: linear (Theorem A) or nonlinear (Theorem B).  They
# share the call signature, so nothing downstream needs to know which it has.
Decoder = LinearDecoder | MLPDecoder

__all__ = [
    "TrainConfig",
    "FitResult",
    "make_dataset",
    "make_behavioural_dataset",
    "fit",
    "fit_many",
]


@dataclass
class TrainConfig:
    steps: int = 1500
    lr: float = 3e-3
    batch: int = 64
    w_recon: float = 1.0
    w_dyn: float = 1.0
    w_white: float = 1.0
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 0  # 0 = silent
    # Route B: penalise u-dependence of the coordinates in [inv_start, inv_stop)
    w_behavior: float = 0.0
    inv_start: int = 0
    inv_stop: int = 0
    # CLAUDE.md §3.12: without this the penalty is paid off by shrinking the
    # block rather than by making it u-invariant.  False reproduces the exp11 /
    # exp12 runs, whose behavioural conclusions are void for that reason.
    behavior_whiten: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def invariant_slice(self) -> slice | None:
        return slice(self.inv_start, self.inv_stop) if self.inv_stop > self.inv_start else None


@dataclass
class FitResult:
    model: LatentDynamicsModel = field(repr=False)
    z_fit: np.ndarray = field(repr=False)
    fit_quality: float = float("nan")
    history: list[float] = field(default_factory=list, repr=False)
    seed: int = 0

    def __repr__(self) -> str:
        return f"FitResult(seed={self.seed}, fit_quality={self.fit_quality:.4e})"


def make_dataset(
    system,
    n_obs: int,
    n_traj: int,
    T: int,
    rng: np.random.Generator,
    radius: float = 1.0,
    noise_std: float = 0.0,
    decoder: Decoder | None = None,
) -> tuple[np.ndarray, np.ndarray, Decoder]:
    """Simulate the true system and observe it.

    ``decoder`` defaults to a random ``LinearDecoder`` (Theorem A, §3.5).  Pass
    an ``MLPDecoder`` for the Theorem B regime, where h is not forced linear.

    Returns ``(X, Z, decoder)`` with X of shape (n_traj, T+1, n_obs) and Z the
    ground-truth latents of shape (n_traj, T+1, d).

    Initial conditions are spread over an annulus rather than a single orbit:
    these systems contract, so a single long trajectory would visit a
    vanishingly small region and the CLAUDE.md §3.6 support caveat would make
    every conclusion vacuous.
    """
    d = system.dim
    z0 = sample_initial_conditions(d, n_traj, rng, radius=radius)
    Z = system.simulate(z0, T)
    dec = decoder if decoder is not None else LinearDecoder.random(n_obs, d, rng, noise_std=noise_std)
    X = dec(Z, rng=rng if dec.noise_std > 0 else None)
    return X, Z, dec


def make_behavioural_dataset(
    system,
    d_a: int,
    d_b: int,
    n_obs: int,
    n_per_u: int,
    T: int,
    u_levels: np.ndarray,
    rng: np.random.Generator,
    mode: str = "variance",
    noise_std: float = 0.0,
    decoder: Decoder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Decoder]:
    """Simulate a modular system from **u-conditioned** initial conditions.

    The first ``d_a`` latent coordinates (block A) have a u-dependent initial law;
    the last ``d_b`` (block B) do not (``idyn.behavior``). Returns
    ``(X, Z, U, decoder)`` with ``X`` of shape ``(n, T+1, n_obs)``, ``Z`` the
    ground-truth latents, and ``U`` the per-trajectory behaviour label. This is the
    B∘C data model: behaviour modulates block A, dynamics are autonomous & modular.
    """
    if system.dim != d_a + d_b:
        raise ValueError(f"system dim {system.dim} != d_a + d_b = {d_a + d_b}")
    sample = conditioned_initial_conditions(d_a, d_b, u_levels, n_per_u, rng, mode=mode)
    Z = system.simulate(sample.Z, T)
    dec = decoder if decoder is not None else LinearDecoder.random(n_obs, system.dim, rng, noise_std=noise_std)
    X = dec(Z, rng=rng if dec.noise_std > 0 else None)
    return X, Z, sample.U, dec


def fit(
    X: np.ndarray, model_cfg: ModelConfig, cfg: TrainConfig, U: np.ndarray | None = None
) -> FitResult:
    """Fit one model.  Returns the model and the latents it assigns to X.

    ``U`` (per-trajectory behaviour labels) is required when ``cfg.w_behavior > 0``:
    the behavioural penalty then drives the ``cfg.invariant_slice`` coordinates to
    be u-invariant (Route B).
    """
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
    inv = cfg.invariant_slice
    behav = cfg.w_behavior > 0.0 and inv is not None
    if behav and U is None:
        raise ValueError("w_behavior > 0 requires U (behaviour labels)")
    Ut = None if U is None else torch.as_tensor(np.asarray(U), device=dev)

    model = LatentDynamicsModel(model_cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 12345)

    n = Xt.shape[0]
    history: list[float] = []
    for step in range(cfg.steps):
        idx = torch.randint(0, n, (min(cfg.batch, n),), generator=gen).to(dev)
        out = model.losses(
            Xt[idx], cfg.w_recon, cfg.w_dyn, cfg.w_white,
            u=(Ut[idx] if behav else None),
            invariant_slice=(inv if behav else None),
            w_behavior=cfg.w_behavior,
            behavior_whiten=cfg.behavior_whiten,
        )
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        opt.step()
        history.append(float(out["total"].item()))
        if cfg.log_every and step % cfg.log_every == 0:
            print(
                f"  step {step:5d}  total {history[-1]:.5f}  "
                f"recon {out['recon'].item():.5f}  dyn {out['dyn'].item():.5f}  "
                f"white {out['white'].item():.5f}  behavior {float(out['behavior']):.5f}"
            )

    model.eval()
    with torch.no_grad():
        full = model.losses(Xt, cfg.w_recon, cfg.w_dyn, cfg.w_white)
        z_fit = model.encode(Xt).cpu().numpy()
    return FitResult(
        model=model,
        z_fit=z_fit,
        fit_quality=float(full["fit_quality"].item()),
        history=history,
        seed=cfg.seed,
    )


def fit_many(
    X: np.ndarray, model_cfg: ModelConfig, cfg: TrainConfig, n_restarts: int = 8
) -> list[FitResult]:
    """Fit ``n_restarts`` times from different initialisations.

    Restarts are the instrument for the §3.1 negative control: if the modular
    constraint really identified the partition, every restart that fits the
    data would have to find the same one.
    """
    out = []
    for r in range(n_restarts):
        c = TrainConfig(**{**cfg.to_dict(), "seed": cfg.seed + 1000 * r})
        out.append(fit(X, model_cfg, c))
    return out
