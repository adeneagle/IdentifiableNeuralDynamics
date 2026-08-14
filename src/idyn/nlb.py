"""Neural Latents Benchmark data: NWB in, trialized spike counts out.

This is the repo's first contact with real recordings (CLAUDE.md §6, "the
empirical program").  Everything else under ``src/idyn`` operates on
``make_dataset`` output, which is a deterministic flow through a known decoder;
here the generator is a monkey.

**No ``nlb_tools`` dependency.**  That package is a convenience wrapper around
the benchmark's split bookkeeping.  Everything needed here -- trial windows,
binned counts, held-out units, hand kinematics -- is in the NWB file itself and
is read with ``h5py``.  Avoiding the dependency keeps the environment pinned as
CLAUDE.md §4.1 describes it.

Scope, and it is a real modelling choice (CLAUDE.md §1.1).  The target class is
**autonomous** latent dynamics with a random initial condition.  Aligning to
movement onset and reading the movement period is the standard regime in which
motor cortex is modelled that way -- it is what autonomous LFADS does -- so the
alignment below is not a neutral preprocessing default.  It is the hypothesis.
Pre-movement epochs, where the target cue drives the population, are outside
the scope §1.1 fixes and are excluded by the window rather than by argument.

Conventions follow CLAUDE.md §8: float64 for anything that reaches the spectrum
code, explicit ``rng``/``seed``, and every parameter recorded so a run can be
reproduced from the results JSON alone.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "DATASETS",
    "fetch",
    "TrialData",
    "load_trials",
    "neuron_split",
]

# dandiset id, asset id, and the local filename each is cached under.
DATASETS: dict[str, dict[str, str]] = {
    "mc_maze_small": {
        "dandiset": "000140",
        "asset": "7821971e-c6a4-4568-8773-1bfa205c13f8",
        "filename": "mc_maze_small_train.nwb",
    },
    "mc_maze": {
        "dandiset": "000128",
        "asset": "26e85f09-39b7-480f-b337-278a8f034007",
        "filename": "mc_maze_full_train.nwb",
    },
    "mc_rtt": {
        "dandiset": "000129",
        "asset": "2ae6bf3c-788b-4ece-8c01-4b4a5680b25b",
        "filename": "mc_rtt_train.nwb",
    },
}

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data"


def fetch(name: str, root: str | os.PathLike | None = None) -> Path:
    """Return the local path to a dataset, downloading from DANDI if absent.

    The NWB blobs are gitignored; this is what makes a run reproducible without
    tracking them.  Raises rather than downloading silently if ``name`` is not
    a known dataset.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[name]
    root = Path(root) if root is not None else _DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    dst = root / spec["filename"]
    if not dst.exists():
        url = (
            f"https://api.dandiarchive.org/api/dandisets/{spec['dandiset']}"
            f"/versions/draft/assets/{spec['asset']}/download/"
        )
        urllib.request.urlretrieve(url, dst)
    return dst


@dataclass
class TrialData:
    """Binned spike counts on a common trial window, plus what they align to.

    ``spikes`` is ``(n_trials, n_bins, n_units)`` of non-negative integers --
    counts, not rates.  Nothing here smooths; a model that wants rates should
    say so itself.
    """

    spikes: np.ndarray
    hand_vel: np.ndarray
    condition: np.ndarray
    unit_ids: np.ndarray
    heldout: np.ndarray
    bin_ms: float
    window_ms: tuple[float, float]
    align: str
    dataset: str
    provenance: dict = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return int(self.spikes.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.spikes.shape[1])

    @property
    def n_units(self) -> int:
        return int(self.spikes.shape[2])

    def summary(self) -> str:
        rate = self.spikes.mean() / (self.bin_ms / 1000.0)
        return (
            f"{self.dataset}: {self.n_trials} trials x {self.n_bins} bins x "
            f"{self.n_units} units @ {self.bin_ms:g} ms, "
            f"align={self.align} window={self.window_ms}, "
            f"mean rate {rate:.1f} Hz, {len(set(self.condition.tolist()))} conditions"
        )

    def condition_average(
        self,
        smooth_ms: float = 30.0,
        min_trials: int = 3,
        sqrt_transform: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trial-averaged rates per condition: ``(n_cond, n_bins, n_units)``.

        **This is the object the theory is about, and that is the reason to use
        it -- not SNR convenience.**  CLAUDE.md §1.1 fixes the target class as a
        *deterministic* flow from a random initial condition, and §1.3 says the
        identifiability content is dynamical rather than distributional.  A
        condition is an initial condition; averaging over its repeats removes the
        Poisson noise that the target class does not model in the first place.
        Single trials would force a distributional equivalence notion that §3.8
        records as explicitly out of scope.

        ``sqrt_transform`` applies the variance-stabilising map for counts, which
        makes the residual noise roughly homoscedastic and lets the existing
        Gaussian observation model be used without misspecification.  The
        decoder then maps latents to square-root rates, which is as good an
        instance of ``x = g(z)`` as any other -- §1.2's Tier 1 needs ``g``
        injective, not ``g`` of a particular form.

        Returns ``(rates, condition_labels, n_trials_per_condition)``.
        """
        labels, inverse, counts = np.unique(
            self.condition, return_inverse=True, return_counts=True
        )
        keep = counts >= min_trials
        out = np.empty((int(keep.sum()), self.n_bins, self.n_units), dtype=float)
        for k, c in enumerate(np.flatnonzero(keep)):
            out[k] = self.spikes[inverse == c].mean(axis=0)
        out /= self.bin_ms / 1000.0                      # counts -> Hz

        if smooth_ms > 0:
            sigma = smooth_ms / self.bin_ms
            half = int(np.ceil(3 * sigma))
            k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
            k /= k.sum()
            pad = np.pad(out, ((0, 0), (half, half), (0, 0)), mode="edge")
            out = np.apply_along_axis(
                lambda v: np.convolve(v, k, mode="valid"), 1, pad
            )
        if sqrt_transform:
            out = np.sqrt(np.maximum(out, 0.0))
        return out, labels[keep], counts[keep]

    def record(self) -> dict:
        """The JSON-serialisable parameter block for a results file."""
        return {
            "dataset": self.dataset,
            "n_trials": self.n_trials,
            "n_bins": self.n_bins,
            "n_units": self.n_units,
            "bin_ms": self.bin_ms,
            "window_ms": list(self.window_ms),
            "align": self.align,
            "n_conditions": len(set(self.condition.tolist())),
            "mean_rate_hz": float(self.spikes.mean() / (self.bin_ms / 1000.0)),
            "n_heldout_units": int(self.heldout.sum()),
            **self.provenance,
        }


def _ragged(group: h5py.Group, name: str) -> list[np.ndarray]:
    """Read an NWB ragged (VectorData + VectorIndex) column into a list."""
    values = group[name][:]
    idx = group[f"{name}_index"][:]
    out, lo = [], 0
    for hi in idx:
        out.append(values[lo:hi])
        lo = int(hi)
    return out


def load_trials(
    dataset: str = "mc_maze",
    bin_ms: float = 20.0,
    window_ms: tuple[float, float] = (-250.0, 450.0),
    align: str = "move_onset_time",
    split: str | None = "train",
    root: str | os.PathLike | None = None,
    min_rate_hz: float = 0.5,
    max_trials: int | None = None,
) -> TrialData:
    """Bin spikes on a fixed window around ``align``, one row per trial.

    ``min_rate_hz`` drops units that essentially never fire in the window --
    they contribute no information and make a Poisson fit ill-conditioned,
    and dropping them is reported rather than silent.

    ``split`` selects the NLB train/val partition; ``None`` keeps both.  The
    benchmark's own *test* split has no spikes for held-out units, so it is a
    separate file and is not used here.
    """
    path = fetch(dataset, root=root)
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        if align not in trials:
            raise KeyError(f"{align!r} not in trials; have {sorted(trials.keys())}")
        t_align = np.asarray(trials[align][:], dtype=float)

        keep = np.isfinite(t_align)
        if split is not None:
            raw = trials["split"][:]
            tags = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in raw])
            keep &= tags == split
        sel = np.flatnonzero(keep)
        if max_trials is not None:
            sel = sel[:max_trials]
        if sel.size == 0:
            raise ValueError("no trials selected")
        t_align = t_align[sel]

        # condition label: maze id x version, the standard MC_Maze grouping
        if "trial_type" in trials and "trial_version" in trials:
            tt = trials["trial_type"][:][sel]
            tv = trials["trial_version"][:][sel]
            condition = np.array([f"{a}_{b}" for a, b in zip(tt, tv)])
        else:
            condition = np.array(["all"] * sel.size)

        units = f["units"]
        spike_times = _ragged(units, "spike_times")
        unit_ids = np.asarray(units["id"][:])
        heldout = (
            np.asarray(units["heldout"][:], dtype=bool)
            if "heldout" in units
            else np.zeros(unit_ids.size, dtype=bool)
        )

        n_bins = int(round((window_ms[1] - window_ms[0]) / bin_ms))
        offsets = (window_ms[0] + bin_ms * np.arange(n_bins + 1)) / 1000.0
        edges = t_align[:, None] + offsets[None, :]          # (n_trials, n_bins+1)

        counts = np.empty((sel.size, n_bins, unit_ids.size), dtype=np.int32)
        flat = edges.ravel()
        for j, st in enumerate(spike_times):
            cum = np.searchsorted(st, flat).reshape(edges.shape)
            counts[:, :, j] = np.diff(cum, axis=1)

        # hand velocity, differentiated from position on the same bin edges
        beh = f["processing/behavior"]
        pos = np.asarray(beh["hand_pos/data"][:], dtype=float)
        ts = np.asarray(beh["hand_pos/timestamps"][:], dtype=float)
        centres = (edges[:, :-1] + edges[:, 1:]) / 2.0
        idx = np.clip(np.searchsorted(ts, centres.ravel()), 1, ts.size - 1)
        dt = ts[idx] - ts[idx - 1]
        vel = (pos[idx] - pos[idx - 1]) / np.where(dt > 0, dt, np.nan)[:, None]
        hand_vel = vel.reshape(*centres.shape, 2)

        provenance = {
            "dandiset": DATASETS[dataset]["dandiset"],
            "asset": DATASETS[dataset]["asset"],
            "file_bytes": int(path.stat().st_size),
            "file_sha256_head": hashlib.sha256(
                path.read_bytes()[:1_000_000]
            ).hexdigest()[:16],
            "split": split,
            "min_rate_hz": min_rate_hz,
        }

    rate = counts.mean(axis=(0, 1)) / (bin_ms / 1000.0)
    live = rate >= min_rate_hz
    provenance["n_units_dropped_low_rate"] = int((~live).sum())

    return TrialData(
        spikes=counts[:, :, live],
        hand_vel=np.nan_to_num(hand_vel),
        condition=condition,
        unit_ids=unit_ids[live],
        heldout=heldout[live],
        bin_ms=float(bin_ms),
        window_ms=(float(window_ms[0]), float(window_ms[1])),
        align=align,
        dataset=dataset,
        provenance=provenance,
    )


def neuron_split(
    n_units: int, seed: int, n_parts: int = 2, rate: np.ndarray | None = None
) -> list[np.ndarray]:
    """Disjoint neuron subsets for the task-40 agreement test.

    Stratified by firing rate when ``rate`` is given, so the two halves are
    comparable populations rather than one fast half and one slow one.  That
    matters: §3.13(b) found recoverability tracks where the orbits carry
    variance, so an unbalanced split would confound "different neurons" with
    "less signal".
    """
    rng = np.random.default_rng(seed)
    if rate is None:
        order = rng.permutation(n_units)
    else:
        if rate.shape != (n_units,):
            raise ValueError("rate must have one entry per unit")
        order = np.argsort(rate)
        # shuffle within rate-ordered blocks of size n_parts, then deal round-robin
        blocks = [order[i : i + n_parts] for i in range(0, n_units, n_parts)]
        order = np.concatenate([rng.permutation(b) for b in blocks])
    return [np.sort(order[i::n_parts]) for i in range(n_parts)]
