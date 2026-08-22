"""Hsu open-field recordings: continuous spikes + ethogram + kinematics.

The repo's **second** real dataset, and it exists because the first one does not
contain the phenomenon the nonlinear theory is about.  `exp15b` closed every
escape on the Neural Latents benchmarks: the MC_Maze latent flow is >=99% linear
across smoothing, window, dimension, dataset, single trials and MC_RTT, so
Theorem B is not merely untested there -- it is inapplicable, and no better
estimator changes that (CLAUDE.md task 42).

What is different here, and each point is a hypothesis to check rather than a
claim:

* **Spontaneous behaviour, no task.**  There is no stimulus and no trial
  structure, so the population is closer to §1.1's *autonomous* target than a
  cued reach is.  The scope note in `nlb.py` -- that aligning to movement onset
  is the hypothesis, not a neutral default -- has no analogue to worry about.
* **A genuine auxiliary variable.**  A 16-state ethogram, plus 36 continuous
  DLC kinematic features.  Route B wants a `u` that conditions the latent law
  non-trivially, and §4.5c (Lemma D''') says a *continuous* covariate is more
  natural than two discrete levels.  Neither was available on MC_Maze.
* **Four areas.**  §1.1 scopes multi-region *out*: a module is a dynamical
  factor inside one population, not a brain area.  So the areas are kept
  separate and an experiment picks one; they are **not** to be concatenated into
  a four-module system.  Their value is as independent replicates of the same
  question.
* **Length.**  ~4.5 h continuous at ~450 sorted units, against MC_Maze's 35-bin
  trials.  §11.6 says the lattice ambiguity bites exactly when the donor module
  does *not* decay, and MC_Maze sits at |lambda| ~ 0.99 over 35 bins with
  nothing to measure.  A long recording spanning many behavioural transitions is
  where genuine contraction could actually show up.

**Two views of the data, and they answer different questions.**
`segments()` chops the continuous recording into uniform windows -- one long
trajectory, no averaging, maximal noise, but nothing smoothed away.
`bout_average()` treats a *behaviour type* as a condition and averages over its
repeats, which is the exact analogue of `nlb.TrialData.condition_average` and
the object §1.1's deterministic-flow target is about.  `exp15b`'s discipline
applies: measure both, because averaging is itself a candidate artefact.

**No absolute paths.**  CLAUDE.md §5 keeps `src/`, `tests/` and `experiments/`
free of them, and §4.1 records that every machine-specific path written into
this repo has gone stale at least once.  The session root therefore comes from
the ``IDYN_HSU_ROOT`` environment variable or an explicit argument.

**No new dependencies.**  The spike table is a 15M-row integer CSV; ``pandas``
reads it in ~2 s and ``numpy.loadtxt`` in minutes, so pandas is used *when
importable* and fallen back on otherwise.  It is deliberately not added to
``environment.yml`` (CLAUDE.md §4.1 pins that list), and the parse is cached to
``data/`` so the slow path is paid at most once.

Conventions follow CLAUDE.md §8: float64 for anything reaching the spectrum
code, explicit ``rng``/``seed``, and every parameter recorded so a run is
reproducible from the results JSON alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "AREAS",
    "BEHAVIOURS",
    "session_root",
    "SessionData",
    "load_session",
]

#: Recorded areas, by the filename stem of their unit-id list.  ``CC`` is
#: present in the files but has only 2 units above threshold, so it is not
#: usable as a population and is excluded here rather than silently returning
#: something too small to fit.
AREAS: tuple[str, ...] = ("VS", "DS", "M56", "M23")

#: The ethogram, read from the session's own labels file when available; this
#: copy is the fallback and the documentation.  ``-1`` is not a behaviour.
BEHAVIOURS: dict[int, str] = {
    -1: "in_nest_sleeping_or_irrelevant",
    0: "investigate_1",
    1: "investigate_2",
    2: "investigate_3",
    3: "rear",
    4: "dive_scrunch",
    5: "paw_groom",
    6: "face_groom_1",
    7: "face_groom_2",
    8: "head_groom",
    9: "contra_body_groom",
    10: "ipsi_body_groom",
    11: "contra_itch",
    12: "ipsi_itch_1",
    13: "contra_orient",
    14: "ipsi_orient",
    15: "locomotion",
}

#: The DLC kinematics are sampled at video rate; bout durations in the processed
#: behaviour file are counts of these frames.  Verified against the gap between
#: consecutive bout onsets (median ratio 60.0).
VIDEO_HZ = 60.0


def session_root(root: str | os.PathLike | None = None) -> Path:
    """Resolve the session directory, preferring an explicit argument.

    Falls back to ``$IDYN_HSU_ROOT``.  Raising here rather than defaulting to a
    hard-coded path is deliberate -- see the module docstring.
    """
    if root is None:
        env = os.environ.get("IDYN_HSU_ROOT")
        if not env:
            raise RuntimeError(
                "no Hsu session directory: pass root=... or set IDYN_HSU_ROOT to "
                "a session folder containing Clu_clock_corrected.csv, "
                "<AREA>Neurons.csv and behavior_labels_*_processed.csv"
            )
        root = env
    p = Path(root)
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")
    return p


@dataclass
class SessionData:
    """One area of one session, binned on a common continuous clock.

    ``spikes`` is ``(n_bins, n_units)`` of non-negative integer counts -- counts,
    not rates, and nothing here smooths, matching ``nlb.TrialData``.  Unlike that
    class there are no trials: this is a single continuous trajectory, and the
    two ways of cutting it into something the flow machinery can read are
    :meth:`segments` and :meth:`bout_average`.
    """

    spikes: np.ndarray
    behaviour: np.ndarray          # (n_bins,) int label per bin, -1 = not a behaviour
    kinematics: np.ndarray         # (n_bins, n_feat) mean over the bin; may be empty
    unit_ids: np.ndarray
    area: str
    bin_ms: float
    session: str
    provenance: dict = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        return int(self.spikes.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.spikes.shape[1])

    @property
    def rate_hz(self) -> np.ndarray:
        """Per-unit mean firing rate, for rate-stratified neuron splits."""
        return self.spikes.mean(axis=0) / (self.bin_ms / 1000.0)

    def summary(self) -> str:
        return (
            f"{self.session}/{self.area}: {self.n_bins} bins x {self.n_units} "
            f"units @ {self.bin_ms:g} ms ({self.n_bins * self.bin_ms / 60000:.1f} min), "
            f"mean rate {self.rate_hz.mean():.2f} Hz, "
            f"{len(set(self.behaviour.tolist())) - 1} behaviours"
        )

    # ---------------------------------------------------------------- views

    def segments(
        self,
        seg_bins: int = 500,
        smooth_ms: float = 40.0,
        sqrt_transform: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Uniform chop of the continuous recording: ``(n_seg, seg_bins, n_units)``.

        The no-averaging view.  Noisy by construction -- there is exactly one
        sample of each moment -- which is why any nonlinearity read off it must
        be scored with ``absolute_gain`` against a shuffle null rather than by a
        drop in linear R2 (CLAUDE.md §3.11, and `exp15b`'s "60% of nothing").

        Smoothing is applied to the *continuous* series before chopping, so
        segment boundaries do not create discontinuities the flow map would then
        be asked to fit.
        """
        R = _smooth(self.spikes.astype(np.float64), self.bin_ms, smooth_ms)
        R /= self.bin_ms / 1000.0
        if sqrt_transform:
            R = np.sqrt(np.maximum(R, 0.0))
        if normalize:
            R = _normalize(R)
        n_seg = self.n_bins // seg_bins
        if n_seg < 2:
            raise ValueError(f"seg_bins={seg_bins} leaves {n_seg} segments")
        return R[: n_seg * seg_bins].reshape(n_seg, seg_bins, self.n_units)

    def bout_average(
        self,
        window_bins: tuple[int, int] = (-10, 30),
        smooth_ms: float = 40.0,
        min_bouts: int = 20,
        sqrt_transform: bool = True,
        normalize: bool = True,
        labels: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Behaviour-type averaged rates: ``(n_behaviour, n_bins, n_units)``.

        The analogue of ``nlb.TrialData.condition_average``, and it is what makes
        this dataset comparable to MC_Maze rather than merely different from it:
        **a behaviour type plays the role of a condition**, i.e. of an initial
        condition, which is what §1.1's deterministic-flow target is about.  With
        ~28k bouts over 16 behaviours there are more repeats per condition here
        than MC_Maze has trials in total.

        Averaging is a candidate artefact in its own right -- it is how `exp15b`
        checked that MC_Maze's linearity was not manufactured -- so an experiment
        should read this *and* :meth:`segments`, never only one.

        Returns ``(rates, labels, n_bouts_per_label)``.
        """
        pre, post = window_bins
        if not pre < post:
            raise ValueError("window_bins must be (pre, post) with pre < post")
        onsets, lab = self.bout_onsets()
        R = _smooth(self.spikes.astype(np.float64), self.bin_ms, smooth_ms)
        R /= self.bin_ms / 1000.0

        want = np.unique(lab) if labels is None else np.asarray(labels)
        want = want[want >= 0]
        out, keep_lab, counts = [], [], []
        for c in want:
            idx = onsets[lab == c]
            idx = idx[(idx + pre >= 0) & (idx + post <= self.n_bins)]
            if len(idx) < min_bouts:
                continue
            acc = np.zeros((post - pre, self.n_units))
            for i in idx:
                acc += R[i + pre : i + post]
            out.append(acc / len(idx))
            keep_lab.append(int(c))
            counts.append(len(idx))
        if not out:
            raise ValueError("no behaviour had enough bouts in-window")
        A = np.stack(out)
        if sqrt_transform:
            A = np.sqrt(np.maximum(A, 0.0))
        if normalize:
            A = _normalize(A)
        return A, np.asarray(keep_lab), np.asarray(counts)

    def circshift_null(self, seed: int) -> "SessionData":
        """Per-neuron circular shift: the null every claim here is scored against.

        Preserves each unit's rate, autocorrelation, burstiness and marginal
        exactly, and destroys **only** the cross-neuron alignment that a shared
        latent would create -- so anything a population-level statistic reports
        on this surrogate is not latent structure.  Same null as `exp15` §10.3.

        It is not optional discipline.  On this dataset the apparent contraction
        of the fitted one-step map survives the shuffle almost unchanged (§14),
        i.e. it is the decay of a smoothed spike train's autocorrelation rather
        than anything dynamical; without the null it reads as a measurement.

        The behaviour labels and kinematics are **not** shifted, so a shuffled
        session also destroys behaviour alignment -- which is what makes it the
        right control for :meth:`bout_average` as well as for :meth:`segments`.
        """
        rng = np.random.default_rng(seed)
        X = np.empty_like(self.spikes)
        for j in range(self.n_units):
            X[:, j] = np.roll(self.spikes[:, j], int(rng.integers(self.n_bins)))
        return SessionData(
            spikes=X,
            behaviour=self.behaviour,
            kinematics=self.kinematics,
            unit_ids=self.unit_ids,
            area=self.area,
            bin_ms=self.bin_ms,
            session=self.session,
            provenance={**self.provenance, "circshift_null_seed": int(seed)},
        )

    def bout_onsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Bin indices where the behaviour label changes, and the new label."""
        b = self.behaviour
        change = np.flatnonzero(np.diff(b) != 0) + 1
        return change, b[change]

    def record(self) -> dict:
        """The JSON-serialisable parameter block for a results file."""
        return {
            "session": self.session,
            "area": self.area,
            "n_bins": self.n_bins,
            "n_units": self.n_units,
            "bin_ms": self.bin_ms,
            "minutes": self.n_bins * self.bin_ms / 60000.0,
            "mean_rate_hz": float(self.rate_hz.mean()),
            "median_rate_hz": float(np.median(self.rate_hz)),
            "n_behaviours": int(len(set(self.behaviour.tolist())) - 1),
            "has_kinematics": bool(self.kinematics.size),
            **self.provenance,
        }


# ------------------------------------------------------------------ helpers


def _smooth(X: np.ndarray, bin_ms: float, smooth_ms: float) -> np.ndarray:
    if smooth_ms <= 0:
        return X.copy()
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(
        X, smooth_ms / bin_ms, axis=0, mode="nearest", truncate=3.0
    )


def _normalize(A: np.ndarray) -> np.ndarray:
    """Soft-normalise per unit, then centre and scale -- as `exp15b.prep`.

    The ``+0.5`` in the range denominator is the standard soft normalisation: it
    stops a near-silent unit from being amplified to the same dynamic range as a
    well-driven one, which otherwise lets sampling noise dominate the PCs.
    """
    axes = tuple(range(A.ndim - 1))
    rng_ = A.max(axis=axes) - A.min(axis=axes)
    A = A / (rng_ + 0.5)
    A = A - A.mean(axis=axes, keepdims=True)
    return A / A.std()


def _load_behaviour(root: Path, n_bins: int, bin_ms: float) -> np.ndarray:
    """Per-bin behaviour label from the processed bout file.

    That file is ``(onset_seconds, label, duration_frames)`` with durations
    counted in 60 Hz video frames -- checked against the gap between consecutive
    onsets rather than assumed.
    """
    hits = sorted(root.glob("behavior_labels_*_processed.csv"))
    if not hits:
        return np.full(n_bins, -1, dtype=np.int16)
    raw = np.loadtxt(hits[0], delimiter=",", ndmin=2)
    out = np.full(n_bins, -1, dtype=np.int16)
    start_bin = (raw[:, 0] * 1000.0 / bin_ms).astype(np.int64)
    stop_bin = start_bin + np.ceil(
        raw[:, 2] / VIDEO_HZ * 1000.0 / bin_ms
    ).astype(np.int64)
    for s, e, lab in zip(start_bin, stop_bin, raw[:, 1].astype(np.int16)):
        if s >= n_bins:
            continue
        out[s : min(e, n_bins)] = lab
    return out


def _load_kinematics(root: Path, n_bins: int, bin_ms: float) -> np.ndarray:
    """DLC features averaged within each bin: ``(n_bins, n_feat)``.

    Returned empty when the file is absent, so an area can still be loaded on a
    session that has no video.
    """
    hits = sorted(root.glob("*_kinematics.npy"))
    if not hits:
        return np.zeros((n_bins, 0))
    K = np.load(hits[0], mmap_mode="r")            # (n_feat, n_frames)
    n_feat, n_frames = K.shape
    frame_bin = (np.arange(n_frames) / VIDEO_HZ * 1000.0 / bin_ms).astype(np.int64)
    valid = frame_bin < n_bins
    frame_bin = frame_bin[valid]
    counts = np.bincount(frame_bin, minlength=n_bins).astype(float)
    out = np.zeros((n_bins, n_feat))
    for f in range(n_feat):
        v = np.asarray(K[f][: len(valid)][valid], dtype=float)
        out[:, f] = np.bincount(frame_bin, weights=np.nan_to_num(v), minlength=n_bins)
    return out / np.maximum(counts, 1.0)[:, None]


_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


def _spike_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(t_ms, cluster_id)`` from the clock-corrected CSV, cached as ``.npy``.

    15M rows: pandas parses it in seconds, ``np.loadtxt`` in minutes.  Neither
    is required after the first call -- the cache is what later runs read.
    """
    cache = _CACHE_DIR / f"hsu_{path.parent.name}_spikes.npy"
    if cache.exists():
        raw = np.load(cache)
        return raw[:, 0], raw[:, 1]
    try:
        import pandas as pd

        raw = pd.read_csv(
            path, header=None, names=["t", "c"], dtype=np.int64
        ).to_numpy()
    except ImportError:                                  # pragma: no cover
        raw = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache, raw)
    except OSError:                                      # pragma: no cover
        pass
    return raw[:, 0], raw[:, 1]


def load_session(
    area: str,
    root: str | os.PathLike | None = None,
    bin_ms: float = 20.0,
    min_rate_hz: float = 0.5,
    with_kinematics: bool = True,
) -> SessionData:
    """Bin one area of one session onto a continuous clock.

    ``min_rate_hz`` drops units too sparse to contribute.  Striatal units here
    are genuinely sparse (median ~0.3-0.5 Hz), so this threshold is doing real
    work rather than removing a handful of dead channels -- report how many
    units survive it, because it sets the population size and §3.13(e) found
    recovery turns on near 32 units per side.
    """
    if area not in AREAS:
        raise ValueError(f"unknown area {area!r}; expected one of {AREAS}")
    p = session_root(root)
    spikes_csv = p / "Clu_clock_corrected.csv"
    if not spikes_csv.exists():
        raise FileNotFoundError(f"missing {spikes_csv}")

    t_ms, clu = _spike_table(spikes_csv)
    n_bins = int(int(t_ms.max()) // bin_ms) + 1
    bin_idx = (t_ms / bin_ms).astype(np.int64)

    ids = np.loadtxt(p / f"{area}Neurons.csv", dtype=np.int64, ndmin=1)
    counts = np.bincount(clu)
    dur_s = n_bins * bin_ms / 1000.0
    keep = np.asarray(
        [i for i in ids if i < len(counts) and counts[i] / dur_s >= min_rate_hz],
        dtype=np.int64,
    )
    if keep.size < 8:
        raise ValueError(
            f"{area}: only {keep.size} units above {min_rate_hz} Hz -- too few to fit"
        )

    lut = np.full(int(clu.max()) + 1, -1, dtype=np.int64)
    lut[keep] = np.arange(keep.size)
    col = lut[clu]
    m = col >= 0
    flat = bin_idx[m] * keep.size + col[m]
    X = np.bincount(flat, minlength=n_bins * keep.size).reshape(n_bins, keep.size)

    return SessionData(
        spikes=X.astype(np.int32),
        behaviour=_load_behaviour(p, n_bins, bin_ms),
        kinematics=(
            _load_kinematics(p, n_bins, bin_ms)
            if with_kinematics
            else np.zeros((n_bins, 0))
        ),
        unit_ids=keep,
        area=area,
        bin_ms=float(bin_ms),
        session=p.name,
        provenance={
            "min_rate_hz": float(min_rate_hz),
            "n_units_listed": int(ids.size),
            "n_units_kept": int(keep.size),
            "source": "Hsu open-field, clock-corrected spike table",
        },
    )
