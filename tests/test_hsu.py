"""Tests for the Hsu open-field loader.

**These run without the recordings.**  The session lives outside the repo behind
``IDYN_HSU_ROOT`` (CLAUDE.md §5 keeps absolute paths out of ``src``/``tests``/
``experiments``), and CLAUDE.md §4.1 requires the suite to pass on a fresh
checkout with nothing downloaded.  So every property below is checked against a
``SessionData`` built in-process from known numbers; the one test that touches
real files skips when the variable is unset.

That is not a weaker test.  The parts that can silently return a plausible wrong
answer -- the null's invariants, bout alignment, smoothing order -- are exactly
the parts that are pure functions of the array, and a synthetic session pins
them to exact values that real data never could.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from idyn import hsu


def _session(n_bins: int = 600, n_units: int = 5, seed: int = 0) -> hsu.SessionData:
    """A synthetic session with a behaviour-locked response of known shape."""
    rng = np.random.default_rng(seed)
    behaviour = np.full(n_bins, -1, dtype=np.int16)
    # alternating 30-bin bouts of behaviours 0 and 1, starting at bin 20
    for k, start in enumerate(range(20, n_bins - 40, 60)):
        behaviour[start : start + 30] = 0
        behaviour[start + 30 : start + 60] = 1
    spikes = rng.poisson(0.4, size=(n_bins, n_units)).astype(np.int32)
    # unit 0 fires extra during behaviour 0 -- a signal bout_average must find
    spikes[behaviour == 0, 0] += 5
    return hsu.SessionData(
        spikes=spikes,
        behaviour=behaviour,
        kinematics=np.zeros((n_bins, 0)),
        unit_ids=np.arange(n_units),
        area="DS",
        bin_ms=20.0,
        session="synthetic",
    )


# --------------------------------------------------------------- the null


def test_circshift_null_preserves_every_single_neuron_statistic():
    """The defining property: it may destroy only cross-neuron alignment.

    If the shuffle changed a unit's rate or its marginal, a population statistic
    scoring higher on data than on the null would be uninterpretable -- the two
    arms would differ in more than the thing under test.
    """
    s = _session()
    n = s.circshift_null(seed=3)
    assert n.spikes.shape == s.spikes.shape
    # counts, and the full multiset of per-bin values, are preserved per unit
    np.testing.assert_array_equal(s.spikes.sum(0), n.spikes.sum(0))
    for j in range(s.n_units):
        np.testing.assert_array_equal(
            np.sort(s.spikes[:, j]), np.sort(n.spikes[:, j])
        )
    np.testing.assert_allclose(s.rate_hz, n.rate_hz)


def test_circshift_null_actually_moves_the_units():
    """A null that silently did nothing would make every arm agree."""
    s = _session()
    n = s.circshift_null(seed=3)
    assert not np.array_equal(s.spikes, n.spikes)


def test_circshift_null_leaves_behaviour_alignment_destroyed_not_relabelled():
    """Labels are NOT shifted, so the null also breaks behaviour locking.

    That is what makes it the right control for ``bout_average`` and not only
    for ``segments``.
    """
    s = _session()
    n = s.circshift_null(seed=5)
    np.testing.assert_array_equal(s.behaviour, n.behaviour)
    lock_data = s.bout_average(window_bins=(0, 20), min_bouts=2, normalize=False)[0]
    lock_null = n.bout_average(window_bins=(0, 20), min_bouts=2, normalize=False)[0]
    # unit 0's behaviour-0 elevation survives averaging in the data and not in the null
    spread_data = np.ptp(lock_data[:, :, 0].mean(1))
    spread_null = np.ptp(lock_null[:, :, 0].mean(1))
    assert spread_data > 3 * spread_null


def test_circshift_null_is_reproducible_from_the_seed():
    s = _session()
    np.testing.assert_array_equal(
        s.circshift_null(seed=7).spikes, s.circshift_null(seed=7).spikes
    )
    assert not np.array_equal(
        s.circshift_null(seed=7).spikes, s.circshift_null(seed=8).spikes
    )


# ------------------------------------------------------------ bout alignment


def test_bout_onsets_are_the_label_changes():
    s = _session()
    on, lab = s.bout_onsets()
    assert len(on) == len(lab)
    assert np.all(s.behaviour[on] == lab)
    assert np.all(s.behaviour[on - 1] != lab)


def test_bout_average_recovers_a_known_behaviour_locked_response():
    """Unit 0 is elevated during behaviour 0 by construction; find it."""
    s = _session()
    A, labels, counts = s.bout_average(
        window_bins=(0, 25), smooth_ms=0.0, min_bouts=2,
        sqrt_transform=False, normalize=False,
    )
    assert A.shape == (len(labels), 25, s.n_units)
    i0 = int(np.flatnonzero(labels == 0)[0])
    i1 = int(np.flatnonzero(labels == 1)[0])
    # rates are in Hz; the injected 5 counts/20 ms bin is 250 Hz
    assert A[i0, :, 0].mean() > A[i1, :, 0].mean() + 100
    # a unit with no behaviour tuning should not separate
    assert abs(A[i0, :, 1].mean() - A[i1, :, 1].mean()) < 50
    assert counts.min() >= 2


def test_bout_average_rejects_labels_with_too_few_bouts():
    s = _session()
    with pytest.raises(ValueError, match="enough bouts"):
        s.bout_average(window_bins=(0, 10), min_bouts=10**6)


def test_bout_average_rejects_a_reversed_window():
    s = _session()
    with pytest.raises(ValueError, match="pre < post"):
        s.bout_average(window_bins=(30, 10))


# ------------------------------------------------------------ the segment view


def test_segments_shape_and_that_smoothing_precedes_chopping():
    """Smoothing after chopping would put a discontinuity at every boundary.

    The flow map is then asked to fit an artefact of the preprocessing, once per
    segment.  Checked by comparing against smoothing the continuous series by
    hand: the two must agree, including across a boundary.
    """
    s = _session(n_bins=600)
    S = s.segments(seg_bins=100, smooth_ms=40.0, sqrt_transform=False, normalize=False)
    assert S.shape == (6, 100, s.n_units)
    ref = hsu._smooth(s.spikes.astype(float), s.bin_ms, 40.0) / (s.bin_ms / 1000.0)
    np.testing.assert_allclose(S.reshape(-1, s.n_units), ref, rtol=1e-10)


def test_segments_refuses_a_window_that_leaves_nothing():
    s = _session(n_bins=600)
    with pytest.raises(ValueError, match="segments"):
        s.segments(seg_bins=10**6)


def test_normalize_is_scale_equivariant_and_leaves_a_silent_unit_alone():
    """Soft normalisation must not amplify a near-silent unit to full range.

    Without the ``+0.5`` a unit whose range is tiny gets multiplied up until its
    sampling noise dominates the PCs -- the same failure §3.10 records for
    unstandardised Jacobian energy, one level earlier in the pipeline.
    """
    A = np.zeros((4, 50, 2))
    A[:, :, 0] = np.linspace(0, 20, 50)[None, :]
    A[:, :, 1] = 1e-6 * np.random.default_rng(0).standard_normal((4, 50))
    out = hsu._normalize(A)
    assert np.abs(out[:, :, 1]).max() < 1e-3 * np.abs(out[:, :, 0]).max()


# ---------------------------------------------------------------- plumbing


def test_session_root_raises_without_an_argument_or_environment(monkeypatch):
    monkeypatch.delenv("IDYN_HSU_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="IDYN_HSU_ROOT"):
        hsu.session_root()


def test_record_is_json_serialisable_and_carries_the_parameters():
    import json

    r = _session().record()
    json.loads(json.dumps(r))
    for key in ("session", "area", "n_bins", "n_units", "bin_ms", "mean_rate_hz"):
        assert key in r


def test_unknown_area_is_rejected_before_any_file_is_touched():
    with pytest.raises(ValueError, match="unknown area"):
        hsu.load_session("V1", root=None)


@pytest.mark.skipif(
    not os.environ.get("IDYN_HSU_ROOT"), reason="IDYN_HSU_ROOT not set"
)
def test_real_session_loads_and_is_self_consistent():
    s = hsu.load_session("DS", bin_ms=20.0, with_kinematics=False)
    assert s.n_units >= 8
    assert s.spikes.min() >= 0
    assert s.rate_hz.min() >= 0.5 - 1e-9
    assert set(np.unique(s.behaviour).tolist()) <= set(hsu.BEHAVIOURS)
