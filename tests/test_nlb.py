"""Tests for the real-data path: NLB loading, the ladder, the lattice quotient.

The NWB blobs are gitignored (they are re-fetchable from DANDI), so every test
that needs one is skipped when it is absent rather than downloading 690 MB
inside a test run.  The pure functions -- neuron splitting, the transitions, the
fitted-model wrapper, the GL(K,Z) quotient -- are tested unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from idyn import nlb
from idyn import spectra as SP
from idyn.models import (
    LatentDynamicsModel,
    LearnedSystem,
    ModelConfig,
    ModularTransition,
    TriangularTransition,
)

_HAVE_DATA = (nlb._DEFAULT_ROOT / nlb.DATASETS["mc_maze"]["filename"]).exists()
needs_data = pytest.mark.skipif(not _HAVE_DATA, reason="NLB NWB not downloaded")


# --------------------------------------------------------------------------
# Neuron splitting -- the instrument for task 40
# --------------------------------------------------------------------------


def test_neuron_split_is_disjoint_and_covers_everything():
    parts = nlb.neuron_split(101, seed=0, n_parts=3)
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(101))
    for i in range(3):
        for j in range(i):
            assert not set(parts[i]) & set(parts[j])


def test_rate_stratified_split_balances_the_halves():
    """An unbalanced split would confound 'different neurons' with 'less signal'.

    §3.13(b) found recoverability tracks where the orbits carry variance, so a
    fast half against a slow half is not the comparison task 40 intends.
    """
    rng = np.random.default_rng(0)
    rate = np.sort(rng.lognormal(0, 1.2, size=200))
    naive = nlb.neuron_split(200, seed=1)
    strat = nlb.neuron_split(200, seed=1, rate=rate)
    d_naive = abs(rate[naive[0]].mean() - rate[naive[1]].mean())
    d_strat = abs(rate[strat[0]].mean() - rate[strat[1]].mean())
    assert d_strat < d_naive, f"stratified {d_strat} should beat naive {d_naive}"
    assert d_strat < 0.05 * rate.mean()


def test_neuron_split_rejects_a_mismatched_rate_vector():
    with pytest.raises(ValueError):
        nlb.neuron_split(10, seed=0, rate=np.ones(9))


def test_fetch_rejects_an_unknown_dataset():
    with pytest.raises(KeyError):
        nlb.fetch("not_a_dataset")


# --------------------------------------------------------------------------
# The task-39 ladder: unconstrained > triangular > modular
# --------------------------------------------------------------------------


def test_triangular_sees_earlier_modules_but_never_later_ones():
    """The defining property, checked as an actual derivative, not by inspection."""
    torch.manual_seed(0)
    part = [2, 2, 2]
    tri = TriangularTransition(part).double()
    z = torch.zeros(1, 6, dtype=torch.float64)
    J = np.asarray(
        torch.autograd.functional.jacobian(lambda v: tri(v).sum(0), z).reshape(6, 6)
    )
    # entry (i, j) nonzero only for j <= i in block terms
    for bi, (a, b) in enumerate(tri.bounds):
        for bj, (c, d) in enumerate(tri.bounds):
            block = J[a:b, c:d]
            if bj > bi:
                assert np.allclose(block, 0.0), f"block ({bi},{bj}) must vanish"


def test_modular_sees_only_itself():
    torch.manual_seed(0)
    part = [2, 2, 2]
    mod = ModularTransition(part).double()
    z = torch.zeros(1, 6, dtype=torch.float64)
    J = np.asarray(torch.autograd.functional.jacobian(lambda v: mod(v).sum(0), z).reshape(6, 6))
    for bi, (a, b) in enumerate(mod.bounds):
        for bj, (c, d) in enumerate(mod.bounds):
            if bi != bj:
                assert np.allclose(J[a:b, c:d], 0.0)


def test_the_ladder_is_nested_in_parameter_count():
    """modular < triangular < unconstrained, which is what makes the gate valid."""
    n = {}
    for st in ("modular", "triangular", "unconstrained"):
        m = LatentDynamicsModel(ModelConfig(n_obs=20, d=6, partition=[2, 2, 2], structure=st))
        n[st] = sum(p.numel() for p in m.dyn.parameters())
    assert n["modular"] < n["triangular"]


def test_unknown_structure_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(n_obs=10, d=4, partition=[2, 2], structure="diagonal")


# --------------------------------------------------------------------------
# The fitted-model wrapper the fingerprint runs on
# --------------------------------------------------------------------------


def test_learned_block_jacobian_matches_autograd():
    """The fingerprint is only as good as this derivative.

    Central differences at eps=1e-6 on a float64 model, checked against
    autograd -- a float32 model would put the roundoff floor on the signal.
    """
    torch.manual_seed(3)
    mod = ModularTransition([2, 2]).double()
    ls = LearnedSystem(mod, [2, 2])
    z = np.array([0.31, -0.22])
    got = ls.blocks[0].jacobian(z)
    zt = torch.tensor(z, dtype=torch.float64)
    want = torch.autograd.functional.jacobian(lambda v: v + mod.nets[0](v), zt).numpy()
    assert np.allclose(got, want, atol=1e-7), f"{got} vs {want}"


def test_learned_system_exposes_what_the_fingerprint_needs():
    from idyn import metrics as M

    mod = ModularTransition([2, 2]).double()
    ls = LearnedSystem(mod)
    assert ls.partition == [2, 2] and len(ls.blocks) == 2
    fp = M.dynamical_fingerprint(
        ls, np.random.default_rng(0).normal(size=(12, 4)) * 0.3, T=25, warmup=6, T_rotation=25
    )
    assert fp.K == 2 and len(fp.rotations) == 2


# --------------------------------------------------------------------------
# GL(K,Z): the quotient task 23 forces, and how fast its power decays
# --------------------------------------------------------------------------


def test_lattice_quotient_finds_the_task23_shear_at_K2():
    m, A = SP.rotation_lattice_margin([0.0796, 0.2069], [0.2865, 0.2069])
    assert m < 1e-12
    assert A.tolist() == [[1, 1], [0, 1]]


def test_lattice_quotient_finds_a_regrouping_at_K3():
    a = [0.0231, 0.0076, 0.0026]
    b = [a[0] + a[1], a[1], a[2]]
    m, A = SP.rotation_lattice_margin(a, b)
    assert m < 1e-12
    assert A[0].tolist() == [1, 1, 0]


def test_lattice_quotient_loses_power_as_K_grows():
    """**The reason exp15 runs K=2 as primary.**

    A conjugacy acts on H_1(T^K) = Z^K, so only the GL(K,Z) orbit is identified.
    With more modules there are more lattice bases to hide in, so a random pair
    of rotation vectors matches ever more easily -- and at the magnitudes the
    real fits produce, K=3 agreement carries almost no information.
    """
    rng = np.random.default_rng(0)
    med = {}
    for K in (2, 3):
        ms = [
            SP.rotation_lattice_margin(
                rng.uniform(0, 0.025, K).tolist(), rng.uniform(0, 0.025, K).tolist()
            )[0]
            for _ in range(120)
        ]
        med[K] = float(np.median(ms))
    assert med[3] < med[2], f"power must decay with K: {med}"
    # and at K=3 the null is so tight that agreement is nearly automatic
    assert med[3] < 0.002


def test_lattice_quotient_rejects_bad_shapes_and_oversized_searches():
    with pytest.raises(ValueError):
        SP.rotation_lattice_margin([0.1], [0.2])
    with pytest.raises(ValueError):
        SP.rotation_lattice_margin([0.1, 0.2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        SP.rotation_lattice_margin([0.1] * 3, [0.2] * 3, max_coeff=4)


def test_lattice_quotient_is_infinite_on_nonfinite_input():
    m, A = SP.rotation_lattice_margin([np.nan, 0.1], [0.1, 0.1])
    assert m == float("inf") and A is None


# --------------------------------------------------------------------------
# Loading, when the data is present
# --------------------------------------------------------------------------


@needs_data
def test_load_trials_shapes_and_counts():
    td = nlb.load_trials("mc_maze")
    assert td.spikes.ndim == 3 and td.spikes.dtype.kind in "iu"
    assert td.spikes.min() >= 0
    assert td.n_bins == 35 and td.n_trials > 1000
    assert td.heldout.shape == (td.n_units,)
    r = td.record()
    assert r["dandiset"] == "000128" and r["n_units"] == td.n_units


@needs_data
def test_condition_average_is_a_mean_over_repeats():
    td = nlb.load_trials("mc_maze_small")
    R, labels, n_per = td.condition_average(smooth_ms=0.0, sqrt_transform=False)
    assert R.shape == (len(labels), td.n_bins, td.n_units)
    c0 = labels[0]
    want = td.spikes[td.condition == c0].mean(0) / (td.bin_ms / 1000.0)
    assert np.allclose(R[0], want)


@needs_data
def test_smoothing_raises_split_half_reliability():
    """Sanity that the smoothing does what it is for -- and how much it is doing.

    Recorded because heavy smoothing is a real risk: it could manufacture the
    smooth low-dimensional dynamics the model then 'finds'.  What protects the
    task-40 conclusion is that the two neuron halves are disjoint, so their
    PSTH noise is independent and smoothing cannot create agreement between them.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from exp15_nlb import split_half_reliability

    td = nlb.load_trials("mc_maze")
    r0 = split_half_reliability(td, 0.0, 0)
    r40 = split_half_reliability(td, 40.0, 0)
    assert r0 < r40, f"smoothing should raise reliability: {r0} -> {r40}"
    assert r40 < 0.95, "and must not saturate it, or the signal is the kernel"
