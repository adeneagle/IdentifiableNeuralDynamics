"""The linear base case: indecomposability, primary decomposition, intertwiners."""

from __future__ import annotations

import numpy as np
import pytest

from idyn import linear as L
from idyn import systems as S


# --------------------------------------------------------------------------
# Indecomposability
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "A, expected, why",
    [
        (np.diag([0.9, 0.7]), 2, "distinct real eigenvalues split"),
        (np.diag([0.9, 0.9]), 2, "a repeated semisimple eigenvalue also splits"),
        (np.array([[0.5, 1.0], [0.0, 0.5]]), 1, "a Jordan block does not"),
        (0.8 * S.rotation(0.4), 1, "a complex pair is one real block"),
        (np.diag([0.9]), 1, "1-D is always indecomposable"),
        (np.diag([0.9, 0.7, 0.5]), 3, "three distinct eigenvalues"),
    ],
)
def test_number_of_indecomposable_summands(A, expected, why):
    assert L.n_indecomposable_summands(A) == expected, why


def test_is_indecomposable_agrees_with_summand_count():
    for A in [np.diag([0.9, 0.7]), 0.8 * S.rotation(0.4), np.array([[0.5, 1.0], [0.0, 0.5]])]:
        assert L.is_indecomposable(A) == (L.n_indecomposable_summands(A) == 1)


def test_block_of_two_rotations_is_not_indecomposable():
    F = np.block([[0.9 * S.rotation(0.3), np.zeros((2, 2))],
                  [np.zeros((2, 2)), 0.7 * S.rotation(1.1)]])
    assert L.n_indecomposable_summands(F) == 2


# --------------------------------------------------------------------------
# Spectral separation and the certificate
# --------------------------------------------------------------------------


def test_spectral_separation_is_zero_for_shared_eigenvalues():
    assert L.spectral_separation([np.diag([0.8]), np.diag([0.8])]) == pytest.approx(0.0)
    assert L.spectral_separation([np.diag([0.8]), np.diag([0.5])]) == pytest.approx(0.3)


def test_certificate_passes_for_separated_indecomposable_blocks():
    F = np.block([[0.95 * S.rotation(0.4), np.zeros((2, 2))],
                  [np.zeros((2, 2)), 0.70 * S.rotation(1.1)]])
    cert = L.certify_finest_decomposition(F, [2, 2])
    assert cert.A1 and cert.A2 and cert.canonical
    assert cert.separation > 0.5


def test_certificate_rejects_mismatched_partition():
    with pytest.raises(ValueError, match="does not sum"):
        L.certify_finest_decomposition(np.eye(4), [2, 3])


# --------------------------------------------------------------------------
# Invariant subspaces
# --------------------------------------------------------------------------


def test_invariant_subspace_recovers_a_coordinate_block():
    A1, A2 = 0.95 * S.rotation(0.4), 0.70 * S.rotation(1.1)
    F = np.block([[A1, np.zeros((2, 2))], [np.zeros((2, 2)), A2]])
    Q = L.invariant_subspace(F, np.linalg.eigvals(A1))
    assert Q.shape == (4, 2)
    assert L.subspace_angle(Q, np.eye(4)[:, :2]) < 1e-9
    # invariance: F Q must stay in span(Q)
    assert np.allclose(Q @ (Q.T @ (F @ Q)), F @ Q, atol=1e-10)


def test_invariant_subspace_is_invariant_after_a_random_change_of_basis():
    rng = np.random.default_rng(0)
    A1, A2 = 0.95 * S.rotation(0.4), 0.70 * S.rotation(1.1)
    F = np.block([[A1, np.zeros((2, 2))], [np.zeros((2, 2)), A2]])
    T = rng.standard_normal((4, 4))
    G = T @ F @ np.linalg.inv(T)
    Q = L.invariant_subspace(G, np.linalg.eigvals(A1))
    assert np.allclose(Q @ (Q.T @ (G @ Q)), G @ Q, atol=1e-8)
    # it should be the image of the original block under T
    expected = np.linalg.qr(T @ np.eye(4)[:, :2])[0]
    assert L.subspace_angle(Q, expected) < 1e-7


def test_primary_decomposition_spans_the_whole_space():
    F = np.diag([0.9, 0.75, 0.6, 0.45])
    parts = L.primary_decomposition(F)
    assert len(parts) == 4
    Q = np.hstack([q for _, q in parts])
    assert np.linalg.matrix_rank(Q) == 4


def test_subspace_angle_basic_properties():
    E = np.eye(4)
    assert L.subspace_angle(E[:, :2], E[:, :2]) == pytest.approx(0.0, abs=1e-12)
    assert L.subspace_angle(E[:, :2], E[:, 2:]) == pytest.approx(np.pi / 2)


# --------------------------------------------------------------------------
# Intertwiner space -- the exact solution set of the linear problem
# --------------------------------------------------------------------------


def test_intertwiner_elements_actually_intertwine():
    rng = np.random.default_rng(1)
    F = np.block([[0.95 * S.rotation(0.4), np.zeros((2, 2))],
                  [np.zeros((2, 2)), 0.70 * S.rotation(1.1)]])
    P = np.eye(4)[[2, 3, 0, 1]]
    Ft = P @ F @ P.T
    H = L.intertwiner_space(F, Ft)
    assert H.shape[0] > 0
    for _ in range(20):
        A = np.tensordot(rng.standard_normal(H.shape[0]), H, axes=(0, 0))
        assert np.allclose(A @ F, Ft @ A, atol=1e-9)


def test_intertwiner_dimension_matches_theory_under_A1_and_A2():
    """dim Hom(F, F~) = sum_i dim End(f_i) when only matched blocks can pair."""
    A1, A2 = 0.95 * S.rotation(0.4), 0.70 * S.rotation(1.1)
    F = np.block([[A1, np.zeros((2, 2))], [np.zeros((2, 2)), A2]])
    P = np.eye(4)[[2, 3, 0, 1]]
    expected = L.intertwiner_space(A1, A1).shape[0] + L.intertwiner_space(A2, A2).shape[0]
    assert L.intertwiner_space(F, P @ F @ P.T).shape[0] == expected


def test_every_invertible_intertwiner_is_a_block_permutation_under_A1_A2():
    """The linear theorem, tested directly."""
    rng = np.random.default_rng(2)
    A1, A2 = 0.95 * S.rotation(0.4), 0.70 * S.rotation(1.1)
    F = np.block([[A1, np.zeros((2, 2))], [np.zeros((2, 2)), A2]])
    P = np.eye(4)[[2, 3, 0, 1]]
    H = L.intertwiner_space(F, P @ F @ P.T)

    n_invertible = 0
    for _ in range(300):
        A = np.tensordot(rng.standard_normal(H.shape[0]), H, axes=(0, 0))
        rep = L.block_permutation_report(A, [2, 2], [2, 2])
        if rep.invertible:
            n_invertible += 1
            assert rep.is_block_permutation
            assert rep.on_block_fraction == pytest.approx(1.0)
    assert n_invertible > 250, "sanity: most random elements should be invertible"


def test_endomorphism_of_a_scaled_rotation_is_two_dimensional():
    """End(sR(w)) = span{I, R(pi/2)} for w not a multiple of pi -- the complex line."""
    assert L.intertwiner_space(0.9 * S.rotation(0.4), 0.9 * S.rotation(0.4)).shape[0] == 2


# --------------------------------------------------------------------------
# Block-permutation reporting
# --------------------------------------------------------------------------


def test_block_permutation_report_on_an_exact_swap():
    P = np.eye(4)[[2, 3, 0, 1]]
    rep = L.block_permutation_report(P, [2, 2], [2, 2])
    assert rep.is_block_permutation
    assert rep.assignment == (1, 0)
    assert rep.on_block_fraction == pytest.approx(1.0)


def test_block_permutation_report_on_a_block_diagonal_map():
    rng = np.random.default_rng(3)
    A = np.zeros((4, 4))
    A[:2, :2] = rng.standard_normal((2, 2))
    A[2:, 2:] = rng.standard_normal((2, 2))
    rep = L.block_permutation_report(A, [2, 2], [2, 2])
    assert rep.is_block_permutation
    assert rep.assignment == (0, 1)


def test_block_permutation_report_rejects_a_mixing_map():
    rng = np.random.default_rng(4)
    rep = L.block_permutation_report(rng.standard_normal((4, 4)), [2, 2], [2, 2])
    assert not rep.is_block_permutation
    assert 0.3 < rep.on_block_fraction < 0.95


def test_block_energy_matrix_partitions_total_energy():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((5, 5))
    E = L.block_energy_matrix(A, [2, 3], [1, 4])
    assert E.sum() == pytest.approx(np.sum(A**2))
    assert E.shape == (2, 2)


def test_block_permutation_requires_square_block_structure():
    with pytest.raises(ValueError, match="equally many"):
        L.block_permutation_report(np.zeros((4, 4)), [2, 2], [1, 1, 2])
