"""(FLAT-D) via the telescoping construction  h = lim Psi^-n Phi^n.

Route A, Tier 2 rests on (FLAT-D): two C^inf contraction germs with equal
inf-jets at 0 are C^inf-conjugate by a flat-tangent map. The existence half is
located (Chaperon 1986, Thm 2(i)); the flat-tangent version follows from the
classical wave-operator limit, which these tests exercise. See
theory/route_a_assessment.md §2.4 and experiments/exp07_flat_tangency.py.
"""

from __future__ import annotations

import numpy as np
import pytest


def flat(x, kind="exp2"):
    """C^inf function flat to infinite order at 0."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    nz = x != 0
    if kind == "exp2":
        out[nz] = np.exp(-1.0 / x[nz] ** 2)
    else:  # exp1: flatter-decaying, the harder case
        out[nz] = np.exp(-1.0 / np.abs(x[nz]))
    return out


def make_pair(lam, beta, c, kind):
    Phi = lambda x: lam * np.asarray(x, float) + beta * np.asarray(x, float) ** 3
    Psi = lambda x: Phi(x) + c * flat(x, kind)

    def Psi_inv(y):
        y = np.asarray(y, float)
        lo, hi = np.full_like(y, -3.0), np.full_like(y, 3.0)
        for _ in range(200):
            m = 0.5 * (lo + hi)
            f = Psi(m) - y
            hi = np.where(f > 0, m, hi)
            lo = np.where(f <= 0, m, lo)
        return 0.5 * (lo + hi)

    return Phi, Psi, Psi_inv


def h_n(Phi, Psi_inv, x, n):
    z = np.asarray(x, float).copy()
    for _ in range(n):
        z = Phi(z)
    for _ in range(n):
        z = Psi_inv(z)
    return z


CASES = [
    (0.60, 0.00, 0.40, "exp2"),
    (0.60, 0.50, 0.30, "exp2"),
    (0.60, 0.50, 0.30, "exp1"),
    (0.45, 0.30, 0.35, "exp1"),
]


@pytest.mark.parametrize("lam,beta,c,kind", CASES)
def test_telescoping_limit_converges(lam, beta, c, kind):
    Phi, Psi, Psi_inv = make_pair(lam, beta, c, kind)
    xs = np.linspace(-0.6, 0.6, 25)
    a = h_n(Phi, Psi_inv, xs, 12)
    b = h_n(Phi, Psi_inv, xs, 20)
    assert np.max(np.abs(a - b)) < 1e-10, "h_n is Cauchy -> the limit exists"


@pytest.mark.parametrize("lam,beta,c,kind", CASES)
def test_limit_is_a_conjugacy(lam, beta, c, kind):
    Phi, Psi, Psi_inv = make_pair(lam, beta, c, kind)
    xs = np.linspace(-0.6, 0.6, 25)
    h = lambda x: h_n(Phi, Psi_inv, x, 40)
    assert np.max(np.abs(h(Phi(xs)) - Psi(h(xs)))) < 1e-10, "h(Phi x) = Psi(h x)"


@pytest.mark.parametrize("lam,beta,c,kind", CASES)
def test_conjugacy_is_flat_tangent_to_identity(lam, beta, c, kind):
    """(h(x) - x)/x^3 -> 0 as x -> 0: the flat-tangency clause of (FLAT-D)."""
    Phi, Psi, Psi_inv = make_pair(lam, beta, c, kind)
    xk = np.array([0.2, 0.1, 0.05, 0.02])
    hk = h_n(Phi, Psi_inv, xk, 40)
    ratio = np.abs((hk - xk) / xk ** 3)
    # at the smallest x the ratio has collapsed to ~1e-11 (h(x)-x at machine
    # precision); a uniform bound would wrongly fail at x=0.2 where h(x)-x ~ x^3.
    assert ratio[-1] < 1e-6


def test_derivative_at_fixed_point_is_one():
    """h'(0) = (Psi^-n)'(0) (Phi^n)'(0) = lam^-n lam^n = 1 exactly."""
    Phi, Psi, Psi_inv = make_pair(0.6, 0.5, 0.3, "exp1")
    dx = 1e-6
    hp = (h_n(Phi, Psi_inv, dx, 40) - h_n(Phi, Psi_inv, -dx, 40)) / (2 * dx)
    assert hp == pytest.approx(1.0, abs=1e-6)
