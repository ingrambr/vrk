"""
Tests for all covariance functions: shape, symmetry, PD, diagonal, gradient FD.
"""
import numpy as np
import pytest
from vrk.covariance.exponential import ExponentialCovariance
from vrk.covariance.spherical import SphericalCovariance
from vrk.covariance.gaussian import GaussianCovariance
from vrk.covariance.matern52 import Matern52Covariance
from vrk.covariance.nugget import NuggetCovariance
from vrk.covariance.sum import SumCovariance

RNG = np.random.default_rng(42)

# All 2-param covariances (sill, range_a)
TWO_PARAM_COVS = [
    ExponentialCovariance(sill=1.5, range_a=0.8),
    SphericalCovariance(sill=1.5, range_a=0.8),
    GaussianCovariance(sill=1.5, range_a=0.8),
    Matern52Covariance(sill=1.5, range_a=0.8),
]

ALL_COVS = TWO_PARAM_COVS + [NuggetCovariance(sill=0.5)]


def _fd_gradient(cov, X, eps=1e-5):
    """Central-difference FD gradient of trace(C(X)) w.r.t. log-params."""
    lp0 = cov.log_params.copy()
    grad = np.zeros(cov.n_params)
    for j in range(cov.n_params):
        lp_p = lp0.copy(); lp_p[j] += eps
        lp_m = lp0.copy(); lp_m[j] -= eps
        cov.log_params = lp_p
        vp = np.trace(cov(X))
        cov.log_params = lp_m
        vm = np.trace(cov(X))
        grad[j] = (vp - vm) / (2.0 * eps)
    cov.log_params = lp0
    return grad


class TestShape:
    def test_square(self):
        X = RNG.standard_normal((5, 2))
        for cov in ALL_COVS:
            C = cov(X)
            assert C.shape == (5, 5), f"{cov.__class__.__name__}: bad shape {C.shape}"

    def test_cross(self):
        X = RNG.standard_normal((4, 2))
        Y = RNG.standard_normal((3, 2))
        for cov in ALL_COVS:
            C = cov(X, Y)
            assert C.shape == (4, 3), f"{cov.__class__.__name__}: bad cross shape {C.shape}"

    def test_diag_shape(self):
        X = RNG.standard_normal((7, 1))
        for cov in ALL_COVS:
            d = cov.diag(X)
            assert d.shape == (7,), f"{cov.__class__.__name__}: bad diag shape"

    def test_diag_matches_diagonal(self):
        X = RNG.standard_normal((6, 1))
        for cov in ALL_COVS:
            assert np.allclose(cov.diag(X), np.diag(cov(X))), \
                f"{cov.__class__.__name__}: diag() mismatch"


class TestSymmetry:
    def test_symmetric(self):
        X = RNG.standard_normal((8, 1))
        for cov in ALL_COVS:
            C = cov(X)
            assert np.allclose(C, C.T, atol=1e-12), \
                f"{cov.__class__.__name__}: not symmetric"


class TestPositiveDefinite:
    def test_pd(self):
        X = np.linspace(0, 3, 10)[:, None]
        for cov in TWO_PARAM_COVS:
            C = cov(X) + 1e-8 * np.eye(10)
            eigvals = np.linalg.eigvalsh(C)
            assert np.all(eigvals > 0), \
                f"{cov.__class__.__name__}: not PD, min eigval={eigvals.min():.3e}"


class TestGetSetParams:
    def test_get_set_roundtrip(self):
        for cov in ALL_COVS:
            p0 = cov.get_params().copy()
            cov.set_params(p0 * 2.0)
            assert np.allclose(cov.get_params(), p0 * 2.0)
            cov.set_params(p0)

    def test_log_params_roundtrip(self):
        for cov in ALL_COVS:
            lp0 = cov.log_params.copy()
            cov.log_params = lp0 + 1.0
            assert np.allclose(cov.log_params, lp0 + 1.0, atol=1e-12)
            cov.log_params = lp0

    def test_n_params(self):
        for cov in TWO_PARAM_COVS:
            assert cov.n_params == 2
        assert NuggetCovariance(sill=1.0).n_params == 1


class TestGradient:
    def _check_grad(self, cov, X, rtol=0.05, atol=1e-4):
        # gradient_matrix returns natural-space gradients;
        # log-space gradient = grad_natural * theta
        GM = cov.gradient_matrix(X)
        assert len(GM) == cov.n_params
        grad_analytic_log = np.array([np.trace(G) for G in GM]) * np.exp(cov.log_params)
        grad_fd_log = _fd_gradient(cov, X)
        for j in range(cov.n_params):
            a, f = grad_analytic_log[j], grad_fd_log[j]
            err = abs(a - f) / max(abs(f), atol)
            assert err < rtol or abs(a - f) < atol, \
                f"{cov.__class__.__name__} param {j}: analytic={a:.6f} fd={f:.6f} err={err:.4f}"

    def test_exponential_gradient(self):
        X = np.linspace(0.1, 2.0, 8)[:, None]
        self._check_grad(ExponentialCovariance(sill=1.2, range_a=0.7), X)

    def test_spherical_gradient(self):
        X = np.linspace(0.1, 1.5, 8)[:, None]
        self._check_grad(SphericalCovariance(sill=1.2, range_a=2.0), X)

    def test_gaussian_gradient(self):
        X = np.linspace(0.1, 2.0, 8)[:, None]
        self._check_grad(GaussianCovariance(sill=1.2, range_a=0.7), X)

    def test_matern52_gradient(self):
        X = np.linspace(0.1, 2.0, 8)[:, None]
        self._check_grad(Matern52Covariance(sill=1.2, range_a=0.7), X)

    def test_nugget_gradient(self):
        X = np.linspace(0.1, 2.0, 5)[:, None]
        self._check_grad(NuggetCovariance(sill=0.3), X)


class TestSumCovariance:
    def test_sum_shape(self):
        cov = SumCovariance(ExponentialCovariance(1.0, 1.0), NuggetCovariance(0.1))
        X = RNG.standard_normal((5, 1))
        assert cov(X).shape == (5, 5)
        assert cov(X, X).shape == (5, 5)

    def test_sum_n_params(self):
        cov = SumCovariance(ExponentialCovariance(1.0, 1.0), NuggetCovariance(0.1))
        assert cov.n_params == 3

    def test_sum_get_set(self):
        cov = SumCovariance(GaussianCovariance(1.0, 1.0), NuggetCovariance(0.2))
        p0 = cov.get_params()
        cov.set_params(p0 * 1.5)
        assert np.allclose(cov.get_params(), p0 * 1.5)
        cov.set_params(p0)

    def test_sum_pd(self):
        cov = SumCovariance(ExponentialCovariance(1.0, 1.0), NuggetCovariance(0.05))
        X = np.linspace(0, 2, 8)[:, None]
        C = cov(X) + 1e-8 * np.eye(8)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_sum_gradient_shape(self):
        cov = SumCovariance(ExponentialCovariance(1.0, 1.0), NuggetCovariance(0.1))
        X = np.linspace(0, 2, 5)[:, None]
        GM = cov.gradient_matrix(X)
        assert len(GM) == 3
        for G in GM:
            assert G.shape == (5, 5)


class TestNuggetOffDiag:
    def test_nugget_cross_zero(self):
        nug = NuggetCovariance(sill=2.0)
        X = RNG.standard_normal((4, 1))
        Y = RNG.standard_normal((3, 1))
        C = nug(X, Y)
        assert np.all(C == 0.0)

    def test_nugget_self_diagonal(self):
        nug = NuggetCovariance(sill=2.0)
        X = RNG.standard_normal((5, 1))
        C = nug(X)
        assert np.allclose(C, 2.0 * np.eye(5))
