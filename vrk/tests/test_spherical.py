"""
Dedicated tests for SphericalCovariance.
"""
import numpy as np
import pytest
from vrk.covariance.spherical import SphericalCovariance


def _fd_grad_logspace(cov, X, eps=1e-5):
    """Central-difference FD gradient of trace(C(X,X)) w.r.t. log-params."""
    lp0 = cov.log_params.copy()
    grad = np.zeros(cov.n_params)
    for j in range(cov.n_params):
        lp_p = lp0.copy(); lp_p[j] += eps
        lp_m = lp0.copy(); lp_m[j] -= eps
        cov.log_params = lp_p; vp = np.trace(cov(X, X))
        cov.log_params = lp_m; vm = np.trace(cov(X, X))
        grad[j] = (vp - vm) / (2.0 * eps)
    cov.log_params = lp0
    return grad


class TestSphericalBasic:
    def test_sill_at_zero_lag(self):
        cov = SphericalCovariance(sill=2.0, range_a=1.5)
        X = np.array([[0.0]])
        assert np.isclose(cov.diag(X)[0], 2.0)
        assert np.isclose(cov(X)[0, 0], 2.0)

    def test_zero_beyond_range(self):
        sill, a = 1.5, 1.0
        cov = SphericalCovariance(sill=sill, range_a=a)
        # Points separated by exactly range and beyond
        X = np.array([[0.0]])
        Y = np.array([[a + 0.01]])
        assert cov(X, Y)[0, 0] == 0.0

        Y2 = np.array([[2 * a]])
        assert cov(X, Y2)[0, 0] == 0.0

    def test_nonzero_inside_range(self):
        cov = SphericalCovariance(sill=1.0, range_a=2.0)
        X = np.array([[0.0]])
        Y = np.array([[1.0]])   # h=1 < a=2
        assert cov(X, Y)[0, 0] > 0.0

    def test_monotone_decreasing(self):
        cov = SphericalCovariance(sill=1.0, range_a=3.0)
        h_vals = np.linspace(0.0, 2.9, 20)
        X0 = np.array([[0.0]])
        values = [cov(X0, np.array([[h]]))[0, 0] for h in h_vals]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-12, \
                f"Not monotone at h={h_vals[i]:.3f}: C={values[i]:.6f} > C={values[i+1]:.6f}"

    def test_symmetric(self):
        cov = SphericalCovariance(sill=1.0, range_a=2.0)
        X = np.linspace(0, 1.5, 6)[:, None]
        C = cov(X)
        assert np.allclose(C, C.T, atol=1e-14)

    def test_diag_equals_sill(self):
        cov = SphericalCovariance(sill=3.0, range_a=1.0)
        X = np.linspace(0, 5, 10)[:, None]
        assert np.allclose(cov.diag(X), 3.0)

    def test_diag_matches_matrix_diag(self):
        cov = SphericalCovariance(sill=1.0, range_a=2.0)
        X = np.linspace(0, 1.5, 8)[:, None]
        assert np.allclose(cov.diag(X), np.diag(cov(X)))

    def test_n_params(self):
        assert SphericalCovariance(1.0, 1.0).n_params == 2

    def test_get_set_params(self):
        cov = SphericalCovariance(sill=2.0, range_a=3.0)
        p = cov.get_params()
        assert np.allclose(p, [2.0, 3.0])
        cov.set_params(np.array([1.0, 1.0]))
        assert np.allclose(cov.get_params(), [1.0, 1.0])


class TestSphericalPD:
    def test_pd_1d(self):
        cov = SphericalCovariance(sill=1.0, range_a=2.0)
        X = np.linspace(0, 1.5, 10)[:, None]
        C = cov(X) + 1e-8 * np.eye(10)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"Min eigval = {eigvals.min():.3e}"

    def test_pd_2d(self):
        rng = np.random.default_rng(7)
        cov = SphericalCovariance(sill=1.0, range_a=2.0)
        X = rng.uniform(0, 1.5, (12, 2))
        C = cov(X) + 1e-8 * np.eye(12)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"Min eigval = {eigvals.min():.3e}"


class TestSphericalGradient:
    def test_gradient_fd_sill(self):
        cov = SphericalCovariance(sill=1.2, range_a=2.0)
        X = np.linspace(0.1, 1.8, 8)[:, None]
        GM = cov.gradient_matrix(X)
        grad_analytic_log = np.array([np.trace(G) for G in GM]) * np.exp(cov.log_params)
        grad_fd = _fd_grad_logspace(cov, X)
        assert abs(grad_analytic_log[0] - grad_fd[0]) / max(abs(grad_fd[0]), 1e-4) < 0.05

    def test_gradient_fd_range(self):
        cov = SphericalCovariance(sill=1.2, range_a=2.0)
        X = np.linspace(0.1, 1.8, 8)[:, None]
        GM = cov.gradient_matrix(X)
        grad_analytic_log = np.array([np.trace(G) for G in GM]) * np.exp(cov.log_params)
        grad_fd = _fd_grad_logspace(cov, X)
        assert abs(grad_analytic_log[1] - grad_fd[1]) / max(abs(grad_fd[1]), 1e-4) < 0.05

    def test_gradient_zero_outside_range(self):
        cov = SphericalCovariance(sill=1.0, range_a=0.5)
        # All points separated by > range_a
        X = np.array([[0.0], [1.0], [2.0]])
        GM = cov.gradient_matrix(X)
        # Off-diagonal of dC_da should be zero (distances > range_a)
        assert GM[1][0, 1] == 0.0
        assert GM[1][0, 2] == 0.0

    def test_gradient_shape(self):
        cov = SphericalCovariance(sill=1.0, range_a=2.0)
        X = np.linspace(0, 1.5, 6)[:, None]
        GM = cov.gradient_matrix(X)
        assert len(GM) == 2
        for G in GM:
            assert G.shape == (6, 6)
