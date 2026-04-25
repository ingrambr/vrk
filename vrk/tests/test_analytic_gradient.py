"""
Tests for analytic gradient of active_set_log_evidence().

Verifies analytic gradient agrees with partial-FD (EP state fixed) to rtol=0.05.
"""
import numpy as np
import scipy.linalg
import pytest
from vrk.core.vrk import VRK
from vrk.covariance.exponential import ExponentialCovariance
from vrk.covariance.spherical import SphericalCovariance
from vrk.covariance.gaussian import GaussianCovariance
from vrk.covariance.matern52 import Matern52Covariance
from vrk.covariance.sum import SumCovariance
from vrk.covariance.nugget import NuggetCovariance
from vrk.likelihoods.gaussian import GaussianLikelihood


def _partial_fd_gradient(model, eps=1e-5):
    """
    Central-difference gradient of active_set_log_evidence() holding
    the EP state (α_, C_, EP sites) fixed — only KB changes via L_.
    """
    cov = model.covariance
    m = model.n_active
    lp0 = cov.log_params.copy()

    alpha0 = model.alpha_.copy()
    C0 = model.C_.copy()
    L0 = model.L_.copy()
    active0 = model.active_set_.copy()
    var_ep0 = model.var_ep_.copy()
    mean_ep0 = model.mean_ep_.copy()
    log_z0 = model.log_z_.copy()

    grad = np.zeros(cov.n_params)
    for j in range(cov.n_params):
        evs = []
        for sign in (+1, -1):
            model.alpha_ = alpha0.copy()
            model.C_ = C0.copy()
            model.active_set_ = active0.copy()
            model.var_ep_ = var_ep0.copy()
            model.mean_ep_ = mean_ep0.copy()
            model.log_z_ = log_z0.copy()
            lp_p = lp0.copy()
            lp_p[j] += sign * eps
            cov.log_params = lp_p
            KB_new = cov(model.active_set_, model.active_set_)
            model.L_ = scipy.linalg.cholesky(KB_new + 1e-10 * np.eye(m), lower=True)
            evs.append(model.active_set_log_evidence())
        grad[j] = (evs[0] - evs[1]) / (2.0 * eps)

    cov.log_params = lp0
    model.alpha_ = alpha0.copy()
    model.C_ = C0.copy()
    model.L_ = L0.copy()
    model.active_set_ = active0.copy()
    model.var_ep_ = var_ep0.copy()
    model.mean_ep_ = mean_ep0.copy()
    model.log_z_ = log_z0.copy()
    return grad


def _fitted(cov, n=25, max_active=8, n_sweeps=2):
    X = np.linspace(0, 5, n)[:, None]
    y = np.sin(X[:, 0])
    lik = GaussianLikelihood(variance=0.05)
    model = VRK(cov, lik, max_active=max_active, n_sweeps=n_sweeps)
    model.fit(X, y)
    return model


class TestAnalyticGradientShape:
    def test_shape_exponential(self):
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        model = _fitted(cov)
        grad = model.analytic_evidence_gradient()
        assert grad.shape == (2,)

    def test_shape_spherical(self):
        cov = SphericalCovariance(sill=1.0, range_a=3.0)
        model = _fitted(cov)
        grad = model.analytic_evidence_gradient()
        assert grad.shape == (2,)

    def test_shape_gaussian(self):
        cov = GaussianCovariance(sill=1.0, range_a=1.0)
        model = _fitted(cov)
        grad = model.analytic_evidence_gradient()
        assert grad.shape == (2,)

    def test_finite_exponential(self):
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        model = _fitted(cov)
        grad = model.analytic_evidence_gradient()
        assert np.all(np.isfinite(grad)), f"Non-finite gradient: {grad}"

    def test_empty_active_set_returns_zeros(self):
        cov = GaussianCovariance(sill=1.0, range_a=1.0)
        lik = GaussianLikelihood(variance=0.1)
        model = VRK(cov, lik, max_active=0)
        X = np.linspace(0, 3, 5)[:, None]
        model.fit(X, np.sin(X[:, 0]))
        grad = model.analytic_evidence_gradient()
        assert np.all(grad == 0.0)


class TestAnalyticVsPartialFD:
    """Analytic gradient should agree with partial-FD (EP state fixed)."""

    def _compare(self, cov, n=20, max_active=6, rtol=0.05, atol=1e-3):
        model = _fitted(cov, n=n, max_active=max_active, n_sweeps=2)
        if model.n_active == 0:
            pytest.skip("No active points")

        grad_analytic = model.analytic_evidence_gradient()
        grad_fd = _partial_fd_gradient(model)

        assert np.all(np.isfinite(grad_analytic)), f"Non-finite: {grad_analytic}"
        assert np.all(np.isfinite(grad_fd)), f"Non-finite FD: {grad_fd}"

        for j in range(len(grad_analytic)):
            a, f = grad_analytic[j], grad_fd[j]
            err = abs(a - f) / max(abs(f), atol)
            assert err < rtol or abs(a - f) < atol, (
                f"Param {j}: analytic={a:.6f}, FD={f:.6f}, rel_err={err:.4f}"
            )

    def test_exponential(self):
        self._compare(ExponentialCovariance(sill=1.0, range_a=1.0))

    def test_spherical(self):
        self._compare(SphericalCovariance(sill=1.0, range_a=3.0))

    def test_gaussian(self):
        self._compare(GaussianCovariance(sill=1.0, range_a=1.0))

    def test_matern52(self):
        self._compare(Matern52Covariance(sill=1.0, range_a=1.0))


class TestSumGradient:
    def test_exponential_plus_nugget(self):
        cov = SumCovariance(ExponentialCovariance(1.0, 1.0), NuggetCovariance(0.01))
        model = _fitted(cov, max_active=8)
        if model.n_active == 0:
            pytest.skip("No active points")
        grad = model.analytic_evidence_gradient()
        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))
