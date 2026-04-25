"""
End-to-end integration tests: VRK fit + predict on simple 1D data.
"""
import numpy as np
import pytest
from vrk.core.vrk import VRK
from vrk.covariance.exponential import ExponentialCovariance
from vrk.covariance.spherical import SphericalCovariance
from vrk.covariance.gaussian import GaussianCovariance
from vrk.covariance.matern52 import Matern52Covariance
from vrk.covariance.sum import SumCovariance
from vrk.covariance.nugget import NuggetCovariance
from vrk.likelihoods.gaussian import GaussianLikelihood

RNG = np.random.default_rng(0)


def _sin_data(n=40, noise=0.1):
    X = np.linspace(0, 2 * np.pi, n)[:, None]
    y = np.sin(X[:, 0]) + noise * RNG.standard_normal(n)
    return X, y


class TestFitPredict:
    """Basic fit + predict sanity checks for each covariance type."""

    def _run(self, cov, n=40, max_active=15, n_sweeps=3):
        X, y = _sin_data(n)
        lik = GaussianLikelihood(variance=0.1)
        model = VRK(cov, lik, max_active=max_active, n_sweeps=n_sweeps)
        model.fit(X, y)
        X_test = np.array([[np.pi / 2], [np.pi], [3 * np.pi / 2]])
        mean, var = model.predict(X_test)
        return model, mean, var

    def test_exponential_fit_predict(self):
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        model, mean, var = self._run(cov)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)
        assert model.n_active > 0

    def test_spherical_fit_predict(self):
        cov = SphericalCovariance(sill=1.0, range_a=2.0)
        model, mean, var = self._run(cov)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)
        assert model.n_active > 0

    def test_gaussian_fit_predict(self):
        cov = GaussianCovariance(sill=1.0, range_a=1.0)
        model, mean, var = self._run(cov)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)
        assert model.n_active > 0

    def test_matern52_fit_predict(self):
        cov = Matern52Covariance(sill=1.0, range_a=1.0)
        model, mean, var = self._run(cov)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)

    def test_sum_covariance_fit_predict(self):
        cov = SumCovariance(ExponentialCovariance(0.9, 1.0), NuggetCovariance(0.05))
        model, mean, var = self._run(cov)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)


class TestActiveSetBound:
    def test_active_set_bounded(self):
        X, y = _sin_data(50)
        cov = GaussianCovariance(sill=1.0, range_a=1.0)
        lik = GaussianLikelihood(variance=0.1)
        model = VRK(cov, lik, max_active=10, n_sweeps=2)
        model.fit(X, y)
        assert model.n_active <= 10


class TestPredictReasonable:
    """Predictions should be in the right ballpark for sin(x)."""

    def test_exponential_rmse(self):
        X, y = _sin_data(50, noise=0.05)
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        lik = GaussianLikelihood(variance=0.05)
        model = VRK(cov, lik, max_active=20, n_sweeps=4)
        model.fit(X, y)
        X_test = X
        mean, _ = model.predict(X_test)
        rmse = np.sqrt(np.mean((mean - np.sin(X_test[:, 0])) ** 2))
        assert rmse < 0.5, f"Exponential RMSE too high: {rmse:.3f}"

    def test_gaussian_rmse(self):
        X, y = _sin_data(50, noise=0.05)
        cov = GaussianCovariance(sill=1.0, range_a=1.0)
        lik = GaussianLikelihood(variance=0.05)
        model = VRK(cov, lik, max_active=20, n_sweeps=4)
        model.fit(X, y)
        mean, _ = model.predict(X)
        rmse = np.sqrt(np.mean((mean - np.sin(X[:, 0])) ** 2))
        assert rmse < 0.5, f"Gaussian RMSE too high: {rmse:.3f}"

    def test_spherical_rmse(self):
        X, y = _sin_data(50, noise=0.05)
        cov = SphericalCovariance(sill=1.0, range_a=3.0)
        lik = GaussianLikelihood(variance=0.05)
        model = VRK(cov, lik, max_active=20, n_sweeps=4)
        model.fit(X, y)
        mean, _ = model.predict(X)
        rmse = np.sqrt(np.mean((mean - np.sin(X[:, 0])) ** 2))
        assert rmse < 0.5, f"Spherical RMSE too high: {rmse:.3f}"


class TestParamsUnchangedByFit:
    def test_covariance_params_unchanged(self):
        cov = GaussianCovariance(sill=2.0, range_a=1.5)
        lik = GaussianLikelihood(variance=0.1)
        model = VRK(cov, lik, max_active=10)
        X, y = _sin_data(20)
        p0 = cov.get_params().copy()
        model.fit(X, y)
        assert np.allclose(cov.get_params(), p0)


class TestImportShortcut:
    def test_top_level_import(self):
        from vrk import VRK, ExponentialCovariance, SphericalCovariance, GaussianLikelihood
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        lik = GaussianLikelihood(variance=0.1)
        model = VRK(cov, lik, max_active=5)
        X = np.linspace(0, 3, 10)[:, None]
        y = np.sin(X[:, 0])
        model.fit(X, y)
        mean, var = model.predict(X)
        assert np.all(np.isfinite(mean))
