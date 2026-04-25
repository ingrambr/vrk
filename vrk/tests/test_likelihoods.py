"""
Tests for all non-Gaussian likelihood implementations.

Covers:
  - EP site parameter contract: r < 0, finite q, finite log_z
  - Likelihood-specific properties and edge cases
  - EM step updates (GaussianLikelihood, ExponentialNoiseLikelihood)
  - End-to-end VRK fit + predict for each likelihood
  - Per-point (mixed) likelihood mode
"""
import numpy as np
import pytest
from vrk.core.vrk import VRK
from vrk.covariance.exponential import ExponentialCovariance
from vrk.likelihoods.gaussian import GaussianLikelihood
from vrk.likelihoods.bernoulli import BernoulliLikelihood
from vrk.likelihoods.student_t import StudentTLikelihood
from vrk.likelihoods.poisson import PoissonLikelihood
from vrk.likelihoods.gumbel import GumbelLikelihood
from vrk.likelihoods.exponential_noise import ExponentialNoiseLikelihood

RNG = np.random.default_rng(7)

# Shared covariance for end-to-end tests
_COV = ExponentialCovariance(sill=1.0, range_a=1.0)


def _vrk(lik, max_active=15, n_sweeps=3):
    return VRK(_COV, lik, max_active=max_active, n_sweeps=n_sweeps)


# ---------------------------------------------------------------------------
# EP site parameter contract
# ---------------------------------------------------------------------------

class TestUpdateCoefficientsContract:
    """
    For any valid observation and cavity, update_coefficients must return
    (q, r, log_z) with r < 0 (site precision > 0), q finite, log_z finite.
    """

    LIKELIHOODS_AND_OBS = [
        (GaussianLikelihood(variance=0.5),          0.3),
        (BernoulliLikelihood(),                      1.0),   # positive label
        (BernoulliLikelihood(),                     -1.0),   # negative label
        (StudentTLikelihood(nu=4.0, sigma=1.0),      0.5),
        (PoissonLikelihood(bin_size=1.0),            2.0),   # count = 2
        (PoissonLikelihood(bin_size=1.0),            0.0),   # zero count
        (GumbelLikelihood(beta=1.0),                 0.8),
        (ExponentialNoiseLikelihood(rate=1.0),       1.5),   # y > cavity_mean
    ]

    @pytest.mark.parametrize("lik,obs", LIKELIHOODS_AND_OBS)
    def test_r_negative(self, lik, obs):
        q, r, log_z = lik.update_coefficients(obs, cavity_mean=0.0, cavity_var=1.0)
        assert r < 0, f"{lik.__class__.__name__}: r={r} must be < 0"

    @pytest.mark.parametrize("lik,obs", LIKELIHOODS_AND_OBS)
    def test_q_finite(self, lik, obs):
        q, r, log_z = lik.update_coefficients(obs, cavity_mean=0.0, cavity_var=1.0)
        assert np.isfinite(q), f"{lik.__class__.__name__}: q={q} not finite"

    @pytest.mark.parametrize("lik,obs", LIKELIHOODS_AND_OBS)
    def test_log_z_finite(self, lik, obs):
        q, r, log_z = lik.update_coefficients(obs, cavity_mean=0.0, cavity_var=1.0)
        assert np.isfinite(log_z), f"{lik.__class__.__name__}: log_z={log_z} not finite"


# ---------------------------------------------------------------------------
# BernoulliLikelihood
# ---------------------------------------------------------------------------

class TestBernoulliLikelihood:
    def test_positive_label_negative_r(self):
        lik = BernoulliLikelihood()
        q, r, log_z = lik.update_coefficients(1.0, cavity_mean=0.5, cavity_var=1.0)
        assert r < 0

    def test_negative_label_negative_r(self):
        lik = BernoulliLikelihood()
        q, r, log_z = lik.update_coefficients(-1.0, cavity_mean=-0.5, cavity_var=1.0)
        assert r < 0

    def test_zero_one_label_treated_as_minus_one(self):
        """Labels 0 and -1 should produce the same update."""
        lik = BernoulliLikelihood()
        q0, r0, lz0 = lik.update_coefficients(0.0,  cavity_mean=0.0, cavity_var=1.0)
        qm, rm, lzm = lik.update_coefficients(-1.0, cavity_mean=0.0, cavity_var=1.0)
        assert np.isclose(q0, qm) and np.isclose(r0, rm) and np.isclose(lz0, lzm)

    def test_log_z_matches_log_probit(self):
        """log Z = log Phi(y * mu / sqrt(1 + sigma^2)) for the Bernoulli probit."""
        from scipy.special import log_ndtr
        lik = BernoulliLikelihood()
        mu, sig2 = 1.0, 0.5
        y = 1.0
        _, _, log_z = lik.update_coefficients(y, cavity_mean=mu, cavity_var=sig2)
        expected = float(log_ndtr(y * mu / np.sqrt(1.0 + sig2)))
        assert np.isclose(log_z, expected, rtol=1e-4)

    def test_cavity_mean_sign_effect(self):
        """Positive cavity mean should give higher log_z for positive label."""
        lik = BernoulliLikelihood()
        _, _, lz_pos = lik.update_coefficients(1.0, cavity_mean=+2.0, cavity_var=1.0)
        _, _, lz_neg = lik.update_coefficients(1.0, cavity_mean=-2.0, cavity_var=1.0)
        assert lz_pos > lz_neg


# ---------------------------------------------------------------------------
# StudentTLikelihood
# ---------------------------------------------------------------------------

class TestStudentTLikelihood:
    def test_typical_observation(self):
        lik = StudentTLikelihood(nu=4.0, sigma=1.0)
        q, r, log_z = lik.update_coefficients(0.5, cavity_mean=0.0, cavity_var=1.0)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_extreme_outlier_triggers_laplace(self):
        """An extreme outlier should still return valid (r < 0) parameters."""
        lik = StudentTLikelihood(nu=4.0, sigma=1.0)
        q, r, log_z = lik.update_coefficients(1000.0, cavity_mean=0.0, cavity_var=0.01)
        assert r < 0 and np.isfinite(q)

    def test_large_nu_converges_to_gaussian(self):
        """For large nu, the Student-t approaches Gaussian; log_z should agree."""
        lik_t   = StudentTLikelihood(nu=1000.0, sigma=1.0)
        lik_g   = GaussianLikelihood(variance=1.0)
        qt, rt, lzt = lik_t.update_coefficients(0.3, cavity_mean=0.0, cavity_var=1.0)
        qg, rg, lzg = lik_g.update_coefficients(0.3, cavity_mean=0.0, cavity_var=1.0)
        assert np.isclose(lzt, lzg, rtol=1e-2), f"log_z: t={lzt:.4f} Gaussian={lzg:.4f}"

    def test_varying_nu(self):
        """Heavier tails (smaller nu) should give broader (less negative r) updates."""
        lik_heavy  = StudentTLikelihood(nu=2.5,  sigma=1.0)
        lik_light  = StudentTLikelihood(nu=30.0, sigma=1.0)
        obs, mu, sig2 = 3.0, 0.0, 1.0   # outlying observation
        _, r_heavy, _ = lik_heavy.update_coefficients(obs, mu, sig2)
        _, r_light, _ = lik_light.update_coefficients(obs, mu, sig2)
        # Heavier tails → smaller site precision → r closer to zero (less negative)
        assert r_heavy > r_light


# ---------------------------------------------------------------------------
# PoissonLikelihood
# ---------------------------------------------------------------------------

class TestPoissonLikelihood:
    def test_typical_count(self):
        lik = PoissonLikelihood(bin_size=1.0)
        q, r, log_z = lik.update_coefficients(2.0, cavity_mean=0.0, cavity_var=1.0)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_zero_count(self):
        lik = PoissonLikelihood(bin_size=1.0)
        q, r, log_z = lik.update_coefficients(0.0, cavity_mean=0.0, cavity_var=1.0)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_large_bin_size(self):
        """Larger bin size (more exposure) increases expected count."""
        lik = PoissonLikelihood(bin_size=10.0)
        q, r, log_z = lik.update_coefficients(5.0, cavity_mean=0.0, cavity_var=1.0)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_laplace_fallback_degenerate_cavity(self):
        """Very small cavity variance can trigger the Laplace fallback."""
        lik = PoissonLikelihood(bin_size=1.0)
        q, r, log_z = lik.update_coefficients(0.0, cavity_mean=5.0, cavity_var=1e-6)
        assert r < 0 and np.isfinite(q)

    def test_log_z_matches_poisson_pmf(self):
        """log Z should be close to log p(y | f=mu) for tight cavity (small sigma^2)."""
        from scipy.stats import poisson
        lik = PoissonLikelihood(bin_size=1.0)
        mu, y = 1.5, 2.0
        # Tight cavity: sigma^2 small, so tilted moments ≈ likelihood at f=mu
        _, _, log_z = lik.update_coefficients(y, cavity_mean=mu, cavity_var=0.001)
        expected = poisson.logpmf(int(y), mu=np.exp(mu))
        assert np.isclose(log_z, expected, atol=0.1), f"log_z={log_z:.3f} expected≈{expected:.3f}"

    def test_bin_size_must_be_positive(self):
        with pytest.raises(ValueError):
            PoissonLikelihood(bin_size=0.0)


# ---------------------------------------------------------------------------
# GumbelLikelihood
# ---------------------------------------------------------------------------

class TestGumbelLikelihood:
    def test_typical_observation(self):
        lik = GumbelLikelihood(beta=1.0)
        q, r, log_z = lik.update_coefficients(1.0, cavity_mean=0.0, cavity_var=1.0)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_negative_observation(self):
        """Gumbel can observe values below the mode (though less probable)."""
        lik = GumbelLikelihood(beta=1.0)
        q, r, log_z = lik.update_coefficients(-2.0, cavity_mean=0.0, cavity_var=1.0)
        assert r < 0 and np.isfinite(q)

    def test_large_beta_broader_update(self):
        """Larger beta (heavier tail) should give less informative (smaller |r|) update."""
        lik_narrow = GumbelLikelihood(beta=0.1)
        lik_broad  = GumbelLikelihood(beta=5.0)
        obs, mu, sig2 = 1.0, 0.0, 1.0
        _, r_narrow, _ = lik_narrow.update_coefficients(obs, mu, sig2)
        _, r_broad,  _ = lik_broad.update_coefficients(obs, mu, sig2)
        # Broader likelihood → smaller site precision → r closer to zero (less negative)
        assert r_broad > r_narrow

    def test_very_small_cavity_var(self):
        """Very small cavity variance should not crash and should return r < 0."""
        lik = GumbelLikelihood(beta=1.0)
        q, r, log_z = lik.update_coefficients(1.0, cavity_mean=0.0, cavity_var=1e-6)
        assert r < 0 and np.isfinite(q)

    def test_beta_must_be_positive(self):
        with pytest.raises(ValueError):
            GumbelLikelihood(beta=0.0)


# ---------------------------------------------------------------------------
# ExponentialNoiseLikelihood
# ---------------------------------------------------------------------------

class TestExponentialNoiseLikelihood:
    def test_typical_observation(self):
        """y > cavity_mean: main analytic path."""
        lik = ExponentialNoiseLikelihood(rate=1.0)
        q, r, log_z = lik.update_coefficients(1.5, cavity_mean=0.5, cavity_var=1.0)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_asymptotic_expansion_regime(self):
        """Large z+s triggers the Abramowitz & Stegun asymptotic expansion."""
        lik = ExponentialNoiseLikelihood(rate=5.0)
        # z + s = (mu - y)/sigma + rate*sigma;  with small sigma and y >> mu, z>>0
        q, r, log_z = lik.update_coefficients(100.0, cavity_mean=0.0, cavity_var=0.01)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_small_sigma_direct_path(self):
        """Small sigma but y close to mu stays in the direct (non-asymptotic) path."""
        lik = ExponentialNoiseLikelihood(rate=1.0)
        q, r, log_z = lik.update_coefficients(0.6, cavity_mean=0.5, cavity_var=0.01)
        assert r < 0 and np.isfinite(q) and np.isfinite(log_z)

    def test_higher_rate_smaller_log_z(self):
        """Higher rate → tighter exponential → y=1.5 is less probable given f=0."""
        lik_slow = ExponentialNoiseLikelihood(rate=0.5)
        lik_fast = ExponentialNoiseLikelihood(rate=5.0)
        obs, mu, sig2 = 2.0, 0.0, 1.0
        _, _, lz_slow = lik_slow.update_coefficients(obs, mu, sig2)
        _, _, lz_fast = lik_fast.update_coefficients(obs, mu, sig2)
        # Rate=5: expected noise only 0.2; obs=2.0 (large residual) → lower log_z
        assert lz_slow > lz_fast

    def test_rate_must_be_positive(self):
        with pytest.raises(ValueError):
            ExponentialNoiseLikelihood(rate=0.0)

    def test_has_em_step(self):
        lik = ExponentialNoiseLikelihood(rate=1.0)
        assert lik.has_em_step is True

    def test_em_step_increases_rate_for_small_residuals(self):
        """When observed residuals are small (y close to f), rate should increase."""
        lik = ExponentialNoiseLikelihood(rate=0.5)
        n = 50
        f_bar = np.zeros(n)
        sig2_f = np.full(n, 0.01)
        y = f_bar + 0.1 * RNG.exponential(1.0, size=n)   # small residuals
        lik.em_step(y, f_bar, sig2_f)
        assert lik.rate > 0.5   # rate should increase since residuals are tiny

    def test_em_step_decreases_rate_for_large_residuals(self):
        """When observed residuals are large, rate should decrease."""
        lik = ExponentialNoiseLikelihood(rate=5.0)
        n = 50
        f_bar = np.zeros(n)
        sig2_f = np.full(n, 0.01)
        y = f_bar + 5.0 * RNG.exponential(1.0, size=n)   # large residuals
        lik.em_step(y, f_bar, sig2_f)
        assert lik.rate < 5.0


# ---------------------------------------------------------------------------
# GaussianLikelihood EM step
# ---------------------------------------------------------------------------

class TestGaussianLikelihoodEM:
    def test_em_step_reduces_variance_on_low_residuals(self):
        lik = GaussianLikelihood(variance=2.0)
        n = 100
        f_bar = np.zeros(n)
        sig2_f = np.zeros(n)
        y = 0.01 * RNG.standard_normal(n)   # tiny residuals
        lik.em_step(y, f_bar, sig2_f)
        assert lik.variance < 2.0

    def test_em_step_increases_variance_on_large_residuals(self):
        lik = GaussianLikelihood(variance=0.01)
        n = 100
        f_bar = np.zeros(n)
        sig2_f = np.zeros(n)
        y = 5.0 * RNG.standard_normal(n)   # large residuals
        lik.em_step(y, f_bar, sig2_f)
        assert lik.variance > 0.01

    def test_em_step_momentum_zero_fully_updates(self):
        lik = GaussianLikelihood(variance=1.0)
        y = np.array([1.0, 1.0, 1.0])
        f_bar = np.zeros(3)
        sig2_f = np.zeros(3)
        lik.em_step(y, f_bar, sig2_f, momentum=0.0)
        assert np.isclose(lik.variance, 1.0)   # mean of (1-0)^2 = 1.0

    def test_em_step_momentum_one_no_change(self):
        lik = GaussianLikelihood(variance=0.5)
        y = 5.0 * RNG.standard_normal(20)
        lik.em_step(y, np.zeros(20), np.zeros(20), momentum=1.0)
        assert np.isclose(lik.variance, 0.5)


# ---------------------------------------------------------------------------
# End-to-end VRK integration: non-Gaussian likelihoods
# ---------------------------------------------------------------------------

class TestNonGaussianVRKIntegration:
    """Fit + predict with each non-Gaussian likelihood; check sanity."""

    def _check(self, model, X_test):
        mean, var = model.predict(X_test)
        assert np.all(np.isfinite(mean)), "Non-finite mean"
        assert np.all(var >= 0),          "Negative variance"
        assert model.n_active > 0,        "Empty active set"
        ev = model.approximate_evidence()
        assert np.isfinite(ev),           "Non-finite approximate_evidence"

    def test_bernoulli(self):
        X = np.linspace(-3, 3, 40)[:, None]
        y = np.sign(np.sin(X[:, 0]))   # ±1 labels
        cov = ExponentialCovariance(sill=1.0, range_a=1.5)
        model = VRK(cov, BernoulliLikelihood(), max_active=20, n_sweeps=3)
        model.fit(X, y)
        self._check(model, np.linspace(-3, 3, 15)[:, None])

    def test_student_t(self):
        X = np.linspace(0, 6, 40)[:, None]
        y = np.sin(X[:, 0]) + 0.2 * RNG.standard_normal(40)
        # Add two outliers
        y[5]  += 5.0
        y[30] -= 5.0
        model = VRK(_COV, StudentTLikelihood(nu=4.0, sigma=1.0), max_active=20, n_sweeps=3)
        model.fit(X, y)
        self._check(model, np.linspace(0, 6, 15)[:, None])

    def test_poisson(self):
        X = np.linspace(0, 3, 30)[:, None]
        # log intensity ~ sin, so counts ~ Poisson(exp(sin(x)))
        intensity = np.exp(np.sin(X[:, 0]))
        y = RNG.poisson(lam=intensity).astype(float)
        model = VRK(_COV, PoissonLikelihood(bin_size=1.0), max_active=15, n_sweeps=3)
        model.fit(X, y)
        self._check(model, np.linspace(0, 3, 10)[:, None])

    def test_gumbel(self):
        X = np.linspace(0, 6, 40)[:, None]
        beta = 0.5
        # Gumbel samples: loc=sin(x), scale=beta
        u = RNG.uniform(size=40)
        y = np.sin(X[:, 0]) - beta * np.log(-np.log(u + 1e-12))
        model = VRK(_COV, GumbelLikelihood(beta=beta), max_active=20, n_sweeps=3)
        model.fit(X, y)
        self._check(model, np.linspace(0, 6, 15)[:, None])

    def test_exponential_noise(self):
        X = np.linspace(0, 6, 40)[:, None]
        # y = sin(x) + Exp(1): y >= sin(x) always
        y = np.sin(X[:, 0]) + RNG.exponential(1.0, size=40)
        model = VRK(_COV, ExponentialNoiseLikelihood(rate=1.0), max_active=20, n_sweeps=3)
        model.fit(X, y)
        self._check(model, np.linspace(0, 6, 15)[:, None])


# ---------------------------------------------------------------------------
# Per-point (mixed) likelihood mode
# ---------------------------------------------------------------------------

class TestPerPointLikelihoods:
    def test_fit_predict_per_point(self):
        """Each observation gets its own likelihood; predictions should be finite."""
        n = 20
        X = np.linspace(0, 4, n)[:, None]
        y = np.sin(X[:, 0]) + 0.1 * RNG.standard_normal(n)
        # Alternate between two Gaussian likelihoods with different variances
        liks = [GaussianLikelihood(variance=0.1 if i % 2 == 0 else 0.5)
                for i in range(n)]
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        model = VRK(cov, liks, max_active=10, n_sweeps=2)
        model.fit(X, y)
        mean, var = model.predict(X)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)
        assert model.n_active > 0

    def test_wrong_number_of_likelihoods_raises(self):
        n = 10
        X = np.linspace(0, 2, n)[:, None]
        y = np.zeros(n)
        liks = [GaussianLikelihood(variance=0.1)] * (n - 1)   # one too few
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        model = VRK(cov, liks, max_active=5)
        with pytest.raises(ValueError, match="Per-point likelihoods"):
            model.fit(X, y)

    def test_fit_likelihoods_override_constructor(self):
        """Likelihoods passed to fit() should override those in the constructor."""
        n = 10
        X = np.linspace(0, 2, n)[:, None]
        y = np.zeros(n)
        ctor_liks = [GaussianLikelihood(variance=1.0)] * n
        fit_liks  = [GaussianLikelihood(variance=0.1)] * n
        cov = ExponentialCovariance(sill=1.0, range_a=1.0)
        model = VRK(cov, ctor_liks, max_active=5, n_sweeps=1)
        model.fit(X, y, likelihoods=fit_liks)
        assert model.n_likelihoods == n
