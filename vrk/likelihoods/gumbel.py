"""
Gumbel (Type I extreme value) likelihood with Gauss-Hermite quadrature EP update.

Observation model
-----------------
    p(y | f) = (1/β) exp(−(y−f)/β − exp(−(y−f)/β))

where f is the location parameter (mode of the distribution) and β > 0 is
the scale parameter.  The Gumbel distribution is right-skewed and is widely
used in:
  - Extreme rainfall, flood, and wind speed modelling (annual maxima)
  - Return-period analysis
  - Reliability and survival analysis

Key moments:
    Mean:     E[y] = f + β γ       (γ ≈ 0.5772 is the Euler–Mascheroni constant)
    Variance: Var[y] = π² β² / 6

The mean exceeds the mode (f) by β γ, so the distribution has a right-skew
proportional to the scale β.

EP update via Gauss-Hermite quadrature
---------------------------------------
The likelihood is not in the exponential family and does not admit a
closed-form EP update, so 20-point Gauss-Hermite quadrature is used to
compute the tilted moments (see student_t.py for the framework description).

Laplace fallback
----------------
For degenerate quadrature (Z ≈ 0), a Laplace approximation is used.  The
mode satisfies:

    d/df [log p(y|f) + log N(f; μ̃, σ̃²)] = 0
    ⟺  (1/β)(exp(−(y−f)/β) − 1) − (f − μ̃)/σ̃² = 0

This is solved numerically by Brent's method.  The second derivative of
log p(y|f) w.r.t. f gives the Laplace site precision:

    −d²/df² log p(y|f) = (1/β²) exp(−(y−f)/β)

References
----------
Gumbel, E. J. (1958). *Statistics of Extremes*. Columbia University Press.

Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*.
    Springer, London.
"""
import numpy as np
from vrk.likelihoods.base import Likelihood

_LAMBDA_TOL = 1e-10
# Pre-compute 20-point Gauss-Hermite nodes and weights once at import time.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(20)

# Euler–Mascheroni constant γ ≈ 0.5772 (shift from mode to mean)
_EULER_MASCHERONI = 0.5772156649015328


class GumbelLikelihood(Likelihood):
    """
    Gumbel (Type I extreme value) likelihood.

        p(y | f) = (1/β) exp(−(y−f)/β − exp(−(y−f)/β))

    The latent f is the location parameter (mode of the Gumbel distribution).
    The predictive mean at f is  f + β γ  and the predictive std is  π β/√6 ≈ 1.283 β.

    Uses 20-point Gauss-Hermite quadrature for the EP site update, with a
    Laplace-approximation fallback.

    Parameters
    ----------
    beta : float > 0   scale parameter β
    """

    def __init__(self, beta: float = 1.0):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self._beta = float(beta)

    @property
    def beta(self) -> float:
        """Scale parameter β."""
        return self._beta

    def _log_likelihood(self, f, y: float) -> np.ndarray:
        """
        log p(y | f) = −(y−f)/β − exp(−(y−f)/β) − log β.

        Evaluated at an array of f values for use in Gauss-Hermite quadrature.
        """
        z = (y - f) / self._beta
        return -z - np.exp(-z) - np.log(self._beta)

    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        EP site update via 20-point Gauss-Hermite quadrature.

        Quadrature change of variables:  f_k = √2 σ̃ x_k + μ̃  (see student_t.py).
        The tilted moments (Z, μ_t, σ²_t) are computed from the weighted sum,
        then converted to (q, r) natural parameters by moment matching.
        Falls back to _laplace_update() if Z is degenerate.

        Returns
        -------
        q, r  : EP natural parameters (r < 0)
        log_z : log EP normalisation constant
        """
        std = np.sqrt(max(cavity_var, _LAMBDA_TOL))
        f_pts = np.sqrt(2.0) * std * _GH_NODES + cavity_mean
        log_lik = self._log_likelihood(f_pts, obs)

        log_max = np.max(log_lik)
        lik = np.exp(log_lik - log_max)

        Z = np.dot(_GH_WEIGHTS, lik) / np.sqrt(np.pi)
        if Z < _LAMBDA_TOL:
            # Cavity is far from the observation; return a neutral (near-zero) update
            return 0.0, -_LAMBDA_TOL, -1e10

        log_z = log_max + np.log(Z)

        mean_t = np.dot(_GH_WEIGHTS, f_pts * lik) / (np.sqrt(np.pi) * Z)
        var_t  = (
            np.dot(_GH_WEIGHTS, f_pts ** 2 * lik) / (np.sqrt(np.pi) * Z)
            - mean_t ** 2
        )
        var_t = max(var_t, _LAMBDA_TOL)

        denom_d = cavity_var - var_t
        if abs(denom_d) < _LAMBDA_TOL:
            return 0.0, -_LAMBDA_TOL, log_z

        var_site = cavity_var * var_t / denom_d
        if var_site <= 0:
            return self._laplace_update(obs, cavity_mean, cavity_var)

        mean_site = (
            mean_t * (cavity_var + var_site) - cavity_mean * var_site
        ) / cavity_var

        r = -1.0 / (var_site + cavity_var)
        q = r * (cavity_mean - mean_site)
        if r >= 0:
            return self._laplace_update(obs, cavity_mean, cavity_var)
        return q, r, log_z

    def _laplace_update(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        Laplace fallback at the mode of  log p(y|f) + log N(f; μ̃, σ̃²).

        Mode equation:
            d/df [log p(y|f) + log N(f; μ̃, σ̃²)] = 0
            ⟺  (1/β)(exp(−(y−f)/β) − 1) − (f − μ̃)/σ̃² = 0

        Second derivative of log p(y|f) w.r.t. f (Hessian for Laplace):
            d²/df² log p(y|f) = −(1/β²) exp(−(y−f)/β)

        Site precision: λ ≈ (1/β²) exp(−(y − f_mode)/β).
        """
        from scipy.optimize import brentq

        def neg_deriv(f):
            z = (obs - f) / self._beta
            grad_lik = (1.0 / self._beta) * (np.exp(-z) - 1.0)
            grad_prior = -(f - cavity_mean) / cavity_var
            return grad_lik + grad_prior

        search_hw = 10.0 * np.sqrt(cavity_var) + 5.0 * self._beta
        try:
            f_mode = brentq(
                neg_deriv,
                cavity_mean - search_hw,
                cavity_mean + search_hw,
            )
        except ValueError:
            f_mode = cavity_mean

        z = (obs - f_mode) / self._beta
        # −d²/df² log p(y|f) = (1/β²) exp(−(y−f)/β)
        d2_lik = -(1.0 / self._beta ** 2) * np.exp(-z)
        prec_site = max(-d2_lik, _LAMBDA_TOL)
        var_site = 1.0 / prec_site

        mean_site = f_mode + var_site * (f_mode - cavity_mean) / cavity_var

        r = -1.0 / (var_site + cavity_var)
        q = r * (cavity_mean - mean_site)
        log_z = float(self._log_likelihood(f_mode, obs))
        return q, r, log_z
