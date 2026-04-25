"""
Poisson likelihood with log link and Gauss-Hermite quadrature EP update.

Observation model
-----------------
    p(y | f) = Poisson(y; Δ · exp(f))
             = exp(y log(Δ exp(f)) − Δ exp(f) − log y!)
             = exp(yf − Δ exp(f) + y log Δ − log y!)

where Δ > 0 is the bin size (exposure / time window), and the latent function
f is the log intensity of the Poisson process.  The mean count in a bin is:
    E[y | f] = Δ · exp(f)

This model is used for count data (e.g. number of events in a spatial cell)
where y ∈ {0, 1, 2, …}.

EP update via Gauss-Hermite quadrature
---------------------------------------
The tilted distribution  p̃(f) ∝ Poisson(y; Δ exp(f)) · N(f; μ̃, σ̃²)  does
not yield a closed-form Gaussian.  We use 20-point Gauss-Hermite quadrature
to numerically integrate the tilted moments (see student_t.py for the
quadrature framework description).

Laplace fallback
----------------
For degenerate quadrature (Z ≈ 0), the method falls back to the Laplace
approximation at the mode:

    d/df [log p(y|f) + log N(f; μ̃, σ̃²)] = 0
    ⟺  y − Δ exp(f) − (f − μ̃)/σ̃² = 0

This is solved by Brent's method.  The Hessian at the mode gives:
    −d²/df² log p(y|f)|_{f_mode} = Δ exp(f_mode)  (Poisson Fisher information)
so the site precision is approximately λ_mode = Δ exp(f_mode).

References
----------
Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine
    Learning*. MIT Press.  (Chapter 9 — GP models for non-Gaussian likelihoods)
"""
import numpy as np
from vrk.likelihoods.base import Likelihood

_LAMBDA_TOL = 1e-10
# Pre-compute 20-point Gauss-Hermite nodes and weights once at import time.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(20)


class PoissonLikelihood(Likelihood):
    """
    Poisson likelihood with log link:  p(y|f) = Poisson(y; Δ exp(f)).

    Uses 20-point Gauss-Hermite quadrature for the EP site update, with a
    Laplace-approximation fallback for degenerate cases.

    Parameters
    ----------
    bin_size : float > 0   Δ, exposure or bin width (default 1.0)
    """

    def __init__(self, bin_size: float = 1.0):
        if bin_size <= 0:
            raise ValueError("bin_size must be positive")
        self._bin_size = float(bin_size)

    @property
    def bin_size(self) -> float:
        """Exposure Δ (bin width or observation window size)."""
        return self._bin_size

    def _log_likelihood(self, f: np.ndarray, y: float) -> np.ndarray:
        """
        log p(y | f) = y·f − Δ·exp(f) + y·log Δ − log(y!)

        Uses gammaln for numerically stable computation of log(y!).
        """
        from scipy.special import gammaln
        lam = self._bin_size * np.exp(f)
        return y * np.log(lam + _LAMBDA_TOL) - lam - float(gammaln(y + 1))

    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        EP update via 20-point Gauss-Hermite quadrature.

        Quadrature change of variables:  f_k = √2 σ̃ x_k + μ̃  (see student_t.py).
        The normalisation Z and tilted moments (μ_t, σ²_t) are computed from
        the weighted sum, then converted to (q, r) site parameters by moment
        matching.  Falls back to _laplace_update() if the quadrature fails.
        """
        std = np.sqrt(cavity_var)
        f_pts = np.sqrt(2.0) * std * _GH_NODES + cavity_mean
        log_lik = self._log_likelihood(f_pts, obs)

        log_max = np.max(log_lik)
        lik = np.exp(log_lik - log_max)
        Z = np.dot(_GH_WEIGHTS, lik) / np.sqrt(np.pi)
        if Z < _LAMBDA_TOL:
            return self._laplace_update(obs, cavity_mean, cavity_var)
        log_z = log_max + np.log(Z)

        mean_t = np.dot(_GH_WEIGHTS, f_pts * lik) / (np.sqrt(np.pi) * Z)
        var_t = np.dot(_GH_WEIGHTS, f_pts ** 2 * lik) / (np.sqrt(np.pi) * Z) - mean_t ** 2
        var_t = max(var_t, _LAMBDA_TOL)

        denom_d = cavity_var - var_t
        if abs(denom_d) < _LAMBDA_TOL:
            return 0.0, -_LAMBDA_TOL, log_z

        var_site = cavity_var * var_t / denom_d
        if var_site <= 0:
            return self._laplace_update(obs, cavity_mean, cavity_var)

        mean_site = (mean_t * (cavity_var + var_site) - cavity_mean * var_site) / cavity_var
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
        Laplace fallback at the mode of log p(y|f) + log N(f; μ̃, σ̃²).

        Mode equation:  y − Δ exp(f) − (f − μ̃)/σ̃² = 0.
        Hessian at mode: −Δ exp(f_mode)  (Poisson Fisher information).
        Site precision: λ ≈ Δ exp(f_mode).
        """
        from scipy.optimize import brentq

        def neg_deriv(f):
            return obs - self._bin_size * np.exp(f) - (f - cavity_mean) / cavity_var

        try:
            f_mode = brentq(neg_deriv, cavity_mean - 20.0, cavity_mean + 20.0, maxiter=200)
        except ValueError:
            f_mode = cavity_mean

        lam_mode = self._bin_size * np.exp(f_mode)
        prec_site = max(lam_mode, _LAMBDA_TOL)   # Poisson Fisher information at mode
        var_site = 1.0 / prec_site
        mean_site = f_mode + var_site * (f_mode - cavity_mean) / cavity_var

        r = -1.0 / (var_site + cavity_var)
        q = r * (cavity_mean - mean_site)
        log_z = float(self._log_likelihood(np.array([f_mode]), obs)[0])
        return q, r, log_z
