"""
Student-t likelihood with Gauss-Hermite quadrature EP update.

Observation model
-----------------
    p(y | f) = t_ν(y; f, σ²)
             = Γ((ν+1)/2) / (Γ(ν/2) √(νπ) σ) · (1 + (y−f)²/(νσ²))^{−(ν+1)/2}

The Student-t likelihood produces heavy-tailed residuals, making the model
robust to outliers.  As ν → ∞, p(y|f) → N(y; f, σ²).  For ν = 4 (the
default), the fourth moment is finite (σ_excess = √(ν/(ν−2)) σ ≈ 1.41σ) but
the tails are substantially heavier than Gaussian.

EP update via Gauss-Hermite quadrature
---------------------------------------
The tilted distribution  p̃(f) ∝ p(y|f) · N(f; μ̃, σ̃²)  does not have a
closed-form Gaussian approximation for the Student-t likelihood.  We use
20-point Gauss-Hermite (GH) quadrature to compute the tilted moments:

    Z       = ∫ p(y|f) N(f; μ̃, σ̃²) df
    μ_t     = (1/Z) ∫ f  p(y|f) N(f; μ̃, σ̃²) df
    σ²_t    = (1/Z) ∫ f² p(y|f) N(f; μ̃, σ̃²) df − μ_t²

The Gauss-Hermite change of variables  f = √2 σ̃ x + μ̃  transforms the
integration over f into  ∫ g(x) exp(−x²) dx,  which is exactly the form
solved by GH quadrature:

    ∫ g(x) exp(−x²) dx ≈ Σ_k w_k g(x_k)

where {x_k, w_k} are the 20 GH nodes and weights.

Laplace fallback
----------------
When the quadrature-derived site variance is non-positive (e.g. for very
outlying observations), the method falls back to a Laplace (mode-based)
approximation.  The mode is found via Brent's root-finding algorithm applied
to the score equation:

    d/df [log p(y|f) + log N(f; μ̃, σ̃²)] = 0
    ⟺  (ν+1)(y−f) / (νσ² + (y−f)²) − (f − μ̃)/σ̃² = 0

The site variance is then approximated by the negative inverse Hessian at
the mode:
    σ²_site ≈ 1 / (−d²/df² log p(y|f)|_{f_mode})

References
----------
Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine
    Learning*. MIT Press.  (Section 3.9 — EP for classification; Appendix A)

Golub, G. H. and Welsch, J. H. (1969). Calculation of Gauss Quadrature Rules.
    *Mathematics of Computation*, 23(106):221–230.
"""
import numpy as np
from vrk.likelihoods.base import Likelihood

_LAMBDA_TOL = 1e-10
# Pre-compute 20-point Gauss-Hermite nodes and weights once at import time.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(20)


class StudentTLikelihood(Likelihood):
    """
    Student-t likelihood  p(y|f) = t_ν(y; f, σ²).

    Provides robustness to outliers via heavy tails.  The EP update uses
    20-point Gauss-Hermite quadrature with a Laplace-approximation fallback.

    Parameters
    ----------
    nu    : float > 2   degrees of freedom (ν = 4 gives finite kurtosis)
    sigma : float > 0   scale parameter σ
    """

    def __init__(self, nu: float = 4.0, sigma: float = 1.0):
        if nu <= 0:
            raise ValueError("nu must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._nu = float(nu)
        self._sigma = float(sigma)

    @property
    def nu(self) -> float:
        """Degrees of freedom ν."""
        return self._nu

    @property
    def sigma(self) -> float:
        """Scale parameter σ."""
        return self._sigma

    def _log_likelihood(self, f: float | np.ndarray, y: float) -> np.ndarray:
        """log p(y | f) = log t_ν(y; f, σ²) evaluated at array of f values."""
        from scipy.stats import t as t_dist
        return t_dist.logpdf(y, df=self._nu, loc=f, scale=self._sigma)

    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        EP update via 20-point Gauss-Hermite quadrature.

        Change of variables:  f_k = √2 σ̃ x_k + μ̃  maps GH nodes x_k to
        quadrature points f_k.  The tilted moments are:

            Z    = (1/√π) Σ_k w_k p(y | f_k) exp(+x_k²)   [absorbed into lik]
            μ_t  = (1/√π Z) Σ_k w_k f_k · lik_k
            σ²_t = (1/√π Z) Σ_k w_k f_k² · lik_k − μ_t²

        Site parameters are extracted by moment-matching:
            σ²_site = σ̃² σ²_t / (σ̃² − σ²_t)
            μ_site  = μ_t (σ̃² + σ²_site) / σ̃² − μ̃ σ²_site / σ̃²
        then converted to (q, r) natural parameters.

        Falls back to Laplace approximation if σ²_site ≤ 0.
        """
        std = np.sqrt(cavity_var)
        # Map GH nodes to f-space: f_k = √2 σ̃ x_k + μ̃
        f_pts = np.sqrt(2.0) * std * _GH_NODES + cavity_mean
        log_lik = self._log_likelihood(f_pts, obs)

        # Numerically stable evaluation: subtract log-max before exponentiating
        log_max = np.max(log_lik)
        lik = np.exp(log_lik - log_max)
        Z = np.dot(_GH_WEIGHTS, lik) / np.sqrt(np.pi)
        log_z = log_max + np.log(Z + _LAMBDA_TOL)

        mean_t = np.dot(_GH_WEIGHTS, f_pts * lik) / (np.sqrt(np.pi) * Z + _LAMBDA_TOL)
        var_t = np.dot(_GH_WEIGHTS, f_pts ** 2 * lik) / (np.sqrt(np.pi) * Z + _LAMBDA_TOL) - mean_t ** 2
        var_t = max(var_t, _LAMBDA_TOL)

        # Moment-matching to extract site variance
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
        Laplace (mode-based) fallback for numerically difficult cases.

        Finds the mode of  log p(y|f) + log N(f; μ̃, σ̃²)  via Brent's method,
        then approximates the site as a Gaussian at that mode with variance
        equal to the negative inverse Hessian:

            σ²_site = 1 / (−d²/df² log p(y|f)|_{f_mode})

        The second derivative of the Student-t log-likelihood is:
            d²/df² log t_ν(y; f, σ²) = (ν+1)(νσ² − (y−f)²) / (νσ² + (y−f)²)²
        """
        from scipy.optimize import brentq
        from scipy.stats import t as t_dist

        def neg_deriv(f):
            dy = obs - f
            grad_lik = (self._nu + 1.0) * dy / (self._nu * self._sigma ** 2 + dy ** 2)
            grad_prior = -(f - cavity_mean) / cavity_var
            return grad_lik + grad_prior

        try:
            f_mode = brentq(neg_deriv, cavity_mean - 10.0 * np.sqrt(cavity_var),
                            cavity_mean + 10.0 * np.sqrt(cavity_var))
        except ValueError:
            f_mode = cavity_mean

        dy = obs - f_mode
        d2_lik = (self._nu + 1.0) * (self._nu * self._sigma ** 2 - dy ** 2) / (
            (self._nu * self._sigma ** 2 + dy ** 2) ** 2
        )
        prec_site = max(d2_lik, _LAMBDA_TOL)
        var_site = 1.0 / prec_site

        mean_site = f_mode + var_site * (f_mode - cavity_mean) / cavity_var

        r = -1.0 / (var_site + cavity_var)
        q = r * (cavity_mean - mean_site)
        log_z = float(t_dist.logpdf(obs, df=self._nu, loc=f_mode, scale=self._sigma))
        return q, r, log_z
