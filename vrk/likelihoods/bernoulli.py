"""
Bernoulli likelihood with probit link and closed-form EP update.

Observation model
-----------------
    p(y | f) = Φ(y · f),   y ∈ {−1, +1}

where Φ is the standard normal CDF (probit link).  This is the standard
model for binary GP classification.

EP update  (closed-form)
-------------------------
The tilted distribution  p̃(f) ∝ Φ(yf) · N(f; μ̃, σ̃²)  has an analytic
Gaussian approximation because the product of a Gaussian CDF and a Gaussian
PDF can be expressed in terms of the normal CDF evaluated at a scaled argument.

Let  z = y μ̃ / √(1 + σ̃²).  Then (Rasmussen & Williams 2006, Section 3.5):

    Z   = Φ(z)
    μ_t = μ̃ + y σ̃² φ(z) / (Φ(z) (1 + σ̃²)^{1/2})
    σ²_t = σ̃² − σ̃⁴ φ(z) [z Φ(z) + φ(z)] / [(1 + σ̃²) Φ(z)²]

where φ = Φ' is the standard normal PDF.  The ratio φ/Φ is the *inverse
Mills ratio* and is evaluated stably using scipy.special.ndtr and log_ndtr.

References
----------
Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine
    Learning*. MIT Press.  (Section 3.5 — probit regression, EP update)

Minka, T. P. (2001). A Family of Algorithms for Approximate Bayesian Inference.
    PhD Thesis, MIT.  (Section 3.6)
"""
import numpy as np
from scipy.special import ndtr, log_ndtr
from vrk.likelihoods.base import Likelihood

_LAMBDA_TOL = 1e-10


class BernoulliLikelihood(Likelihood):
    """
    Probit Bernoulli likelihood:  p(y | f) = Φ(y · f),  y ∈ {−1, +1}.

    The EP update is closed-form (no quadrature needed).  Binary labels are
    accepted as ±1 or as 0/1 (0 is treated as −1 internally).
    """

    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        Closed-form EP update for the probit Bernoulli likelihood.

        The cavity N(f; μ̃, σ̃²) and likelihood Φ(yf) combine analytically.
        Let  denom = √(1 + σ̃²),  z = y μ̃ / denom.  Then:

            Z     = Φ(z)                              (normalisation)
            ratio = φ(z) / Φ(z)                       (inverse Mills ratio)
            μ_t   = μ̃ + y σ̃² ratio / denom            (tilted mean)
            σ²_t  = σ̃² − σ̃⁴ ratio (z/denom + ratio) / (1 + σ̃²)
                                                       (tilted variance)

        Site parameters are extracted by moment-matching, then converted to
        the (q, r) natural-parameter convention used by vrk.py.
        """
        y = 1.0 if obs > 0 else -1.0

        denom = np.sqrt(1.0 + cavity_var)
        z = y * cavity_mean / denom

        log_z = log_ndtr(z)                         # log Φ(z), numerically stable
        Phi_z = np.exp(log_z)                        # Φ(z)
        phi_z = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)   # φ(z)

        ratio = phi_z / (Phi_z + _LAMBDA_TOL)        # φ(z) / Φ(z), inverse Mills ratio

        # Tilted mean and variance
        mean_t = cavity_mean + y * cavity_var * ratio / denom
        var_t = cavity_var - (cavity_var ** 2) * ratio * (z / denom + ratio) / (1.0 + cavity_var)
        var_t = max(var_t, _LAMBDA_TOL)

        # Moment-matching to extract site variance
        denom_d = cavity_var - var_t
        if abs(denom_d) < _LAMBDA_TOL:
            return 0.0, -_LAMBDA_TOL, log_z

        var_site = cavity_var * var_t / denom_d
        if var_site <= 0:
            var_site = _LAMBDA_TOL

        mean_site = (mean_t * (cavity_var + var_site) - cavity_mean * var_site) / cavity_var

        r = -1.0 / (var_site + cavity_var)
        q = r * (cavity_mean - mean_site)
        return q, r, log_z
