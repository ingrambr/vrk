"""
Gaussian likelihood with closed-form EP updates.

For a Gaussian observation model  p(y | f) = N(y; f, σ²),  the EP update has
an exact closed form because the product of two Gaussians is Gaussian:

    p̃(f) ∝ N(y; f, σ²) · N(f; μ̃, σ̃²) = N(f; μ_t, σ²_t) · const

The tilted mean and variance are:
    σ²_t  = σ̃² σ² / (σ̃² + σ²)     (harmonic mean of the two variances)
    μ_t   = (μ̃/σ̃² + y/σ²) / (1/σ̃² + 1/σ²)

which simplifies to the formulas below.  Since the update is exact, no
quadrature or approximation is needed.

EM update
---------
Given predictive moments (f̄_i, σ²_f_i) at training locations, the MLE update
for σ² maximises  E_q[Σ log p(y_i | f_i)]:

    σ²_new = (1/n) Σ_i [ (y_i − f̄_i)² + σ²_f_i ]

This is the standard empirical risk plus a variance correction that accounts
for uncertainty in f_i.
"""
import numpy as np
from vrk.likelihoods.base import Likelihood


class GaussianLikelihood(Likelihood):
    """
    Gaussian observation model:  p(y | f) = N(y; f, σ²).

    The EP update is exact (closed-form) because Gaussian × Gaussian = Gaussian.

        r      = −1 / (σ̃² + σ²)              (site precision parameter)
        q      = −r · (y − μ̃)                (site mean parameter)
        log Z  = log N(y; μ̃, σ̃² + σ²)       (normalisation)

    where μ̃ and σ̃² are the cavity mean and variance.

    Parameters
    ----------
    variance : float > 0   σ², observation noise variance
    """

    def __init__(self, variance: float = 1.0):
        if variance <= 0:
            raise ValueError("variance must be positive")
        self._variance = float(variance)

    @property
    def variance(self) -> float:
        """Observation noise variance σ²."""
        return self._variance

    @variance.setter
    def variance(self, v: float) -> None:
        if v <= 0:
            raise ValueError("variance must be positive")
        self._variance = float(v)

    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        Closed-form EP update for the Gaussian likelihood.

        The cavity N(f; μ̃, σ̃²) and likelihood N(y; f, σ²) combine as:

            p̃(f) ∝ N(y; f, σ²) N(f; μ̃, σ̃²)
                  = N(f; μ_t, σ²_t) · Z

        with total variance  σ²_total = σ̃² + σ²  (variances sum for
        independent Gaussians in the observation direction).

        Returns  r = −1/σ²_total < 0,  q = (y − μ̃)/σ²_total,
        log Z = log N(y; μ̃, σ²_total).
        """
        total_var = cavity_var + self._variance
        r = -1.0 / total_var
        q = -r * (obs - cavity_mean)
        log_z = -0.5 * (np.log(2.0 * np.pi * total_var) + (obs - cavity_mean) ** 2 / total_var)
        return q, r, log_z

    def observation_noise_variance(self) -> float:
        """Return σ² (used in active_set_log_evidence)."""
        return self._variance

    @property
    def has_em_step(self) -> bool:
        return True

    def em_step(
        self,
        y: np.ndarray,
        mean_pred: np.ndarray,
        var_pred: np.ndarray,
        momentum: float = 0.0,
    ) -> None:
        """
        EM M-step: σ²_new = mean((y_i − f̄_i)² + σ²_f_i).

        This is the MLE update for σ² given the posterior predictive moments
        (f̄_i, σ²_f_i) from the E-step (EP sweep).  The σ²_f_i term corrects
        for the uncertainty in f_i — it would be zero if f_i were observed.
        """
        residuals_sq = (y - mean_pred) ** 2 + var_pred
        new_var = float(np.mean(residuals_sq))
        self._variance = momentum * self._variance + (1.0 - momentum) * new_var
