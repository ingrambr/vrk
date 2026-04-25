"""
One-sided exponential noise likelihood for EP.

Observation model
-----------------
    p(y | f) = λ · exp(−λ (y − f))   for y ≥ f
               0                       for y < f

Equivalently:  y = f + ε,   ε ~ Exp(λ),  so observations are always ≥ f.

This models sensors that produce a systematic positive bias: the latent field f
is the true value, and the observation y is always at or above it.  Applications
include one-sided measurement errors (e.g. geometric distance sensors that
measure along a longer path than the straight-line distance, or optical sensors
that record the first reflection above the surface).

EP update derivation
--------------------
The tilted distribution is:

    p̃(f) ∝ p(y|f) · N(f; μ̃, σ̃²)
          = λ exp(−λ(y−f)) · N(f; μ̃, σ̃²)   for f ≤ y
          = 0                                  for f > y

Completing the square in the exponent:

    −λ(y−f) − (f−μ̃)²/(2σ̃²)
    = −(f − (μ̃ + λσ̃²))² / (2σ̃²)  +  λ(λσ̃²/2 + μ̃ − y)

Let  s = λ σ̃,  z = (μ̃ − y) / σ̃,  so  (z + s) captures the normalised
deviation of the observation below the cavity mean.

The EP normalisation constant is:

    Z = λ · exp(s²/2 + sz) · Φ(−(z + s))

where  Φ  is the standard normal CDF.  In log-space:

    log Z = log λ + s(s/2 + z) + log Φ(−(z+s))

Asymptotic expansion for z + s > 6 (tail regime)
-------------------------------------------------
When z + s is large, Φ(−(z+s)) is extremely small and the direct evaluation
of log Φ is numerically unstable.  We use the continued-fraction / asymptotic
expansion (Abramowitz & Stegun 26.2.12):

    log Φ(−u) ≈ −u²/2 − ½ log(2π) − log u + L(1/u²)

where  L(t) = log(1 − 5t/2 + (37/3)t² − (353/4)t³ + …)  is a series
correction.  This ensures stable computation for large positive z+s.

EM update
---------
Given posterior moments (f̄_i, σ²_f_i) at training locations, the MLE rate
update is:

    λ_new = n / Σ_i E[y_i − f_i]

The conditional expectation E[y − f | f̄, σ²_f] for a truncated Gaussian is:

    E[y − f] = (y − f̄) Φ(ζ) + σ_f φ(ζ)    where  ζ = (y − f̄) / σ_f

This is the standard truncated normal mean formula, using the fact that
ε = y − f > 0 and the posterior of f is approximately N(f̄, σ²_f).
"""
import numpy as np
from scipy.special import erf, erfc
from vrk.likelihoods.base import Likelihood

_LAMBDA_TOL = 1e-8
_LOG2PI = np.log(2.0 * np.pi)


class ExponentialNoiseLikelihood(Likelihood):
    """
    One-sided exponential noise likelihood.

    Observation model:  y = f + ε,   ε ~ Exp(λ),  y ≥ f always.

    The EP update does not have a closed form (unlike the Gaussian case) but
    can be computed analytically via the log-normal CDF formula described in
    the module docstring.  A numerically stable asymptotic expansion is used
    when the argument of the log-CDF is large.

    Parameters
    ----------
    rate : float > 0   λ, exponential rate (mean noise = 1/λ)
    """

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)

    @property
    def rate(self) -> float:
        """Exponential rate λ (mean noise = 1/λ)."""
        return self._rate

    @rate.setter
    def rate(self, v: float) -> None:
        if v <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(v)

    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        EP site update for the one-sided exponential noise model.

        Using the notation:
            s  = λ σ̃                      (scaled rate)
            z  = (μ̃ − y) / σ̃             (normalised cavity-minus-observation)

        the log normalisation is:
            log Z = log λ + s(s/2 + z) + log Φ(−(z+s))

        For z + s > 6 (tail regime), the asymptotic expansion replaces log Φ
        to avoid underflow.

        The EP site parameters are derived from the derivative of log Z:
            q = λ + d(log Z)/dμ̃
            r = −[d(log Z)/dμ̃] · ((z+s)/σ̃ + d(log Z)/dμ̃)

        A site precision cap is applied to keep the EP iteration numerically
        stable (effective precision ≤ 100).
        """
        rate = self._rate
        sig_x2 = max(cavity_var, _LAMBDA_TOL)
        sq_sig = np.sqrt(sig_x2)     # σ̃
        s = rate * sq_sig            # s = λ σ̃
        z = (cavity_mean - obs) / sq_sig   # z = (μ̃ − y) / σ̃

        zs = z + s   # z + s = (μ̃ − y + λσ̃²) / σ̃

        if zs > 6.0:
            # Asymptotic expansion for log Φ(−zs) when zs >> 1
            # (Abramowitz & Stegun 26.2.12 continued fraction)
            tt = 1.0 / zs ** 2
            l_f = (1.0 - (5.0 / 2.0 - (37.0 / 3.0 - 353.0 / 4.0 * tt) * tt) * tt) * tt
            log_z = -(z ** 2 + _LOG2PI) / 2.0 - np.log(zs) + l_f + np.log(rate)
            dt = -(zs) * np.exp(l_f)
        else:
            # Direct computation via erfc: Φ(−u) = erfc(u/√2) / 2
            log_Phi_neg = np.log(max((1.0 - erf(zs / np.sqrt(2.0))) / 2.0, 1e-300))
            log_z = s * (s / 2.0 + z) + log_Phi_neg + np.log(rate)

            # Derivative of log Φ(−zs) w.r.t. zs: −φ(zs) / Φ(−zs)
            erfc_val = max(1.0 - erf(zs / np.sqrt(2.0)), 1e-300)
            dt = -np.exp(-zs ** 2 / 2.0) / erfc_val * 2.0 / np.sqrt(2.0 * np.pi)

        dt = dt / sq_sig   # Convert derivative from zs-space to μ̃-space

        # EP natural parameters (see module docstring derivation)
        q = rate + dt
        r = -dt * (zs / sq_sig + dt)

        # Clamp site precision to avoid numerical instability in EP sweeps
        tol_l = 100.0
        effective_prec = -r / (1.0 + r * sig_x2)
        if effective_prec > tol_l:
            r = -1.0 / (sig_x2 + 1.0 / tol_l)
        if -r < 1.0 / tol_l ** 2:
            r = -1.0 / tol_l ** 2

        return q, r, log_z

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
        EM M-step: λ_new = n / Σ_i E[y_i − f_i].

        The expected excess  E[y_i − f_i]  is computed using the truncated
        normal expectation formula (ε > 0):

            E[ε_i] = (y_i − f̄_i) Φ(ζ_i) + σ_f_i φ(ζ_i)
                     where  ζ_i = (y_i − f̄_i) / σ_f_i

        This integral accounts for the positivity constraint on ε_i.
        """
        y = np.asarray(y, dtype=float)
        var_pred = np.maximum(var_pred, _LAMBDA_TOL)
        std = np.sqrt(var_pred)
        zeta = (y - mean_pred) / std
        phi_zeta = np.exp(-0.5 * zeta ** 2) / np.sqrt(2.0 * np.pi)
        Phi_zeta = 0.5 * erfc(-zeta / np.sqrt(2.0))
        E_excess = (y - mean_pred) * Phi_zeta + std * phi_zeta
        E_excess = np.maximum(E_excess, 1e-12)
        new_rate = float(len(y) / np.sum(E_excess))
        self._rate = momentum * self._rate + (1.0 - momentum) * new_rate
