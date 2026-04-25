"""
One-sided exponential noise likelihood for EP, via Gauss-Hermite quadrature.

This is a numerically approximate alternative to the closed-form implementation
in exponential_noise.py.  It is provided for comparison purposes: to quantify
how well Gauss-Hermite (GH) quadrature handles the truncated exponential
likelihood, and to identify where it breaks down relative to the exact formula.

Observation model
-----------------
    p(y | f) = λ · exp(−λ(y − f))   for y ≥ f     (truncated exponential)
               0                      for y < f

EP update via Gauss-Hermite quadrature
---------------------------------------
The tilted distribution is:

    p̃(f) ∝ p(y|f) · N(f; μ̃, σ̃²)

whose normalisation Z and first two moments (μ_t, σ²_t) are computed by
K-point Gauss-Hermite quadrature.  The change of variables

    f = √2 σ̃ x + μ̃   ⟹   N(f; μ̃, σ̃²) df = exp(−x²)/√π dx

transforms the integrals into ∫ g(x) exp(−x²) dx form, solvable by GH.

The truncation f ≤ y is handled by setting p(y|f) = 0 (i.e., log p = −∞)
for nodes f_k > y, effectively zeroing those weights.

Limitations vs. the analytical method
---------------------------------------
1. **Tail regime (z+s >> 1):** When z = (μ̃ − y)/σ̃ is large and positive, the
   valid region f ≤ y lies far below the cavity mean, so nearly all GH nodes
   violate the truncation constraint.  The effective number of live nodes drops
   rapidly, destroying accuracy.  The closed-form code handles this via the
   Abramowitz & Stegun asymptotic expansion.

2. **Accuracy:** GH is a numerical approximation with K-point truncation error.
   For smooth likelihoods without truncation, K=20 is usually sufficient.  For
   the truncated exponential, the effective support is one-sided and accuracy
   depends strongly on z.

3. **Cost:** O(K) evaluations per update vs. O(1) for the analytical formula.

This class uses K=20 by default (matching StudentTLikelihood), but exposes K
as a constructor argument so accuracy/speed trade-offs can be studied.

References
----------
Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine
    Learning*. MIT Press. Appendix A (Gauss-Hermite quadrature in EP).
"""
import numpy as np
from vrk.likelihoods.base import Likelihood

_LAMBDA_TOL = 1e-10


class ExponentialNoiseGHLikelihood(Likelihood):
    """
    One-sided exponential noise likelihood, EP update via Gauss-Hermite quadrature.

    Observation model:  y = f + ε,  ε ~ Exp(λ),  y ≥ f always.

    The EP update is computed numerically using K-point GH quadrature.  For
    comparison, ExponentialNoiseLikelihood provides the exact analytical update.

    Parameters
    ----------
    rate : float > 0   λ, exponential rate (mean noise = 1/λ)
    n_gh : int > 0     Number of Gauss-Hermite quadrature points (default 20)
    """

    def __init__(self, rate: float = 1.0, n_gh: int = 20):
        if rate <= 0:
            raise ValueError("rate must be positive")
        if n_gh < 1:
            raise ValueError("n_gh must be a positive integer")
        self._rate = float(rate)
        self._n_gh = int(n_gh)
        self._gh_nodes, self._gh_weights = np.polynomial.hermite.hermgauss(self._n_gh)

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, v: float) -> None:
        if v <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(v)

    @property
    def n_gh(self) -> int:
        """Number of Gauss-Hermite quadrature points."""
        return self._n_gh

    def _log_likelihood(self, f: np.ndarray, y: float) -> np.ndarray:
        """
        Compute log p(y | f) = log λ − λ(y − f)  for f ≤ y,  −∞ for f > y.

        Parameters
        ----------
        f : (K,) array of candidate latent values (GH nodes in f-space)
        y : scalar observation

        Returns
        -------
        (K,) array of log-likelihoods; entries where f > y are −∞.
        """
        ll = np.where(f <= y,
                      np.log(self._rate) - self._rate * (y - f),
                      -np.inf)
        return ll

    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        EP site update via Gauss-Hermite quadrature.

        Change of variables:  f_k = √2 σ̃ x_k + μ̃  maps GH nodes x_k to
        quadrature points in f-space.  Tilted moments:

            Z    = (1/√π) Σ_k w_k · p(y | f_k)   [absorbing Gaussian into nodes]
            μ_t  = (1/√π Z) Σ_k w_k · f_k · p(y | f_k)
            σ²_t = (1/√π Z) Σ_k w_k · f_k² · p(y | f_k) − μ_t²

        Site parameters by moment-matching (following StudentTLikelihood):
            var_site  = σ̃² σ²_t / (σ̃² − σ²_t)
            mean_site = (μ_t (σ̃² + var_site) − μ̃ var_site) / σ̃²
            r = −1 / (var_site + σ̃²)
            q = r · (μ̃ − mean_site)
        """
        sq_sig = np.sqrt(max(cavity_var, _LAMBDA_TOL))

        # Map GH nodes to f-space
        f_pts = np.sqrt(2.0) * sq_sig * self._gh_nodes + cavity_mean

        # Log-likelihood at each node (−∞ where f_k > obs, i.e. truncation)
        log_lik = self._log_likelihood(f_pts, obs)

        # Numerically stable: subtract log-max of finite entries
        finite_mask = np.isfinite(log_lik)
        if not np.any(finite_mask):
            # All nodes violate the truncation: observation far below cavity.
            # Return near-zero site (no information from this observation).
            return 0.0, -_LAMBDA_TOL, -1e10

        log_max = np.max(log_lik[finite_mask])
        lik = np.exp(log_lik - log_max)   # shape (K,); zero where f_k > obs

        # Tilted normalisation and moments
        Z = np.dot(self._gh_weights, lik) / np.sqrt(np.pi)
        if Z < _LAMBDA_TOL:
            # Numerically zero: no effective quadrature nodes below obs.
            return 0.0, -_LAMBDA_TOL, -1e10

        log_z = log_max + np.log(Z)

        norm = np.sqrt(np.pi) * Z
        mean_t = np.dot(self._gh_weights, f_pts * lik) / norm
        var_t  = np.dot(self._gh_weights, f_pts ** 2 * lik) / norm - mean_t ** 2
        var_t  = max(var_t, _LAMBDA_TOL)

        # Moment-matching → site parameters
        denom = cavity_var - var_t
        if abs(denom) < _LAMBDA_TOL:
            return 0.0, -_LAMBDA_TOL, log_z

        var_site = cavity_var * var_t / denom
        if var_site <= 0:
            # GH failed (too few nodes in valid region): return uninformative site
            return 0.0, -_LAMBDA_TOL, log_z

        mean_site = (mean_t * (cavity_var + var_site) - cavity_mean * var_site) / cavity_var

        r = -1.0 / (var_site + cavity_var)
        q = r * (cavity_mean - mean_site)

        if r >= 0:
            return 0.0, -_LAMBDA_TOL, log_z

        # Apply the same precision clamp as the analytical implementation
        tol_l = 100.0
        effective_prec = -r / (1.0 + r * cavity_var)
        if effective_prec > tol_l:
            r = -1.0 / (cavity_var + 1.0 / tol_l)
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
        EM M-step: same truncated-normal formula as ExponentialNoiseLikelihood.

        The EM update does not use quadrature; it uses the closed-form
        truncated-Gaussian expectation that is exact regardless of how the
        EP update was computed.
        """
        from scipy.special import erfc
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
