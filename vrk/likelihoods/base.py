"""
Abstract base class for EP-compatible likelihood functions.

EP likelihood interface
-----------------------
Expectation Propagation (Csáto 2002 thesis; Minka 2001)
approximates each non-Gaussian factor p(y_i | f_i) by a Gaussian site:

    t̃_i(f_i) = Z_i · N(f_i; ã_i, λ_i^{-1})

parameterised by a site precision λ_i ≥ 0 and a site mean ã_i.

The update_coefficients() method is called once per observation per EP sweep.
It receives the current *cavity distribution* — the posterior with site i
removed — and returns the natural parameters of the updated site together with
the EP normalisation constant.

Natural-parameter convention
-----------------------------
The return tuple (q, r, log_z) follows the C++ PSGP convention (matching the
VRK Cholesky-based update in vrk.py):

    q  — site mean natural parameter:  q = λ_i · ã_i
    r  — site precision natural parameter:  r = −λ_i < 0  (always negative)

The site precision λ_i is recovered as:
    λ_i = −r / (1 + r · σ̃²_cav)   > 0

and the site mean as:
    ã_i = μ̃_cav − q/r

where μ̃_cav and σ̃²_cav are the cavity mean and variance.

References
----------
Csáto, L. (2002). *Gaussian Processes — Iterative Sparse Approximations*.
    PhD Thesis, Aston University.

Minka, T. P. (2001). A Family of Algorithms for Approximate Bayesian Inference.
    PhD Thesis, MIT.

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
from abc import ABC, abstractmethod
import numpy as np


class Likelihood(ABC):
    """
    Abstract base class for EP-compatible likelihoods.

    Each subclass implements one observation model p(y | f) and provides the
    EP update_coefficients() method to compute the site parameters.  The rest
    of the VRK inference machinery is likelihood-agnostic.

    EP return convention
    --------------------
        q     : site mean natural parameter  (q = λ_i ã_i)
        r     : site precision natural parameter  (r = −λ_i < 0)
        log_z : log EP normalisation constant  log Z_i
    """

    @abstractmethod
    def update_coefficients(
        self,
        obs: float,
        cavity_mean: float,
        cavity_var: float,
    ) -> tuple[float, float, float]:
        """
        Compute EP site update for one observation.

        Given the cavity distribution N(f; μ̃, σ̃²), compute the EP
        *tilted distribution* moments and return the site parameters.

        The tilted distribution is:
            p̃(f) ∝ p(y_i | f) · N(f; μ̃, σ̃²)

        Parameters
        ----------
        obs          : scalar observation y_i
        cavity_mean  : scalar cavity mean  μ̃_i
        cavity_var   : scalar cavity variance  σ̃²_i  (> 0)

        Returns
        -------
        q     : float  site mean natural parameter  (= λ_i ã_i)
        r     : float  site precision natural parameter  (< 0)
        log_z : float  log normalisation:  log ∫ p(y_i | f) N(f; μ̃_i, σ̃²_i) df
        """

    def observation_noise_variance(self) -> float:
        """
        Effective observation noise variance, used in some evidence computations.

        Returns 0.0 for non-Gaussian likelihoods (where noise variance is not
        a single scalar).  Overridden by GaussianLikelihood.
        """
        return 0.0

    @property
    def has_em_step(self) -> bool:
        """Return True if this likelihood supports an EM hyperparameter update."""
        return False

    def em_step(
        self,
        y: np.ndarray,
        mean_pred: np.ndarray,
        var_pred: np.ndarray,
        momentum: float = 0.0,
    ) -> None:
        """
        Perform one EM M-step update of likelihood hyperparameters.

        The E-step is implicitly carried out during the EP sweep (predictive
        moments serve as the sufficient statistics).  The M-step maximises the
        expected complete-data log-likelihood w.r.t. the likelihood parameters.

        Parameters
        ----------
        y          : (n,) observations
        mean_pred  : (n,) posterior predictive means at training locations
        var_pred   : (n,) posterior predictive variances at training locations
        momentum   : float in [0, 1], mixing weight for the previous parameter
                     value (0 = full update, 1 = no update)
        """
        raise NotImplementedError
