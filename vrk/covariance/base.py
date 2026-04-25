"""
Abstract base class for stationary covariance (kernel) functions.

Geostatistics background
------------------------
A covariance function (semivariogram model) encodes the spatial correlation
structure of the latent field.  All models here are *stationary* and
*isotropic*: the covariance depends only on the Euclidean distance h = ‖x − y‖
between two locations, not on their absolute positions.

The covariance C(h) characterises two key features of the spatial field:

  Sill   (σ²):  C(0) = σ², the total variance of the field.
  Range  (a):   The characteristic distance beyond which observations
                become approximately uncorrelated.

For composite models (e.g. structural + nugget), sill and range refer to
the component, and the total sill is their sum.

Hyperparameter handling
-----------------------
All parameters are stored internally in natural (positive) space.  The
log_params property exposes them in log-space, which is convenient for
unconstrained numerical optimisation (see optimization/hyperparameters.py).

Gradient convention
-------------------
gradient_matrix(X) returns a list of ∂C/∂θ_i matrices evaluated at C(X, X),
in natural parameter space.  These are used by analytic_evidence_gradient()
in vrk.py to compute ∂F/∂ log θ via the chain rule:

    ∂F/∂ log θ_i = (∂F/∂θ_i) · θ_i = ½ tr(G_i · M) · θ_i
"""
from abc import ABC, abstractmethod
import numpy as np


class CovarianceFunction(ABC):
    """
    Abstract base class for stationary, isotropic covariance functions.

    All covariance functions represent kernels K_0(x, y) = C(‖x − y‖) that
    depend only on the Euclidean lag distance h = ‖x − y‖.  They must be
    positive semi-definite to guarantee a valid GP prior.

    Subclasses must implement:
        __call__(X, Y)           full cross-covariance matrix C(X, Y)
        diag(X)                  diagonal c(x_i, x_i) for each row x_i in X
        n_params                 number of hyperparameters (property)
        get_params()             return parameters in natural space
        set_params(p)            set parameters from natural-space array
        gradient_matrix(X)       list of ∂C/∂θ_i matrices (natural space)

    Parameters are stored in natural (positive) space internally.  The
    log_params property provides log-space access for optimisation.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate the covariance matrix K_0(X, Y).

        Parameters
        ----------
        X : (n, d) or (n,) array of input locations
        Y : (m, d) or (m,) array, optional.  If None, evaluate C(X, X).

        Returns
        -------
        C : (n, m) array  (or (n, n) when Y is None)
            Entry C[i, j] = K_0(X[i], Y[j]).
        """

    @abstractmethod
    def diag(self, X: np.ndarray) -> np.ndarray:
        """
        Return the self-covariance  c(x_i, x_i) = K_0(X[i], X[i])  for each row.

        For a stationary covariance this equals the sill σ² (or sum of component
        sills for SumCovariance) at every location.

        Returns
        -------
        (n,) array of non-negative variances
        """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of hyperparameters."""

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return hyperparameters as a 1-D array in natural (positive) space."""

    @abstractmethod
    def set_params(self, p: np.ndarray) -> None:
        """Set hyperparameters from a natural-space array p."""

    @abstractmethod
    def gradient_matrix(self, X: np.ndarray) -> list[np.ndarray]:
        """
        Return gradient matrices  [∂C/∂θ_0, …, ∂C/∂θ_{n-1}]  at C(X, X).

        Each element is an (n, n) matrix of partial derivatives in natural
        parameter space, evaluated at the current hyperparameter values.

        Used by VRK.analytic_evidence_gradient() to compute:
            ∂F/∂θ_i = ½ tr(M · ∂C/∂θ_i)
        where M is the gradient pre-factor assembled in vrk.py.
        """

    # ------------------------------------------------------------------
    # Log-space convenience
    # ------------------------------------------------------------------

    @property
    def log_params(self) -> np.ndarray:
        """Hyperparameters in log-space: log θ.  Used by the optimiser."""
        return np.log(self.get_params())

    @log_params.setter
    def log_params(self, lp: np.ndarray) -> None:
        """Set hyperparameters from log-space array (exp maps back to natural space)."""
        self.set_params(np.exp(lp))

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Squared Euclidean distances ‖x_i − y_j‖² → (n, m) matrix.

        Computed via the identity  ‖x−y‖² = ‖x‖² − 2 xᵀy + ‖y‖²  using
        broadcasting, which is efficient for large arrays.  The result is
        clipped to zero to avoid small negative values from floating-point.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        sq_X = np.sum(X ** 2, axis=1, keepdims=True)    # (n, 1)
        sq_Y = np.sum(Y ** 2, axis=1, keepdims=True).T  # (1, m)
        return np.maximum(sq_X + sq_Y - 2.0 * (X @ Y.T), 0.0)

    @staticmethod
    def _dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Euclidean distances ‖x_i − y_j‖ → (n, m) matrix.

        Computed as √_sq_dists(X, Y).  The sqrt is safe because _sq_dists
        is clipped to zero.
        """
        return np.sqrt(CovarianceFunction._sq_dists(X, Y))
