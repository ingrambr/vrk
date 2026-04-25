"""
Gaussian (squared-exponential / RBF) covariance function.

The squared-exponential is the most widely used kernel in GP literature,
producing infinitely differentiable (analytic) sample paths.  While
convenient for computation, this extreme smoothness can be unrealistic for
physical spatial fields; the Matérn family (see matern52.py) is often preferred
for geostatistical applications where only finite smoothness is appropriate.

References
----------
Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine
    Learning*. MIT Press, Cambridge, MA.  (Section 4.2.1)

Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*.
    Springer, New York.  (Chapter 2 — critique of the Gaussian model)

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
import numpy as np
from vrk.covariance.base import CovarianceFunction


class GaussianCovariance(CovarianceFunction):
    """
    Gaussian (squared-exponential) covariance function.

        C(h) = σ² exp(−h² / (2a²))

    where h = ‖x − y‖ is the Euclidean lag distance.

    Properties
    ----------
    * Infinitely mean-square differentiable sample paths.
    * Isotropic and stationary; sill C(0) = σ².
    * The Fourier transform is also Gaussian, making spectral analysis tractable.

    Note: for spatial prediction of rough fields (e.g. terrain, soil properties)
    the Matérn-5/2 model is usually preferable as it allows a more realistic
    degree of roughness.

    Parameters
    ----------
    sill    : float > 0   σ², marginal variance of the field
    range_a : float > 0   a, length-scale (correlation ≈ 0.61 at h = a)

    Gradients (natural space)
    -------------------------
        ∂C/∂σ²  = C / σ²
        ∂C/∂a   = C ⊙ H² / a³    where H² is the squared-distance matrix
    """

    def __init__(self, sill: float = 1.0, range_a: float = 1.0):
        if sill <= 0 or range_a <= 0:
            raise ValueError("sill and range_a must be positive")
        self._sill = float(sill)
        self._range_a = float(range_a)

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        H2 = self._sq_dists(X, Y)
        return self._sill * np.exp(-H2 / (2.0 * self._range_a ** 2))

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonal C(0) = σ² at every location."""
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self._sill)

    @property
    def n_params(self) -> int:
        return 2

    def get_params(self) -> np.ndarray:
        return np.array([self._sill, self._range_a])

    def set_params(self, p: np.ndarray) -> None:
        self._sill = float(p[0])
        self._range_a = float(p[1])

    def gradient_matrix(self, X: np.ndarray) -> list[np.ndarray]:
        """
        Returns [∂C/∂σ², ∂C/∂a] evaluated at C(X, X).

        Derivation for ∂C/∂a:
            C = σ² exp(−H²/(2a²))
            ∂C/∂a = C · H² / a³   from  d/da exp(−H²/(2a²)) = H²/a³ · exp(−H²/(2a²))
        """
        X = np.atleast_2d(X)
        H2 = self._sq_dists(X, X)
        C = self._sill * np.exp(-H2 / (2.0 * self._range_a ** 2))
        dC_dsill = C / self._sill
        dC_da = C * H2 / (self._range_a ** 3)
        return [dC_dsill, dC_da]
