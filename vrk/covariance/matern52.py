"""
Matérn covariance with smoothness parameter ν = 5/2.

The Matérn class of covariance functions (Matérn 1960; Stein 1999) offers a
principled way to control the smoothness of GP sample paths.  For ν = 5/2,
sample paths are twice mean-square differentiable — a good default for
physical fields such as terrain elevation, which are smooth but not infinitely
so.

References
----------
Matérn, B. (1960). *Spatial Variation*. Meddelanden fran Statens Skogsforskningsinstitut, 49(5).

Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*.
    Springer, New York.

Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine
    Learning*. MIT Press, Cambridge, MA.  (Section 4.2)

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
import numpy as np
from vrk.covariance.base import CovarianceFunction


class Matern52Covariance(CovarianceFunction):
    """
    Matérn-5/2 covariance function.

        C(h) = σ² (1 + √5 h/a + 5h²/(3a²)) exp(−√5 h/a)

    where h = ‖x − y‖ is the Euclidean lag distance.

    Properties
    ----------
    * Twice mean-square differentiable sample paths (smoothness ν = 5/2).
    * Decays to zero as h → ∞; effectively zero for h ≳ 3a.
    * Preferred over the squared-exponential for spatial fields that are
      smooth but exhibit some local roughness.

    Parameters
    ----------
    sill    : float > 0   σ², total marginal variance of the field
    range_a : float > 0   a, practical range (correlation ≈ 0.05 at h ≈ 3a)

    Gradients (natural space, for analytic_evidence_gradient in vrk.py)
    -------------------------------------------------------------------
    Let z = √5 h/a.  Then C(h) = σ² (1 + z + z²/3) exp(−z), and:

        ∂C/∂σ²  = C / σ²
        ∂C/∂a   = σ² · z²(1 + z)/(3a) · exp(−z)

    The range gradient follows from  dz/da = −z/a  and the product rule:

        d/da [(1+z+z²/3) exp(−z)] = (−z/3)(1+z) exp(−z) · (−z/a)
                                   = z²(1+z) exp(−z) / (3a)
    """

    def __init__(self, sill: float = 1.0, range_a: float = 1.0):
        if sill <= 0 or range_a <= 0:
            raise ValueError("sill and range_a must be positive")
        self._sill = float(sill)
        self._range_a = float(range_a)

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        H = self._dists(X, Y)
        return self._covariance_from_H(H)

    def _covariance_from_H(self, H: np.ndarray) -> np.ndarray:
        """Evaluate C(h) = σ²(1 + z + z²/3) exp(−z) with z = √5 h/a."""
        sqrt5 = np.sqrt(5.0)
        z = sqrt5 * H / self._range_a
        return self._sill * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonal C(0) = σ² for every location."""
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
            z = √5 h/a,  dz/da = −z/a
            d/da [(1+z+z²/3) exp(−z)] = [(1+z) − (1+z+z²/3)] dz/da · exp(−z)
                                       = (−z²/3)(−z/a) exp(−z) = z²(1+z) exp(−z)/(3a)
        """
        X = np.atleast_2d(X)
        H = self._dists(X, X)
        sqrt5 = np.sqrt(5.0)
        z = sqrt5 * H / self._range_a
        exp_z = np.exp(-z)
        C = self._sill * (1.0 + z + z ** 2 / 3.0) * exp_z

        dC_dsill = C / self._sill

        # ∂C/∂a = σ² · z²(1+z) · exp(−z) / (3a)
        dC_da = self._sill * (z ** 2) * (1.0 + z) * exp_z / (3.0 * self._range_a)

        return [dC_dsill, dC_da]
