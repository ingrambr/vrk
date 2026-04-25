"""
Spherical covariance function — classical geostatistics model.

The spherical model is one of the three original covariance models introduced
by Matheron (1963) and forms the core of classical kriging practice.  Its
defining property is *compact support*: the correlation is exactly zero for
lags h > a, so observations beyond the range have no influence on predictions.

Positive-definiteness restriction
----------------------------------
The spherical model is positive definite only in ℝ^d for d ≤ 3.  In higher
dimensions it is not a valid covariance function and should not be used.
For d > 3, the Matérn family (matern52.py) is appropriate.

References
----------
Journel, A. G. and Huijbregts, Ch. J. (1978). *Mining Geostatistics*.
    Academic Press, London.  (Chapter III — the spherical model)

Chilès, J.-P. and Delfiner, P. (2012). *Geostatistics: Modeling Spatial
    Uncertainty*, 2nd ed.  Wiley.

Cressie, N. A. C. (1993). *Statistics for Spatial Data*, revised ed.
    Wiley, New York.  (Section 2.4)

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
import numpy as np
from vrk.covariance.base import CovarianceFunction


class SphericalCovariance(CovarianceFunction):
    """
    Spherical covariance function.

        C(h) = σ² (1 − 1.5 t + 0.5 t³)   for t = h/a ≤ 1
               0                            for h > a

    where h = ‖x − y‖ is the Euclidean lag distance and a is the *range*
    (the distance at which C(h) first reaches zero).

    Properties
    ----------
    * Compact support: C(h) = 0 for all h > a.  This makes the covariance
      matrix sparse for spatially localised data — a computational advantage
      over global models like the Matérn.
    * The derivative at h = 0 is non-zero (linear near-origin behaviour),
      indicating sample paths that are continuous but not differentiable.
    * Jump discontinuity in the derivative at h = a (kink at the range).
    * Valid only in ℝ^d for d ≤ 3.

    Parameters
    ----------
    sill    : float > 0   σ², total marginal variance
    range_a : float > 0   a, range (correlation = 0 at h = a)

    Gradients (natural space)
    -------------------------
    Let t = h/a.  Then:

        ∂C/∂σ²  = (1 − 1.5t + 0.5t³)     for h ≤ a, else 0
        ∂C/∂a   = σ² · 1.5(1 − t²) · h/a²  for h ≤ a, else 0

    The range gradient follows from  d/da (1 − 1.5 h/a + 0.5 (h/a)³):

        d/da [−1.5 h/a + 0.5 h³/a³] = 1.5 h/a² − 1.5 h³/a⁴ = 1.5(h/a²)(1 − h²/a²)
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
        t = H / self._range_a
        C = self._sill * (1.0 - 1.5 * t + 0.5 * t ** 3)
        C[t > 1.0] = 0.0   # Compact support: zero beyond the range
        return C

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

        Both gradients are zero for pairs with h > a (compact support).
        """
        X = np.atleast_2d(X)
        H = self._dists(X, X)
        t = H / self._range_a
        inside = t <= 1.0

        # ∂C/∂σ² = (1 − 1.5t + 0.5t³) for h ≤ a, else 0
        dC_dsill = np.where(inside, 1.0 - 1.5 * t + 0.5 * t ** 3, 0.0)

        # ∂C/∂a = σ² · 1.5(1 − t²) · h/a²  for h ≤ a, else 0
        dC_da = np.where(inside, self._sill * 1.5 * (1.0 - t ** 2) * (H / self._range_a ** 2), 0.0)

        return [dC_dsill, dC_da]
