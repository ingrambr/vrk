"""
Exponential (Ornstein-Uhlenbeck) covariance function.

This is the Matérn covariance with smoothness parameter ν = 1/2, the simplest
member of the Matérn family.  It is equivalent to the covariance of the
Ornstein-Uhlenbeck process (continuous-time first-order autoregressive model)
and produces sample paths that are continuous but nowhere mean-square
differentiable — appropriate for rough, irregular spatial fields.

Convention note
---------------
This implementation uses the standard geostatistics parameterisation
  C(h) = σ² exp(−h/a)
where the range a corresponds to a practical range of approximately 3a (the
distance at which the correlation falls to exp(−3) ≈ 0.05).

This differs from some GP literature (e.g. Rasmussen & Williams 2006) which
writes  C(h) = σ² exp(−h/(2ℓ))  with a different length-scale convention.

References
----------
Matérn, B. (1960). *Spatial Variation*. Meddelanden fran Statens Skogsforskningsinstitut, 49(5).

Chilès, J.-P. and Delfiner, P. (2012). *Geostatistics: Modeling Spatial
    Uncertainty*, 2nd ed.  Wiley.  (Section 2.4 — exponential model)

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
import numpy as np
from vrk.covariance.base import CovarianceFunction


class ExponentialCovariance(CovarianceFunction):
    """
    Exponential covariance function (Matérn ν = 1/2, Ornstein-Uhlenbeck).

        C(h) = σ² exp(−h / a)

    where h = ‖x − y‖ is the Euclidean lag distance.

    Properties
    ----------
    * Continuous but not differentiable sample paths (roughest Matérn).
    * Finite correlation length; C(h) → 0 as h → ∞.
    * The correlation structure decays monotonically; no hole effects.

    Parameters
    ----------
    sill    : float > 0   σ², marginal variance of the field
    range_a : float > 0   a, range parameter (effective range ≈ 3a)

    Gradients (natural space)
    -------------------------
        ∂C/∂σ²  = C / σ²
        ∂C/∂a   = C ⊙ H / a²    where H is the distance matrix
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
        return self._sill * np.exp(-H / self._range_a)

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
            C = σ² exp(−h/a)
            ∂C/∂a = C · h / a²   from  d/da exp(−h/a) = h/a² · exp(−h/a)
        """
        X = np.atleast_2d(X)
        H = self._dists(X, X)
        C = self._sill * np.exp(-H / self._range_a)
        dC_dsill = C / self._sill
        dC_da = C * H / (self._range_a ** 2)
        return [dC_dsill, dC_da]
