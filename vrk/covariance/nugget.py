"""
Nugget (white-noise / pure measurement-error) covariance function.

In geostatistics, the *nugget effect* represents two distinct physical phenomena:

  1. **Measurement error**: Repeated observations at the same location differ
     due to instrument noise, sampling variability, or digitisation error.

  2. **Micro-scale variability**: Spatial variation at scales below the
     sampling resolution that cannot be resolved by the data.

The nugget contributes a constant variance σ²_n to every observation but
carries no spatial structure — observations at different locations are
uncorrelated regardless of distance.  This is reflected in the discontinuity
of the variogram at the origin (the semivariogram jumps from 0 to σ²_n/2
as h → 0⁺).

Usage in composite models
--------------------------
The nugget is almost always combined with a structural covariance via
SumCovariance:

    C_total(h) = C_struct(h) + C_nugget(h)

The structural term (Matérn52, Exponential, etc.) models the spatial
correlation of the true latent field; the nugget absorbs measurement noise.
This composite is equivalent to GP regression with an explicit noise model:

    y_i = f(x_i) + ε_i,   ε_i ~ N(0, σ²_n) i.i.d.

under a Gaussian likelihood with variance σ²_n.

Note on off-diagonal blocks
----------------------------
When called with two distinct arrays X ≠ Y (cross-covariance block), the
nugget returns a zero matrix because observations at different locations are
uncorrelated.  The nugget only acts on the diagonal of C(X, X).

References
----------
Journel, A. G. and Huijbregts, Ch. J. (1978). *Mining Geostatistics*.
    Academic Press, London.  (Section III.A.2 — nugget effect)

Chilès, J.-P. and Delfiner, P. (2012). *Geostatistics: Modeling Spatial
    Uncertainty*, 2nd ed.  Wiley.

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
import numpy as np
from vrk.covariance.base import CovarianceFunction


class NuggetCovariance(CovarianceFunction):
    """
    Nugget (pure diagonal) covariance function.

        C(h) = σ²_n   if h = 0  (same location, i.e. X[i] == Y[i])
               0       if h > 0  (distinct locations)

    When evaluated as C(X, X) (self-covariance), returns σ²_n I_n.
    When evaluated as C(X, Y) with Y ≠ X, returns the zero matrix.

    Parameters
    ----------
    sill : float > 0   σ²_n, nugget variance (observation noise or micro-scale)
    """

    def __init__(self, sill: float = 1.0):
        if sill <= 0:
            raise ValueError("sill must be positive")
        self._sill = float(sill)

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = np.atleast_2d(X)
        if Y is None:
            # Self-covariance: σ²_n × identity (diagonal nugget matrix)
            return self._sill * np.eye(X.shape[0])
        # Cross-covariance between distinct location sets: all zeros
        Y = np.atleast_2d(Y)
        return np.zeros((X.shape[0], Y.shape[0]))

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonal σ²_n at every location (C(0) = σ²_n)."""
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self._sill)

    @property
    def n_params(self) -> int:
        return 1

    def get_params(self) -> np.ndarray:
        return np.array([self._sill])

    def set_params(self, p: np.ndarray) -> None:
        self._sill = float(p[0])

    def gradient_matrix(self, X: np.ndarray) -> list[np.ndarray]:
        """
        Returns [∂C/∂σ²_n] = [I_n].

        The gradient of the nugget w.r.t. its sill parameter is the
        identity matrix: ∂(σ²_n I)/∂σ²_n = I.
        """
        X = np.atleast_2d(X)
        return [np.eye(X.shape[0])]
