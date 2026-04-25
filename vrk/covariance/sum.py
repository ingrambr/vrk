"""
Sum-of-covariances composite kernel.

The sum of any finite number of valid (positive semi-definite) covariance
functions is itself a valid covariance function.  This follows directly from
the fact that a non-negative linear combination of positive semi-definite
matrices is positive semi-definite.

The most common usage in spatial statistics is the *structural + nugget*
decomposition:

    C_total(h) = C_struct(h) + C_nugget(h)

where C_struct captures spatial correlation (Matérn52, Exponential, etc.) and
the nugget absorbs uncorrelated measurement error (see nugget.py).  This is
equivalent to the GP + noise observation model:

    y_i = f(x_i) + ε_i,   f ~ GP(0, C_struct),   ε_i ~ N(0, σ²_n) i.i.d.

More complex decompositions are also possible, for example separating short-range
and long-range structural components:

    C(h) = C_long(h) + C_short(h) + C_nugget(h)

Parameters are concatenated in the order the component covariances are supplied
to the constructor, and split accordingly in get_params() / set_params().
"""
import numpy as np
from vrk.covariance.base import CovarianceFunction


class SumCovariance(CovarianceFunction):
    """
    Additive composite covariance:  C(h) = C₁(h) + C₂(h) + …

    Hyperparameters of each component are concatenated into a single vector
    in the order the covariances were supplied, and split back by set_params().

    Parameters
    ----------
    *covariances : CovarianceFunction instances
        Two or more covariance functions to sum.
    """

    def __init__(self, *covariances: CovarianceFunction):
        if not covariances:
            raise ValueError("SumCovariance requires at least one covariance function")
        self._covariances = list(covariances)

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """C_total(X, Y) = Σ_k C_k(X, Y)."""
        return sum(c(X, Y) for c in self._covariances)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Total diagonal variance = Σ_k C_k(x_i, x_i) for each row."""
        return sum(c.diag(X) for c in self._covariances)

    @property
    def n_params(self) -> int:
        """Total number of hyperparameters across all component covariances."""
        return sum(c.n_params for c in self._covariances)

    def get_params(self) -> np.ndarray:
        """Concatenated hyperparameters from all components in order."""
        return np.concatenate([c.get_params() for c in self._covariances])

    def set_params(self, p: np.ndarray) -> None:
        """Distribute the concatenated parameter vector to each component."""
        idx = 0
        for c in self._covariances:
            c.set_params(p[idx: idx + c.n_params])
            idx += c.n_params

    def gradient_matrix(self, X: np.ndarray) -> list[np.ndarray]:
        """
        Concatenated list of gradient matrices from all components.

        Because  C = Σ_k C_k,  the gradient w.r.t. any parameter of component k
        is simply the gradient of C_k alone:  ∂C/∂θ_k = ∂C_k/∂θ_k.
        """
        grads = []
        for c in self._covariances:
            grads.extend(c.gradient_matrix(X))
        return grads

    @property
    def covariances(self) -> list[CovarianceFunction]:
        """List of the constituent covariance function objects."""
        return self._covariances
