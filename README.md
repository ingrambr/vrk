# vrk
Variable Rank Kriging

## Installation

```bash
pip install -e vrk/
```

## Quick start

```python
import numpy as np
from vrk import VRK, ExponentialCovariance, SphericalCovariance, GaussianLikelihood

# 1-D example: fit sin(x) with an Exponential covariance
X = np.linspace(0, 6, 50)[:, None]
y = np.sin(X[:, 0]) + 0.1 * np.random.randn(50)

cov = ExponentialCovariance(sill=1.0, range_a=1.0)
lik = GaussianLikelihood(variance=0.05)
model = VRK(cov, lik, max_active=20, n_sweeps=3)
model.fit(X, y)

X_test = np.linspace(0, 6, 100)[:, None]
mean, var = model.predict(X_test)

# Spherical covariance (valid for d <= 3)
cov_sph = SphericalCovariance(sill=1.0, range_a=2.0)
model2 = VRK(cov_sph, lik, max_active=20, n_sweeps=3)
model2.fit(X, y)
```

