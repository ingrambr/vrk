"""
Example: 1-D kriging with Exponential + Nugget covariance.

Total covariance:
    C(h) = sigma^2 * exp(-h / a)  +  nugget * delta(h)

The nugget (white noise) component captures measurement error and micro-scale
variability at zero lag.  The exponential structure function produces rough
(non-differentiable) realisations — equivalent to Matern nu=1/2.

Standard geostatistics parameterisation:
    nugget    — variance at h=0+ (discontinuity at the origin)
    sill      — structural variance (asymptotic level of the structural part)
    total sill = nugget + sill
    range     — distance at which spatial correlation effectively dies out
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import VRK, ExponentialCovariance, NuggetCovariance, SumCovariance, GaussianLikelihood

rng = np.random.default_rng(0)

# --- data ---
n_train = 40
X_train = np.sort(rng.uniform(0, 2 * np.pi, n_train))[:, None]
noise_std = 0.2
y_train = np.sin(X_train[:, 0]) + noise_std * rng.standard_normal(n_train)

# --- covariance model ---
# nugget captures measurement error; structural part captures spatial correlation
nugget_var  = noise_std ** 2        # 0.04 — matches the data noise
struct_sill = 1.0                   # structural sill
range_a     = 1.0                   # range

struct_cov = ExponentialCovariance(sill=struct_sill, range_a=range_a)
nugget_cov = NuggetCovariance(sill=nugget_var)
cov = SumCovariance(struct_cov, nugget_cov)

# GaussianLikelihood variance is near-zero: nugget already models the noise
lik = GaussianLikelihood(variance=1e-6)
model = VRK(cov, lik, max_active=15, n_sweeps=3)
model.fit(X_train, y_train)

# --- predict ---
X_test = np.linspace(0, 2 * np.pi, 200)[:, None]
mean, var = model.predict(X_test)
std = np.sqrt(np.maximum(var, 0.0))

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(8, 7))

ax = axes[0]
ax.fill_between(X_test[:, 0], mean - 2 * std, mean + 2 * std,
                alpha=0.25, color="steelblue", label="95% CI")
ax.plot(X_test[:, 0], mean, color="steelblue", lw=2, label="Posterior mean")
ax.plot(X_test[:, 0], np.sin(X_test[:, 0]), "k--", lw=1, label="True sin(x)")
ax.scatter(X_train[:, 0], y_train, color="tomato", s=20, zorder=5, label="Training data")
if model.n_active > 0:
    ax.vlines(model.active_set_[:, 0], -2, -1.7,
              color="steelblue", lw=1.5, label=f"Active set (m={model.n_active})")
ax.set_ylabel("y")
ax.set_title(
    f"VRK — Exponential + Nugget  "
    f"(sill={struct_sill:.2f}, range={range_a:.2f}, nugget={nugget_var:.3f})\n"
    f"n={n_train} training points, m={model.n_active} active"
)
ax.legend(fontsize=8)
ax.set_ylim(-2.2, 2.2)

# --- covariance function plot ---
ax2 = axes[1]
h = np.linspace(0, range_a * 3, 300)
X0 = np.array([[0.0]])
C_struct = np.array([struct_cov(X0, np.array([[hi]]))[0, 0] for hi in h])
C_total_nonzero = nugget_var + C_struct   # total at h > 0 (nugget only visible at h=0)

ax2.plot(h, C_struct, color="steelblue", lw=2, ls="--", label="Structural C(h)")
ax2.plot(h, C_total_nonzero, color="steelblue", lw=2, label="Total C(h) [h > 0]")
ax2.plot(0, nugget_var + struct_sill, "o", color="steelblue", ms=8, label=f"C(0) = {nugget_var + struct_sill:.3f}")
ax2.axhline(nugget_var, color="gray", ls=":", lw=1, label=f"Nugget = {nugget_var:.3f}")
ax2.axvline(range_a, color="tomato", ls="--", lw=1, label=f"Range a = {range_a:.1f}")
ax2.set_xlabel("Lag h")
ax2.set_ylabel("C(h)")
ax2.set_title("Covariance function: Exponential + Nugget")
ax2.legend(fontsize=8)
ax2.set_ylim(-0.05, (nugget_var + struct_sill) * 1.1)

plt.tight_layout()
plt.savefig("vrk_exponential_1d.png", dpi=150)
print("Saved vrk_exponential_1d.png")

# --- summary ---
print(f"\nCovariance model: Exponential + Nugget")
print(f"  structural sill = {struct_sill:.4f}")
print(f"  range_a         = {range_a:.4f}")
print(f"  nugget          = {nugget_var:.4f}")
print(f"  total sill      = {struct_sill + nugget_var:.4f}")
print(f"Active set size: {model.n_active} / {model.max_active}")
print(f"EP approximate evidence: {model.approximate_evidence():.4f}")

X_pts = np.array([[np.pi / 4], [np.pi / 2], [np.pi], [3 * np.pi / 2]])
m_pts, v_pts = model.predict(X_pts)
print("\nPredictions at selected points:")
print(f"  {'x':>8}  {'true':>8}  {'mean':>8}  {'std':>8}")
for xi, yi_true, mi, vi in zip(X_pts[:, 0], np.sin(X_pts[:, 0]), m_pts, np.sqrt(v_pts)):
    print(f"  {xi:8.4f}  {yi_true:8.4f}  {mi:8.4f}  {vi:8.4f}")
