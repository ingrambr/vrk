"""
Example: 1-D kriging with Gaussian + Nugget covariance.

Total covariance:
    C(h) = sigma^2 * exp(-h^2 / (2*a^2))  +  nugget * delta(h)

The Gaussian covariance produces infinitely differentiable (very smooth) realisations.
It is the most common covariance in the machine-learning GP literature (RBF / SE kernel).
In geostatistics it is sometimes considered unrealistically smooth; the Matern family
is often preferred for physical data.  Adding a nugget models measurement error.

Standard geostatistics parameterisation:
    nugget    — variance at h=0+ (measurement error / micro-scale variability)
    sill      — structural variance
    total sill = nugget + sill
    range     — effective range (correlation length)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import VRK, GaussianCovariance, NuggetCovariance, SumCovariance, GaussianLikelihood

rng = np.random.default_rng(2)

# --- data ---
n_train = 40
X_train = np.sort(rng.uniform(0, 2 * np.pi, n_train))[:, None]
noise_std = 0.2
y_train = np.sin(X_train[:, 0]) + noise_std * rng.standard_normal(n_train)

X_test = np.linspace(0, 2 * np.pi, 300)[:, None]
y_true = np.sin(X_test[:, 0])

# --- covariance model ---
nugget_var  = noise_std ** 2        # 0.04
struct_sill = 1.0
range_a     = 1.0

struct_cov = GaussianCovariance(sill=struct_sill, range_a=range_a)
nugget_cov = NuggetCovariance(sill=nugget_var)
cov = SumCovariance(struct_cov, nugget_cov)

lik = GaussianLikelihood(variance=1e-6)
model = VRK(cov, lik, max_active=15, n_sweeps=3)
model.fit(X_train, y_train)
mean, var = model.predict(X_test)
std = np.sqrt(np.maximum(var, 0.0))

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(8, 7))

ax = axes[0]
ax.fill_between(X_test[:, 0], mean - 2 * std, mean + 2 * std,
                alpha=0.25, color="mediumseagreen", label="95% CI")
ax.plot(X_test[:, 0], mean, color="mediumseagreen", lw=2, label="Posterior mean")
ax.plot(X_test[:, 0], y_true, "k--", lw=1, label="True sin(x)")
ax.scatter(X_train[:, 0], y_train, color="tomato", s=20, zorder=5, label="Training data")
if model.n_active > 0:
    ax.vlines(model.active_set_[:, 0], -2.1, -1.8,
              color="mediumseagreen", lw=1.5, label=f"Active set (m={model.n_active})")
ax.set_ylabel("y")
ax.set_title(
    f"VRK — Gaussian + Nugget  "
    f"(sill={struct_sill:.2f}, range={range_a:.2f}, nugget={nugget_var:.3f})\n"
    f"n={n_train} training points, m={model.n_active} active"
)
ax.legend(fontsize=8)
ax.set_ylim(-2.2, 2.2)

# --- compare three range values, each with the same nugget ---
ax2 = axes[1]
h = np.linspace(1e-6, 4.0, 300)
X0 = np.array([[0.0]])
colors = ["mediumseagreen", "steelblue", "darkorange"]
for a_val, col in zip([0.5, 1.0, 2.0], colors):
    sc = GaussianCovariance(sill=struct_sill, range_a=a_val)
    C_h = np.array([sc(X0, np.array([[hi]]))[0, 0] for hi in h]) + nugget_var
    ax2.plot(h, C_h, color=col, lw=2, label=f"range_a={a_val:.1f}")
ax2.axhline(nugget_var, color="gray", ls=":", lw=1, label=f"Nugget = {nugget_var:.3f}")
ax2.set_xlabel("Lag h")
ax2.set_ylabel("C(h)")
ax2.set_title("Gaussian + Nugget covariance for different range values")
ax2.legend(fontsize=8)
ax2.set_ylim(-0.02, struct_sill + nugget_var + 0.05)

plt.tight_layout()
plt.savefig("vrk_gaussian_1d.png", dpi=150)
print("Saved vrk_gaussian_1d.png")

# --- summary ---
print(f"\nCovariance model: Gaussian + Nugget")
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
