"""
Example: 1-D kriging with Spherical + Nugget covariance.

Total covariance:
    C(h) = sigma^2 * (1 - 1.5*(h/a) + 0.5*(h/a)^3)  +  nugget * delta(h)
           for h <= a  (structural part = 0 for h > a)

The spherical model is the most widely used covariance in classical geostatistics
(mining, hydrology).  It has a finite range: at lag h = a the structural
contribution drops to exactly zero.  Adding a nugget gives the full nugget-effect
model, with a jump discontinuity at the origin.

Note: The spherical covariance is only positive definite in R^d for d <= 3.

Standard geostatistics parameterisation:
    nugget    — variance at h=0+ (jump at the origin)
    sill      — structural variance
    total sill = nugget + sill
    range     — lag at which the spherical component reaches zero
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import VRK, SphericalCovariance, NuggetCovariance, SumCovariance, GaussianLikelihood

rng = np.random.default_rng(1)

# --- data ---
n_train = 40
X_train = np.sort(rng.uniform(0, 2 * np.pi, n_train))[:, None]
noise_std = 0.2
y_train = np.sin(X_train[:, 0]) + noise_std * rng.standard_normal(n_train)

# --- covariance model ---
nugget_var  = noise_std ** 2        # 0.04
struct_sill = 1.0
range_a     = 3.0                   # covers most of [0, 2*pi]

struct_cov = SphericalCovariance(sill=struct_sill, range_a=range_a)
nugget_cov = NuggetCovariance(sill=nugget_var)
cov = SumCovariance(struct_cov, nugget_cov)

lik = GaussianLikelihood(variance=1e-6)
model = VRK(cov, lik, max_active=15, n_sweeps=3)
model.fit(X_train, y_train)

# --- predict (with slight extrapolation) ---
X_test = np.linspace(-0.5, 2 * np.pi + 0.5, 300)[:, None]
mean, var = model.predict(X_test)
std = np.sqrt(np.maximum(var, 0.0))

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

ax = axes[0]
ax.fill_between(X_test[:, 0], mean - 2 * std, mean + 2 * std,
                alpha=0.25, color="darkorange", label="95% CI")
ax.plot(X_test[:, 0], mean, color="darkorange", lw=2, label="Posterior mean")
ax.plot(X_test[:, 0], np.sin(X_test[:, 0]), "k--", lw=1, label="True sin(x)")
ax.scatter(X_train[:, 0], y_train, color="tomato", s=20, zorder=5, label="Training data")
if model.n_active > 0:
    ax.vlines(model.active_set_[:, 0], -2.1, -1.8,
              color="darkorange", lw=1.5, label=f"Active set (m={model.n_active})")
ax.set_ylabel("y")
ax.set_title(
    f"VRK — Spherical + Nugget  "
    f"(sill={struct_sill:.2f}, range={range_a:.2f}, nugget={nugget_var:.3f})\n"
    f"n={n_train} training points, m={model.n_active} active"
)
ax.legend(fontsize=8)
ax.set_ylim(-2.3, 2.3)

# --- covariance function (showing nugget jump) ---
ax2 = axes[1]
h_pos = np.linspace(1e-6, range_a * 1.4, 400)
C_struct = np.array([struct_cov(np.array([[0.0]]), np.array([[hi]]))[0, 0] for hi in h_pos])
C_total = nugget_var + C_struct

# Show nugget jump at h=0
ax2.plot([0, 0], [0, nugget_var + struct_sill], color="darkorange", lw=1.5, ls=":")
ax2.plot(0, nugget_var + struct_sill, "o", color="darkorange", ms=8,
         label=f"C(0) = {nugget_var + struct_sill:.3f}  (total sill)")
ax2.plot(h_pos, C_total, color="darkorange", lw=2, label="C(h) [h > 0]")
ax2.plot(h_pos, C_struct, color="darkorange", lw=2, ls="--", label="Structural only")
ax2.axhline(nugget_var, color="gray", ls=":", lw=1, label=f"Nugget = {nugget_var:.3f}")
ax2.axvline(range_a, color="tomato", ls="--", lw=1, label=f"Range a = {range_a:.1f}")
ax2.set_xlabel("Lag h")
ax2.set_ylabel("C(h)")
ax2.set_title("Covariance function: Spherical + Nugget  (note jump at h=0)")
ax2.legend(fontsize=8)
ax2.set_ylim(-0.05, (nugget_var + struct_sill) * 1.1)
ax2.set_xlim(-0.1, range_a * 1.4)

plt.tight_layout()
plt.savefig("vrk_spherical_1d.png", dpi=150)
print("Saved vrk_spherical_1d.png")

# --- summary ---
print(f"\nCovariance model: Spherical + Nugget")
print(f"  structural sill = {struct_sill:.4f}")
print(f"  range_a         = {range_a:.4f}")
print(f"  nugget          = {nugget_var:.4f}")
print(f"  total sill      = {struct_sill + nugget_var:.4f}")
c_at_range = struct_cov(np.array([[0.0]]), np.array([[range_a]]))[0, 0]
print(f"  C_struct(a)     = {c_at_range:.6f}  (should be 0)")
print(f"Active set size: {model.n_active} / {model.max_active}")
print(f"EP approximate evidence: {model.approximate_evidence():.4f}")

X_pts = np.array([[np.pi / 4], [np.pi / 2], [np.pi], [3 * np.pi / 2]])
m_pts, v_pts = model.predict(X_pts)
print("\nPredictions at selected points:")
print(f"  {'x':>8}  {'true':>8}  {'mean':>8}  {'std':>8}")
for xi, yi_true, mi, vi in zip(X_pts[:, 0], np.sin(X_pts[:, 0]), m_pts, np.sqrt(v_pts)):
    print(f"  {xi:8.4f}  {yi_true:8.4f}  {mi:8.4f}  {vi:8.4f}")
