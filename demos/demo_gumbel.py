"""
Demo: Gumbel (extreme value type I) likelihood.

The Gumbel distribution is right-skewed and widely used in:
  - Annual maximum rainfall / flood return-period analysis
  - Extreme wind speed modelling
  - Reliability / fatigue analysis

Model:
  p(y | f) = (1/β) * exp(-(y - f)/β - exp(-(y - f)/β))

where f is the latent GP (location parameter = mode) and β is a fixed scale.

Properties:
  Mean:     E[y|f] = f + β·γ   (γ ≈ 0.5772, Euler–Mascheroni constant)
  Variance: Var[y|f] = π²β²/6

This demo simulates annual maximum precipitation, fits a VRK with Gumbel
likelihood, and compares against a Gaussian likelihood baseline.

True latent function: f(x) = sin(x) + 2  (location parameter of Gumbel)
Scale: β = 0.5

Uses ExponentialCovariance + NuggetCovariance.  The exponential covariance
(Matérn 1/2) is often used for environmental extremes which show rough,
non-differentiable spatial variation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

from vrk import (VRK, ExponentialCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood, GumbelLikelihood)

RNG = np.random.default_rng(42)

BETA       = 0.5                       # Gumbel scale
EULER_MASCHERONI = 0.5772156649015328  # γ

# ── Simulate extreme value data ────────────────────────────────────────────────
n = 60
X_train  = np.sort(RNG.uniform(0, 2 * np.pi, n))[:, None]
f_true   = np.sin(X_train[:, 0]) + 2.0         # true location (mode)
y_train  = f_true + gumbel_r.rvs(scale=BETA, size=n, random_state=RNG)
# y is always right-skewed: most values near f, heavy right tail

X_test      = np.linspace(0, 2 * np.pi, 200)[:, None]
f_test_true = np.sin(X_test[:, 0]) + 2.0        # true mode
mean_test   = f_test_true + BETA * EULER_MASCHERONI   # true mean
std_test    = np.pi * BETA / np.sqrt(6)          # true std of data

print("=" * 65)
print("Demo: Gumbel (extreme value type I) likelihood")
print(f"      Covariance: ExponentialCovariance + NuggetCovariance")
print("=" * 65)
print(f"\nTrue latent (mode): f(x) = sin(x) + 2")
print(f"Gumbel scale: β = {BETA}")
print(f"True data mean offset from mode: β·γ = {BETA * EULER_MASCHERONI:.4f}")
print(f"True data std: π·β/√6 = {std_test:.4f}")
print(f"\nObserved y range: [{y_train.min():.3f}, {y_train.max():.3f}]")
print(f"True mode  range: [{f_true.min():.3f}, {f_true.max():.3f}]")


def make_model(lik, sill=1.0, range_a=1.0, nugget=0.01):
    struct_cov = ExponentialCovariance(sill=sill, range_a=range_a)
    nug_cov    = NuggetCovariance(sill=nugget)
    cov = SumCovariance(struct_cov, nug_cov)
    return VRK(cov, lik, max_active=15, n_sweeps=3)


# ── Gaussian baseline ─────────────────────────────────────────────────────────
print("\n--- Gaussian likelihood (symmetric, ignores right-skew) ---")
model_g = make_model(GaussianLikelihood(variance=0.5))
model_g.fit(X_train, y_train)
mean_g, var_g = model_g.predict(X_test)
# Gaussian fits the mean (not the mode)
rmse_mode_g = np.sqrt(np.mean((mean_g - f_test_true) ** 2))
rmse_mean_g = np.sqrt(np.mean((mean_g - mean_test) ** 2))
print(f"  n_active={model_g.n_active}")
print(f"  RMSE(vs mode) = {rmse_mode_g:.4f}")
print(f"  RMSE(vs mean) = {rmse_mean_g:.4f}")

# ── Gumbel likelihood ─────────────────────────────────────────────────────────
print("\n--- Gumbel likelihood (EP via Gauss-Hermite quadrature) ---")
model_gum = make_model(GumbelLikelihood(beta=BETA))
model_gum.fit(X_train, y_train)
mean_gum, var_gum = model_gum.predict(X_test)
# Gumbel fits the mode, not the mean
rmse_mode_gum = np.sqrt(np.mean((mean_gum - f_test_true) ** 2))
# Adjust Gumbel posterior mean by β·γ for fair comparison
mean_gum_adj  = mean_gum + BETA * EULER_MASCHERONI
rmse_mean_gum = np.sqrt(np.mean((mean_gum_adj - mean_test) ** 2))
print(f"  n_active={model_gum.n_active}")
print(f"  RMSE(vs mode)       = {rmse_mode_gum:.4f}  (should be smaller than Gaussian)")
print(f"  RMSE(adjusted mean) = {rmse_mean_gum:.4f}")
print(f"  EP approximate evidence: {model_gum.approximate_evidence():.4f}")

# ── Return period quantiles ────────────────────────────────────────────────────
print("\n--- Return period quantiles at x = π/2 ---")
x_query = np.array([[np.pi / 2]])
m_q, v_q = model_gum.predict(x_query)
f_mode = m_q[0]
print(f"  Predicted mode at x=π/2: {f_mode:.4f}  (true: {np.sin(np.pi/2)+2:.4f})")
for T in [2, 10, 50, 100]:
    # Gumbel quantile: F^{-1}(1-1/T) = mode - β·ln(-ln(1-1/T))
    q = f_mode - BETA * np.log(-np.log(1.0 - 1.0 / T))
    print(f"  T={T:4d}-year event quantile: {q:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "VRK — Gumbel likelihood: extreme value regression  "
    "(ExponentialCovariance + Nugget)",
    fontsize=13,
)

x_te = X_test[:, 0]
x_tr = X_train[:, 0]
std_g   = np.sqrt(np.maximum(var_g, 0))
std_gum = np.sqrt(np.maximum(var_gum, 0))

# ── Panel 1: data ──────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.scatter(x_tr, y_train, s=20, c="k", alpha=0.5, zorder=5,
            label=f"Observed y (Gumbel, β={BETA})")
ax1.plot(x_te, f_test_true, "k--", lw=1.5, label="True mode f(x)", zorder=4)
ax1.plot(x_te, mean_test,   "k:",  lw=1.0, label="True mean f+β·γ", zorder=4)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Training data\n(right-skewed: observations above mode)")
ax1.legend(fontsize=8)
ax1.set_xlim(x_te[0], x_te[-1])

# ── Panel 2: Gaussian baseline ─────────────────────────────────────────────────
ax2 = axes[1]
ax2.scatter(x_tr, y_train, s=12, c="k", alpha=0.3, zorder=5)
ax2.plot(x_te, f_test_true, "k--", lw=1.5, label="True mode f(x)")
ax2.plot(x_te, mean_g, "#3498db", lw=2, label=f"Gaussian mean (RMSE={rmse_mode_g:.3f})")
ax2.fill_between(x_te, mean_g - 2 * std_g, mean_g + 2 * std_g,
                 color="#3498db", alpha=0.18, label="±2σ")
if model_g.n_active > 0:
    ax2.vlines(model_g.active_set_[:, 0], 0.8, 1.0,
               color="#3498db", lw=1.5, label=f"Active set (m={model_g.n_active})")
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")
ax2.set_title("Gaussian likelihood baseline\n(biased: fits mean, not mode)")
ax2.legend(fontsize=8)
ax2.set_xlim(x_te[0], x_te[-1])

# ── Panel 3: Gumbel likelihood ─────────────────────────────────────────────────
ax3 = axes[2]
ax3.scatter(x_tr, y_train, s=12, c="k", alpha=0.3, zorder=5)
ax3.plot(x_te, f_test_true, "k--", lw=1.5, label="True mode f(x)")
ax3.plot(x_te, mean_gum, "#e74c3c", lw=2,
         label=f"Gumbel mode (RMSE={rmse_mode_gum:.3f})")
ax3.fill_between(x_te, mean_gum - 2 * std_gum, mean_gum + 2 * std_gum,
                 color="#e74c3c", alpha=0.18, label="±2σ (latent)")
# Plot T-year return levels using posterior mode
for T, ls in [(10, "--"), (50, ":")]:
    rl = mean_gum - BETA * np.log(-np.log(1.0 - 1.0 / T))
    ax3.plot(x_te, rl, color="#e74c3c", lw=1, ls=ls, alpha=0.7,
             label=f"T={T}-year level")
if model_gum.n_active > 0:
    ax3.vlines(model_gum.active_set_[:, 0], 0.8, 1.0,
               color="#e74c3c", lw=1.5,
               label=f"Active set (m={model_gum.n_active})")
ax3.set_xlabel("x")
ax3.set_ylabel("f(x) / return level")
ax3.set_title("Gumbel likelihood (EP)\nCorrectly identifies mode; shows return levels")
ax3.legend(fontsize=8)
ax3.set_xlim(x_te[0], x_te[-1])

plt.tight_layout()
plt.savefig("vrk_demo_gumbel.png", dpi=150)
print("\nSaved vrk_demo_gumbel.png")

# ── Gumbel distribution illustration ──────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 4))
y_range = np.linspace(-0.5, 5.0, 300)
for f_val, col in [(1.5, "#3498db"), (2.0, "#e74c3c"), (2.5, "#2ecc71")]:
    pdf = gumbel_r.pdf(y_range, loc=f_val, scale=BETA)
    ax.plot(y_range, pdf, color=col, lw=2, label=f"f={f_val:.1f}")
    ax.axvline(f_val, color=col, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(f_val + BETA * EULER_MASCHERONI, color=col, lw=0.8, ls=":", alpha=0.6)
ax.set_xlabel("y")
ax.set_ylabel("p(y | f)")
ax.set_title(f"Gumbel likelihood (β={BETA})\nDashed = mode (f), dotted = mean (f+β·γ)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("vrk_demo_gumbel_distribution.png", dpi=150)
print("Saved vrk_demo_gumbel_distribution.png")
