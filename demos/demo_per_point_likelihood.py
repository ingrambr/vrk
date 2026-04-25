"""
Demo: Per-point likelihood API.

Sensor fusion scenario:
  - 20 accurate sensors: sin(x) + N(0, 0.05²)  in [0, π]
  - 10 noisy sensors:    sin(x) + N(0, 1.0²)   in [π, 2π]

Uses ExponentialCovariance + NuggetCovariance.

Compares:
  - Shared likelihood model (wrong: treats all sensors equally)
  - Per-point likelihood model (correct: each point gets its own noise level)

Demonstrates the explicit fit(X, y, likelihoods=[...]) API.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, ExponentialCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood)

RNG = np.random.default_rng(123)

n_accurate = 20
n_noisy    = 10
n_total    = n_accurate + n_noisy

# Accurate sensors: densely sampled in [0, π]
X_acc = np.linspace(0, np.pi, n_accurate)[:, None]
y_acc = np.sin(X_acc[:, 0]) + 0.05 * RNG.standard_normal(n_accurate)

# Noisy sensors: sparse in [π, 2π]
X_nsy = np.linspace(np.pi, 2 * np.pi, n_noisy)[:, None]
y_nsy = np.sin(X_nsy[:, 0]) + 1.0 * RNG.standard_normal(n_noisy)

X_train = np.vstack([X_acc, X_nsy])
y_train = np.concatenate([y_acc, y_nsy])

X_test = np.linspace(-0.2, 2 * np.pi + 0.2, 300)[:, None]
y_true = np.sin(X_test[:, 0])

# Per-point likelihoods
liks_per_point = (
    [GaussianLikelihood(variance=0.05 ** 2)] * n_accurate +
    [GaussianLikelihood(variance=1.0 ** 2)]  * n_noisy
)

print("=" * 65)
print("Demo: Per-point likelihood API — sensor fusion scenario")
print(f"      Covariance: ExponentialCovariance + NuggetCovariance")
print("=" * 65)
print(f"\n  Accurate sensors (σ=0.05): n={n_accurate}, x ∈ [0, π]")
print(f"  Noisy sensors   (σ=1.00):  n={n_noisy}, x ∈ [π, 2π]")

# Small background nugget for numerical stability (main noise from GaussianLikelihood)
def make_model():
    struct_cov = ExponentialCovariance(sill=1.0, range_a=1.5)
    nug_cov    = NuggetCovariance(sill=1e-4)   # tiny structural nugget
    cov = SumCovariance(struct_cov, nug_cov)
    # Default likelihood used only as fallback; per-point will override
    return VRK(cov, GaussianLikelihood(variance=0.25), max_active=15, n_sweeps=3)

# Shared likelihood model
model_shared = make_model()
model_shared.fit(X_train, y_train)    # no per-point
mean_shared, var_shared = model_shared.predict(X_test)
rmse_shared = np.sqrt(np.mean((mean_shared - y_true) ** 2))

print(f"\n[Shared likelihood σ=0.5 (wrong)]")
print(f"  n_active = {model_shared.n_active}")
print(f"  RMSE     = {rmse_shared:.5f}")

# Per-point likelihood model
model_pp = make_model()
model_pp.fit(X_train, y_train, likelihoods=liks_per_point)
mean_pp, var_pp = model_pp.predict(X_test)
rmse_pp = np.sqrt(np.mean((mean_pp - y_true) ** 2))

print(f"\n[Per-point likelihoods (correct)]")
print(f"  n_active = {model_pp.n_active}")
print(f"  RMSE     = {rmse_pp:.5f}")
print(f"\nRMSE improvement: {rmse_shared:.5f} → {rmse_pp:.5f} "
      f"({100 * (rmse_shared - rmse_pp) / rmse_shared:.1f}% reduction)")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
x = X_test[:, 0]

for ax, (mean, var, model, title, color) in zip(axes, [
    (mean_shared, var_shared, model_shared, "Shared likelihood (σ=0.5, wrong)", "#e74c3c"),
    (mean_pp,     var_pp,     model_pp,    "Per-point likelihoods (correct)",   "#3498db"),
]):
    std = np.sqrt(np.maximum(var, 0))
    ax.plot(x, y_true, "k--", lw=1.5, label="True sin(x)", zorder=5)
    ax.scatter(X_acc[:, 0], y_acc, s=30, c="#2ecc71", alpha=0.9, zorder=4,
               label="Accurate (σ=0.05)", marker="o")
    ax.scatter(X_nsy[:, 0], y_nsy, s=30, c="#e74c3c", alpha=0.9, zorder=4,
               label="Noisy (σ=1.0)", marker="x", linewidths=2)
    ax.plot(x, mean, color=color, lw=2.5, label="Prediction")
    ax.fill_between(x, mean - 2 * std, mean + 2 * std,
                    color=color, alpha=0.15, label="±2σ")
    if model.n_active > 0:
        ax.scatter(model.active_set_[:, 0], np.full(model.n_active, 1.4),
                   marker="|", s=200, c=color, linewidths=2, zorder=6,
                   label="Active set")
    rmse = np.sqrt(np.mean((mean - y_true) ** 2))
    ax.set_title(f"{title}\nn_active={model.n_active}, RMSE={rmse:.4f}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-2.5, 2.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.axvline(np.pi, color="gray", ls=":", lw=1, alpha=0.6)
    ax.grid(True, alpha=0.2)
    ax.axvspan(np.pi, 2 * np.pi, alpha=0.04, color="red")

plt.suptitle(
    "Sensor fusion: per-point likelihood API  (ExponentialCovariance + Nugget)\n"
    "Green=accurate sensors  Red×=noisy sensors  │=active set",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("vrk_demo_per_point_likelihood.png", dpi=120)
print("\nSaved vrk_demo_per_point_likelihood.png")
