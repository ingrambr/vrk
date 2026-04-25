"""
Demo: Gaussian likelihood — fixed and optimised hyperparameters.

Fits sin(x) + N(0, 0.05²) noise with two covariance configurations:
  1. Fixed covariance (Matern52 + Nugget, range_a=1)
  2. After hyperparameter optimisation via scipy (L-BFGS-B on log evidence)

Uses Matern52 + NuggetCovariance to demonstrate the standard geostatistics
nugget-effect model.  The Matérn 5/2 is preferred over Gaussian (RBF) for
physical data because it produces once-differentiable realisations.

Prints RMSE and displays predictions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, Matern52Covariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood)
from vrk.optimization.hyperparameters import optimise_hyperparameters

RNG = np.random.default_rng(42)

noise_std = 0.05
nugget_var = noise_std ** 2   # 0.0025

# ── Training data ──────────────────────────────────────────────────────────────
n = 60
X_train = np.linspace(0, 2 * np.pi, n)[:, None]
y_train = np.sin(X_train[:, 0]) + noise_std * RNG.standard_normal(n)

# ── Test grid ──────────────────────────────────────────────────────────────────
X_test = np.linspace(-0.5, 2 * np.pi + 0.5, 200)[:, None]
y_true = np.sin(X_test[:, 0])


def make_model(range_a, max_active=15, n_sweeps=3):
    """Matern52 + Nugget covariance model."""
    struct_cov = Matern52Covariance(sill=1.0, range_a=range_a)
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    lik = GaussianLikelihood(variance=1e-6)
    return VRK(cov, lik, max_active=max_active, n_sweeps=n_sweeps)


def run_fixed(range_a=1.0, max_active=15, n_sweeps=3):
    """Fit with fixed hyperparameters."""
    model = make_model(range_a, max_active, n_sweeps)
    model.fit(X_train, y_train)
    mean, var = model.predict(X_test)
    rmse = np.sqrt(np.mean((mean[50:150] - y_true[50:150]) ** 2))
    print(f"  [Fixed  range_a={range_a:.2f}]  "
          f"n_active={model.n_active}  RMSE={rmse:.4f}")
    return mean, var, model


def run_optimised(init_range_a=0.8, max_active=15, n_sweeps=3):
    """Fit with hyperparameter optimisation."""
    model = make_model(init_range_a, max_active, n_sweeps)
    ev = optimise_hyperparameters(model, X_train, y_train, n_restarts=1)
    mean, var = model.predict(X_test)
    params = model.covariance.get_params()  # [sill, range_a, nugget_sill]
    rmse = np.sqrt(np.mean((mean[50:150] - y_true[50:150]) ** 2))
    print(f"  [Optim. range_a={params[1]:.3f}, sill={params[0]:.3f}, nugget={params[2]:.4f}]  "
          f"n_active={model.n_active}  RMSE={rmse:.4f}  evidence={ev:.2f}")
    return mean, var, model


print("=" * 60)
print("Demo: Gaussian likelihood — 1D sin(x) regression (Matern52 + Nugget)")
print("=" * 60)

print("\n--- Fixed hyperparameters (varying range_a) ---")
mean_short, var_short, model_s = run_fixed(range_a=0.5)
mean_good,  var_good,  model_g = run_fixed(range_a=1.0)
mean_long,  var_long,  model_l = run_fixed(range_a=2.5)

print("\n--- Optimised hyperparameters ---")
mean_opt, var_opt, model_opt = run_optimised(init_range_a=0.8)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
fig.suptitle("VRK — Gaussian likelihood: sin(x) regression (Matern52 + Nugget)",
             fontsize=12)

x = X_test[:, 0]

# Left panel: fixed hyperparameter sweep
ax1.plot(x, y_true, "k--", lw=1.5, label="True sin(x)", zorder=4)
ax1.scatter(X_train[:, 0], y_train, s=18, c="k", alpha=0.35, zorder=3, label="Train")
colors = ["#e74c3c", "#2ecc71", "#3498db"]
labels = ["range_a=0.5", "range_a=1.0", "range_a=2.5"]
for mean, var, c, lbl in zip(
    [mean_short, mean_good, mean_long],
    [var_short, var_good, var_long],
    colors, labels,
):
    std = np.sqrt(np.maximum(var, 0))
    ax1.plot(x, mean, color=c, lw=1.5, label=lbl)
    ax1.fill_between(x, mean - 2 * std, mean + 2 * std, color=c, alpha=0.12)
ax1.set_title("Fixed hyperparameters")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlim(x[0], x[-1])

# Right panel: optimised hyperparameters
std_opt = np.sqrt(np.maximum(var_opt, 0))
params_opt = model_opt.covariance.get_params()
ax2.plot(x, y_true, "k--", lw=1.5, label="True sin(x)", zorder=4)
ax2.scatter(X_train[:, 0], y_train, s=18, c="k", alpha=0.35, zorder=3, label="Train")
ax2.plot(x, mean_opt, "#e74c3c", lw=2, label=f"Optimised range_a={params_opt[1]:.2f}")
ax2.fill_between(
    x, mean_opt - 2 * std_opt, mean_opt + 2 * std_opt,
    color="#e74c3c", alpha=0.18, label="±2σ",
)
if model_opt.n_active > 0:
    ax2.vlines(model_opt.active_set_[:, 0], -1.6, -1.4,
               color="#e74c3c", lw=1.5, label=f"Active set (m={model_opt.n_active})")
ax2.set_title(f"Optimised (range_a={params_opt[1]:.2f}, sill={params_opt[0]:.2f})")
ax2.set_xlabel("x")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_xlim(x[0], x[-1])

plt.tight_layout()
plt.savefig("vrk_demo_gaussian.png", dpi=150)
print("\nSaved vrk_demo_gaussian.png")
