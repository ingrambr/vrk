"""
Demo: Mixed per-point likelihoods — FIXED covariance hyperparameters.

True latent function: f(x) = sin(x)

Covariance: Matern52 + NuggetCovariance (range_a=1.5, sill=1, nugget=0.01).
See demo_mixed_likelihood_optimised.py for the version that estimates them.

Data generation:
  Left  half  x ∈ [0, π]:   one-sided exponential noise, rate=3
                              y_i = f(x_i) + Exp(1/3),  so y_i >= f(x_i)
  Right half  x ∈ [π, 2π]:  symmetric Gaussian noise, σ=0.1
                              y_i = f(x_i) + N(0, 0.1²)

Three models compared:
  1. All-Exponential  — ExponentialNoiseLikelihood applied to every point
  2. All-Gaussian     — GaussianLikelihood applied to every point
  3. Mixed (correct)  — per-point likelihoods matching the data generation
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, Matern52Covariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood, ExponentialNoiseLikelihood)

RNG = np.random.default_rng(42)

# ── Data generation ────────────────────────────────────────────────────────────
n_half = 20

X_exp = np.sort(RNG.uniform(0.0, np.pi, n_half))
f_exp = np.sin(X_exp)
y_exp = f_exp + RNG.exponential(scale=1.0 / 3.0, size=n_half)

X_gau = np.sort(RNG.uniform(np.pi, 2.0 * np.pi, n_half))
f_gau = np.sin(X_gau)
y_gau = f_gau + RNG.normal(0.0, 0.1, n_half)

X_train = np.concatenate([X_exp, X_gau])[:, None]
y_train = np.concatenate([y_exp, y_gau])

likelihoods_mixed = (
    [ExponentialNoiseLikelihood(rate=3.0)] * n_half
    + [GaussianLikelihood(variance=0.1 ** 2)] * n_half
)

X_test = np.linspace(0.0, 2.0 * np.pi, 400)[:, None]
f_test_true = np.sin(X_test[:, 0])

print("=" * 65)
print("Demo: Mixed per-point likelihoods — f(x) = sin(x)  [Matern52 + Nugget]")
print("=" * 65)
print(f"Left  half (x ∈ [0, π]):    Exponential noise, rate=3  (y ≥ f)")
print(f"Right half (x ∈ [π, 2π]):  Gaussian noise,    σ=0.1")
print(f"n_train={len(y_train)},  n_test={len(X_test)}")


def make_model(default_lik):
    struct_cov = Matern52Covariance(sill=1.0, range_a=1.5)
    nug_cov    = NuggetCovariance(sill=0.01)
    cov = SumCovariance(struct_cov, nug_cov)
    return VRK(cov, default_lik, max_active=20, n_sweeps=3)


def fit_and_report(label, likelihood, per_point_liks=None):
    model = make_model(likelihood)
    if per_point_liks is not None:
        model.fit(X_train, y_train, likelihoods=per_point_liks)
    else:
        model.fit(X_train, y_train)
    mean, var = model.predict(X_test)
    rmse = np.sqrt(np.mean((mean - f_test_true) ** 2))
    bias = float(np.mean(mean - f_test_true))
    left  = X_test[:, 0] <= np.pi
    right = ~left
    rmse_l = np.sqrt(np.mean((mean[left]  - f_test_true[left])  ** 2))
    rmse_r = np.sqrt(np.mean((mean[right] - f_test_true[right]) ** 2))
    print(f"\n  [{label}]")
    print(f"    n_active={model.n_active}  "
          f"RMSE={rmse:.4f}  bias={bias:+.4f}  "
          f"(left={rmse_l:.4f}, right={rmse_r:.4f})")
    return mean, var, model


print("\n--- All-Exponential likelihood ---")
mean_ae, var_ae, model_ae = fit_and_report(
    "All-Exp",   ExponentialNoiseLikelihood(rate=3.0))

print("\n--- All-Gaussian likelihood ---")
mean_ag, var_ag, model_ag = fit_and_report(
    "All-Gauss", GaussianLikelihood(variance=0.1 ** 2))

print("\n--- Mixed (correct per-point) likelihoods ---")
mean_mx, var_mx, model_mx = fit_and_report(
    "Mixed",     GaussianLikelihood(variance=0.1 ** 2),
    per_point_liks=likelihoods_mixed)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle(
    "VRK — Mixed per-point likelihoods (Matern52 + Nugget):  f(x) = sin(x)",
    fontsize=12,
)

x_te     = X_test[:, 0]
boundary = np.pi

configs = [
    ("All-Exponential",  mean_ae, var_ae, model_ae, "#e74c3c"),
    ("All-Gaussian",     mean_ag, var_ag, model_ag, "#3498db"),
    ("Mixed (correct)",  mean_mx, var_mx, model_mx, "#2ecc71"),
]

for ax, (title, mean, var, model, color) in zip(axes, configs):
    std = np.sqrt(np.maximum(var, 0.0))

    ax.axvspan(0.0,      boundary,     color="#e74c3c", alpha=0.06)
    ax.axvspan(boundary, 2.0 * np.pi, color="#3498db", alpha=0.06)
    ax.axvline(boundary, color="gray", lw=0.8, ls=":", zorder=2)

    ax.scatter(X_exp, y_exp, s=16, c="#c0392b", alpha=0.55, zorder=5,
               label="Exp. obs.")
    ax.scatter(X_gau, y_gau, s=16, c="#2980b9", alpha=0.55, zorder=5,
               label="Gauss. obs.")
    ax.plot(x_te, f_test_true, "k--", lw=1.4, label="True f(x)", zorder=4)
    ax.plot(x_te, mean, color=color, lw=2.0, label="VRK mean", zorder=3)
    ax.fill_between(x_te, mean - 2 * std, mean + 2 * std,
                    color=color, alpha=0.18, label="±2σ", zorder=2)

    if model.n_active > 0:
        mean_at_active = np.interp(model.active_set_[:, 0], x_te, mean)
        ax.scatter(model.active_set_[:, 0], mean_at_active,
                   s=80, facecolors="none", edgecolors=color, lw=1.5,
                   zorder=6, label=f"Basis ({model.n_active})")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_xlim(x_te[0], x_te[-1])
    ax.legend(fontsize=7, loc="upper right")

axes[0].set_ylabel("f(x) / y")
plt.tight_layout()
plt.savefig("vrk_demo_mixed_likelihood_fixed.png", dpi=120)
print("\nSaved vrk_demo_mixed_likelihood_fixed.png")
