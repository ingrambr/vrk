"""
Demo: Student-t likelihood — robustness to outliers.

Compares Gaussian and Student-t (heavy-tailed) EP likelihoods on data
that contains several gross outliers.

True function: f(x) = sin(x)
Training data: mostly f(x) + N(0, 0.1²), but 10% are large outliers.

Uses ExponentialCovariance + NuggetCovariance.  The exponential covariance
(Matérn 1/2) is appropriate here because physical phenomena with outliers
often also display rough (non-differentiable) behaviour.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, ExponentialCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood, StudentTLikelihood)

RNG = np.random.default_rng(3)

noise_std  = 0.1
nugget_var = noise_std ** 2

# ── Data with outliers ─────────────────────────────────────────────────────────
n = 50
X_train  = np.sort(RNG.uniform(0, 2 * np.pi, n))[:, None]
f_true   = np.sin(X_train[:, 0])
y_clean  = f_true + noise_std * RNG.standard_normal(n)
n_outliers   = max(1, n // 10)
outlier_idx  = RNG.choice(n, n_outliers, replace=False)
y_train = y_clean.copy()
y_train[outlier_idx] += RNG.choice([-3.0, 3.0], n_outliers)

X_test = np.linspace(0, 2 * np.pi, 150)[:, None]
y_true = np.sin(X_test[:, 0])

print("=" * 60)
print("Demo: Student-t likelihood — robust regression with outliers")
print(f"      Covariance: ExponentialCovariance + NuggetCovariance")
print("=" * 60)
print(f"n_train={n}, n_outliers={n_outliers}, outlier magnitude ≈ ±3")


def make_model(default_lik):
    struct_cov = ExponentialCovariance(sill=1.0, range_a=1.0)
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    return VRK(cov, default_lik, max_active=12, n_sweeps=3)


def fit_eval(label, lik):
    model = make_model(lik)
    model.fit(X_train, y_train)
    mean, var = model.predict(X_test)
    rmse = np.sqrt(np.mean((mean - y_true) ** 2))
    print(f"\n  [{label}]  n_active={model.n_active}  RMSE={rmse:.4f}")
    return mean, var, model


print("\n--- Gaussian likelihood (sensitive to outliers) ---")
mean_g,  var_g,  model_g  = fit_eval("Gaussian  σ²=0.01", GaussianLikelihood(variance=0.01))

print("\n--- Gaussian likelihood (inflated variance for robustness) ---")
mean_g2, var_g2, model_g2 = fit_eval("Gaussian  σ²=0.5 ", GaussianLikelihood(variance=0.5))

print("\n--- Student-t  ν=4 (robust heavy-tailed) ---")
mean_t4, var_t4, model_t4 = fit_eval("Student-t ν=4  ", StudentTLikelihood(nu=4.0, sigma=0.2))

print("\n--- Student-t  ν=10 (moderate heavy tail) ---")
mean_t10, var_t10, _ = fit_eval("Student-t ν=10 ", StudentTLikelihood(nu=10.0, sigma=0.2))

# ── ASCII residuals ────────────────────────────────────────────────────────────
print("\n--- ASCII residuals (|pred - true| < 0.15 → '.' else '|') ---")
idx = np.round(np.linspace(0, 149, 50)).astype(int)


def residual_row(mean, label):
    row = "".join(
        "." if abs(mean[i] - y_true[i]) < 0.15 else "|"
        for i in idx
    )
    good = row.count(".")
    print(f"  {label:20s}: {row}  ({good}/{len(idx)} within 0.15)")


residual_row(mean_g,   "Gaussian σ²=0.01")
residual_row(mean_g2,  "Gaussian σ²=0.5 ")
residual_row(mean_t4,  "Student-t ν=4   ")
residual_row(mean_t10, "Student-t ν=10  ")

print(f"\nOutlier positions (indices): {sorted(outlier_idx.tolist())}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
fig.suptitle(
    "VRK — Student-t likelihood: robust regression  (ExponentialCovariance + Nugget)",
    fontsize=12,
)

x_te = X_test[:, 0]
x_tr = X_train[:, 0]
inlier_mask = np.ones(len(X_train), dtype=bool)
inlier_mask[outlier_idx] = False


def _plot_model(ax, mean, var, color, label):
    std = np.sqrt(np.maximum(var, 0))
    ax.plot(x_te, mean, color=color, lw=1.8, label=label)
    ax.fill_between(x_te, mean - 2 * std, mean + 2 * std, color=color, alpha=0.14)


for ax in (ax1, ax2):
    ax.plot(x_te, y_true, "k--", lw=1.5, label="True sin(x)", zorder=5)
    ax.scatter(x_tr[inlier_mask],  y_train[inlier_mask],  s=18, c="k",
               alpha=0.4, zorder=4, label="Inliers")
    ax.scatter(x_tr[~inlier_mask], y_train[~inlier_mask], s=60, c="red",
               alpha=0.8, marker="x", lw=2, zorder=4, label="Outliers")
    ax.set_xlabel("x")
    ax.set_xlim(x_te[0], x_te[-1])

_plot_model(ax1, mean_g,  var_g,  "#3498db", "Gaussian σ²=0.01")
_plot_model(ax1, mean_g2, var_g2, "#2ecc71", "Gaussian σ²=0.5")
ax1.set_title("Gaussian likelihood")
ax1.set_ylabel("f(x)")
ax1.legend(fontsize=8)

_plot_model(ax2, mean_t4,  var_t4,  "#e74c3c", "Student-t ν=4")
_plot_model(ax2, mean_t10, var_t10, "#9b59b6", "Student-t ν=10")
ax2.set_title("Student-t likelihood (robust)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("vrk_demo_student_t.png", dpi=150)
print("\nSaved vrk_demo_student_t.png")
