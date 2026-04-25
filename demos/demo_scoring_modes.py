"""
Demo: Active-set scoring mode comparison.

Fits sin(x) with max_active=10 using all 5 scoring modes:
  - geometric
  - mean_component
  - full_kl          (default)
  - entropy_reduction
  - loo_score

Uses SphericalCovariance + NuggetCovariance.  The spherical model is widely
used in classical geostatistics and has a finite effective range, making it
an interesting choice to contrast with the infinitely-supported kernels.

Shows:
  - Which training points are selected as active
  - Prediction quality (RMSE on test set)
  - Active set visualised with different markers per mode
  - Summary table: mode, active size, RMSE, active-set log evidence
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, SphericalCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood)

RNG = np.random.default_rng(0)

noise_std  = 0.05
nugget_var = noise_std ** 2

n_train = 50
X_train = np.linspace(0, 2 * np.pi, n_train)[:, None]
y_train = np.sin(X_train[:, 0]) + noise_std * RNG.standard_normal(n_train)

n_test = 200
X_test  = np.linspace(0, 2 * np.pi, n_test)[:, None]
y_test  = np.sin(X_test[:, 0])

MAX_ACTIVE = 10
N_SWEEPS   = 3

MODES   = ["geometric", "mean_component", "full_kl", "entropy_reduction", "loo_score"]
COLORS  = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
MARKERS = ["o", "s", "^", "D", "v"]

print("=" * 70)
print(f"Demo: Active-set scoring mode comparison (max_active={MAX_ACTIVE})")
print(f"      Covariance: SphericalCovariance + NuggetCovariance")
print("=" * 70)

results = {}
for mode in MODES:
    struct_cov = SphericalCovariance(sill=1.0, range_a=3.0)
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    lik = GaussianLikelihood(variance=1e-6)
    model = VRK(cov, lik, max_active=MAX_ACTIVE, n_sweeps=N_SWEEPS, scoring=mode)
    model.fit(X_train, y_train)

    mean, var = model.predict(X_test)
    rmse = np.sqrt(np.mean((mean - y_test) ** 2))
    ase  = model.active_set_log_evidence()
    results[mode] = {
        "model": model, "mean": mean, "var": var,
        "rmse": rmse, "ase": ase,
        "active_x": model.active_set_[:, 0].tolist() if model.n_active > 0 else [],
    }

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'Mode':<20}  {'n_active':>8}  {'RMSE':>8}  {'ASE':>9}")
print("  " + "-" * 50)
for mode in MODES:
    r = results[mode]
    n_act = r["model"].n_active
    print(f"{mode:<20}  {n_act:8d}  {r['rmse']:8.5f}  {r['ase']:9.4f}")

best_mode = min(MODES, key=lambda m: results[m]["rmse"])
print(f"\nBest RMSE: {best_mode} ({results[best_mode]['rmse']:.5f})")

print("\nActive x-locations per mode:")
for mode in MODES:
    xs = [f"{x:.2f}" for x in sorted(results[mode]["active_x"])]
    print(f"  {mode:<20}: {', '.join(xs)}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

x = X_test[:, 0]

for idx, (mode, color, marker) in enumerate(zip(MODES, COLORS, MARKERS)):
    ax = axes[idx]
    r  = results[mode]
    model = r["model"]
    mean  = r["mean"]
    std   = np.sqrt(np.maximum(r["var"], 0))

    ax.plot(x, y_test, "k--", lw=1.5, alpha=0.6, label="True sin(x)")
    ax.scatter(X_train[:, 0], y_train, s=12, c="gray", alpha=0.4, zorder=2)
    ax.plot(x, mean, color=color, lw=2, label="Prediction")
    ax.fill_between(x, mean - 2 * std, mean + 2 * std, color=color, alpha=0.15)

    if model.n_active > 0:
        ax.scatter(
            model.active_set_[:, 0],
            np.full(model.n_active, 1.2),
            marker=marker, s=80, color=color, zorder=5, label="Active set",
        )

    ax.set_title(
        f"{mode}\nRMSE={r['rmse']:.4f}, ASE={r['ase']:.2f}",
        fontsize=10,
        fontweight="bold" if mode == best_mode else "normal",
    )
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

# Summary bar chart in last panel
ax_bar = axes[5]
rmses = [results[m]["rmse"] for m in MODES]
bars = ax_bar.bar(range(len(MODES)), rmses, color=COLORS, alpha=0.85)
ax_bar.set_xticks(range(len(MODES)))
ax_bar.set_xticklabels([m.replace("_", "\n") for m in MODES], fontsize=8)
ax_bar.set_ylabel("RMSE")
ax_bar.set_title("RMSE comparison\n(lower is better)")
ax_bar.grid(True, axis="y", alpha=0.3)
best_idx = MODES.index(best_mode)
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(2)

plt.suptitle(
    f"Active-set scoring modes — SphericalCovariance + Nugget  "
    f"(n={n_train}, max_active={MAX_ACTIVE})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("vrk_demo_scoring_modes.png", dpi=120)
print("\nSaved vrk_demo_scoring_modes.png")
