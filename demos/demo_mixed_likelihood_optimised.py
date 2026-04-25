"""
Demo: Mixed per-point likelihoods — OPTIMISED covariance hyperparameters.

True latent function: f(x) = sin(x)

Mirrors the MATLAB ogphyplearn / demogp_reg approach:
  for iHyp in 1..n_rounds:
      model.fit()                  →  EP sweeps (ogpreset + ogptrain)
      optimise_hyperparameters()   →  L-BFGS-B on active_set_log_evidence

Evidence objective: active_set_log_evidence (MATLAB ogpevid formula).
Works for ALL likelihood types, including Gaussian.

Covariance: Matern52 + NuggetCovariance (sill, range_a, nugget_sill).
Three parameters in log-space.  Likelihood parameters are fixed.

Data generation:
  Left  half  x ∈ [0, π]:   one-sided exponential noise, rate=3  (y ≥ f)
  Right half  x ∈ [π, 2π]:  symmetric Gaussian noise, σ=0.1

Three models compared (each with optimised covariance):
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
from vrk.optimization.hyperparameters import optimise_hyperparameters

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
print("Demo: Mixed per-point likelihoods — optimised covariance  [Matern52 + Nugget]")
print("=" * 65)
print(f"Left  half (x ∈ [0, π]):    Exponential noise, rate=3  (y ≥ f)")
print(f"Right half (x ∈ [π, 2π]):  Gaussian noise,    σ=0.1")
print(f"n_train={len(y_train)},  n_test={len(X_test)}")
print(f"\nEvidence: active_set_log_evidence  (MATLAB ogpevid formula)")
print(f"Covariance: Matern52(sill, range_a) + Nugget(sill) — 3 params in log-space")
print(f"Initial: sill=1.0, range_a=1.5, nugget_sill=0.01")


def make_model(default_lik):
    struct_cov = Matern52Covariance(sill=1.0, range_a=1.5)
    nug_cov    = NuggetCovariance(sill=0.01)
    cov = SumCovariance(struct_cov, nug_cov)
    return VRK(cov, default_lik, max_active=20, n_sweeps=3)


def cov_summary(model):
    p = model.covariance.get_params()  # [sill, range_a, nugget_sill]
    return f"sill={p[0]:.3f}  range_a={p[1]:.3f}  nugget={p[2]:.4f}"


def fit_and_optimise(label, likelihood, per_point_liks=None, n_rounds=4):
    """Alternating EP training + covariance optimisation."""
    model = make_model(likelihood)

    print(f"\n  [{label}]")
    print(f"    Init:   {cov_summary(model)}")

    ev_history = []

    for rnd in range(n_rounds):
        if per_point_liks is not None:
            model.fit(X_train, y_train, likelihoods=per_point_liks)
        else:
            model.fit(X_train, y_train)
        ev_history.append(("T", model.active_set_log_evidence()))

        ev_opt = optimise_hyperparameters(
            model, X_train, y_train,
            n_restarts=1, method="L-BFGS-B",
            evidence="active_set",
        )
        ev_history.append(("O", ev_opt))

    mean, var = model.predict(X_test)
    rmse  = np.sqrt(np.mean((mean - f_test_true) ** 2))
    bias  = float(np.mean(mean - f_test_true))
    left  = X_test[:, 0] <= np.pi
    right = ~left
    rmse_l = np.sqrt(np.mean((mean[left]  - f_test_true[left])  ** 2))
    rmse_r = np.sqrt(np.mean((mean[right] - f_test_true[right]) ** 2))

    print(f"    Final:  {cov_summary(model)}")
    print(f"    n_active={model.n_active}  RMSE={rmse:.4f}  bias={bias:+.4f}  "
          f"(left={rmse_l:.4f}, right={rmse_r:.4f})")
    trace = "  ".join(f"{tag}:{ev:.1f}" for tag, ev in ev_history)
    print(f"    Trace:  {trace}")

    return mean, var, model, ev_history


print("\n--- All-Exponential likelihood ---")
mean_ae, var_ae, model_ae, ev_ae = fit_and_optimise(
    "All-Exp",   ExponentialNoiseLikelihood(rate=3.0))

print("\n--- All-Gaussian likelihood ---")
mean_ag, var_ag, model_ag, ev_ag = fit_and_optimise(
    "All-Gauss", GaussianLikelihood(variance=0.1 ** 2))

print("\n--- Mixed (correct per-point) likelihoods ---")
mean_mx, var_mx, model_mx, ev_mx = fit_and_optimise(
    "Mixed",     GaussianLikelihood(variance=0.1 ** 2),
    per_point_liks=likelihoods_mixed)

# ── ASCII sanity check ─────────────────────────────────────────────────────────
print("\n--- ASCII comparison at 60 test pts ---")
print("  (O=close to f, +=above f, -=below f)")
idx = np.round(np.linspace(0, len(X_test) - 1, 60)).astype(int)
for label, mean in [
    ("All-Exp ", mean_ae),
    ("All-Gaus", mean_ag),
    ("Mixed   ", mean_mx),
]:
    row = "".join(
        "O" if abs(mean[i] - f_test_true[i]) < 0.15 else
        ("+" if mean[i] > f_test_true[i] else "-")
        for i in idx
    )
    print(f"  {label}: {row}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
fig.suptitle(
    "VRK — Mixed per-point likelihoods (optimised Matern52 + Nugget):  f(x) = sin(x)",
    fontsize=12,
)

x_te     = X_test[:, 0]
boundary = np.pi

configs = [
    ("All-Exponential", mean_ae, var_ae, model_ae, ev_ae, "#e74c3c"),
    ("All-Gaussian",    mean_ag, var_ag, model_ag, ev_ag, "#3498db"),
    ("Mixed (correct)", mean_mx, var_mx, model_mx, ev_mx, "#2ecc71"),
]

for col, (title, mean, var, model, ev_hist, color) in enumerate(configs):
    ax_pred = axes[0, col]
    ax_ev   = axes[1, col]
    std    = np.sqrt(np.maximum(var, 0.0))
    params = model.covariance.get_params()

    ax_pred.axvspan(0.0,      boundary,     color="#e74c3c", alpha=0.06)
    ax_pred.axvspan(boundary, 2.0 * np.pi,  color="#3498db", alpha=0.06)
    ax_pred.axvline(boundary, color="gray", lw=0.8, ls=":", zorder=2)
    ax_pred.scatter(X_exp, y_exp, s=16, c="#c0392b", alpha=0.55, zorder=5,
                    label="Exp. obs.")
    ax_pred.scatter(X_gau, y_gau, s=16, c="#2980b9", alpha=0.55, zorder=5,
                    label="Gauss. obs.")
    ax_pred.plot(x_te, f_test_true, "k--", lw=1.4, label="True f(x)", zorder=4)
    ax_pred.plot(x_te, mean, color=color, lw=2.0, label="VRK mean", zorder=3)
    ax_pred.fill_between(x_te, mean - 2 * std, mean + 2 * std,
                         color=color, alpha=0.18, zorder=2)
    if model.n_active > 0:
        mean_at_active = np.interp(model.active_set_[:, 0], x_te, mean)
        ax_pred.scatter(model.active_set_[:, 0], mean_at_active,
                        s=80, facecolors="none", edgecolors=color, lw=1.5,
                        zorder=6, label=f"Basis ({model.n_active})")
    ax_pred.set_title(
        f"{title}\na={params[1]:.2f}  sill={params[0]:.2f}  nug={params[2]:.4f}",
        fontsize=9,
    )
    ax_pred.set_xlabel("x")
    ax_pred.set_xlim(x_te[0], x_te[-1])
    ax_pred.legend(fontsize=7, loc="upper right")
    if col == 0:
        ax_pred.set_ylabel("f(x) / y")

    tags  = [t for t, _ in ev_hist]
    evs   = [e for _, e in ev_hist]
    steps = np.arange(1, len(evs) + 1)
    ax_ev.plot(steps, evs, "o-", color=color, lw=1.5, ms=5)
    for i, tag in enumerate(tags):
        if tag == "T":
            ax_ev.axvspan(steps[i] - 0.4, steps[i] + 0.4,
                          color="gray", alpha=0.10)
    ax_ev.set_xticks(steps)
    ax_ev.set_xticklabels(tags, fontsize=8)
    ax_ev.set_xlabel("T=train  O=optimise")
    ax_ev.set_ylabel("active_set_log_evidence")
    ax_ev.set_title("Evidence trace (↑ better)", fontsize=9)
    ax_ev.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("vrk_demo_mixed_likelihood_optimised.png", dpi=120)
print("\nSaved vrk_demo_mixed_likelihood_optimised.png")
