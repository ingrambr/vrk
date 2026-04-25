"""
Demo: Poisson likelihood — count data regression.

Models count observations y_i ~ Poisson(λ(x_i)) where
λ(x) = exp(f(x)) and f(x) is the latent GP.

True latent: f(x) = sin(x) + 1   → intensity λ(x) = exp(sin(x)+1)
             giving λ in [exp(0), exp(2)] ≈ [1, 7.4]

Uses Matern52Covariance + NuggetCovariance.  The Matérn 5/2 is a good
default for count data over a continuous domain: it is smoother than the
exponential (Matérn 1/2) but less restrictively smooth than the Gaussian,
making it well-suited to the log-intensity of ecological and traffic data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, Matern52Covariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood, PoissonLikelihood)

RNG = np.random.default_rng(5)

# ── Data ───────────────────────────────────────────────────────────────────────
n = 60
X_train  = np.sort(RNG.uniform(0, 2 * np.pi, n))[:, None]
f_true   = np.sin(X_train[:, 0]) + 1.0
lam_true = np.exp(f_true)
y_train  = RNG.poisson(lam_true).astype(float)

X_test   = np.linspace(0, 2 * np.pi, 150)[:, None]
f_test   = np.sin(X_test[:, 0]) + 1.0
lam_test = np.exp(f_test)

print("=" * 60)
print("Demo: Poisson likelihood — count data regression  [Matern52 + Nugget]")
print("=" * 60)
print(f"n_train={n}")
print(f"Observed counts range: {int(y_train.min())} – {int(y_train.max())}")
print(f"True intensity λ range: [{lam_true.min():.1f}, {lam_true.max():.1f}]")

# Small structural nugget; Poisson likelihood handles overdispersion
nugget_var = 1e-4


def make_model(bin_size=1.0):
    struct_cov = Matern52Covariance(sill=1.0, range_a=1.0)
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    lik = PoissonLikelihood(bin_size=bin_size)
    return VRK(cov, lik, max_active=12, n_sweeps=3)


def fit_and_eval(label, bin_size=1.0):
    model = make_model(bin_size)
    model.fit(X_train, y_train)
    mean_f, var_f = model.predict(X_test)
    lam_pred = np.exp(mean_f)
    rmse_lam = np.sqrt(np.mean((lam_pred - lam_test) ** 2))
    rmse_f   = np.sqrt(np.mean((mean_f - f_test) ** 2))
    print(f"\n  [{label}]  n_active={model.n_active}")
    print(f"    RMSE(λ)={rmse_lam:.4f}  RMSE(f)={rmse_f:.4f}")
    return mean_f, var_f, lam_pred, model


print("\n--- Poisson likelihood (bin_size=1) ---")
mean_p,  var_p,  lam_p,  model_p  = fit_and_eval("Poisson bin=1")

print("\n--- Poisson likelihood (bin_size=2, longer exposure) ---")
mean_p2, var_p2, lam_p2, model_p2 = fit_and_eval("Poisson bin=2", bin_size=2.0)

# ── ASCII intensity ────────────────────────────────────────────────────────────
print("\n--- ASCII intensity (50 test pts, normalised to 5 levels) ---")
idx   = np.round(np.linspace(0, 149, 50)).astype(int)
chars = " .:+#@"

def to_ascii(vals, vmin, vmax):
    return "".join(
        chars[min(5, int((v - vmin) / (vmax - vmin + 1e-9) * 5))]
        for v in vals[idx]
    )

vmin = min(lam_p.min(), lam_test[idx].min())
vmax = max(lam_p.max(), lam_test[idx].max())
print(f"Predicted λ: [{to_ascii(lam_p, vmin, vmax)}]")
print(f"True λ:      [{to_ascii(lam_test, vmin, vmax)}]")
print(f"(legend: {chars!r} = low→high intensity)")

# ── Plot ───────────────────────────────────────────────────────────────────────
x_tr   = X_train[:, 0]
x_te   = X_test[:, 0]
std_f  = np.sqrt(np.maximum(var_p, 0))
lam_lo = np.exp(mean_p - 2 * std_f)
lam_hi = np.exp(mean_p + 2 * std_f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(
    "VRK — Poisson likelihood: count data regression  (Matern52 + Nugget)",
    fontsize=12,
)

ax1.bar(x_tr, y_train, width=0.08, color="#95a5a6", alpha=0.6, label="Observed counts y")
ax1.plot(x_te, lam_test, "k--", lw=1.5, label="True λ(x)")
ax1.plot(x_te, lam_p, "#e74c3c", lw=2, label="Predicted λ (bin=1)")
ax1.fill_between(x_te, lam_lo, lam_hi, color="#e74c3c", alpha=0.18, label="±2σ band")
ax1.set_xlabel("x")
ax1.set_ylabel("λ(x)  /  count")
ax1.set_title("Intensity λ = exp(f)")
ax1.legend(fontsize=8)
ax1.set_xlim(x_te[0], x_te[-1])

std_f2 = np.sqrt(np.maximum(var_p2, 0))
ax2.plot(x_te, f_test, "k--", lw=1.5, label="True f(x)")
ax2.scatter(x_tr, np.log(y_train + 0.5), s=16, c="k", alpha=0.35, label="log(y+0.5)")
ax2.plot(x_te, mean_p, "#e74c3c", lw=2, label="VRK mean (bin=1)")
ax2.fill_between(x_te, mean_p - 2 * std_f, mean_p + 2 * std_f,
                 color="#e74c3c", alpha=0.18, label="±2σ")
ax2.plot(x_te, mean_p2, "#3498db", lw=1.5, ls="--", label="VRK mean (bin=2)")
ax2.fill_between(x_te, mean_p2 - 2 * std_f2, mean_p2 + 2 * std_f2,
                 color="#3498db", alpha=0.12)
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")
ax2.set_title("Latent function space")
ax2.legend(fontsize=8)
ax2.set_xlim(x_te[0], x_te[-1])

plt.tight_layout()
plt.savefig("vrk_demo_poisson.png", dpi=150)
print("\nSaved vrk_demo_poisson.png")
