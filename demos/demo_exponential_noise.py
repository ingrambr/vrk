"""
Demo: Exponential / one-sided noise likelihood.

Models non-negative data where observations are always >= the underlying
function (e.g., flood levels, precipitation maxima).

Setting: f(x) = 0.5 + sin(x),  y_i = f(x_i) + |N(0, σ²)|  (one-sided)
So observations are always above the latent function.

Uses Matern52Covariance + NuggetCovariance.  The Matérn 5/2 covariance is
a common choice for environmental/hydrological data: it is twice
differentiable (smoother than exponential) but not unrealistically smooth.

Compares:
  1. Gaussian likelihood (doesn't know about one-sidedness)
  2. Exponential noise likelihood (correctly models the asymmetry)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, Matern52Covariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood, ExponentialNoiseLikelihood)

RNG = np.random.default_rng(0)

# ── Data: latent function + one-sided noise ────────────────────────────────────
n = 50
X_train = np.sort(RNG.uniform(0, 2 * np.pi, n))[:, None]
f_true  = 0.5 + np.sin(X_train[:, 0])
noise   = np.abs(RNG.normal(0, 0.3, n))
y_train = f_true + noise

X_test = np.linspace(0, 2 * np.pi, 500)[:, None]
f_test_true = 0.5 + np.sin(X_test[:, 0])

print("=" * 60)
print("Demo: Exponential (one-sided) noise likelihood  [Matern52 + Nugget]")
print("=" * 60)
print(f"Observations always >= latent function (noise = |N(0, 0.3²)|)")
print(f"y_min={y_train.min():.3f}  y_max={y_train.max():.3f}")
print(f"f_true range: [{f_true.min():.3f}, {f_true.max():.3f}]")


def make_model(default_lik):
    struct_cov = Matern52Covariance(sill=1.0, range_a=1.5)
    nug_cov    = NuggetCovariance(sill=0.01)
    cov = SumCovariance(struct_cov, nug_cov)
    return VRK(cov, default_lik, max_active=12, n_sweeps=3)


def fit_and_report(lik_name, lik):
    model = make_model(lik)
    model.fit(X_train, y_train)
    mean, var = model.predict(X_test)
    rmse      = np.sqrt(np.mean((mean - f_test_true) ** 2))
    mean_bias = np.mean(mean - f_test_true)
    print(f"\n  [{lik_name}]")
    print(f"    n_active={model.n_active}  RMSE={rmse:.4f}  bias={mean_bias:+.4f}")
    return mean, var, model


print("\n--- Gaussian likelihood (symmetric, ignores one-sidedness) ---")
mean_g, var_g, model_g = fit_and_report(
    "Gaussian", GaussianLikelihood(variance=0.1))

print("\n--- Exponential noise likelihood (asymmetric) ---")
mean_e, var_e, model_e = fit_and_report(
    "Exp.noise", ExponentialNoiseLikelihood(rate=3.0))

print("\n--- ASCII comparison at 40 test pts ---")
idx   = np.round(np.linspace(0, len(X_test) - 1, 40)).astype(int)
row_g = "".join(
    "G" if abs(mean_g[i] - f_test_true[i]) < 0.15 else
    ("+" if mean_g[i] > f_test_true[i] else "-")
    for i in idx
)
row_e = "".join(
    "E" if abs(mean_e[i] - f_test_true[i]) < 0.15 else
    ("+" if mean_e[i] > f_test_true[i] else "-")
    for i in idx
)
print(f"Gaussian: {row_g}  (G=close, +=above, -=below f_true)")
print(f"ExpNoise: {row_e}")

# EM step demo
print("\n--- EM hyperparameter update (exp. noise rate) ---")
struct_cov_em = Matern52Covariance(sill=1.0, range_a=1.5)
nug_cov_em    = NuggetCovariance(sill=0.01)
cov_em = SumCovariance(struct_cov_em, nug_cov_em)
lik_em = ExponentialNoiseLikelihood(rate=1.0)
model_em = VRK(cov_em, lik_em, max_active=12, n_sweeps=1)
model_em.fit(X_train, y_train)
mean_pred, var_pred = model_em.predict(X_train)
rate_before = lik_em.rate
lik_em.em_step(y_train, mean_pred, var_pred, momentum=0.5)
print(f"  Rate before EM: {rate_before:.4f}")
print(f"  Rate after EM:  {lik_em.rate:.4f}  (true≈3.33=1/0.3)")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
fig.suptitle(
    "VRK — Exponential (one-sided) noise likelihood  (Matern52 + Nugget)",
    fontsize=12,
)

x_tr = X_train[:, 0]
x_te = X_test[:, 0]
std_g = np.sqrt(np.maximum(var_g, 0))
std_e = np.sqrt(np.maximum(var_e, 0))

ax.scatter(x_tr, y_train, s=20, c="k", alpha=0.45, zorder=5,
           label="Observations (y ≥ f)")
ax.plot(x_te, f_test_true, "k--", lw=1.5, label="True latent f(x)", zorder=4)

ax.plot(x_te, mean_g, "#3498db", lw=2, label="Gaussian mean")
ax.fill_between(x_te, mean_g - 2 * std_g, mean_g + 2 * std_g,
                color="#3498db", alpha=0.15, label="Gaussian ±2σ")

ax.plot(x_te, mean_e, "#e74c3c", lw=2, label="ExpNoise mean")
ax.fill_between(x_te, mean_e - 2 * std_e, mean_e + 2 * std_e,
                color="#e74c3c", alpha=0.15, label="ExpNoise ±2σ")

if model_g.n_active > 0:
    mean_g_at_active = np.interp(model_g.active_set_[:, 0], x_te, mean_g)
    ax.scatter(model_g.active_set_[:, 0], mean_g_at_active,
               s=80, facecolors="none", edgecolors="#3498db", lw=1.5, zorder=6,
               label=f"Gaussian basis ({model_g.n_active})")
if model_e.n_active > 0:
    mean_e_at_active = np.interp(model_e.active_set_[:, 0], x_te, mean_e)
    ax.scatter(model_e.active_set_[:, 0], mean_e_at_active,
               s=80, facecolors="none", edgecolors="#e74c3c", lw=1.5, zorder=6,
               label=f"ExpNoise basis ({model_e.n_active})")

ax.set_xlabel("x")
ax.set_ylabel("f(x) / y")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(x_te[0], x_te[-1])
ax.set_title("Observations always ≥ latent f  (noise ~ |N(0, σ²)|)")

plt.tight_layout()
plt.savefig("vrk_demo_exponential_noise.png", dpi=150)
print("\nSaved vrk_demo_exponential_noise.png")
