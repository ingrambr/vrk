"""
Demo: KL divergence analysis.

Shows three views of KL divergence for a GP fit to sin(x),
using SphericalCovariance + NuggetCovariance.

The spherical covariance has a finite range and is the most widely used model
in classical geostatistics.  Here we study:

  1. KL(q || prior) vs number of EP sweeps (convergence monitoring)
  2. KL between models with different range_a values (model distance)
  3. KL from prior vs max_active (how much information is captured)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, SphericalCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood)

RNG = np.random.default_rng(42)

noise_std  = 0.05
nugget_var = noise_std ** 2

n = 40
X = np.linspace(0, 2 * np.pi, n)[:, None]
y = np.sin(X[:, 0]) + noise_std * RNG.standard_normal(n)

print("=" * 60)
print("Demo: KL divergence analysis  (SphericalCovariance + Nugget)")
print("=" * 60)


def make_model(range_a, max_active=15, n_sweeps=3):
    struct_cov = SphericalCovariance(sill=1.0, range_a=range_a)
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    lik = GaussianLikelihood(variance=1e-6)
    return VRK(cov, lik, max_active=max_active, n_sweeps=n_sweeps)


# ── 1. KL(q || prior) vs EP sweeps ────────────────────────────────────────────
print("\n[1] KL from prior vs EP sweeps (range_a=3.0)")
kl_vs_sweeps = []
for s in range(1, 9):
    model = make_model(range_a=3.0, n_sweeps=s)
    model.fit(X, y)
    kl   = model.kl_from_prior()
    elbo = model.elbo()
    kl_vs_sweeps.append((s, kl, elbo))
    print(f"  n_sweeps={s}: KL={kl:.4f}, ELBO={elbo:.4f}, n_active={model.n_active}")

# ── 2. KL between models with different range_a ───────────────────────────────
print("\n[2] KL between models (range_a=1.0 vs varying)")
model_ref = make_model(range_a=1.0)
model_ref.fit(X, y)

log_range_grid = np.linspace(-1.0, 1.5, 20)  # log(range_a) sweep
kl_model_dist  = []
for log_r in log_range_grid:
    model2 = make_model(range_a=np.exp(log_r))
    model2.fit(X, y)
    kl = model_ref.kl_to(model2)
    kl_model_dist.append(kl)

kl_model_dist = np.array(kl_model_dist)
kl_self = model_ref.kl_to(model_ref)
print(f"  KL(ref || ref) = {kl_self:.2e} (should be ~0)")
print(f"  KL range: [{kl_model_dist.min():.3f}, {kl_model_dist.max():.3f}]")

# ── 3. KL from prior vs max_active ────────────────────────────────────────────
print("\n[3] KL from prior vs max_active (range_a=3.0)")
max_actives  = [2, 4, 6, 8, 10, 12, 15, 20, 25]
kl_vs_active = []
for ma in max_actives:
    model = make_model(range_a=3.0, max_active=ma)
    model.fit(X, y)
    kl = model.kl_from_prior()
    kl_vs_active.append((ma, model.n_active, kl))
    print(f"  max_active={ma:3d}, n_active={model.n_active:3d}, KL={kl:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

sweeps_arr, kl_arr, elbo_arr = zip(*kl_vs_sweeps)
ax1 = axes[0]
ax1.plot(sweeps_arr, kl_arr, "o-", color="#e74c3c", lw=2, label="KL(q || prior)")
ax1_r = ax1.twinx()
ax1_r.plot(sweeps_arr, elbo_arr, "s--", color="#3498db", lw=2, label="ELBO")
ax1.set_xlabel("Number of EP sweeps")
ax1.set_ylabel("KL(q || prior)", color="#e74c3c")
ax1_r.set_ylabel("ELBO", color="#3498db")
ax1.set_title("KL vs EP sweeps")
ax1.tick_params(axis="y", labelcolor="#e74c3c")
ax1_r.tick_params(axis="y", labelcolor="#3498db")
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9)

axes[1].plot(log_range_grid, kl_model_dist, "o-", color="#2ecc71", lw=2)
axes[1].axvline(np.log(1.0), color="gray", ls="--", lw=1.5, label="Reference (a=1.0)")
axes[1].set_xlabel("log range_a of comparison model")
axes[1].set_ylabel("KL(q_ref || q_comparison)")
axes[1].set_title("KL between models")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

ma_arr, na_arr, kl_ma_arr = zip(*kl_vs_active)
axes[2].plot(na_arr, kl_ma_arr, "o-", color="#9b59b6", lw=2)
axes[2].set_xlabel("Actual active set size")
axes[2].set_ylabel("KL(q || prior)")
axes[2].set_title("KL vs active set size\n(how much info is captured)")
axes[2].grid(True, alpha=0.3)

plt.suptitle(
    "KL divergence analysis — SphericalCovariance + Nugget",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("vrk_demo_kl_divergence.png", dpi=120)
print("\nSaved vrk_demo_kl_divergence.png")
