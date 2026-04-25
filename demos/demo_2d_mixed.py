"""
2-D VRK demo — spatially varying mixed likelihoods.

The unit domain [0,1]² is split into four quadrants, each with a different
noise model and magnitude:

  ┌─────────────────┬─────────────────┐
  │  Top-left       │  Top-right      │
  │  Gaussian       │  Exponential    │
  │  σ²_n = 1.0    │  rate = 5.0     │
  │  (5× BL)        │  (mean = 0.20)  │
  ├─────────────────┼─────────────────┤
  │  Bottom-left    │  Bottom-right   │
  │  Gaussian       │  Exponential    │
  │  σ²_n = 0.2    │  rate = 1.67    │
  │  (base)         │  (3× TR noise)  │
  └─────────────────┴─────────────────┘

Left half  — Gaussian noise (symmetric, can go below f)
Right half — Exponential noise (one-sided, always y ≥ f)

Uses GaussianCovariance + NuggetCovariance.  The smooth Gaussian covariance
is well-suited to the smooth synthetic GP field used here.

True GP parameters:  sill σ² = 2.5,  range_a = 0.15
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from vrk import (VRK, GaussianCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood, ExponentialNoiseLikelihood)
from vrk.optimization.hyperparameters import optimise_hyperparameters

# ── noise parameters ───────────────────────────────────────────────────────────
TRUE_SILL     = 2.5
TRUE_RANGE    = 0.15
GAUSS_VAR_BL  = 0.2
GAUSS_VAR_TL  = GAUSS_VAR_BL * 5
EXP_RATE_TR   = 5.0
EXP_RATE_BR   = EXP_RATE_TR / 3.0

N_TRAIN    = 400
N_GRID     = 200
MAX_ACTIVE = 70
N_SWEEPS   = 3
RNG        = np.random.default_rng(7)


def rbf_cov(A, B, var, rng_a):
    diff    = A[:, None, :] - B[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    return var * np.exp(-0.5 * sq_dist / rng_a ** 2)


def quadrant_mask(X):
    x1, x2 = X[:, 0], X[:, 1]
    bl = (x1 < 0.5) & (x2 < 0.5)
    tl = (x1 < 0.5) & (x2 >= 0.5)
    tr = (x1 >= 0.5) & (x2 >= 0.5)
    br = (x1 >= 0.5) & (x2 < 0.5)
    return bl, tl, tr, br


def make_likelihoods(X):
    bl, tl, tr, br = quadrant_mask(X)
    liks = [None] * X.shape[0]
    for i in np.where(bl)[0]:
        liks[i] = GaussianLikelihood(variance=GAUSS_VAR_BL)
    for i in np.where(tl)[0]:
        liks[i] = GaussianLikelihood(variance=GAUSS_VAR_TL)
    for i in np.where(tr)[0]:
        liks[i] = ExponentialNoiseLikelihood(rate=EXP_RATE_TR)
    for i in np.where(br)[0]:
        liks[i] = ExponentialNoiseLikelihood(rate=EXP_RATE_BR)
    return liks


def add_noise(f_vals, X):
    y = f_vals.copy()
    bl, tl, tr, br = quadrant_mask(X)
    y[bl] += RNG.normal(0, np.sqrt(GAUSS_VAR_BL), bl.sum())
    y[tl] += RNG.normal(0, np.sqrt(GAUSS_VAR_TL), tl.sum())
    y[tr] += RNG.exponential(1.0 / EXP_RATE_TR,   tr.sum())
    y[br] += RNG.exponential(1.0 / EXP_RATE_BR,   br.sum())
    return y


def draw_quadrants(ax, lw=0.8):
    ax.axvline(0.5, color="k", lw=lw, ls="--", alpha=0.5)
    ax.axhline(0.5, color="k", lw=lw, ls="--", alpha=0.5)
    kw = dict(fontsize=6.5, alpha=0.75, va="center", ha="center")
    ax.text(0.25, 0.25, f"Gauss\nσ²={GAUSS_VAR_BL}",  transform=ax.transData,
            color="#2471a3", **kw)
    ax.text(0.25, 0.75, f"Gauss\nσ²={GAUSS_VAR_TL}",  transform=ax.transData,
            color="#2471a3", **kw)
    ax.text(0.75, 0.75, f"Exp\nr={EXP_RATE_TR:.1f}",   transform=ax.transData,
            color="#922b21", **kw)
    ax.text(0.75, 0.25, f"Exp\nr={EXP_RATE_BR:.2f}",   transform=ax.transData,
            color="#922b21", **kw)


# ── simulate training data ─────────────────────────────────────────────────────
X_train = RNG.uniform(0.0, 1.0, (N_TRAIN, 2))
K_nn    = rbf_cov(X_train, X_train, TRUE_SILL, TRUE_RANGE)
L       = np.linalg.cholesky(K_nn + 1e-8 * np.eye(N_TRAIN))
f_true  = L @ RNG.standard_normal(N_TRAIN)
y_train = add_noise(f_true, X_train)

# ── prediction grid ────────────────────────────────────────────────────────────
g       = np.linspace(0.0, 1.0, N_GRID)
gx, gy  = np.meshgrid(g, g)
X_grid  = np.column_stack([gx.ravel(), gy.ravel()])

# ── true GP field ──────────────────────────────────────────────────────────────
K_gn       = rbf_cov(X_grid, X_train, TRUE_SILL, TRUE_RANGE)
alpha_true = np.linalg.solve(K_nn + 1e-8 * np.eye(N_TRAIN), f_true)
f_grid     = K_gn @ alpha_true
f_true_img = f_grid.reshape(N_GRID, N_GRID)

y_noisy_grid = add_noise(f_grid, X_grid)
y_noisy_img  = y_noisy_grid.reshape(N_GRID, N_GRID)

# ── fit VRK with per-point likelihoods ────────────────────────────────────────
liks_train = make_likelihoods(X_train)
struct_cov = GaussianCovariance(sill=1.0, range_a=0.5)   # deliberately off-truth init
nug_cov    = NuggetCovariance(sill=0.1)
cov = SumCovariance(struct_cov, nug_cov)
model = VRK(cov, liks_train, max_active=MAX_ACTIVE, n_sweeps=N_SWEEPS)
model.fit(X_train, y_train)

print(f"\nVRK fitted  active={model.n_active}/{MAX_ACTIVE}")

# ── multi-round hyperparameter optimisation ────────────────────────────────────
N_ROUNDS       = 4
params_before  = model.covariance.get_params()
print(f"\nHyperparameter optimisation ({N_ROUNDS} rounds)")
print(f"  Initial: sill={params_before[0]:.3f}, range_a={params_before[1]:.3f}, "
      f"nugget={params_before[2]:.4f}")
print(f"  True:    sill={TRUE_SILL:.3f}, range_a={TRUE_RANGE:.3f}")

for rnd in range(1, N_ROUNDS + 1):
    model.fit(X_train, y_train)
    best_ev = optimise_hyperparameters(
        model, X_train, y_train, n_restarts=1, method="L-BFGS-B",
        evidence="active_set",
    )
    params = model.covariance.get_params()
    print(f"  Round {rnd}: sill={params[0]:.3f}, range_a={params[1]:.3f}, "
          f"nugget={params[2]:.4f}, evidence={best_ev:.2f}")

# ── predictions ───────────────────────────────────────────────────────────────
mu, var   = model.predict(X_grid)
std       = np.sqrt(np.maximum(var, 0.0))
mu_img    = mu.reshape(N_GRID, N_GRID)
std_img   = std.reshape(N_GRID, N_GRID)
err_img   = np.abs(mu_img - f_true_img)

grid_rmse = float(np.sqrt(np.mean((mu - f_grid) ** 2)))
print(f"\nGrid RMSE (pred vs true): {grid_rmse:.4f}")

# ── colour scales ──────────────────────────────────────────────────────────────
all_vals   = np.concatenate([f_grid, y_noisy_grid, y_train, mu])
vmin, vmax = all_vals.min(), all_vals.max()
norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap       = "RdYlBu_r"


def add_cbar(ax, mappable):
    plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)


# ── 2×3 figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.35})

ax = axes[0, 0]
im = ax.imshow(f_true_img, origin="lower", extent=[0, 1, 0, 1],
               cmap=cmap, norm=norm, aspect="equal", interpolation="bilinear")
draw_quadrants(ax)
ax.set_title("True GP field (noiseless)", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im)

ax = axes[0, 1]
im_n = ax.imshow(y_noisy_img, origin="lower", extent=[0, 1, 0, 1],
                 cmap=cmap, norm=norm, aspect="equal", interpolation="bilinear")
draw_quadrants(ax)
ax.set_title("Simulated data + noise (dense grid)", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im_n)

ax = axes[0, 2]
bl, tl, tr, br = quadrant_mask(X_train)
for mask, marker in [(bl, "o"), (tl, "o"), (tr, "^"), (br, "^")]:
    ax.scatter(X_train[mask, 0], X_train[mask, 1],
               c=y_train[mask], cmap=cmap, norm=norm,
               s=20, marker=marker, edgecolors="k", linewidths=0.25, zorder=3)
if model.n_active > 0:
    active = model.active_set_
    ax.scatter(active[:, 0], active[:, 1],
               s=90, facecolors="none", edgecolors="k",
               linewidths=1.2, zorder=5)
ax.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
patches = [
    mpatches.Patch(color="#2471a3", label=f"Gauss BL σ²={GAUSS_VAR_BL}"),
    mpatches.Patch(color="#5dade2", label=f"Gauss TL σ²={GAUSS_VAR_TL}"),
    mpatches.Patch(color="#922b21", label=f"Exp TR rate={EXP_RATE_TR:.0f}"),
    mpatches.Patch(color="#e59866", label=f"Exp BR rate={EXP_RATE_BR:.2f}"),
    mpatches.Patch(facecolor="none", edgecolor="k",
                   label=f"Active set m={model.n_active}"),
]
ax.legend(handles=patches, fontsize=7, loc="upper right")
ax.set_title(f"Training observations  (n={N_TRAIN})", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
sc_dummy = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
add_cbar(ax, sc_dummy)

ax = axes[1, 0]
im2 = ax.imshow(mu_img, origin="lower", extent=[0, 1, 0, 1],
                cmap=cmap, norm=norm, aspect="equal", interpolation="bilinear")
draw_quadrants(ax)
ax.scatter(X_train[:, 0], X_train[:, 1], c="k", s=4, alpha=0.25, zorder=3)
ax.set_title(f"Predicted mean  (RMSE={grid_rmse:.3f})", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im2)

ax = axes[1, 1]
im3 = ax.imshow(std_img, origin="lower", extent=[0, 1, 0, 1],
                cmap="Greens", aspect="equal", interpolation="bilinear")
draw_quadrants(ax)
ax.scatter(X_train[:, 0], X_train[:, 1], c="k", s=4, alpha=0.25, zorder=3)
ax.set_title("Predictive std dev", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im3)

ax = axes[1, 2]
im4 = ax.imshow(err_img, origin="lower", extent=[0, 1, 0, 1],
                cmap="Oranges", aspect="equal", interpolation="bilinear")
draw_quadrants(ax)
ax.scatter(X_train[:, 0], X_train[:, 1], c="k", s=4, alpha=0.25, zorder=3)
ax.set_title("|Predicted − True|", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im4)

params_final = model.covariance.get_params()
fig.suptitle(
    f"2-D VRK — mixed likelihoods  |  GaussianCovariance + Nugget  |  "
    f"Left=Gaussian, Right=Exponential  |  n={N_TRAIN}, m={model.n_active}",
    fontsize=10,
)
plt.savefig("vrk_demo_2d_mixed.png", dpi=150, bbox_inches="tight")
print("Saved vrk_demo_2d_mixed.png")
