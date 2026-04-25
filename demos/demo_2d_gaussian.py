"""
2-D VRK demo — Gaussian noise on a 2D domain.

Simulates data from a known GP (GaussianCovariance + Nugget), fits a VRK,
and predicts on a regular grid so the result can be visualised as a raster.

The GaussianCovariance (RBF / squared-exponential) is infinitely differentiable
and well-suited to smooth 2-D fields.  Nugget models sensor measurement error.

True parameters
---------------
  sill       σ² = 1.5
  range_a    a  = 0.25   (relative to a unit domain [0,1]²)
  nugget     σ²_n = 0.02
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from vrk import (VRK, GaussianCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood)

# ── simulation parameters ─────────────────────────────────────────────────────
TRUE_SILL   = 1.5
TRUE_RANGE  = 0.25
TRUE_NUGGET = 0.02
N_TRAIN     = 200
N_GRID      = 60          # grid cells per axis  (60×60 = 3600 prediction pts)
MAX_ACTIVE  = 60
N_SWEEPS    = 3
RNG         = np.random.default_rng(7)


def rbf_cov(A, B, var, rng):
    """GaussianCovariance: C(h) = var * exp(-h²/(2*rng²))."""
    diff    = A[:, None, :] - B[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    return var * np.exp(-0.5 * sq_dist / rng ** 2)


# ── sample training locations uniformly in [0,1]² ─────────────────────────────
X_train = RNG.uniform(0.0, 1.0, (N_TRAIN, 2))

# ── draw a GP sample at training locations ────────────────────────────────────
K_nn   = rbf_cov(X_train, X_train, TRUE_SILL, TRUE_RANGE)
L      = np.linalg.cholesky(K_nn + 1e-8 * np.eye(N_TRAIN))
f_true = L @ RNG.standard_normal(N_TRAIN)
y_train = f_true + RNG.normal(0.0, np.sqrt(TRUE_NUGGET), N_TRAIN)

# ── prediction grid ───────────────────────────────────────────────────────────
g       = np.linspace(0.0, 1.0, N_GRID)
gx, gy  = np.meshgrid(g, g)
X_grid  = np.column_stack([gx.ravel(), gy.ravel()])

# ── true GP field on grid ─────────────────────────────────────────────────────
K_gn       = rbf_cov(X_grid, X_train, TRUE_SILL, TRUE_RANGE)
alpha_true = np.linalg.solve(K_nn + 1e-8 * np.eye(N_TRAIN), f_true)
f_grid     = K_gn @ alpha_true
f_true_img = f_grid.reshape(N_GRID, N_GRID)

y_noisy_grid = f_grid + RNG.normal(0.0, np.sqrt(TRUE_NUGGET), f_grid.shape)
y_noisy_img  = y_noisy_grid.reshape(N_GRID, N_GRID)

# ── fit VRK ───────────────────────────────────────────────────────────────────
struct_cov = GaussianCovariance(sill=TRUE_SILL, range_a=TRUE_RANGE)
nug_cov    = NuggetCovariance(sill=TRUE_NUGGET)
cov = SumCovariance(struct_cov, nug_cov)
lik = GaussianLikelihood(variance=1e-6)
model = VRK(cov, lik, max_active=MAX_ACTIVE, n_sweeps=N_SWEEPS)
model.fit(X_train, y_train)

print(f"VRK fitted — active set size: {model.n_active} / {MAX_ACTIVE}")

# ── VRK predictions on grid ───────────────────────────────────────────────────
mu, var   = model.predict(X_grid)
std       = np.sqrt(np.maximum(var, 0.0))
mu_img    = mu.reshape(N_GRID, N_GRID)
std_img   = std.reshape(N_GRID, N_GRID)

grid_rmse = float(np.sqrt(np.mean((mu - f_grid) ** 2)))
print(f"Grid RMSE (pred vs true): {grid_rmse:.4f}  (noise σ = {np.sqrt(TRUE_NUGGET):.4f})")

# ── shared colour scale ────────────────────────────────────────────────────────
all_vals = np.concatenate([f_grid, y_noisy_grid, y_train, mu])
vmin, vmax = all_vals.min(), all_vals.max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "RdYlBu_r"


def add_cbar(ax, mappable):
    plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)


# ── 2×3 figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                         gridspec_kw={"hspace": 0.38, "wspace": 0.35})

ax = axes[0, 0]
im = ax.imshow(f_true_img, origin="lower", extent=[0, 1, 0, 1],
               cmap=cmap, norm=norm, aspect="equal", interpolation="bilinear")
ax.set_title("True GP field (noiseless)", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im)

ax = axes[0, 1]
im_n = ax.imshow(y_noisy_img, origin="lower", extent=[0, 1, 0, 1],
                 cmap=cmap, norm=norm, aspect="equal", interpolation="bilinear")
ax.set_title(f"Simulated data + noise  (σ_n={np.sqrt(TRUE_NUGGET):.2f})", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im_n)

ax = axes[0, 2]
sc = ax.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, cmap=cmap, norm=norm,
                s=22, edgecolors="k", linewidths=0.3, zorder=3)
if model.n_active > 0:
    active = model.active_set_
    ax.scatter(active[:, 0], active[:, 1],
               s=90, facecolors="none", edgecolors="k",
               linewidths=1.2, zorder=5, label=f"Active set (m={model.n_active})")
ax.set_title(f"Training observations  (n={N_TRAIN})", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper right")
add_cbar(ax, sc)

ax = axes[1, 0]
im2 = ax.imshow(mu_img, origin="lower", extent=[0, 1, 0, 1],
                cmap=cmap, norm=norm, aspect="equal", interpolation="bilinear")
ax.scatter(X_train[:, 0], X_train[:, 1], c="k", s=4, alpha=0.3, zorder=3)
ax.set_title(f"Predicted mean  (grid RMSE={grid_rmse:.3f})", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im2)

ax = axes[1, 1]
im3 = ax.imshow(std_img, origin="lower", extent=[0, 1, 0, 1],
                cmap="Greens", aspect="equal", interpolation="bilinear")
ax.scatter(X_train[:, 0], X_train[:, 1], c="k", s=4, alpha=0.3, zorder=3)
ax.set_title("Predictive std dev", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im3)

ax = axes[1, 2]
err_img = np.abs(mu_img - f_true_img)
im4 = ax.imshow(err_img, origin="lower", extent=[0, 1, 0, 1],
                cmap="Oranges", aspect="equal", interpolation="bilinear")
ax.scatter(X_train[:, 0], X_train[:, 1], c="k", s=4, alpha=0.3, zorder=3)
ax.set_title("|Predicted − True|", fontsize=11)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
add_cbar(ax, im4)

fig.suptitle(
    f"2-D VRK  |  GaussianCovariance + Nugget: sill={TRUE_SILL}, range={TRUE_RANGE}  |  "
    f"nugget={TRUE_NUGGET}  |  n={N_TRAIN}, m={model.n_active}",
    fontsize=10,
)
plt.savefig("vrk_demo_2d_gaussian.png", dpi=150, bbox_inches="tight")
print("Saved vrk_demo_2d_gaussian.png")
