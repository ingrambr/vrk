"""
Demo: Bernoulli (probit) likelihood — binary classification.

Latent function: f(x) = sin(x)   (positive → class +1, negative → class -1)
Observations: y_i ∈ {-1, +1}   with probit link P(y=1|f) = Φ(f)

Uses GaussianCovariance + NuggetCovariance.  The smooth (infinitely
differentiable) Gaussian covariance is appropriate for classification
because the latent probability surface is typically smooth.

Demonstrates:
  - Fixed range_a sweep: accuracy vs EP evidence trade-off
  - For classification, EP evidence may prefer overfitting solutions
    (smaller range → more active points → higher per-point log_z, but
     worse generalisation). Cross-validation is more reliable.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import ndtr

from vrk import (VRK, GaussianCovariance, NuggetCovariance, SumCovariance,
                 BernoulliLikelihood)

RNG = np.random.default_rng(7)

nugget_var = 1e-3   # small structural nugget for classification

n = 80
X_train      = np.sort(RNG.uniform(-np.pi, 2 * np.pi, n))[:, None]
f_true_tr    = np.sin(X_train[:, 0])
p_pos        = ndtr(f_true_tr)
y_train      = 2.0 * (RNG.uniform(size=n) < p_pos).astype(float) - 1.0

X_test       = np.linspace(-np.pi, 2 * np.pi, 200)[:, None]
f_test_true  = np.sin(X_test[:, 0])
y_test_true  = np.sign(f_test_true)

print("=" * 60)
print("Demo: Bernoulli / probit likelihood — 1D classification")
print(f"      Covariance: GaussianCovariance + NuggetCovariance")
print("=" * 60)
print(f"n_train={n}, class balance: {int((y_train==1).sum())}/{int((y_train==-1).sum())}")


def make_model(range_a):
    struct_cov = GaussianCovariance(sill=1.0, range_a=range_a)
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    lik = BernoulliLikelihood()
    return VRK(cov, lik, max_active=15, n_sweeps=3)


def fit_and_eval(label, range_a):
    model = make_model(range_a)
    model.fit(X_train, y_train)
    mean_f, var_f = model.predict(X_test)
    y_pred   = np.sign(mean_f)
    accuracy = np.mean(y_pred == y_test_true)
    ev       = model.approximate_evidence()
    print(f"  [{label}]  n_active={model.n_active}  "
          f"accuracy={accuracy:.3f}  EP evidence={ev:.1f}")
    return mean_f, var_f, model


print("\n--- Range sweep (accuracy vs EP evidence) ---")
mean_f1, var_f1, _           = fit_and_eval("range_a=0.5", 0.5)
mean_f2, var_f2, _           = fit_and_eval("range_a=1.0", 1.0)
mean_f3, var_f3, model_best  = fit_and_eval("range_a=2.0", 2.0)
mean_f4, var_f4, _           = fit_and_eval("range_a=3.0", 3.0)

print("\n  Note: EP evidence prefers small range_a (more active points),")
print("        but best accuracy is usually at intermediate range_a.")
print("        Cross-validation is more reliable for model selection")
print("        in classification.")

# ── ASCII decision boundary ────────────────────────────────────────────────────
print("\n--- ASCII decision boundary (range_a=2.0, 60 test pts) ---")
idx       = np.round(np.linspace(0, 199, 60)).astype(int)
row_pred  = "".join("+" if v > 0 else "-" for v in mean_f3[idx])
row_true  = "".join("+" if v > 0 else "-" for v in y_test_true[idx])
print(f"Predicted: {row_pred}")
print(f"True sign: {row_true}")
correct = sum(p == t for p, t in zip(row_pred, row_true))
print(f"Character accuracy: {correct}/{len(row_pred)}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(
    "VRK — Bernoulli (probit) likelihood: binary classification  (GaussianCovariance + Nugget)",
    fontsize=12,
)

x_te      = X_test[:, 0]
x_tr      = X_train[:, 0]
pos_mask  = y_train == 1
neg_mask  = y_train == -1

ax1.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax1.plot(x_te, f_test_true, "k--", lw=1.5, label="True f = sin(x)", zorder=5)
colors_l = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
for mean, var, c, lbl in zip(
    [mean_f1, mean_f2, mean_f3, mean_f4],
    [var_f1,  var_f2,  var_f3,  var_f4],
    colors_l,
    ["a=0.5", "a=1.0", "a=2.0", "a=3.0"],
):
    std = np.sqrt(np.maximum(var, 0))
    ax1.plot(x_te, mean, color=c, lw=1.6, label=lbl)
    ax1.fill_between(x_te, mean - 2 * std, mean + 2 * std, color=c, alpha=0.08)
ax1.scatter(x_tr[pos_mask], np.ones(pos_mask.sum())  *  1.8,
            marker="^", s=30, c="#2c3e50", alpha=0.5, label="+1 class")
ax1.scatter(x_tr[neg_mask], np.ones(neg_mask.sum()) * -1.8,
            marker="v", s=30, c="#7f8c8d", alpha=0.5, label="-1 class")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.set_title("Latent function for each range_a")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlim(x_te[0], x_te[-1])

std3         = np.sqrt(np.maximum(var_f3, 0))
p_pos_pred   = ndtr(mean_f3 / np.sqrt(1 + var_f3))
cmap_class   = LinearSegmentedColormap.from_list("cls", ["#3498db", "white", "#e74c3c"])
ax2.imshow(
    p_pos_pred[np.newaxis, :], aspect="auto",
    extent=[x_te[0], x_te[-1], -2.5, 2.5],
    origin="lower", cmap=cmap_class, vmin=0, vmax=1, alpha=0.25,
)
ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax2.plot(x_te, f_test_true, "k--", lw=1.5, label="True f = sin(x)", zorder=5)
ax2.plot(x_te, mean_f3, "#e74c3c", lw=2, label="VRK mean (a=2.0)", zorder=4)
ax2.fill_between(x_te, mean_f3 - 2 * std3, mean_f3 + 2 * std3,
                 color="#e74c3c", alpha=0.18, label="±2σ", zorder=3)
ax2.scatter(x_tr[pos_mask], np.full(pos_mask.sum(),  2.2),
            marker="^", s=35, c="#c0392b", alpha=0.7, zorder=6, label="+1")
ax2.scatter(x_tr[neg_mask], np.full(neg_mask.sum(), -2.2),
            marker="v", s=35, c="#2980b9", alpha=0.7, zorder=6, label="-1")
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")
ax2.set_title("Best model (a=2.0) — background = P(y=+1|x)")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_xlim(x_te[0], x_te[-1])
ax2.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig("vrk_demo_bernoulli.png", dpi=150)
print("\nSaved vrk_demo_bernoulli.png")
