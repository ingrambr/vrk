"""
Demo: Analytic vs finite-difference gradient comparison.

Fits sin(x) using Matern52Covariance + NuggetCovariance and compares:
  - Analytic gradient of active_set_log_evidence()  [O(m²) per param]
  - Finite-difference gradient (active_set_evidence_gradient())

Prints:
  - Side-by-side table: param, analytic, FD, relative error
  - Timing comparison
  - Convergence curve using each gradient method for a small optimisation run

The Matérn 5/2 is a good default for physical data:  it is twice
differentiable, smoother than the exponential (Matérn 1/2) but less
unrealistically smooth than the Gaussian (RBF).
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, Matern52Covariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood)

RNG = np.random.default_rng(7)

noise_std  = 0.05
nugget_var = noise_std ** 2

n = 30
X = np.linspace(0, 2 * np.pi, n)[:, None]
y = np.sin(X[:, 0]) + noise_std * RNG.standard_normal(n)

MAX_ACTIVE   = 12
N_SWEEPS     = 3
# SumCovariance(Matern52, Nugget) has 3 params: sill, range_a, nugget_sill
PARAM_NAMES  = ["log sill", "log range_a", "log nugget"]


def make_model():
    struct_cov = Matern52Covariance(sill=1.0, range_a=1.0)
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    lik = GaussianLikelihood(variance=1e-6)
    return VRK(cov, lik, max_active=MAX_ACTIVE, n_sweeps=N_SWEEPS)


model = make_model()
model.fit(X, y)

print("=" * 60)
print("Demo: Analytic vs FD gradient of active_set_log_evidence()")
print(f"      Covariance: Matern52Covariance + NuggetCovariance")
print("=" * 60)
print(f"\nModel: Matern52 + Nugget, n={n}, max_active={MAX_ACTIVE}, n_sweeps={N_SWEEPS}")
print(f"Active points: {model.n_active}")
print(f"ASE = {model.active_set_log_evidence():.4f}")

# ── Gradient comparison table ──────────────────────────────────────────────────
t0 = time.perf_counter()
grad_analytic = model.analytic_evidence_gradient()
t_analytic = time.perf_counter() - t0

t0 = time.perf_counter()
grad_fd = model.active_set_evidence_gradient()
t_fd = time.perf_counter() - t0

print(f"\n{'Param':<14}  {'Analytic':>12}  {'FD':>12}  {'Rel err':>10}")
print("  " + "-" * 50)
for j, name in enumerate(PARAM_NAMES[:len(grad_analytic)]):
    a = grad_analytic[j]
    f = grad_fd[j]
    denom = max(abs(f), 1e-8)
    rel_err = abs(a - f) / denom
    print(f"{name:<14}  {a:12.6f}  {f:12.6f}  {rel_err:10.4f}")

print(f"\nTiming:")
print(f"  Analytic gradient: {t_analytic * 1e3:.2f} ms")
print(f"  FD gradient:       {t_fd * 1e3:.2f} ms")
if t_analytic > 0:
    print(f"  Speedup (FD/analytic): {t_fd / max(t_analytic, 1e-9):.1f}×")

# ── Optimisation convergence ───────────────────────────────────────────────────
print("\n--- Convergence: 30 gradient ascent steps ---")

N_STEPS = 30
STEP    = 3e-3


def run_gradient_ascent(grad_fn, label):
    m = make_model()
    m.fit(X, y)
    history = [m.active_set_log_evidence()]
    for _ in range(N_STEPS):
        g    = grad_fn(m)
        norm = np.linalg.norm(g)
        if norm < 1e-12:
            break
        lp = m.covariance.log_params + STEP * g / norm
        m.covariance.log_params = lp
        m.fit(X, y)
        history.append(m.active_set_log_evidence())
    print(f"  {label}: init={history[0]:.4f}, final={history[-1]:.4f}")
    return history


hist_analytic = run_gradient_ascent(
    lambda m: m.analytic_evidence_gradient(), "Analytic"
)
hist_fd = run_gradient_ascent(
    lambda m: m.active_set_evidence_gradient(), "FD      "
)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x_pos = np.arange(len(grad_analytic))
width = 0.35
axes[0].bar(x_pos - width / 2, grad_analytic, width, label="Analytic", color="#3498db")
axes[0].bar(x_pos + width / 2, grad_fd, width, label="FD", color="#e74c3c", alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(PARAM_NAMES[:len(grad_analytic)], fontsize=9)
axes[0].set_title("Gradient comparison: analytic vs FD")
axes[0].set_ylabel("dASE / d(log θ)")
axes[0].legend()
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].plot(hist_analytic, color="#3498db", lw=2, label="Analytic grad")
axes[1].plot(hist_fd, color="#e74c3c", lw=2, ls="--", label="FD grad")
axes[1].set_xlabel("Gradient step")
axes[1].set_ylabel("Active-set log evidence")
axes[1].set_title("Convergence: gradient ascent on ASE")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    "Analytic vs FD gradient — Matern52 + Nugget",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("vrk_demo_gradient_comparison.png", dpi=120)
print("\nSaved vrk_demo_gradient_comparison.png")
