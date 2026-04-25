"""
Demo: Evidence formula comparison.

Fits sin(x) with Gaussian noise and sweeps over covariance range_a,
comparing all four evidence quantities:
  1. approximate_evidence()     = Σ log Z_i (EP lower bound)
  2. active_set_log_evidence()  (EP approximation, MATLAB ogpevid formula)
  3. elbo()                     (ELBO = approx_ev - KL_from_prior, genuine lower bound)
  4. alt_log_evidence()         (cross-check via KBinv + C formula)

Uses ExponentialCovariance + NuggetCovariance.  Sweeps over log(range_a)
to show how each evidence objective behaves under misspecification.

Prints a table at the optimal range_a from each criterion and shows a line
plot of all four vs log(range_a).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrk import (VRK, ExponentialCovariance, NuggetCovariance, SumCovariance,
                 GaussianLikelihood)

RNG = np.random.default_rng(0)

noise_std  = 0.05
nugget_var = noise_std ** 2

n = 40
X = np.linspace(0, 2 * np.pi, n)[:, None]
y = np.sin(X[:, 0]) + noise_std * RNG.standard_normal(n)

MAX_ACTIVE       = 15
N_SWEEPS         = 3
LOG_RANGE_GRID   = np.linspace(-2.0, 1.5, 35)   # log(range_a) sweep

ev_ep   = []
ev_ase  = []
ev_elbo = []
ev_alt  = []

print("Evidence comparison — sweeping log(range_a)  [ExponentialCovariance + Nugget]")
print(f"  {'log_a':>8}  {'EP':>9}  {'ASE':>9}  {'ELBO':>9}  {'Alt':>9}")
print("  " + "-" * 44)

for log_a in LOG_RANGE_GRID:
    struct_cov = ExponentialCovariance(sill=1.0, range_a=np.exp(log_a))
    nug_cov    = NuggetCovariance(sill=nugget_var)
    cov = SumCovariance(struct_cov, nug_cov)
    lik = GaussianLikelihood(variance=1e-6)
    model = VRK(cov, lik, max_active=MAX_ACTIVE, n_sweeps=N_SWEEPS)
    model.fit(X, y)

    ep  = model.approximate_evidence()
    ase = model.active_set_log_evidence()
    elb = model.elbo()
    alt = model.alt_log_evidence()

    ev_ep.append(ep)
    ev_ase.append(ase)
    ev_elbo.append(elb)
    ev_alt.append(alt)

    if abs(log_a - round(log_a)) < 0.05 or log_a in LOG_RANGE_GRID[[0, -1]]:
        print(f"  {log_a:8.2f}  {ep:9.3f}  {ase:9.3f}  {elb:9.3f}  {alt:9.3f}")

ev_ep   = np.array(ev_ep)
ev_ase  = np.array(ev_ase)
ev_elbo = np.array(ev_elbo)
ev_alt  = np.array(ev_alt)


def best(ev):
    i = np.nanargmax(ev)
    return LOG_RANGE_GRID[i], np.exp(LOG_RANGE_GRID[i]), ev[i]


bp_ep,  ra_ep,  v_ep  = best(ev_ep)
bp_ase, ra_ase, v_ase = best(ev_ase)
bp_elb, ra_elb, v_elb = best(ev_elbo)
bp_alt, ra_alt, v_alt = best(ev_alt)

print("\nOptimal range_a per criterion:")
print(f"  {'Criterion':<20} {'log_a':>8}  {'range_a':>7}  {'value':>9}")
print("  " + "-" * 48)
for name, bp, ra, v in [
    ("EP (Σ log Z)",     bp_ep,  ra_ep,  v_ep),
    ("Active-set (ASE)", bp_ase, ra_ase, v_ase),
    ("ELBO",             bp_elb, ra_elb, v_elb),
    ("Alt (KBinv+C)",    bp_alt, ra_alt, v_alt),
]:
    print(f"  {name:<20} {bp:8.3f}  {ra:7.3f}  {v:9.3f}")

n_below = np.sum(ev_elbo < ev_ase)
print(f"\nELBO < ASE at {n_below}/{len(LOG_RANGE_GRID)} grid points "
      f"(ELBO is genuine lower bound; ASE is an approximation).")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(LOG_RANGE_GRID, ev_ep,   label="EP (Σ log Z)",     color="#e74c3c", lw=2)
ax.plot(LOG_RANGE_GRID, ev_ase,  label="Active-set (ASE)", color="#3498db", lw=2)
ax.plot(LOG_RANGE_GRID, ev_elbo, label="ELBO",             color="#2ecc71", lw=2, ls="--")
ax.plot(LOG_RANGE_GRID, ev_alt,  label="Alt (KBinv+C)",    color="#9b59b6", lw=1.5, ls=":")

for bp, v, c in [(bp_ep, v_ep, "#e74c3c"), (bp_ase, v_ase, "#3498db"),
                 (bp_elb, v_elb, "#2ecc71"), (bp_alt, v_alt, "#9b59b6")]:
    ax.axvline(bp, color=c, alpha=0.3, lw=1)

ax.set_xlabel("log(range_a)", fontsize=12)
ax.set_ylabel("Evidence value", fontsize=12)
ax.set_title("Evidence formula comparison: ExponentialCovariance + Nugget", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("vrk_demo_evidence_comparison.png", dpi=120)
print("\nSaved vrk_demo_evidence_comparison.png")
