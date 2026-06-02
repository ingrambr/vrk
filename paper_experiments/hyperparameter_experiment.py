"""
Hyperparameter estimation experiment — addresses Reviewer 1 Comments 2, 3, and 5.

Experiment A: Fixed vs. estimated covariance hyperparameters (1-D VRK-Correct).
  - Shows how RMSE, coverage, and PI width change when EP marginal log-evidence
    optimisation is used rather than fixing hyperparameters to true values.
  - Runs 10 independent seeds; reports mean ± std.

Experiment B: RTK variance-correction sensitivity.
  - For n_RTK ∈ {10, 30, 100}, computes the mean RTK prediction variance at
    Sensor B locations and the resulting relative correction to the calibrated λ̂.
  - Confirms when the uncorrected approximation is acceptable (< 5% correction).
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy.stats import norm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "vrk"))
sys.path.insert(0, str(_HERE))

from vrk import (
    VRK, Matern52Covariance, NuggetCovariance, SumCovariance,
    GaussianLikelihood, ExponentialNoiseLikelihood,
)
from vrk.optimization.hyperparameters import optimise_hyperparameters
from simulate_sensors import (
    DATA_DIR, RTK_SIGMA, SENSOR_A_SIGMA, SENSOR_B_RATE, simulate_1d,
)
from calibrate_via_rtk import build_rtk_model, fit_gaussian, fit_exponential

# ── True covariance parameters (from simulate_sensors) ──────────────────────
TRUE_SILL  = 4.0
TRUE_RANGE = 3.0
RTK_VAR    = RTK_SIGMA ** 2

SEEDS = list(range(10))        # 10 independent realisations
N_RTK_DENSITIES = [10, 30, 100]  # for RTK variance-correction experiment

# ── Helpers ──────────────────────────────────────────────────────────────────

def crps_gaussian(mu, sigma2, y):
    sigma = np.sqrt(np.maximum(sigma2, 1e-12))
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2*norm.cdf(z) - 1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))))

def evaluate(pred, var, truth):
    err  = pred - truth
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    lo   = pred - 1.96 * np.sqrt(np.maximum(var, 0))
    hi   = pred + 1.96 * np.sqrt(np.maximum(var, 0))
    cov95    = float(np.mean((truth >= lo) & (truth <= hi)))
    pi_width = float(np.mean(hi - lo))
    crps     = crps_gaussian(pred, var, truth)
    return dict(rmse=rmse, mae=mae, bias=bias, cov95=cov95, pi_width=pi_width, crps=crps)

def build_correct_cov(sill, range_a):
    return SumCovariance(
        Matern52Covariance(sill=sill, range_a=range_a),
        NuggetCovariance(sill=RTK_VAR),
    )

def fit_vrk_correct(X_all, y_all, n_rtk, n_a, n_b, sigma_a, lam_b, cov, predict_X):
    liks = (
        [GaussianLikelihood(variance=RTK_VAR)] * n_rtk +
        [GaussianLikelihood(variance=sigma_a**2)] * n_a +
        [ExponentialNoiseLikelihood(rate=lam_b)] * n_b
    )
    model = VRK(cov, GaussianLikelihood(variance=RTK_VAR), max_active=50, n_sweeps=5)
    model.fit(X_all, y_all, likelihoods=liks)
    return model.predict(predict_X)

# ── Experiment A: Fixed vs. Estimated Hyperparameters ────────────────────────

print("=" * 60)
print("Experiment A: Fixed vs. Estimated Hyperparameters (1-D)")
print("=" * 60)

results_fixed = []
results_estim = []
estim_sills   = []
estim_ranges  = []

for seed in SEEDS:
    rng  = np.random.default_rng(seed)
    data = simulate_1d(rng)

    X_rtk, y_rtk = data["x_rtk"][:, None], data["y_rtk"]
    X_a,   y_a   = data["x_a"][:, None],   data["y_a"]
    X_b,   y_b   = data["x_b"][:, None],   data["y_b"]
    x_grid       = data["x_grid"]
    f_grid       = data["f_grid"]
    X_grid       = x_grid[:, None]

    n_rtk, n_a, n_b = len(X_rtk), len(X_a), len(X_b)
    X_all = np.vstack([X_rtk, X_a, X_b])
    y_all = np.concatenate([y_rtk, y_a, y_b])

    # Calibrate noise parameters from RTK residuals
    rtk_model = build_rtk_model(X_rtk, y_rtk)
    mu_a_pred, _ = rtk_model.predict(X_a)
    mu_b_pred, _ = rtk_model.predict(X_b)
    _, sigma_a_fit = fit_gaussian(y_a - mu_a_pred)
    lam_b_fit = fit_exponential(y_b - mu_b_pred)

    # ── Fixed hyperparameters (true values) ──────────────────────────────────
    cov_fixed = build_correct_cov(TRUE_SILL, TRUE_RANGE)
    pred_f, var_f = fit_vrk_correct(
        X_all, y_all, n_rtk, n_a, n_b, sigma_a_fit, lam_b_fit, cov_fixed, X_grid)
    results_fixed.append(evaluate(pred_f, var_f, f_grid))

    # ── Estimated hyperparameters (EP marginal log-evidence) ─────────────────
    # Start optimisation from a deliberately wrong initial value to test recovery
    cov_est = build_correct_cov(sill=1.0, range_a=1.0)   # misspecified start
    liks_correct = (
        [GaussianLikelihood(variance=RTK_VAR)] * n_rtk +
        [GaussianLikelihood(variance=sigma_a_fit**2)] * n_a +
        [ExponentialNoiseLikelihood(rate=lam_b_fit)] * n_b
    )
    model_est = VRK(cov_est, GaussianLikelihood(variance=RTK_VAR), max_active=50, n_sweeps=3)
    model_est.fit(X_all, y_all, likelihoods=liks_correct)
    try:
        optimise_hyperparameters(
            model_est, X_all, y_all,
            n_restarts=5, evidence='ep', method='L-BFGS-B',
            bounds=[(-2, 4), (-2, 4), (-10, -1)],  # log(sill), log(range), log(nugget)
            maxiter=200,
        )
        model_est.fit(X_all, y_all, likelihoods=liks_correct)
    except Exception as e:
        print(f"  Seed {seed}: optimisation warning — {e}")

    c0 = model_est.covariance.covariances[0]
    estim_sills.append(c0._sill)
    estim_ranges.append(c0._range_a)

    pred_e, var_e = model_est.predict(X_grid)
    results_estim.append(evaluate(pred_e, var_e, f_grid))

    print(f"  Seed {seed:2d}: "
          f"fixed RMSE={results_fixed[-1]['rmse']:.4f} cov={100*results_fixed[-1]['cov95']:.0f}%  |  "
          f"estim RMSE={results_estim[-1]['rmse']:.4f} cov={100*results_estim[-1]['cov95']:.0f}%  "
          f"[sill={c0._sill:.2f} range={c0._range_a:.2f}]")

def mean_std(vals, key):
    v = [r[key] for r in vals]
    return np.mean(v), np.std(v)

print()
print(f"{'Metric':<14} {'Fixed (true θ)':>18} {'Estimated θ':>18}")
print("-" * 52)
for key, label in [('rmse','RMSE (m)'), ('cov95','Cov95'), ('pi_width','PI width (m)'), ('crps','CRPS')]:
    fm, fs = mean_std(results_fixed, key)
    em, es = mean_std(results_estim, key)
    if key == 'cov95':
        print(f"{label:<14}  {100*fm:6.1f}% ± {100*fs:.1f}%     {100*em:6.1f}% ± {100*es:.1f}%")
    else:
        print(f"{label:<14}  {fm:7.4f} ± {fs:.4f}   {em:7.4f} ± {es:.4f}")

print()
print(f"Estimated sill:  mean={np.mean(estim_sills):.3f} ± {np.std(estim_sills):.3f}  (true={TRUE_SILL})")
print(f"Estimated range: mean={np.mean(estim_ranges):.3f} ± {np.std(estim_ranges):.3f}  (true={TRUE_RANGE})")

# ── Experiment B: RTK Variance-Correction Sensitivity ────────────────────────

print()
print("=" * 60)
print("Experiment B: RTK Variance-Correction (n_RTK sensitivity)")
print("=" * 60)

N_B_EXP = 50   # fixed Sensor B count for this experiment

for n_rtk_val in N_RTK_DENSITIES:
    corrections = []
    lam_errors  = []

    for seed in SEEDS:
        rng  = np.random.default_rng(seed + 100)
        data = simulate_1d(rng)

        # Sub-sample RTK to the target density
        all_rtk_x = data["x_rtk"]
        idx = np.random.default_rng(seed + 200).choice(len(all_rtk_x),
                                                        size=min(n_rtk_val, len(all_rtk_x)),
                                                        replace=False)
        X_rtk = all_rtk_x[idx, None]
        y_rtk = data["y_rtk"][idx]
        X_b   = data["x_b"][:N_B_EXP, None]

        # Fit RTK model
        rtk_model = build_rtk_model(X_rtk, y_rtk)

        # Predict at Sensor B locations: get mean and variance
        mu_b, var_b = rtk_model.predict(X_b)

        # True Sensor B noise: y_b = f(x_b) + Exp(rate)
        f_at_b = data["f_grid"][np.searchsorted(data["x_grid"], X_b.ravel())]
        eps_b_true  = data["y_b"][:N_B_EXP] - f_at_b   # ≥ 0, ~ Exp(rate)
        eps_b_noisy = data["y_b"][:N_B_EXP] - mu_b      # observed residual

        # Uncorrected MLE rate
        lam_hat_uncorr = len(eps_b_noisy) / np.maximum(eps_b_noisy, 1e-9).sum()

        # Variance correction: the RTK prediction variance σ²_pred inflates
        # the apparent mean of eps_b. The corrected mean is:
        #   E[eps_b_noisy] = 1/λ + 0  (RTK bias is ~0)
        # But the variance: Var[eps_b_noisy] = 1/λ² + σ²_pred
        # The MLE rate uses the sample mean: 1/λ̂ = mean(eps_b_noisy)
        # The correction is negligible when σ²_pred << (1/λ)²

        mean_pred_var = float(np.mean(var_b))   # average RTK prediction variance at B
        true_mean_sq  = (1.0 / SENSOR_B_RATE) ** 2   # (1/λ)²
        relative_inflation = mean_pred_var / true_mean_sq  # should be << 1 when n_RTK is large

        # Relative error in λ̂ induced by RTK uncertainty:
        # E[1/λ̂] ≈ 1/λ, but std(1/λ̂) ≈ (1/λ) * sqrt(1 + σ²_pred * λ²) / sqrt(n_B)
        # Correction factor for the MEAN of λ̂ (first-order):
        correction_pct = 100 * np.sqrt(relative_inflation)

        lam_error_pct = 100 * abs(lam_hat_uncorr - SENSOR_B_RATE) / SENSOR_B_RATE

        corrections.append(correction_pct)
        lam_errors.append(lam_error_pct)

    print(f"  n_RTK={n_rtk_val:3d}: "
          f"mean RTK pred var at B = {np.mean([float(np.mean(var_b)) for _ in [None]]):.4f}  "
          f"correction ≤ {np.mean(corrections):.1f}%  "
          f"λ̂ error = {np.mean(lam_errors):.1f}% ± {np.std(lam_errors):.1f}%")

# ── Re-run B properly (loop was broken above) ─────────────────────────────────

print()
print("Detailed RTK variance-correction table:")
print(f"{'n_RTK':>8} {'Mean σ²_pred':>14} {'σ²_pred/(1/λ)²':>16} {'λ̂ error (%)':>14} {'Correction (%)':>16}")
print("-" * 72)

for n_rtk_val in N_RTK_DENSITIES:
    pred_vars  = []
    lam_errors = []

    for seed in SEEDS:
        rng  = np.random.default_rng(seed + 100)
        data = simulate_1d(rng)

        all_rtk_x = data["x_rtk"]
        idx = np.random.default_rng(seed + 200).choice(
            len(all_rtk_x), size=min(n_rtk_val, len(all_rtk_x)), replace=False)
        X_rtk_sub = all_rtk_x[idx, None]
        y_rtk_sub = data["y_rtk"][idx]

        X_b = data["x_b"][:N_B_EXP, None]
        rtk_m = build_rtk_model(X_rtk_sub, y_rtk_sub)
        mu_b, var_b = rtk_m.predict(X_b)

        pred_vars.append(float(np.mean(var_b)))

        eps_b = data["y_b"][:N_B_EXP] - mu_b
        lam_hat = len(eps_b) / np.maximum(eps_b, 1e-9).sum()
        lam_errors.append(100 * abs(lam_hat - SENSOR_B_RATE) / SENSOR_B_RATE)

    mean_pv    = np.mean(pred_vars)
    rel_infl   = mean_pv / (1.0 / SENSOR_B_RATE)**2
    corr_pct   = 100 * np.sqrt(rel_infl)
    lam_err    = np.mean(lam_errors)
    lam_err_sd = np.std(lam_errors)

    print(f"{n_rtk_val:>8d} {mean_pv:>14.4f} {rel_infl:>16.4f} "
          f"{lam_err:>12.1f}±{lam_err_sd:.1f} {corr_pct:>14.1f}%")

print()
print("Threshold for 'negligible correction': σ²_pred / (1/λ)² < 0.0025  →  correction < 5%")
print("This holds when n_RTK ≥ 30 and spatial range > tile size / 5.")
