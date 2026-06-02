"""
NEON Harvard Forest multi-seed sensitivity experiment.

Re-samples the RTK / Sensor-A / Sensor-B observation locations from the same
NEON raster tile across N independent seeds, runs the full calibration-fusion
pipeline for each seed, and reports per-method metric distributions.

This addresses Reviewer feedback that the headline NEON numbers come from a
single seed (SEED=42 in create_observation_files.py) and therefore carry no
error bar.
"""

import sys
import json
import numpy as np
import rasterio
from pathlib import Path
from scipy.spatial import KDTree
from scipy.stats import norm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "vrk"))

from vrk import (
    VRK, Matern52Covariance, NuggetCovariance, SumCovariance,
    GaussianLikelihood, ExponentialNoiseLikelihood,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE  = _HERE / "data" / "neon_harv"
TILE  = "727000_4706000"
STEP  = 10

N_RTK_SWEEP = [15]
N_A_SWEEP   = [50, 120]
N_B   = 200

SIGMA_RTK = 0.05
SIGMA_A   = 0.5
RTK_VAR   = SIGMA_RTK ** 2

SEEDS = list(range(10))

# ── Load rasters once ────────────────────────────────────────────────────────
def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr

print("Loading rasters …")
dsm = load_raster(BASE / f"NEON_D01_HARV_DP3_{TILE}_DSM.tif")
dtm = load_raster(BASE / f"NEON_D01_HARV_DP3_{TILE}_DTM.tif")
chm = dsm - dtm

rows = np.arange(0, 1000, STEP)
cols = np.arange(0, 1000, STEP)
R, C = np.meshgrid(rows, cols, indexing="ij")

easting_origin  = int(TILE.split("_")[0])
northing_origin = int(TILE.split("_")[1])

grid_e = (easting_origin  + C * 1.0) / 1000.0
grid_n = (northing_origin + R * 1.0) / 1000.0

XY_grid_full  = np.column_stack([grid_e.ravel(), grid_n.ravel()])
DTM_full      = dtm[R, C].ravel()
DSM_full      = dsm[R, C].ravel()
CHM_full      = chm[R, C].ravel()

valid = np.isfinite(DTM_full) & np.isfinite(DSM_full)
XY_grid  = XY_grid_full[valid]
DTM_grid = DTM_full[valid]
DSM_grid = DSM_full[valid]
CHM_grid = CHM_full[valid]

elev_mean = float(np.mean(DTM_grid))
DTM_norm  = DTM_grid - elev_mean
dtm_var   = float(np.var(DTM_norm))

print(f"  Grid: {len(DTM_grid)} valid pixels  |  elev_mean = {elev_mean:.2f} m  |  DTM var = {dtm_var:.2f}")

# Fixed physically-motivated covariance hyperparameters (matches neon_experiment.py)
SILL  = dtm_var * 0.95
RANGE = 0.17

# ── Helpers ──────────────────────────────────────────────────────────────────
all_idx = np.arange(len(DTM_grid))
veg_idx = np.where(CHM_grid > 5.0)[0]

def sample_observations(seed, n_rtk, n_a):
    """Return (X_rtk, y_rtk, X_a, y_a, X_b, y_b, y_b_raw, chm_b_true)
       — normalised to elev_mean and sampled with the given seed."""
    rng = np.random.default_rng(seed)

    rtk_idx = rng.choice(all_idx, size=n_rtk, replace=False)
    b_pool  = np.setdiff1d(veg_idx, rtk_idx)
    b_idx   = rng.choice(b_pool, size=N_B, replace=False)
    used    = set(b_idx.tolist()) | set(rtk_idx.tolist())
    remain  = np.array([i for i in all_idx if i not in used])
    a_idx   = rng.choice(remain, size=n_a, replace=False)

    X_rtk = XY_grid[rtk_idx]
    y_rtk = (DTM_grid[rtk_idx] + rng.normal(0, SIGMA_RTK, n_rtk)) - elev_mean

    X_a   = XY_grid[a_idx]
    y_a   = (DTM_grid[a_idx] + rng.normal(0, SIGMA_A, n_a)) - elev_mean

    X_b        = XY_grid[b_idx]
    y_b_raw    = DSM_grid[b_idx]
    y_b        = y_b_raw - elev_mean
    chm_b_true = CHM_grid[b_idx]

    return X_rtk, y_rtk, X_a, y_a, X_b, y_b, y_b_raw, chm_b_true

def fresh_cov():
    return SumCovariance(
        Matern52Covariance(sill=SILL, range_a=RANGE),
        NuggetCovariance(sill=RTK_VAR),
    )

def crps_gaussian(mu, sigma2, y):
    sigma = np.sqrt(np.maximum(sigma2, 1e-12))
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2*norm.cdf(z) - 1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))))

def evaluate(pred, var, truth, mask=None):
    if mask is not None:
        pred, var, truth = pred[mask], var[mask], truth[mask]
    err = pred - truth
    lo  = pred - 1.96 * np.sqrt(np.maximum(var, 0))
    hi  = pred + 1.96 * np.sqrt(np.maximum(var, 0))
    return dict(
        rmse     = float(np.sqrt(np.mean(err**2))),
        mae      = float(np.mean(np.abs(err))),
        bias     = float(np.mean(err)),
        cov95    = float(np.mean((truth >= lo) & (truth <= hi))),
        pi_width = float(np.mean(hi - lo)),
        crps     = crps_gaussian(pred, var, truth),
    )

# ── Per-seed pipeline ────────────────────────────────────────────────────────
def run_seed(seed, n_rtk, n_a):
    X_rtk, y_rtk, X_a, y_a, X_b, y_b, y_b_raw, chm_b_true = sample_observations(seed, n_rtk, n_a)

    # ── Calibration: fit RTK-only model, extract residuals at sensor locations
    rtk_cal = VRK(fresh_cov(), GaussianLikelihood(variance=RTK_VAR),
                  max_active=30, n_sweeps=5)
    rtk_cal.fit(X_rtk, y_rtk)
    mu_a, _ = rtk_cal.predict(X_a)
    mu_b, _ = rtk_cal.predict(X_b)
    eps_a = y_a - mu_a
    eps_b = y_b - mu_b
    sigma_a = float(np.std(eps_a, ddof=1))
    lam_b   = float(len(eps_b) / np.maximum(eps_b, 1e-6).sum())

    # ── Fusion data
    X_all = np.vstack([X_rtk, X_a, X_b])
    y_all = np.concatenate([y_rtk, y_a, y_b])
    n_rtk, n_a, n_b = len(X_rtk), len(X_a), len(X_b)

    # ── Zone masks
    tree_b = KDTree(X_b)
    dists, _ = tree_b.query(XY_grid)
    zone_b   = dists < 0.025
    zone_op  = ~zone_b

    out = {}

    def fit_and_score(label, liks, y_override=None):
        y_use = y_all if y_override is None else y_override
        m = VRK(fresh_cov(), GaussianLikelihood(variance=RTK_VAR),
                max_active=50, n_sweeps=3)
        m.fit(X_all, y_use, likelihoods=liks)
        pred, var = m.predict(XY_grid)
        out[label] = dict(
            full = evaluate(pred, var, DTM_norm),
            zone_b = evaluate(pred, var, DTM_norm, zone_b),
            open = evaluate(pred, var, DTM_norm, zone_op),
        )

    # RTK-Only
    m_rtk = VRK(fresh_cov(), GaussianLikelihood(variance=RTK_VAR),
                max_active=30, n_sweeps=5)
    m_rtk.fit(X_rtk, y_rtk)
    pred_rtk, var_rtk = m_rtk.predict(XY_grid)
    out["RTK-Only"] = dict(
        full = evaluate(pred_rtk, var_rtk, DTM_norm),
        zone_b = evaluate(pred_rtk, var_rtk, DTM_norm, zone_b),
        open = evaluate(pred_rtk, var_rtk, DTM_norm, zone_op),
    )

    fit_and_score("HGPR",
        [GaussianLikelihood(variance=RTK_VAR)] * n_rtk +
        [GaussianLikelihood(variance=sigma_a**2)] * n_a +
        [GaussianLikelihood(variance=(1/lam_b)**2)] * n_b)

    fit_and_score("VRK-Correct",
        [GaussianLikelihood(variance=RTK_VAR)] * n_rtk +
        [GaussianLikelihood(variance=sigma_a**2)] * n_a +
        [ExponentialNoiseLikelihood(rate=lam_b)] * n_b)

    y_b_oracle = y_b_raw - chm_b_true - elev_mean
    fit_and_score("VRK-Oracle",
        [GaussianLikelihood(variance=RTK_VAR)] * n_rtk +
        [GaussianLikelihood(variance=sigma_a**2)] * n_a +
        [GaussianLikelihood(variance=RTK_VAR)] * n_b,
        y_override=np.concatenate([y_rtk, y_a, y_b_oracle]))

    out["_meta"] = dict(sigma_a=sigma_a, lam_b=lam_b,
                        chm_mean=float(chm_b_true.mean()))
    return out

# ── Run sweep over (n_rtk, n_a) ──────────────────────────────────────────────
methods = ["RTK-Only", "HGPR", "VRK-Correct", "VRK-Oracle"]
sweep_results = {}

for n_rtk in N_RTK_SWEEP:
    for n_a in N_A_SWEEP:
        cfg = (n_rtk, n_a)
        print(f"\n{'='*70}\nN_RTK={n_rtk}, N_A={n_a}  ({len(SEEDS)} seeds)\n{'='*70}")
        rs = []
        for seed in SEEDS:
            print(f"  Seed {seed:2d} …", end=" ", flush=True)
            r = run_seed(seed, n_rtk, n_a)
            rs.append(r)
            print(f"VRK-Correct RMSE={r['VRK-Correct']['full']['rmse']:.3f}  "
                  f"zoneB bias VRK={r['VRK-Correct']['zone_b']['bias']:+.3f} "
                  f"HGPR={r['HGPR']['zone_b']['bias']:+.3f}  "
                  f"1/λ̂={1/r['_meta']['lam_b']:.1f}m")
        sweep_results[cfg] = rs

# ── Reporting: focus on vegetated zone ───────────────────────────────────────
def stats(results, method, zone, key):
    v = np.array([r[method][zone][key] for r in results])
    return v.mean(), v.std()

cfgs = [(r, a) for r in N_RTK_SWEEP for a in N_A_SWEEP]
print("\n\n" + "="*70)
print("VEGETATED (Sensor B) ZONE — mean ± std across 10 seeds")
print("="*70)
print(f"\n{'':<14} " + "  ".join(f"{'RTK='+str(r)+',A='+str(a):^22s}" for r,a in cfgs))
for key, fmt, label in [
    ("rmse",     "{:6.3f} ± {:5.3f}",  "RMSE (m)"),
    ("bias",     "{:+6.3f} ± {:5.3f}", "Bias (m)"),
    ("cov95",    "{:6.3f} ± {:5.3f}",  "Cov95"),
    ("pi_width", "{:6.2f} ± {:5.2f}",  "PI width (m)"),
    ("crps",     "{:6.3f} ± {:5.3f}",  "CRPS"),
]:
    print(f"\n  {label}")
    for m in methods:
        row = f"    {m:<12}"
        for cfg in cfgs:
            mean, sd = stats(sweep_results[cfg], m, "zone_b", key)
            row += "  " + fmt.format(mean, sd).rjust(22)
        print(row)

print("\n\n" + "="*70)
print("VEGETATED ZONE — HGPR → VRK-Correct improvement (per-seed paired)")
print("="*70)
for cfg in cfgs:
    rs = sweep_results[cfg]
    rmse_v = np.array([r["VRK-Correct"]["zone_b"]["rmse"] for r in rs])
    rmse_h = np.array([r["HGPR"]["zone_b"]["rmse"] for r in rs])
    bias_v = np.abs(np.array([r["VRK-Correct"]["zone_b"]["bias"] for r in rs]))
    bias_h = np.abs(np.array([r["HGPR"]["zone_b"]["bias"] for r in rs]))
    crps_v = np.array([r["VRK-Correct"]["zone_b"]["crps"] for r in rs])
    crps_h = np.array([r["HGPR"]["zone_b"]["crps"] for r in rs])
    d_rmse = 100 * (rmse_h - rmse_v) / rmse_h
    d_bias = 100 * (bias_h - bias_v) / bias_h
    d_crps = 100 * (crps_h - crps_v) / crps_h
    print(f"\n  N_RTK={cfg[0]}, N_A={cfg[1]}:")
    print(f"    ΔRMSE  = {d_rmse.mean():+5.2f}% ± {d_rmse.std():4.2f}%  "
          f"(seeds with VRK better: {(d_rmse > 0).sum()}/10)")
    print(f"    Δ|bias|= {d_bias.mean():+5.2f}% ± {d_bias.std():4.2f}%  "
          f"(seeds with VRK better: {(d_bias > 0).sum()}/10)")
    print(f"    ΔCRPS  = {d_crps.mean():+5.2f}% ± {d_crps.std():4.2f}%  "
          f"(seeds with VRK better: {(d_crps > 0).sum()}/10)")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = BASE / "neon_multiseed_results.json"
serial = {}
for cfg in cfgs:
    key = f"rtk{cfg[0]}_a{cfg[1]}"
    serial[key] = []
    for r in sweep_results[cfg]:
        s = {m: {z: r[m][z] for z in ["full", "zone_b", "open"]} for m in methods}
        s["_meta"] = r["_meta"]
        serial[key].append(s)
with open(out_path, "w") as f:
    json.dump({"seeds": SEEDS, "configs": [list(c) for c in cfgs], "results": serial}, f, indent=2)
print(f"\nResults written → {out_path}")
