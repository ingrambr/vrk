"""
Create fixed observation data files for the NEON Harvard Forest experiment.

Outputs:
  data/neon_harv/rtk_observations.csv      — 15 RTK-GPS reference points
  data/neon_harv/gps_observations.csv      — 120 GPS (Sensor A) points

Coordinates are given in:
  - UTM Zone 18N (easting_m, northing_m)   — native raster coordinates
  - WGS84 decimal degrees (lon, lat)        — for reference/publication

Elevations are raw metres above sea level (not mean-normalised).
The normalisation offset (elev_mean) is saved in data/neon_harv/experiment_meta.json
so the experiment script can apply it consistently.
"""

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from pathlib import Path
import csv
import json

SEED = 42
rng = np.random.default_rng(SEED)

BASE   = Path(__file__).parent / "data" / "neon_harv"
TILE   = "727000_4706000"
STEP   = 10          # subsampling step (10 m grid)
N_RTK  = 15
N_A    = 50
N_B    = 200
SIGMA_RTK = 0.05     # m
SIGMA_A   = 0.5      # m

# ---------------------------------------------------------------------------
# Load rasters
# ---------------------------------------------------------------------------
def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.crs, src.transform

dsm, crs, transform = load_raster(BASE / f"NEON_D01_HARV_DP3_{TILE}_DSM.tif")
dtm, _,   _         = load_raster(BASE / f"NEON_D01_HARV_DP3_{TILE}_DTM.tif")
chm = dsm - dtm

easting_origin  = int(TILE.split("_")[0])
northing_origin = int(TILE.split("_")[1])

rows = np.arange(0, 1000, STEP)
cols = np.arange(0, 1000, STEP)
R, C = np.meshgrid(rows, cols, indexing="ij")

# UTM coordinates (metres)
utm_e = (easting_origin  + C * 1.0).ravel()
utm_n = (northing_origin + R * 1.0).ravel()

dtm_flat  = dtm[R, C].ravel()
dsm_flat  = dsm[R, C].ravel()
chm_flat  = chm[R, C].ravel()

valid = np.isfinite(dtm_flat) & np.isfinite(dsm_flat)
utm_e    = utm_e[valid];   utm_n    = utm_n[valid]
dtm_flat = dtm_flat[valid]; dsm_flat = dsm_flat[valid]
chm_flat = chm_flat[valid]

# Convert UTM → WGS84
lons, lats = warp_transform(crs, "EPSG:4326", utm_e, utm_n)
lons = np.array(lons); lats = np.array(lats)

print(f"Grid: {len(dtm_flat)} valid 10-m pixels")

# ---------------------------------------------------------------------------
# Sample observations (same logic as neon_experiment.py)
# ---------------------------------------------------------------------------
all_idx = np.arange(len(dtm_flat))
veg_idx = np.where(chm_flat > 5.0)[0]

rtk_idx = rng.choice(all_idx, size=min(N_RTK, len(all_idx)), replace=False)

b_pool  = np.setdiff1d(veg_idx, rtk_idx)
b_idx   = rng.choice(b_pool, size=min(N_B, len(b_pool)), replace=False)

used      = set(b_idx.tolist()) | set(rtk_idx.tolist())
remaining = np.array([i for i in all_idx if i not in used])
a_idx     = rng.choice(remaining, size=min(N_A, len(remaining)), replace=False)

elev_mean = float(np.mean(dtm_flat))
print(f"Elevation mean (normalisation offset): {elev_mean:.3f} m")

# ---------------------------------------------------------------------------
# Add noise
# ---------------------------------------------------------------------------
y_rtk = dtm_flat[rtk_idx] + rng.normal(0, SIGMA_RTK, size=len(rtk_idx))
y_a   = dtm_flat[a_idx]   + rng.normal(0, SIGMA_A,   size=len(a_idx))
y_b   = dsm_flat[b_idx]   # no extra noise; one-sided canopy bias is the signal

# ---------------------------------------------------------------------------
# Write RTK file
# ---------------------------------------------------------------------------
rtk_path = BASE / "rtk_observations.csv"
with open(rtk_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "point_id", "easting_m", "northing_m", "lon_wgs84", "lat_wgs84",
        "elevation_obs_m", "sigma_m", "sensor_type"
    ])
    for k, i in enumerate(rtk_idx):
        writer.writerow([
            f"RTK_{k+1:02d}",
            f"{utm_e[i]:.1f}",
            f"{utm_n[i]:.1f}",
            f"{lons[i]:.6f}",
            f"{lats[i]:.6f}",
            f"{y_rtk[k]:.4f}",
            f"{SIGMA_RTK:.3f}",
            "RTK-GPS"
        ])
print(f"Wrote {len(rtk_idx)} RTK observations → {rtk_path}")

# ---------------------------------------------------------------------------
# Write GPS/Sensor-A file
# ---------------------------------------------------------------------------
gps_path = BASE / "gps_observations.csv"
with open(gps_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "point_id", "easting_m", "northing_m", "lon_wgs84", "lat_wgs84",
        "elevation_obs_m", "sigma_m", "sensor_type"
    ])
    for k, i in enumerate(a_idx):
        writer.writerow([
            f"GPS_{k+1:03d}",
            f"{utm_e[i]:.1f}",
            f"{utm_n[i]:.1f}",
            f"{lons[i]:.6f}",
            f"{lats[i]:.6f}",
            f"{y_a[k]:.4f}",
            f"{SIGMA_A:.3f}",
            "GPS"
        ])
print(f"Wrote {len(a_idx)} GPS observations → {gps_path}")

# ---------------------------------------------------------------------------
# Write Lidar/Sensor-B file
# ---------------------------------------------------------------------------
lidar_path = BASE / "lidar_observations.csv"
with open(lidar_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "point_id", "easting_m", "northing_m", "lon_wgs84", "lat_wgs84",
        "dsm_obs_m", "chm_true_m", "sensor_type"
    ])
    for k, i in enumerate(b_idx):
        writer.writerow([
            f"LIDAR_{k+1:03d}",
            f"{utm_e[i]:.1f}",
            f"{utm_n[i]:.1f}",
            f"{lons[i]:.6f}",
            f"{lats[i]:.6f}",
            f"{y_b[k]:.4f}",
            f"{chm_flat[i]:.4f}",
            "Lidar-DSM"
        ])
print(f"Wrote {len(b_idx)} lidar observations → {lidar_path}")

# ---------------------------------------------------------------------------
# Write metadata JSON (for experiment script to load consistently)
# ---------------------------------------------------------------------------
meta = {
    "tile":              TILE,
    "crs":               "EPSG:32618",
    "step_m":            STEP,
    "seed":              SEED,
    "elev_mean_m":       round(elev_mean, 3),
    "sigma_rtk_m":       SIGMA_RTK,
    "sigma_a_m":         SIGMA_A,
    "n_rtk":             len(rtk_idx),
    "n_gps":             len(a_idx),
    "n_lidar":           len(b_idx),
    "rtk_file":          "rtk_observations.csv",
    "gps_file":          "gps_observations.csv",
    "lidar_file":        "lidar_observations.csv",
    "bounds_utm": {
        "left":   easting_origin,
        "bottom": northing_origin,
        "right":  easting_origin  + 1000,
        "top":    northing_origin + 1000
    },
    "bounds_wgs84": {
        "lon_min": round(float(np.min(lons)), 5),
        "lon_max": round(float(np.max(lons)), 5),
        "lat_min": round(float(np.min(lats)), 5),
        "lat_max": round(float(np.max(lats)), 5)
    }
}
meta_path = BASE / "experiment_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Wrote metadata → {meta_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"RTK:   n={len(rtk_idx)}, elev range {y_rtk.min():.1f}–{y_rtk.max():.1f} m asl")
print(f"GPS:   n={len(a_idx)},  elev range {y_a.min():.1f}–{y_a.max():.1f} m asl")
print(f"Lidar: n={len(b_idx)},  DSM range  {y_b.min():.1f}–{y_b.max():.1f} m asl")
print(f"Normalisation offset: {elev_mean:.3f} m")
