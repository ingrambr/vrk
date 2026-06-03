# Paper Experiments

Reproduction code and fixed observation data for:

> **"Integrating Spatial Data with Heterogeneous Error Characteristics
> in a Geostatistical Framework"**

---

## Requirements

```
numpy scipy rasterio
```

The `vrk` package must be importable. Place these scripts in the repository
root (alongside the `vrk/` directory), or add the `vrk/` path to `PYTHONPATH`.

---

## NEON raster data

The raw NEON AOP lidar rasters are **not** included here (each tile is ~50 MB).
Download the DSM and DTM products for Harvard Forest (HARV, Domain D01) from
the NEON Data Portal:

- **Product:** Discrete Return Lidar (DP3.30024.001)
- **Site:** HARV — Harvard Forest
- **Tile:** UTM Zone 18N, 727000\_4706000 (1 km²)
- **Collection year:** 2022
- **DOI:** https://doi.org/10.48443/2haw-yd70

Place the downloaded `.tif` files in `data/neon_harv/` before running any script.

Expected filenames:
```
data/neon_harv/NEON_D01_HARV_DP3_727000_4706000_DSM.tif
data/neon_harv/NEON_D01_HARV_DP3_727000_4706000_DTM.tif
```

---

## Fixed observation files (included)

| File | Contents |
|---|---|
| `rtk_observations.csv` | 15 RTK-GPS reference points |
| `gps_observations.csv` | 50 GPS / Sensor-A points |
| `lidar_observations.csv` | 200 lidar DSM / Sensor-B points |
| `experiment_meta.json` | Tile metadata and normalisation offset |

---

## Scripts

Run in order:

### 1. `neon_experiment.py`
Single-seed NEON pipeline (seed 42). Produces calibration diagnostics and
the figures used in Section 7 of the paper.

### 2. `neon_multiseed.py`
Multi-seed sensitivity analysis (10 seeds). Resamples observation locations
10 times and reports mean ± std for all methods across vegetated and full-grid
zones. Output saved to `data/neon_harv/neon_multiseed_results.json`.
This script produces the Table 6 numbers reported in the paper.

### 3. `hyperparameter_experiment.py`
Supplementary hyperparameter estimation experiment (Section 10.5 and Response
to Reviewer 1, Comment 2). Runs EP marginal-likelihood optimisation starting
from a misspecified covariance on the 1-D synthetic field.

---

## Expected runtimes (single CPU)

| Script | Approx. time |
|---|---|
| `neon_experiment.py` | ~2 min |
| `neon_multiseed.py` | ~20 min (10 seeds × 4 methods) |
| `hyperparameter_experiment.py` | ~5 min |
