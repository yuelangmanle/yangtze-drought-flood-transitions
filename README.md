# Yangtze River Basin Sequential Drought-to-Flood Transition Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab.svg)](requirements.txt)
[![Interactive Portal](https://img.shields.io/badge/Portal-GitHub%20Pages-0f766e.svg)](https://yuelangmanle.github.io/yangtze-drought-flood-transitions/)

Reproducible dataset archive and Python diagnostic suite for sequential drought-to-flood abrupt alternation (DFAA) transitions across the Yangtze River Basin (1961–2025), integrating HydroBASINS Level-6 fractional topology, gridded precipitation, temperature, and Penman–Monteith potential evapotranspiration (`CRU TS v4.10`), independent precipitation validation (`GPCC Monthly v2025`), Datong mainstem streamflow (`1961–2024`), atmospheric reanalyses (`NCEP/NCAR Reanalysis 1`, `NCEP-DOE Reanalysis 2`), and NOAA seasonal climate indices.

**Interactive Project Explorer & Event Catalog:** [https://yuelangmanle.github.io/yangtze-drought-flood-transitions/](https://yuelangmanle.github.io/yangtze-drought-flood-transitions/)

## Key Quantitative Findings (1961–2025)

| Diagnostic Scope | Primary Result | Verification & Bounds |
| :--- | :--- | :--- |
| **Basin-wide chronology (`SPI-1`)** | **39 transitions** (`0.60 events yr⁻¹`), **10 (25.6%) cross-year** | Flood season `May–Sep`: 16 (41.0%); cool half-year (`Oct–Apr`): 23 (59.0%; `Dec–Feb` 13, shoulder 10) |
| **Tributary masking & physical excess** | Peak tributary `F_w` amplifies basin mean by **1.84×** (`F_w < 1.0`) vs. **1.24×** (`F_w ≥ 1.0`) | MW *p* = 1.8 × 10⁻⁴; uncoupled level `r_s = +0.689` (*p* = 1.0 × 10⁻⁶), log–log slope `β = 0.622`; physical excess **3.29×** (`14.23 km³`) |
| **Severity independence & power** | No detectable `S_d–F_w` scaling across basin, subbasins, or 748 grid cells | Basin *τ* = −0.139 (*p* = 0.213); 80%-power MDE \|*τ*\| = 0.313 (\|*r*\| = 0.436); `0/748` BH-FDR `q < 0.05` (spatial `AR(1)` field *p* = 0.183) |
| **Prospective month-2 transitions** | 4-mo `F_w ≥ 1.0` rates: **44.7%** Basin, **51.1%** Upper, **55.1%** Middle, **47.8%** Lower | Matches seasonally weighted non-wet-origin base (`46.4–48.2%`) & seasonal `AR(1)`/`AR(2)` nulls |
| **Datong hydrometric filtering** | `F_w < 1.0`: **−0.409σ** (*p* = 0.031) vs. `F_w ≥ 1.0`: **+0.402σ** (*p* = 0.016); Flood `F_w ≥ 1.0`: **+0.489σ** | Matched non-DFAA ordinary wet runs: similar `Q_wet` (*p* = 0.204), larger `ΔQ_z` swing (**+0.626σ** for `F_w ≥ 1.0`) |
| **Synoptic circulation & teleconnections** | Surface-bounded `ΔDIV_ps = −1.136σ` (`−1.10 mm day⁻¹`, *p* = 0.0001); `ΔDI500 = +1.179σ` (*p* = 0.0001) | Exceeds matched wet controls (`DI500 +0.464σ`, `EAP850 −0.678σ`), `NCEP-DOE-2 DIV_ps` (*p* = 0.0001); causal `ONI/SOI/NAO LOOCV AUC = 0.312–0.325` |

## Repository Structure

```text
yangtze-drought-flood-transitions/
├── run_pipeline.py                 # Full raw execution (--raw-dir) & derived CSV mode (--derived-only)
├── requirements.txt
├── CITATION.cff
├── scripts/
│   └── fetch_public_data.py        # Helper to download public PSL/CPC subsets & Natural Earth coastlines
├── src/
│   ├── basin.py                    # HydroBASINS Level-6 fractional topology & cos(lat) area weighting
│   ├── spi.py                      # Mixed-Gamma SPI (0.5 q_0), PM/Thornthwaite GLO SPEI, run theory, AR(1)/AR(2) nulls
│   ├── dependence.py               # Marginals, 9 copulas + Genest CvM bootstrap, cluster prospective CIs, spatial AR(1) BH-FDR
│   ├── runoff.py                   # Datong monthly/pre-TGP standardization, ordinary wet controls, CRU stn & GPCC v2025 benchmark
│   ├── circulation.py              # p_s-bounded moisture flux, DI500/DI200/EAP850, detrended WNPSH, W flux, causal teleconnections
│   └── figures.py                  # Publication figures (Fig. 1–6 & ESM_Fig1)
├── data/                           # Derived CSV outputs, yangtze_mask.npz, circulation_fields.npz & SHA256SUMS.txt
├── figures/                        # Rendered PNG figures (600 dpi)
└── docs/                           # Static GitHub Pages interactive portal
```

## Quick Start & Reproducibility

All derived catalogs, summary tables, vector rings, and composite 2D circulation arrays are stored in `data/` with SHA-256 verification checksums (`data/SHA256SUMS.txt`).

```bash
git clone https://github.com/yuelangmanle/yangtze-drought-flood-transitions.git
cd yangtze-drought-flood-transitions
pip install -r requirements.txt

# Verify derived CSV tables and regenerate Figs. 1–6 and ESM_Fig1 without raw archives
python3 run_pipeline.py --derived-only
```

### Full End-to-End Execution From Primary Archives
Obtain the primary grids (`cru_pre.nc.gz`, `cru_tmp.nc.gz`, `cru_pet.nc.gz`) from [CRU TS v4.10](https://crudata.uea.ac.uk/), pressure-level and surface fields from [NOAA PSL (`NCEP-1`, `NCEP-DOE-2`, `GPCC v2025`)](https://psl.noaa.gov/), and HydroBASINS Asia Level-6 from [HydroSHEDS](https://www.hydrosheds.org/), then run:

```bash
python3 scripts/fetch_public_data.py --raw-dir /path/to/raw
python3 run_pipeline.py --raw-dir /path/to/raw --mask data/yangtze_mask.npz --out-dir . --formats png
```

## Derived Data Inventory (`data/`)

| File | Description |
| :--- | :--- |
| `event_catalog.csv` | Complete chronological catalog of 39 basin-wide transitions (`S_d`, `F_w`, `J_w`, season, Datong `Q_z` and `Q_z_preTGP`) |
| `tributary_amplification.csv` & `tributary_summary.csv` | Basin vs. synchronous peak tributary `F_w`, physical depth (`mm`), volume (`km³`), and uncoupled scaling tests |
| `subregion_summary.csv` & `seasonal_dependence.csv` | Whole-basin, Upper/Middle/Lower, and seasonal/duration/era `S_d–F_w` statistics, MDE bounds, and `Q_50`/`Q_90` slopes |
| `marginal_fits.csv` & `copula_fits.csv` | Threshold-anchored marginal fits (bootstrap KS) and 9-copula comparisons (Genest et al. 2009 CvM tests) |
| `prospective_skill.csv` & `tributary_prospective_skill.csv` | Forward month-2 transitions across All/Flood/Cool seasons for 4 regions and 7 tributaries with cluster CIs |
| `sensitivity_schemes.csv` & `index_robustness.csv` | 40-rule threshold/gap/severity matrix plus `SPI-3`, Penman–Monteith & Thornthwaite `SPEI-1/3`, and bridge checks |
| `null_surrogates.csv` & `annual_trends_dispersion.csv` | 4-region white-noise, seasonal `AR(1)`/`AR(2)`, and block surrogates; dispersion `χ²` and Quasi-Poisson GLM |
| `datong_strata.csv` & `datong_ordinary_wet_benchmark.csv` | Reservoir-era discharge stratification (`1961–2024`) and matched non-DFAA ordinary wet-spell paired controls |
| `station_density.csv` & `cru_gpcc_validation.csv` | `1961–2019` vs. `2020–2025` CRU `stn` audit and multi-domain benchmark against `GPCC v2025` |
| `circulation_summary.csv`, `circulation_windows.csv`, & `circulation_metadata.csv` | `DIV_ps` (`σ` & `mm day⁻¹`), `DIV_1000hPa`, `DI500`, `DI200`, terrain-masked `EAP850`, lags, and detrended WNPSH longitudes |
| `circulation_matched_controls.csv` & `reanalysis_verification.csv` | Strict (`tol=0.1`) & all-39 matched wet controls (`pool_perm_p`, `q_FDR`), OOS deciles, `PL_west`, and `NCEP-DOE-2 DIV_ps` |
| `spatial_field_grid.csv` & `spatial_field_summary.csv` | 748-grid-cell `τ`, BH-FDR `q`-values, and 1,000-realization spatially coherent seasonal `AR(1)` field null |
| `whiplash_comparison.csv` & `seasonal_indices_skill.csv` | Interannual precipitation whiplash overlap (`Jaccard = 0.079`) and strictly causal DJF/MAM ONI–SOI–NAO LOOCV AUC |

## Citation & License

Source code is licensed under the [MIT License](LICENSE). See [CITATION.cff](CITATION.cff) for software and dataset citation metadata.
