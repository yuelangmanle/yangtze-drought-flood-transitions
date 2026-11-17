# Yangtze River Basin Sequential Drought-to-Flood Transition Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab.svg)](requirements.txt)
[![Interactive Portal](https://img.shields.io/badge/Portal-GitHub%20Pages-0f766e.svg)](https://yuelangmanle.github.io/yangtze-drought-flood-transitions/)

Reproducible dataset archive and Python diagnostic suite for sequential drought-to-flood abrupt alternation (DFAA) transitions across the Yangtze River Basin (1961–2025), integrating multi-scale HydroBASINS Level-6 topology, gridded precipitation and temperature (`CRU TS v4.10`), observed mainstem runoff at Datong Station (`1961–2024`), atmospheric reanalysis (`NCEP/NCAR Reanalysis 1`, `NCEP-DOE Reanalysis 2`), and NOAA seasonal climate indices.

**Interactive Project Explorer & Event Catalog:** [https://yuelangmanle.github.io/yangtze-drought-flood-transitions/](https://yuelangmanle.github.io/yangtze-drought-flood-transitions/)

## Key Quantitative Findings (1961–2025)

| Diagnostic Scope | Primary Result | Verification & Bounds |
| :--- | :--- | :--- |
| **Basin-wide chronology (`SPI-1`)** | **39 transitions** (`0.60 events yr⁻¹`), **10 (25.6%) cross-year** | Flood season `May–Sep`: 16 (41.0%); `Dec–Feb`: 13 (33.3%); shoulder: 10 (25.6%) |
| **Tributary masking & physical excess** | Peak tributary `F_w` amplifies basin mean by **1.82×** (`F_w < 1.0`) vs. **1.23×** (`F_w ≥ 1.0`) | `r_s = −0.592` (*p* = 7.3 × 10⁻⁵); physical rainfall excess median **3.26×** (mm) |
| **Severity independence & power** | No detectable `S_d–F_w` scaling across basin, subbasins, or 748 grid cells | Basin *τ* = −0.123 (*p* = 0.271); 80%-power MDE \|*r*\| = 0.436; `0/748` BH-FDR `q < 0.05` |
| **Prospective month-2 transitions** | 4-mo `F_w ≥ 1.0` rates: **42.6%** Basin, **52.1%** Upper, **55.1%** Middle | Matches unconditional 4-mo base (`48.6–49.5%`) & seasonal `AR(1)` null (`50.2%`) |
| **Datong hydrometric filtering** | `F_w < 1.0`: **−0.406σ** (*p* = 0.018) vs. `F_w ≥ 1.0`: **+0.440σ** (*p* = 0.011) | `2003–2012`: −0.787σ (`n = 5`); `2013–2024` `F_w ≥ 1.0`: +0.743σ (`n = 4`) |
| **Synoptic circulation signatures** | Surface-bounded `ΔDIV_ps* = −1.198σ` (*p* = 0.0001); `ΔDI500 = +1.080σ` (*p* = 0.0001) | Exceeds matched wet controls (`ΔDI500 +0.402σ`, *p* = 0.022), `1993–2025` OOS (*p* = 0.0037), `NCEP-DOE-2` (*p* = 0.0001) |

## Repository Structure

```text
yangtze-drought-flood-transitions/
├── run_pipeline.py                 # Full raw execution (--raw-dir) & derived CSV mode (--derived-only)
├── requirements.txt
├── CITATION.cff
├── scripts/
│   └── fetch_public_data.py        # Helper to download public PSL/CPC subsets & Natural Earth coastlines
├── src/
│   ├── basin.py                    # HydroBASINS Level-6 tracing & cosine-area weighting
│   ├── spi.py                      # Positive-Gamma SPI, Thornthwaite-GLO SPEI, run theory, AR(1) nulls
│   ├── dependence.py               # Marginals, 5 copula models, prospective skill, 748-cell BH-FDR
│   ├── runoff.py                   # Datong monthly standardization, era strata, Quasi-Poisson GLM, CRU stn
│   ├── circulation.py              # p_s-bounded moisture flux, DI500/DI200/EAP850, Takaya-Nakamura W flux
│   └── figures.py                  # Publication figures (Fig. 1–6 & Fig. S1)
├── data/                           # Derived CSV outputs, yangtze_mask.npz, & SHA256SUMS.txt
├── figures/                        # Rendered PNG figures (600 dpi)
└── docs/                           # Static GitHub Pages interactive portal
```

## Quick Start & Reproducibility

### 1. Verify Results & Regenerate Diagnostic Figures Without Raw Archives
All derived catalogs and summary tables are stored in `data/` with cryptographic verification checksums (`data/SHA256SUMS.txt`).

```bash
git clone https://github.com/yuelangmanle/yangtze-drought-flood-transitions.git
cd yangtze-drought-flood-transitions
pip install -r requirements.txt

# Verify derived CSV tables and regenerate Figs. 2–5
python3 run_pipeline.py --derived-only
```

### 2. Full End-to-End Execution From Primary Archives
Obtain the primary precipitation/temperature grids (`cru_pre.nc.gz`, `cru_tmp.nc.gz`) from [CRU TS v4.10](https://crudata.uea.ac.uk/), pressure-level NCEP/NCAR Reanalysis 1 monthly fields from [NOAA PSL](https://psl.noaa.gov/), and HydroBASINS Asia Level-6 from [HydroSHEDS](https://www.hydrosheds.org/). Then fetch remaining subsets automatically and run:

```bash
python3 scripts/fetch_public_data.py --raw-dir /path/to/raw
python3 run_pipeline.py --raw-dir /path/to/raw --mask data/yangtze_mask.npz --out-dir . --formats png
```

## Derived Data Inventory (`data/`)

| File | Description |
| :--- | :--- |
| `event_catalog.csv` | Complete chronological catalog of 39 basin-wide transitions (1961–2025) with Datong anomalies |
| `tributary_amplification.csv` | Basin vs. synchronous peak tributary `F_w`, SPI ratio, and physical rainfall excess (`mm`, `%`) |
| `subregion_summary.csv` | Whole-basin, Upper, Middle, and Lower event counts, correlation tests, and 80%-power MDE |
| `marginal_fits.csv` & `copula_fits.csv` | Parametric marginal screening (6 distributions) and copula AICc / CvM comparisons |
| `prospective_skill.csv` | Forward transitions from month-2 trigger confirmation with 95% binomial CIs & base rates |
| `sensitivity_schemes.csv` & `index_robustness.csv` | 36-scheme threshold matrix plus `SPI-3`, `SPEI-1`, `SPEI-3`, and 1-month bridge checks |
| `null_surrogates.csv` & `annual_trends_dispersion.csv` | White-noise, seasonal `AR(1)`, and year-block permutations; dispersion `χ²` and Quasi-Poisson GLM |
| `datong_strata.csv` & `station_density.csv` | Reservoir-era discharge stratification (`1961–2024`) and CRU TS `stn` gauge-density summary |
| `circulation_summary.csv` & `circulation_windows.csv` | `DIV_ps*`, `DIV_1000`, `DI500`, `DI200`, and `EAP850` permutation, LOO, BH-FDR, and lag results |
| `circulation_matched_controls.csv` & `reanalysis_verification.csv` | Ordinary wet-month controls (`n = 35/15`), split-period OOS checks, and `NCEP-DOE-2` verification |
| `spatial_field_grid.csv` & `spatial_field_summary.csv` | 748-grid-cell `τ`, BH-FDR `q`-values, and synchronous calendar-year field-permutation null |
| `whiplash_comparison.csv` & `seasonal_indices_skill.csv` | Interannual precipitation whiplash overlap (`Jaccard = 0.079`) and DJF/MAM ONI–SOI–NAO LOOCV AUC |

## Citation & License

Source code is licensed under the [MIT License](LICENSE). See [CITATION.cff](CITATION.cff) for software and dataset citation metadata.
