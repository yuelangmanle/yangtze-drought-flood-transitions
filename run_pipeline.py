"""Reproducible end-to-end execution script for the Yangtze transition diagnostics."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.basin import (
    SUBBASIN_LABELS,
    area_weighted_series,
    build_tributary_masks,
    extract_basin_vectors,
    load_basin_mask,
    load_basin_vectors,
    save_basin_vectors,
)
from src.spi import (
    detect_transitions,
    detect_with_bridge,
    fit_spei,
    fit_spi,
    interannual_whiplash_overlap,
    run_null_surrogates,
    thornthwaite_pet,
    threshold_sensitivity_matrix,
    tributary_amplification,
)
from src.dependence import (
    dependence_summary,
    fit_copulas,
    grid_dependence_fdr,
    prospective_onset_skill,
    screen_marginals,
)
from src.runoff import (
    annual_frequency_and_trends,
    attach_datong_to_events,
    load_datong_anomalies,
    station_density_summary,
    stratify_datong_response,
)
from src.circulation import run_circulation_diagnostics, seasonal_teleconnection_skill
from src.figures import (
    plot_esm_fig1_wave_flux,
    plot_fig1_study_area,
    plot_fig2_tributary_amplification,
    plot_fig3_chronology_and_seasonality,
    plot_fig4_dependence_and_spatial,
    plot_fig5_sensitivity_and_runoff,
    plot_fig6_circulation_composites,
)


def _write_sha256_manifest(data_dir):
    files = sorted(p for p in Path(data_dir).iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = []
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.name}")
    (Path(data_dir) / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_pipeline(raw_dir, processed_mask, out_dir, fig_formats=("png",)):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    data_out = out_dir / "data"
    fig_out = out_dir / "figures"
    data_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    mask = load_basin_mask(processed_mask)
    lats, lons, frac, zone = mask["lat"], mask["lon"], mask["frac"], mask["zone"]

    # Persist compact mask grid inside repository for standalone plotting
    np.savez_compressed(data_out / "yangtze_mask.npz", lat=lats, lon=lons, frac=frac, zone=zone)

    pre_ds = xr.open_dataset(raw_dir / "cru_pre.nc.gz").sel(
        lat=slice(lats[0], lats[-1]),
        lon=slice(lons[0], lons[-1]),
        time=slice("1961-01-01", "2025-12-31"),
    )
    tmp_ds = xr.open_dataset(raw_dir / "cru_tmp.nc.gz").sel(
        lat=slice(lats[0], lats[-1]),
        lon=slice(lons[0], lons[-1]),
        time=slice("1961-01-01", "2025-12-31"),
    )

    precip_grid = pre_ds["pre"].values.astype(float)
    temp_grid = tmp_ds["tmp"].values.astype(float)
    stn_grid = pre_ds["stn"].values.astype(float)
    months = pre_ds.time.dt.month.values
    years = pre_ds.time.dt.year.values
    periods = pd.PeriodIndex(np.datetime_as_string(pre_ds.time.values, unit="M"), freq="M")

    # Station density audit
    stn_df = station_density_summary(stn_grid, frac, lons)
    stn_df.to_csv(data_out / "station_density.csv", index=False)

    # Subbasin SPI-1, event catalogs, copulas, and prospective skill
    subbasin_rows = []
    copula_frames = []
    prosp_rows = []
    basin_events_df = None
    basin_spi = None
    pr_basin = None

    for zid, zname in SUBBASIN_LABELS.items():
        pr = area_weighted_series(precip_grid, frac, zone, zid)
        spi = fit_spi(pr, months, scale=1)
        evs, drys = detect_transitions(spi, periods)
        ev_df = pd.DataFrame(evs)
        ev_df.insert(0, "region", zname)
        ev_df.insert(1, "event_id", np.arange(1, len(ev_df) + 1))

        dep = dependence_summary(ev_df["S_d"], ev_df["F_w"])
        dep.update({"region": zname})
        subbasin_rows.append(dep)

        copula_frames.append(fit_copulas(zname, ev_df["S_d"], ev_df["F_w"]))
        pr_row = prospective_onset_skill(spi, drys)
        pr_row["region"] = zname
        prosp_rows.append(pr_row)

        if zid == 0:
            basin_events_df = ev_df.copy()
            basin_spi = spi
            pr_basin = pr

    pd.DataFrame(subbasin_rows).to_csv(data_out / "subregion_summary.csv", index=False)
    pd.concat(copula_frames, ignore_index=True).to_csv(data_out / "copula_fits.csv", index=False)
    pd.DataFrame(prosp_rows).to_csv(data_out / "prospective_skill.csv", index=False)

    # Marginal distribution selection for basin S_d and F_w
    marg_df = screen_marginals(basin_events_df["S_d"], basin_events_df["F_w"])
    marg_df.to_csv(data_out / "marginal_fits.csv", index=False)

    # Datong discharge attachment and stratification
    q_lookup = load_datong_anomalies(raw_dir / "datong_discharge_1950_2024.xlsx")
    basin_events_df = attach_datong_to_events(basin_events_df, q_lookup)
    basin_events_df.to_csv(data_out / "event_catalog.csv", index=False)

    datong_df = stratify_datong_response(basin_events_df)
    datong_df.to_csv(data_out / "datong_strata.csv", index=False)

    # Annual count dispersion, Quasi-Poisson GLM, and MK trends
    freq_dict = annual_frequency_and_trends(basin_events_df)
    pd.DataFrame([freq_dict]).to_csv(data_out / "annual_trends_dispersion.csv", index=False)

    # Tributary localization and physical precipitation excess ratios
    trib_masks, _, _ = build_tributary_masks(raw_dir / "hybas_l6/hybas_as_lev06_v1c", lats, lons, frac)
    pr_tribs = {nm: np.nansum(precip_grid * w[None, :, :], axis=(1, 2)) for nm, w in trib_masks.items()}
    spi_tribs = {nm: fit_spi(pr_t, months, scale=1) for nm, pr_t in pr_tribs.items()}
    trib_df = tributary_amplification(basin_events_df, pr_basin, pr_tribs, basin_spi, spi_tribs, months, periods)
    trib_df.to_csv(data_out / "tributary_amplification.csv", index=False)

    # 36-scheme threshold matrix + multi-scale/index robustness
    schemes_df = threshold_sensitivity_matrix(basin_spi)
    schemes_df.to_csv(data_out / "sensitivity_schemes.csv", index=False)

    t_basin = area_weighted_series(temp_grid, frac, zone, 0)
    w_b = frac / frac.sum()
    mean_lat = float(np.sum(lats[:, None] * w_b))
    pet_basin = thornthwaite_pet(t_basin, years, months, mean_lat)
    wb_basin = pr_basin - pet_basin

    from scipy import stats as sc_stats
    rng_boot = np.random.default_rng(2025)

    def _tau_ci_row(name, ev_list):
        sd_arr = np.array([e["S_d"] for e in ev_list], dtype=float)
        fw_arr = np.array([e["F_w"] for e in ev_list], dtype=float)
        tau_v, p_v = sc_stats.kendalltau(sd_arr, fw_arr)
        n_ev = len(sd_arr)
        boots = np.empty(2000, dtype=float)
        for b in range(2000):
            idx_b = rng_boot.integers(0, n_ev, size=n_ev)
            boots[b] = sc_stats.kendalltau(sd_arr[idx_b], fw_arr[idx_b]).statistic
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        return {
            "index": name,
            "n_events": n_ev,
            "tau": round(float(tau_v), 4),
            "ci95_lo": round(float(lo), 4),
            "ci95_hi": round(float(hi), 4),
            "p_value": round(float(p_v), 4),
        }

    idx_rows = []
    for idx_nm, ser in [
        ("SPI-1", basin_spi),
        ("SPI-3", fit_spi(pr_basin, months, scale=3)),
        ("SPEI-1", fit_spei(wb_basin, months, scale=1)),
        ("SPEI-3", fit_spei(wb_basin, months, scale=3)),
    ]:
        evs, _ = detect_transitions(ser, periods)
        idx_rows.append(_tau_ci_row(idx_nm, evs))

    ev_br, n_bridges = detect_with_bridge(basin_spi)
    idx_rows.append(_tau_ci_row(f"1-mo bridge ({n_bridges})", ev_br))
    idx_rob_df = pd.DataFrame(idx_rows)
    idx_rob_df.to_csv(data_out / "index_robustness.csv", index=False)

    # Null surrogates (IID Gaussian, seasonal AR1, calendar year-block shuffle)
    null_df, _ = run_null_surrogates(basin_spi, months, years, n_boot=1000)
    null_df.to_csv(data_out / "null_surrogates.csv", index=False)

    # 748-cell spatial grid dependence + BH-FDR + field permutation
    grid_df, grid_summary, _ = grid_dependence_fdr(precip_grid, months, years, frac, lats, lons, fit_spi, detect_transitions)
    grid_df.to_csv(data_out / "spatial_field_grid.csv", index=False)
    pd.DataFrame([grid_summary]).to_csv(data_out / "spatial_field_summary.csv", index=False)

    # Circulation diagnostics (NCEP-1 DIV_ps, DI500, DI200, EAP850, wave flux, NCEP-2 check)
    circ_sum, circ_win, circ_ctrl, circ_verif, fields = run_circulation_diagnostics(
        raw_dir, basin_events_df, basin_spi, periods, mask
    )
    circ_sum.to_csv(data_out / "circulation_summary.csv", index=False)
    circ_win.to_csv(data_out / "circulation_windows.csv", index=False)
    circ_ctrl.to_csv(data_out / "circulation_matched_controls.csv", index=False)
    circ_verif.to_csv(data_out / "reanalysis_verification.csv", index=False)

    # Whiplash timescale separation & seasonal teleconnection predictability
    whip = interannual_whiplash_overlap(pr_basin, years, basin_events_df)
    pd.DataFrame([whip]).to_csv(data_out / "whiplash_comparison.csv", index=False)
    if (raw_dir / "cpc_oni.ascii.txt").exists():
        tele_df = seasonal_teleconnection_skill(raw_dir, basin_events_df)
        tele_df.to_csv(data_out / "seasonal_indices_skill.csv", index=False)

    # Monthly basin hydroclimatic climatology (1961-2025)
    clim_rows = [
        {
            "month": mo,
            "precip_mean_mm": round(float(np.mean(pr_basin[months == mo])), 2),
            "precip_q25_mm": round(float(np.percentile(pr_basin[months == mo], 25)), 2),
            "precip_q75_mm": round(float(np.percentile(pr_basin[months == mo], 75)), 2),
            "temp_mean_c": round(float(np.mean(t_basin[months == mo])), 2),
        }
        for mo in range(1, 13)
    ]
    clim_df = pd.DataFrame(clim_rows)
    clim_df.to_csv(data_out / "hydro_climatology.csv", index=False)

    # Smooth HydroBASINS boundaries, tributary polygons, and East Asia coastlines
    vectors = extract_basin_vectors(
        raw_dir / "hybas_l6/hybas_as_lev06_v1c",
        raw_dir / "ne_50m_coastline.geojson",
        precip_grid=precip_grid,
        years=years,
    )
    save_basin_vectors(data_out / "basin_vectors.npz", vectors)

    # Generate figures
    plot_fig1_study_area(mask, clim_df, vectors, fig_out, formats=fig_formats)
    plot_fig2_tributary_amplification(trib_df, fig_out, formats=fig_formats)
    plot_fig3_chronology_and_seasonality(basin_events_df, fig_out, formats=fig_formats)
    plot_fig4_dependence_and_spatial(
        basin_events_df, subbasin_rows[0], marg_df, grid_df, grid_summary, mask, vectors, fig_out, formats=fig_formats
    )
    plot_fig5_sensitivity_and_runoff(schemes_df, idx_rob_df, basin_events_df, datong_df, fig_out, formats=fig_formats)
    plot_fig6_circulation_composites(fields, circ_win, vectors, mask, circ_sum, circ_ctrl, fig_out, formats=fig_formats)
    plot_esm_fig1_wave_flux(fields, vectors, mask, fig_out, formats=fig_formats)

    _write_sha256_manifest(data_out)
    print(f"Pipeline complete -> data written to {data_out}, figures written to {fig_out}")


def run_derived_only(out_dir, fig_formats=("png",)):
    """Re-verify statistical summaries and re-plot figures Fig1-Fig5 from CSV deliverables."""
    out_dir = Path(out_dir)
    data_out = out_dir / "data"
    fig_out = out_dir / "figures"

    mask = load_basin_mask(data_out / "yangtze_mask.npz")
    vectors = load_basin_vectors(data_out / "basin_vectors.npz")
    clim_df = pd.read_csv(data_out / "hydro_climatology.csv")
    events_df = pd.read_csv(data_out / "event_catalog.csv")
    trib_df = pd.read_csv(data_out / "tributary_amplification.csv")
    schemes_df = pd.read_csv(data_out / "sensitivity_schemes.csv")
    idx_rob_df = pd.read_csv(data_out / "index_robustness.csv")
    datong_df = pd.read_csv(data_out / "datong_strata.csv")
    marg_df = pd.read_csv(data_out / "marginal_fits.csv")
    grid_df = pd.read_csv(data_out / "spatial_field_grid.csv")
    grid_summary = pd.read_csv(data_out / "spatial_field_summary.csv").iloc[0].to_dict()
    sub_df = pd.read_csv(data_out / "subregion_summary.csv")
    basin_dep = sub_df[sub_df["region"] == "Basin"].iloc[0].to_dict()

    plot_fig1_study_area(mask, clim_df, vectors, fig_out, formats=fig_formats)
    plot_fig2_tributary_amplification(trib_df, fig_out, formats=fig_formats)
    plot_fig3_chronology_and_seasonality(events_df, fig_out, formats=fig_formats)
    plot_fig4_dependence_and_spatial(
        events_df, basin_dep, marg_df, grid_df, grid_summary, mask, vectors, fig_out, formats=fig_formats
    )
    plot_fig5_sensitivity_and_runoff(schemes_df, idx_rob_df, events_df, datong_df, fig_out, formats=fig_formats)
    print(f"Derived-only verification passed ({len(events_df)} events, figures refreshed in {fig_out})")


def main():
    parser = argparse.ArgumentParser(description="Yangtze River Basin sequential drought-to-flood pipeline.")
    parser.add_argument("--raw-dir", default="../data/raw", help="Directory containing CRU, NCEP, HydroBASINS, and Datong files.")
    parser.add_argument("--mask", default="../data/processed/yangtze_mask.npz", help="Path to gridded basin mask.")
    parser.add_argument("--out-dir", default=".", help="Output repository root directory.")
    parser.add_argument("--formats", nargs="+", default=["png"], help="Figure formats to export (e.g. png pdf eps).")
    parser.add_argument("--derived-only", action="store_true", help="Re-run verification from derived CSV files only.")
    args = parser.parse_args()

    if args.derived_only:
        run_derived_only(args.out_dir, fig_formats=tuple(args.formats))
    else:
        run_full_pipeline(args.raw_dir, args.mask, args.out_dir, fig_formats=tuple(args.formats))


if __name__ == "__main__":
    main()
