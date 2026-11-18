"""End-to-end reproducible workflow for Yangtze River Basin drought-to-flood transitions."""

import argparse
import gzip
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
from src.circulation import run_circulation_diagnostics, seasonal_teleconnection_skill
from src.dependence import (
    bh_fdr,
    dependence_summary,
    fit_copulas,
    grid_dependence_fdr,
    prospective_onset_skill,
    screen_marginals,
)
from src.figures import (
    plot_esm_fig1_wave_flux,
    plot_fig1_study_area,
    plot_fig2_chronology_and_seasonality,
    plot_fig3_dependence_and_spatial,
    plot_fig4_sensitivity_and_runoff,
    plot_fig5_tributary_amplification,
    plot_fig6_circulation_composites,
)
from src.runoff import (
    annual_frequency_and_trends,
    attach_datong_to_events,
    benchmark_cru_vs_gpcc,
    benchmark_datong_ordinary_wet,
    load_datong_anomalies,
    station_density_summary,
    stratify_datong_response,
)
from src.spi import (
    detect_transitions,
    detect_with_bridge,
    fit_spei,
    fit_spi,
    interannual_whiplash_overlap,
    run_null_surrogates,
    summarize_tributary_amplification,
    threshold_sensitivity_matrix,
    tributary_amplification,
)


def _open_gz_nc(gz_path):
    """Decompress a gzipped NetCDF file into a temporary dataset loaded in memory."""
    with gzip.open(gz_path, "rb") as gz_in:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
            tf.write(gz_in.read())
            tmp_path = tf.name
    try:
        ds = xr.open_dataset(tmp_path)
        ds.load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return ds


def _resolve_existing(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def _bootstrap_tau_ci(sd, fw, n_boot=1000, seed=505):
    rng = np.random.default_rng(seed)
    sd_arr = np.asarray(sd, dtype=float)
    fw_arr = np.asarray(fw, dtype=float)
    n = len(sd_arr)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = float(stats.kendalltau(sd_arr[idx], fw_arr[idx]).statistic)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _write_sha256_manifest(data_dir):
    files = sorted(
        p for p in Path(data_dir).iterdir()
        if p.is_file() and p.name != "SHA256SUMS.txt" and not p.name.startswith(".")
    )
    lines = []
    for fpath in files:
        h = hashlib.sha256(fpath.read_bytes()).hexdigest()
        lines.append(f"{h}  {fpath.name}")
    (Path(data_dir) / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_figures(mask, vectors, circ_fields, tables, fig_dir, formats):
    """Generate main manuscript Figures 1-6 and Online Resource ESM_Fig1."""
    basin_dep = tables["subregion_summary"].loc[
        tables["subregion_summary"]["region"] == "Basin"
    ].iloc[0].to_dict()

    plot_fig1_study_area(mask, tables["hydro_climatology"], vectors, fig_dir, formats=formats)
    plot_fig2_chronology_and_seasonality(tables["event_catalog"], fig_dir, formats=formats)
    plot_fig3_dependence_and_spatial(
        tables["event_catalog"],
        basin_dep,
        tables["marginal_fits"],
        tables["spatial_field_grid"],
        tables["spatial_field_summary"].iloc[0].to_dict(),
        mask,
        vectors,
        fig_dir,
        formats=formats,
    )
    plot_fig4_sensitivity_and_runoff(
        tables["sensitivity_schemes"],
        tables["index_robustness"],
        tables["event_catalog"],
        tables["datong_strata"],
        fig_dir,
        formats=formats,
    )
    plot_fig5_tributary_amplification(
        tables["tributary_amplification"],
        fig_dir,
        formats=formats,
        trib_sum_df=tables["tributary_summary"],
    )
    plot_fig6_circulation_composites(
        circ_fields,
        tables["circulation_windows"],
        vectors,
        mask,
        tables["circulation_summary"],
        tables["circulation_matched_controls"],
        fig_dir,
        formats=formats,
    )
    plot_esm_fig1_wave_flux(circ_fields, vectors, mask, fig_dir, formats=formats)


def run_derived_only(mask_path, out_dir, fig_dir, formats):
    """Re-render figures directly from archived derived CSV and NPZ artifacts."""
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    mask = load_basin_mask(mask_path)
    vectors = load_basin_vectors(data_dir / "basin_vectors.npz")
    cf_npz = np.load(data_dir / "circulation_fields.npz", allow_pickle=False)
    circ_fields = {k: cf_npz[k] for k in cf_npz.files}

    table_names = [
        "hydro_climatology",
        "tributary_amplification",
        "tributary_summary",
        "event_catalog",
        "subregion_summary",
        "marginal_fits",
        "spatial_field_grid",
        "spatial_field_summary",
        "sensitivity_schemes",
        "index_robustness",
        "datong_strata",
        "circulation_windows",
        "circulation_summary",
        "circulation_matched_controls",
    ]
    tables = {name: pd.read_csv(data_dir / f"{name}.csv") for name in table_names}
    render_figures(mask, vectors, circ_fields, tables, fig_dir, formats)


def run_full_pipeline(args):
    """Execute complete physical, statistical, runoff, and circulation computations."""
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir) if args.fig_dir else (out_dir / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = _resolve_existing([args.raw_dir, out_dir / "data/raw", out_dir.parent / "data/raw"])
    mask_path = _resolve_existing([args.mask, data_dir / "yangtze_mask.npz", out_dir.parent / "data/processed/yangtze_mask.npz"])
    hybas_prefix = _resolve_existing([f"{args.hybas}.shp", raw_dir / "hybas_l6/hybas_as_lev06_v1c.shp"]).with_suffix("")
    coast_path = _resolve_existing([args.coast, raw_dir / "ne_50m_coastline.geojson"])

    mask = load_basin_mask(mask_path)
    lats, lons, frac, zone = mask["lat"], mask["lon"], mask["frac"], mask["zone"]
    np.savez(data_dir / "yangtze_mask.npz", lat=lats, lon=lons, frac=frac, zone=zone)

    print("[1/7] Loading CRU TS v4.10 pre, stn, tmp, and Penman-Monteith pet...")
    ds_pre = _open_gz_nc(raw_dir / "cru_pre.nc.gz")
    ds_tmp = _open_gz_nc(raw_dir / "cru_tmp.nc.gz")
    ds_pet = _open_gz_nc(raw_dir / "cru_pet.nc.gz")

    lat_sl = slice(float(lats[0]), float(lats[-1]))
    lon_sl = slice(float(lons[0]), float(lons[-1]))
    t_sl = slice("1961-01-01", "2025-12-31")

    sub_pre = ds_pre["pre"].sel(lat=lat_sl, lon=lon_sl, time=t_sl)
    sub_stn = ds_pre["stn"].sel(lat=lat_sl, lon=lon_sl, time=t_sl)
    sub_tmp = ds_tmp["tmp"].sel(lat=lat_sl, lon=lon_sl, time=t_sl)
    sub_pet = ds_pet["pet"].sel(lat=lat_sl, lon=lon_sl, time=t_sl)

    assert np.allclose(sub_pre.lat.values, lats) and np.allclose(sub_pre.lon.values, lons)
    assert np.allclose(sub_pet.lat.values, lats) and np.allclose(sub_pet.lon.values, lons)

    periods = pd.PeriodIndex(np.datetime_as_string(sub_pre.time.values, unit="M"), freq="M")
    years = np.array([p.year for p in periods], dtype=int)
    months = np.array([p.month for p in periods], dtype=int)
    days_per_mo = np.array([p.days_in_month for p in periods], dtype=float)

    pre = sub_pre.values.astype(float)
    stn = sub_stn.values.astype(float)
    tmp = sub_tmp.values.astype(float)
    pet_mm_grid = sub_pet.values.astype(float) * days_per_mo[:, None, None]

    station_df = station_density_summary(stn, frac, zone, lons, years=years)
    station_df.to_csv(data_dir / "station_density.csv", index=False)

    pr_basin = area_weighted_series(pre, frac, zone=None, zone_id=0, lats=lats)
    tmp_basin = area_weighted_series(tmp, frac, zone=None, zone_id=0, lats=lats)
    pet_pm_basin = area_weighted_series(pet_mm_grid, frac, zone=None, zone_id=0, lats=lats)

    clim_rows = []
    for mo in range(1, 13):
        sel_m = months == mo
        pm = pr_basin[sel_m]
        tm = tmp_basin[sel_m]
        clim_rows.append({
            "month": mo,
            "precip_mean_mm": round(float(np.mean(pm)), 2),
            "precip_sd_mm": round(float(np.std(pm, ddof=1)), 2),
            "precip_median_mm": round(float(np.median(pm)), 2),
            "precip_q25_mm": round(float(np.percentile(pm, 25)), 2),
            "precip_q75_mm": round(float(np.percentile(pm, 75)), 2),
            "temp_mean_c": round(float(np.mean(tm)), 2),
            "temp_median_c": round(float(np.median(tm)), 2),
            "temp_sd_c": round(float(np.std(tm, ddof=1)), 2),
        })
    clim_df = pd.DataFrame(clim_rows)
    clim_df.to_csv(data_dir / "hydro_climatology.csv", index=False)

    print("[2/7] Detecting regional transitions, seasonal stratification, copulas, and prospective skill...")
    reg_series = {}
    subreg_rows = []
    copula_frames = []
    prosp_rows = []
    null_frames = []

    for zid, rname in SUBBASIN_LABELS.items():
        pr_r = area_weighted_series(pre, frac, zone=zone, zone_id=zid, lats=lats)
        spi_r = fit_spi(pr_r, months, scale=1)
        evs_r, drys_r = detect_transitions(spi_r, periods=periods)
        ev_df_r = pd.DataFrame(evs_r)
        reg_series[rname] = (pr_r, spi_r, evs_r, drys_r, ev_df_r)

        sd_r = ev_df_r["S_d"].to_numpy()
        fw_r = ev_df_r["F_w"].to_numpy()
        dep_r = dependence_summary(sd_r, fw_r)
        dep_r.update({"region": rname, "n_dry_spells": len(drys_r)})
        subreg_rows.append(dep_r)

        copula_frames.append(fit_copulas(rname, sd_r, fw_r, seed=101 + zid, n_boot=400))

        for s_filt in (None, "Flood (May-Sep)", "Spring (Mar-Apr)", "Autumn (Oct-Nov)", "Winter (Dec-Feb)", "Cool (Oct-Apr)"):
            pr_res = prospective_onset_skill(
                spi_r, drys_r, periods=periods, season_filter=s_filt, rng_seed=404 + zid
            )
            pr_res.update({"region": rname, "season": s_filt or "All (1961-2025)"})
            prosp_rows.append(pr_res)

        null_df_r, _ = run_null_surrogates(spi_r, months, years, n_boot=1000, seed=2026 + zid, region=rname)
        null_frames.append(null_df_r)

    subreg_df = pd.DataFrame(subreg_rows)
    subreg_df.to_csv(data_dir / "subregion_summary.csv", index=False)
    copula_df = pd.concat(copula_frames, ignore_index=True)
    copula_df.to_csv(data_dir / "copula_fits.csv", index=False)
    prosp_df = pd.DataFrame(prosp_rows)
    prosp_df.to_csv(data_dir / "prospective_skill.csv", index=False)
    null_df = pd.concat(null_frames, ignore_index=True)
    null_df.to_csv(data_dir / "null_surrogates.csv", index=False)

    _, spi_basin, evs_basin, _, events_df = reg_series["Basin"]
    events_df = events_df.copy()
    events_df.insert(0, "event_id", np.arange(1, len(events_df) + 1))

    flood_ov_mask = events_df.apply(
        lambda r: any(
            5 <= p.month <= 9
            for p in pd.period_range(r["wet_start"], r["wet_end"], freq="M")
        ),
        axis=1,
    ).to_numpy()
    seas_rows = []
    seas_masks = [
        ("All (1961-2025)", np.ones(len(events_df), dtype=bool), False),
        ("Flood (May-Sep)", events_df["trans_month"].between(5, 9).to_numpy(), True),
        ("Spring (Mar-Apr)", events_df["trans_month"].isin([3, 4]).to_numpy(), True),
        ("Autumn (Oct-Nov)", events_df["trans_month"].isin([10, 11]).to_numpy(), True),
        ("Winter (Dec-Feb)", events_df["trans_month"].isin([12, 1, 2]).to_numpy(), True),
        ("Short dry (D = 2)", (events_df["D"] == 2).to_numpy(), True),
        ("Extended dry (D >= 3)", (events_df["D"] >= 3).to_numpy(), True),
        ("Pre-2003 (1961-2002)", (events_df["year"] <= 2002).to_numpy(), True),
        ("Post-2003 (2003-2025)", (events_df["year"] >= 2003).to_numpy(), True),
        ("Flood Overlap (May-Sep)", flood_ov_mask, False),
        ("Immediate g=1", (events_df["gap"] == 1).to_numpy(), False),
        ("Short-gap g<=2", (events_df["gap"] <= 2).to_numpy(), False),
        ("Cool (Oct-Apr)", (~events_df["trans_month"].between(5, 9)).to_numpy(), False),
    ]
    for s_lbl, smask, is_fdr_group in seas_masks:
        sub_e = events_df.loc[smask]
        if len(sub_e) < 4:
            continue
        row_d = dependence_summary(sub_e["S_d"], sub_e["F_w"])
        row_d["stratum"] = s_lbl
        row_d["in_primary_fdr"] = is_fdr_group
        seas_rows.append(row_d)
    seas_df = pd.DataFrame(seas_rows)
    seas_df["q_FDR_tau"] = np.nan
    seas_df["q_FDR_rs"] = np.nan
    prim_f = seas_df["in_primary_fdr"].to_numpy()
    seas_df.loc[prim_f, "q_FDR_tau"] = np.round(bh_fdr(seas_df.loc[prim_f, "kendall_p"].to_numpy()), 4)
    seas_df.loc[prim_f, "q_FDR_rs"] = np.round(bh_fdr(seas_df.loc[prim_f, "spearman_p"].to_numpy()), 4)
    seas_df.drop(columns=["in_primary_fdr"]).to_csv(data_dir / "seasonal_dependence.csv", index=False)

    marg_df = screen_marginals(events_df["S_d"].to_numpy(), events_df["F_w"].to_numpy(), n_boot=400, seed=303)
    marg_df.to_csv(data_dir / "marginal_fits.csv", index=False)

    print("[3/7] Running 748-cell spatial seasonal AR(1) field significance and FDR...")
    grid_df, grid_sum, _ = grid_dependence_fdr(
        pre, months, years, frac, lats, lons, fit_spi, detect_transitions, n_perm=args.n_perm, seed=42
    )
    grid_df.to_csv(data_dir / "spatial_field_grid.csv", index=False)
    grid_sum_df = pd.DataFrame([grid_sum])
    grid_sum_df.to_csv(data_dir / "spatial_field_summary.csv", index=False)

    print("[4/7] Computing threshold sensitivity, Penman-Monteith vs Thornthwaite SPEI, and GPCC v2025 validation...")
    schemes_df = threshold_sensitivity_matrix(spi_basin)
    schemes_df.to_csv(data_dir / "sensitivity_schemes.csv", index=False)

    mean_lat = float(np.average(lats, weights=frac.sum(axis=1)))
    pet_th_basin = np.array(
        [0.0] if False else __import__("src.spi", fromlist=["thornthwaite_pet"]).thornthwaite_pet(tmp_basin, years, months, mean_lat)
    )
    z_crit = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)

    idx_specs = [
        ("SPI-1", fit_spi(pr_basin, months, scale=1), False),
        ("SPI-3", fit_spi(pr_basin, months, scale=3), False),
        ("SPEI-1 (Thorn.)", fit_spei(pr_basin - pet_th_basin, months, scale=1), False),
        ("SPEI-1 (PM)", fit_spei(pr_basin - pet_pm_basin, months, scale=1), False),
        ("SPEI-3 (Thorn.)", fit_spei(pr_basin - pet_th_basin, months, scale=3), False),
        ("SPEI-3 (PM)", fit_spei(pr_basin - pet_pm_basin, months, scale=3), False),
        ("Bridge-1 SPI-1", fit_spi(pr_basin, months, scale=1), True),
    ]
    idx_rows = []
    for idx_i, (ilabel, iser, is_bridge) in enumerate(idx_specs):
        if is_bridge:
            evs_i, _ = detect_with_bridge(iser)
        else:
            evs_i, _ = detect_transitions(iser)
        sd_i = [e["S_d"] for e in evs_i]
        fw_i = [e["F_w"] for e in evs_i]
        n_ev = len(evs_i)
        tau_i, p_i = stats.kendalltau(sd_i, fw_i)
        ci_lo, ci_hi = _bootstrap_tau_ci(sd_i, fw_i, n_boot=1000, seed=505 + idx_i)
        mde_t = float(z_crit * np.sqrt(2.0 * (2 * n_ev + 5) / (9.0 * n_ev * max(n_ev - 1, 1))))
        idx_rows.append({
            "index": ilabel,
            "n_events": n_ev,
            "tau": round(float(tau_i), 4),
            "p_value": round(float(p_i), 4),
            "ci95_lo": round(ci_lo, 4),
            "ci95_hi": round(ci_hi, 4),
            "mde_tau_80pct": round(mde_t, 3),
        })
    idx_rob_df = pd.DataFrame(idx_rows)
    idx_rob_df.to_csv(data_dir / "index_robustness.csv", index=False)

    gpcc_df = benchmark_cru_vs_gpcc(
        pre, raw_dir / "gpcc_precip_050_v2025.nc", frac, zone, lats, lons, months, fit_spi, detect_transitions, years=years
    )
    gpcc_df.to_csv(data_dir / "cru_gpcc_validation.csv", index=False)

    print("[5/7] Computing 7-tributary fractional HydroBASINS amplification, prospective tiers, and Datong runoff...")
    masks, trib_zid, trib_names, trib_areas_km2 = build_tributary_masks(hybas_prefix, lats, lons, frac)
    pr_tribs, spi_tribs = {}, {}
    trib_prosp_rows = []
    for t_i, tname in enumerate(trib_names):
        pr_t = area_weighted_series(pre, masks[tname], zone=None, zone_id=0, lats=None)
        spi_t = fit_spi(pr_t, months, scale=1)
        pr_tribs[tname] = pr_t
        spi_tribs[tname] = spi_t
        _, drys_t = detect_transitions(spi_t, periods=periods)
        for s_filt in (None, "Flood (May-Sep)", "Spring (Mar-Apr)", "Autumn (Oct-Nov)", "Winter (Dec-Feb)", "Cool (Oct-Apr)"):
            res_tp = prospective_onset_skill(
                spi_t, drys_t, periods=periods, season_filter=s_filt, rng_seed=600 + t_i
            )
            res_tp.update({
                "tributary": tname,
                "area_km2": trib_areas_km2[tname],
                "season": s_filt or "All (1961-2025)",
            })
            trib_prosp_rows.append(res_tp)
    pd.DataFrame(trib_prosp_rows).to_csv(data_dir / "tributary_prospective_skill.csv", index=False)

    cos_cell = frac * np.cos(np.deg2rad(lats))[:, None]
    resid_rows = []
    for r_lbl, w_sub in [
        ("Residual non-tributary (All)", np.where(trib_zid < 0, cos_cell, 0.0)),
        ("Upper Jinsha + Chongqing-Yichang TGP interval", np.where((trib_zid < 0) & (zone == 1), cos_cell, 0.0)),
        ("Middle-Lower mainstem corridor", np.where((trib_zid < 0) & (zone >= 2), cos_cell, 0.0)),
    ]:
        if w_sub.sum() <= 0:
            continue
        pr_res_s = area_weighted_series(pre, w_sub, zone=None, zone_id=0, lats=None)
        spi_res_s = fit_spi(pr_res_s, months, scale=1)
        cl_res = np.array([pr_res_s[months == m].mean() for m in range(1, 13)])
        anom_res = np.array([pr_res_s[t] - cl_res[months[t] - 1] for t in range(len(pr_res_s))])
        ev_spis = []
        n_exceed_tribs = 0
        for _, er in events_df.iterrows():
            ks, j2e = int(er["k"]), int(er["j2"]) + 1
            seg_r = spi_res_s[ks:j2e]
            fw_r = float(np.sum(seg_r[seg_r >= 0.5]))
            ev_spis.append(fw_r)
            t_mx = max(float(np.sum(spi_tribs[nm][ks:j2e][spi_tribs[nm][ks:j2e] >= 0.5])) for nm in trib_names)
            if fw_r > t_mx:
                n_exceed_tribs += 1
        resid_rows.append({
            "zone": r_lbl,
            "n_cells": int((w_sub > 0).sum()),
            "cos_frac_of_basin_pct": round(float(100.0 * w_sub.sum() / cos_cell.sum()), 2),
            "mean_event_fw_spi": round(float(np.mean(ev_spis)), 3),
            "median_event_fw_spi": round(float(np.median(ev_spis)), 3),
            "median_event_mm_excess": round(float(np.median([np.sum(np.maximum(anom_res[int(er['k']):int(er['j2'])+1], 0.0)) for _, er in events_df.iterrows()])), 2),
            "n_exceed_all_7tribs": n_exceed_tribs,
        })
    pd.DataFrame(resid_rows).to_csv(data_dir / "residual_mask_coverage.csv", index=False)

    trib_df = tributary_amplification(
        events_df, pr_basin, pr_tribs, spi_basin, spi_tribs, months, periods, trib_areas_km2=trib_areas_km2
    )
    trib_df.to_csv(data_dir / "tributary_amplification.csv", index=False)
    trib_sum_df = summarize_tributary_amplification(
        trib_df, basin_spi=spi_basin, spi_tribs=spi_tribs, events_df=events_df, periods=periods
    )
    trib_sum_df.to_csv(data_dir / "tributary_summary.csv", index=False)

    datong_xlsx = raw_dir / "datong_discharge_1950_2024.xlsx"
    q_lookup = load_datong_anomalies(datong_xlsx, base_start=1961, base_end=2024)
    q_lookup_pre = load_datong_anomalies(datong_xlsx, base_start=1961, base_end=2002)
    events_df = attach_datong_to_events(events_df, q_lookup, z_lookup_pre=q_lookup_pre)
    events_df.to_csv(data_dir / "event_catalog.csv", index=False)

    datong_df = stratify_datong_response(events_df)
    datong_df.to_csv(data_dir / "datong_strata.csv", index=False)

    ord_bench_df = benchmark_datong_ordinary_wet(q_lookup, q_lookup_pre, spi_basin, periods, events_df)
    ord_bench_df.to_csv(data_dir / "datong_ordinary_wet_benchmark.csv", index=False)

    trends_df = pd.DataFrame([annual_frequency_and_trends(events_df)])
    trends_df.to_csv(data_dir / "annual_trends_dispersion.csv", index=False)

    whip_df = pd.DataFrame([interannual_whiplash_overlap(pr_basin, years, events_df)])
    whip_df.to_csv(data_dir / "whiplash_comparison.csv", index=False)

    print("[6/7] Running surface-bounded atmospheric moisture budget, EAP850, NCEP-DOE-2, and causal teleconnections...")
    circ_sum, circ_win, circ_ctrl, circ_verif, circ_fields = run_circulation_diagnostics(
        raw_dir, events_df, spi_basin, periods, mask, n_perm=10000, seed=777
    )
    circ_sum.to_csv(data_dir / "circulation_summary.csv", index=False)
    circ_win.to_csv(data_dir / "circulation_windows.csv", index=False)
    circ_ctrl.to_csv(data_dir / "circulation_matched_controls.csv", index=False)
    circ_verif.to_csv(data_dir / "reanalysis_verification.csv", index=False)

    edge_c = circ_fields["wnpsh_edge_clim_lon"]
    edge_f = circ_fields["wnpsh_edge_flood_lon"]
    edge_o = circ_fields["wnpsh_edge_ord_wet_lon"]
    pd.DataFrame([{
        "wnpsh_edge_clim_lon": round(float(edge_c), 2) if np.isfinite(edge_c) else np.nan,
        "wnpsh_edge_flood_lon": round(float(edge_f), 2) if np.isfinite(edge_f) else np.nan,
        "wnpsh_edge_ord_wet_lon": round(float(edge_o), 2) if np.isfinite(edge_o) else np.nan,
    }]).to_csv(data_dir / "circulation_metadata.csv", index=False)

    tele_df = seasonal_teleconnection_skill(
        raw_dir, events_df, dry_spells=reg_series["Basin"][3], periods=periods, spi_basin=spi_basin
    )
    tele_df.to_csv(data_dir / "seasonal_indices_skill.csv", index=False)

    field_save_keys = [
        "lat", "lon", "Z500_all", "Z500_flood", "p_Z500_all", "p_Z500_flood",
        "q_Z500_all", "q_Z500_flood", "Z850_flood", "p_Z850_flood", "q_Z850_flood",
        "Z200_flood", "h500_flood", "h500_flood_clim", "h500_ord_wet_flood",
        "wnpsh_edge_flood_lon", "wnpsh_edge_clim_lon", "wnpsh_edge_ord_wet_lon",
        "qu_anom_all", "qv_anom_all", "qu_anom_flood", "qv_anom_flood",
        "u200_clim_flood", "wx_200_flood", "wy_200_flood",
    ]
    np.savez_compressed(data_dir / "circulation_fields.npz", **{k: circ_fields[k] for k in field_save_keys})

    vectors = extract_basin_vectors(hybas_prefix, coast_path, precip_grid=pre, years=years)
    save_basin_vectors(data_dir / "basin_vectors.npz", vectors)
    _write_sha256_manifest(data_dir)

    print("[7/7] Rendering Figures 1-6 and Online Resource ESM_Fig1...")
    tables = {
        "hydro_climatology": clim_df,
        "tributary_amplification": trib_df,
        "tributary_summary": trib_sum_df,
        "event_catalog": events_df,
        "subregion_summary": subreg_df,
        "marginal_fits": marg_df,
        "spatial_field_grid": grid_df,
        "spatial_field_summary": grid_sum_df,
        "sensitivity_schemes": schemes_df,
        "index_robustness": idx_rob_df,
        "datong_strata": datong_df,
        "circulation_windows": circ_win,
        "circulation_summary": circ_sum,
        "circulation_matched_controls": circ_ctrl,
    }
    render_figures(mask, vectors, circ_fields, tables, fig_dir, tuple(args.formats))
    print("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Yangtze River Basin DFAA diagnostic pipeline")
    parser.add_argument("--raw-dir", default="../data/raw", help="Directory containing raw CRU/NCEP/GPCC/Datong inputs")
    parser.add_argument("--mask", default="data/yangtze_mask.npz", help="Path to Yangtze basin NPZ mask")
    parser.add_argument("--hybas", default="../data/raw/hybas_l6/hybas_as_lev06_v1c", help="HydroBASINS Level-6 shapefile prefix")
    parser.add_argument("--coast", default="../data/raw/ne_50m_coastline.geojson", help="Natural Earth coastline GeoJSON")
    parser.add_argument("--out-dir", default=".", help="Repository output root (writes data/ and figures/)")
    parser.add_argument("--fig-dir", default=None, help="Optional override for figure output directory")
    parser.add_argument("--derived-only", action="store_true", help="Re-render figures from derived CSV/NPZ artifacts")
    parser.add_argument("--formats", nargs="+", default=["png"], help="Figure export formats (e.g. png pdf eps)")
    parser.add_argument("--n-perm", type=int, default=1000, help="Spatial AR(1) field significance permutations")
    args = parser.parse_args()

    if args.derived_only:
        out_dir = Path(args.out_dir)
        mask_p = _resolve_existing([args.mask, out_dir / "data/yangtze_mask.npz"])
        fig_dir = Path(args.fig_dir) if args.fig_dir else (out_dir / "figures")
        run_derived_only(mask_p, out_dir, fig_dir, tuple(args.formats))
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
