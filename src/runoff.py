"""Runoff response stratification, non-DFAA wet controls, count trends, and precipitation gauge audits."""

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import xarray as xr

from src.basin import area_weighted_series


def load_datong_anomalies(xlsx_path, base_start=1961, base_end=2024):
    """Load Datong monthly discharge and compute standardized anomalies for given baseline years."""
    raw = pd.read_excel(xlsx_path, sheet_name="Sheet1", header=None)
    years = raw.iloc[1:, 0].astype(int).to_numpy()
    q_mat = raw.iloc[1:, 1:13].astype(float).to_numpy()
    base = (years >= base_start) & (years <= base_end)

    z_lookup = {}
    for mo in range(1, 13):
        col = q_mat[:, mo - 1]
        mu = float(np.mean(col[base]))
        sd = float(np.std(col[base], ddof=1))
        for i, yr in enumerate(years):
            z_lookup[pd.Period(f"{yr}-{mo:02d}", freq="M")] = (float(col[i]) - mu) / sd
    return z_lookup


def _expand_months(start_str, end_str):
    p0 = pd.Period(start_str, freq="M")
    p1 = pd.Period(end_str, freq="M")
    return [p0 + k for k in range((p1 - p0).n + 1)]


def attach_datong_to_events(events_df, z_lookup, z_lookup_pre=None):
    """Attach dry-spell, wet-spell, peak, net, and symmetric 2-month pre-wet Datong anomalies."""
    q_dry_list, q_wet_list, q_wet_max_list, dq_list, dq_sym_list = [], [], [], [], []
    q_dry_p_list, q_wet_p_list, dq_p_list = [], [], []

    for _, row in events_df.iterrows():
        dry_ps = _expand_months(row["dry_start"], row["dry_end"])
        wet_ps = _expand_months(row["wet_start"], row["wet_end"])
        w0 = pd.Period(row["wet_start"], freq="M")
        prior2_ps = [w0 - 2, w0 - 1]
        if all(p in z_lookup for p in dry_ps):
            q_d = float(np.mean([z_lookup[p] for p in dry_ps]))
            q_dp = float(np.mean([z_lookup_pre[p] for p in dry_ps])) if z_lookup_pre else np.nan
        else:
            q_d, q_dp = np.nan, np.nan
        if all(p in z_lookup for p in wet_ps):
            wvals = [z_lookup[p] for p in wet_ps]
            q_w = float(np.mean(wvals))
            q_wm = float(np.max(wvals))
            q_wp = float(np.mean([z_lookup_pre[p] for p in wet_ps])) if z_lookup_pre else np.nan
        else:
            q_w, q_wm, q_wp = np.nan, np.nan, np.nan
        dq = (q_w - q_d) if (np.isfinite(q_w) and np.isfinite(q_d)) else np.nan
        dqp = (q_wp - q_dp) if (np.isfinite(q_wp) and np.isfinite(q_dp)) else np.nan
        p2_vals = [z_lookup[p] for p in prior2_ps if p in z_lookup]
        dq_sym = (q_w - float(np.mean(p2_vals))) if (np.isfinite(q_w) and p2_vals) else np.nan

        q_dry_list.append(round(q_d, 3) if np.isfinite(q_d) else np.nan)
        q_wet_list.append(round(q_w, 3) if np.isfinite(q_w) else np.nan)
        q_wet_max_list.append(round(q_wm, 3) if np.isfinite(q_wm) else np.nan)
        dq_list.append(round(dq, 3) if np.isfinite(dq) else np.nan)
        dq_sym_list.append(round(dq_sym, 3) if np.isfinite(dq_sym) else np.nan)
        q_dry_p_list.append(round(q_dp, 3) if np.isfinite(q_dp) else np.nan)
        q_wet_p_list.append(round(q_wp, 3) if np.isfinite(q_wp) else np.nan)
        dq_p_list.append(round(dqp, 3) if np.isfinite(dqp) else np.nan)

    out = events_df.copy()
    out["Q_dry_z"] = q_dry_list
    out["Q_wet_z"] = q_wet_list
    out["Q_wet_peak_z"] = q_wet_max_list
    out["delta_Q_z"] = dq_list
    out["delta_Q_sym_z"] = dq_sym_list
    out["Q_dry_z_preTGP"] = q_dry_p_list
    out["Q_wet_z_preTGP"] = q_wet_p_list
    out["delta_Q_z_preTGP"] = dq_p_list
    return out


def _wet_overlaps_flood(df_in):
    flags = []
    for _, r in df_in.iterrows():
        ps = _expand_months(r["wet_start"], r["wet_end"])
        flags.append(any(5 <= int(p.month) <= 9 for p in ps))
    return np.asarray(flags, dtype=bool)


def stratify_datong_response(events_df):
    """Evaluate Datong discharge across eras, seasons, gap/overlap strata, severity tiers, and dispersion."""
    df = events_df[events_df["Q_wet_z"].notna()].copy()
    df["fw_intensity"] = df["F_w"] / df["W"]
    df["flood_overlap"] = _wet_overlaps_flood(df)

    rs_cum, p_cum = stats.spearmanr(df["F_w"], df["Q_wet_z"])
    rs_int, p_int = stats.spearmanr(df["fw_intensity"], df["Q_wet_z"])
    rs_pk, p_pk = stats.spearmanr(df["F_w"], df["Q_wet_peak_z"])

    groups = [
        ("Full (1961-2024)", df),
        ("Flood (May-Sep)", df[df["trans_month"].between(5, 9)]),
        ("Flood Overlap (May-Sep)", df[df["flood_overlap"]]),
        ("Cool (Oct-Apr)", df[~df["trans_month"].between(5, 9)]),
        ("Spring (Mar-Apr)", df[df["trans_month"].isin([3, 4])]),
        ("Autumn (Oct-Nov)", df[df["trans_month"].isin([10, 11])]),
        ("Winter (Dec-Feb)", df[df["trans_month"].isin([12, 1, 2])]),
        ("Immediate (g = 1)", df[df["gap"] == 1]),
        ("Short-gap (g <= 2)", df[df["gap"] <= 2]),
        ("Pre-2003 (1961-2002)", df[(df["year"] >= 1961) & (df["year"] <= 2002)]),
        ("Pre-2003 Flood (May-Sep)", df[(df["year"] <= 2002) & df["trans_month"].between(5, 9)]),
        ("Pre-2003 Cool (Oct-Apr)", df[(df["year"] <= 2002) & (~df["trans_month"].between(5, 9))]),
        ("Impoundment (2003-2012, caveat)", df[(df["year"] >= 2003) & (df["year"] <= 2012)]),
        ("Mega-cascade (2013-2024, caveat)", df[(df["year"] >= 2013) & (df["year"] <= 2024)]),
    ]
    rows = []
    for era_name, sub in groups:
        for cls_name, mask in [
            ("All", np.ones(len(sub), dtype=bool)),
            ("Fw < 1.0", sub["F_w"].to_numpy() < 1.0),
            ("Fw >= 1.0", sub["F_w"].to_numpy() >= 1.0),
        ]:
            vals = sub.loc[mask, "Q_wet_z"].to_numpy()
            if len(vals) == 0:
                continue
            dvals = sub.loc[mask, "Q_dry_z"].to_numpy()
            dqvals = sub.loc[mask, "delta_Q_z"].to_numpy()
            p_w = float(stats.wilcoxon(vals).pvalue) if len(vals) >= 6 else np.nan
            p_dq = float(stats.wilcoxon(dqvals).pvalue) if len(dqvals) >= 6 else np.nan
            q25, q75 = np.percentile(vals, [25, 75])
            rows.append({
                "era": era_name,
                "wet_class": cls_name,
                "n_events": len(vals),
                "Q_wet_mean_z": round(float(np.mean(vals)), 3),
                "Q_wet_sd_z": round(float(np.std(vals, ddof=1)), 3) if len(vals) >= 2 else np.nan,
                "Q_wet_median_z": round(float(np.median(vals)), 3),
                "Q_wet_iqr_z": f"{q25:+.2f}..{q75:+.2f}",
                "Q_wet_peak_mean_z": round(float(sub.loc[mask, "Q_wet_peak_z"].mean()), 3),
                "Q_dry_mean_z": round(float(np.mean(dvals)), 3),
                "Q_dry_sd_z": round(float(np.std(dvals, ddof=1)), 3) if len(dvals) >= 2 else np.nan,
                "delta_Q_mean_z": round(float(np.mean(dqvals)), 3),
                "delta_Q_sd_z": round(float(np.std(dqvals, ddof=1)), 3) if len(dqvals) >= 2 else np.nan,
                "Q_wet_mean_preTGP": round(float(sub.loc[mask, "Q_wet_z_preTGP"].mean()), 3),
                "Q_dry_mean_preTGP": round(float(sub.loc[mask, "Q_dry_z_preTGP"].mean()), 3),
                "delta_Q_mean_preTGP": round(float(sub.loc[mask, "delta_Q_z_preTGP"].mean()), 3),
                "wilcoxon_p": round(p_w, 4) if np.isfinite(p_w) else np.nan,
                "wilcoxon_dq_p": round(p_dq, 4) if np.isfinite(p_dq) else np.nan,
                "spearman_cumFw_Qwet": round(float(rs_cum), 4) if era_name.startswith("Full") and cls_name == "All" else np.nan,
                "spearman_meanFwW_Qwet": round(float(rs_int), 4) if era_name.startswith("Full") and cls_name == "All" else np.nan,
                "spearman_meanFwW_p": round(float(p_int), 5) if era_name.startswith("Full") and cls_name == "All" else np.nan,
                "spearman_cumFw_Qpeak": round(float(rs_pk), 4) if era_name.startswith("Full") and cls_name == "All" else np.nan,
            })
    return pd.DataFrame(rows)


def _mean_ci95(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return np.nan, np.nan
    se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
    tc = float(stats.t.ppf(0.975, df=len(a) - 1))
    m = float(np.mean(a))
    return m - tc * se, m + tc * se


def benchmark_datong_ordinary_wet(q_lookup, q_lookup_pre, spi_basin, periods, events_df, wet_th=0.5):
    """Compare DFAA wet-spell Datong discharge against non-DFAA ordinary wet spells matched by month, era, and F_w."""
    dfaa_months = set()
    for _, r in events_df.iterrows():
        dfaa_months.update(_expand_months(r["wet_start"], r["wet_end"]))

    ord_runs = []
    n = len(spi_basin)
    i = 0
    while i < n:
        if np.isfinite(spi_basin[i]) and spi_basin[i] >= wet_th:
            j = i
            while j < n and np.isfinite(spi_basin[j]) and spi_basin[j] >= wet_th:
                j += 1
            ps = periods[i:j]
            if not any(p in dfaa_months for p in ps) and all(p in q_lookup for p in ps):
                fw_o = float(spi_basin[i:j].sum())
                qw_o = float(np.mean([q_lookup[p] for p in ps]))
                qw_op = float(np.mean([q_lookup_pre[p] for p in ps]))
                prior = [periods[t] for t in range(max(0, i - 2), i) if periods[t] in q_lookup]
                dq_o = qw_o - float(np.mean([q_lookup[p] for p in prior])) if prior else np.nan
                ord_runs.append({
                    "start_month": int(ps[0].month),
                    "year": int(ps[0].year),
                    "F_w": fw_o,
                    "Q_wet_z": qw_o,
                    "Q_wet_z_preTGP": qw_op,
                    "delta_Q_z": dq_o,
                })
            i = j
        else:
            i += 1
    ord_df = pd.DataFrame(ord_runs)

    sub_ev = events_df[events_df["Q_wet_z"].notna()].copy()
    strata_defs = [
        ("All (1961-2024)", np.ones(len(sub_ev), dtype=bool), np.ones(len(ord_df), dtype=bool), False),
        ("Fw < 1.0", sub_ev["F_w"].to_numpy() < 1.0, ord_df["F_w"].to_numpy() < 1.0, False),
        ("Fw >= 1.0", sub_ev["F_w"].to_numpy() >= 1.0, ord_df["F_w"].to_numpy() >= 1.0, False),
        ("Flood (May-Sep)", sub_ev["trans_month"].between(5, 9).to_numpy(), ord_df["start_month"].between(5, 9).to_numpy(), False),
        ("Pre-2003 Era-Matched (1961-2002)", (sub_ev["year"] <= 2002).to_numpy(), (ord_df["year"] <= 2002).to_numpy(), True),
        ("Post-2003 Era-Matched (2003-2024)", (sub_ev["year"] >= 2003).to_numpy(), (ord_df["year"] >= 2003).to_numpy(), True),
    ]

    rows = []
    for cls_lbl, cond_ev, cond_ord, same_era in strata_defs:
        ev_s = sub_ev.loc[cond_ev]
        or_s = ord_df.loc[cond_ord]
        if ev_s.empty or or_s.empty:
            continue
        diffs_q, diffs_qp, diffs_dq, diffs_dqs, matched_ord_q = [], [], [], [], []
        n_fallback = 0
        for _, r in ev_s.iterrows():
            mo = int(r["trans_month"])
            fw_r = float(r["F_w"])
            cands = or_s[or_s["start_month"] == mo]
            if cands.empty:
                n_fallback += 1
                cands = or_s
            nearest = cands.iloc[(cands["F_w"] - fw_r).abs().argsort()[:3]]
            oq = float(nearest["Q_wet_z"].mean())
            matched_ord_q.append(oq)
            diffs_q.append(float(r["Q_wet_z"]) - oq)
            diffs_qp.append(float(r["Q_wet_z_preTGP"]) - float(nearest["Q_wet_z_preTGP"].mean()))
            if np.isfinite(r["delta_Q_z"]) and nearest["delta_Q_z"].notna().any():
                diffs_dq.append(float(r["delta_Q_z"]) - float(nearest["delta_Q_z"].mean()))
            if "delta_Q_sym_z" in r and np.isfinite(r["delta_Q_sym_z"]) and nearest["delta_Q_z"].notna().any():
                diffs_dqs.append(float(r["delta_Q_sym_z"]) - float(nearest["delta_Q_z"].mean()))

        _, p_mw = stats.mannwhitneyu(ev_s["Q_wet_z"], or_s["Q_wet_z"], alternative="two-sided")
        _, p_pair = stats.wilcoxon(diffs_q) if len(diffs_q) >= 6 else (np.nan, np.nan)
        _, p_dq = stats.wilcoxon(diffs_dq) if len(diffs_dq) >= 6 else (np.nan, np.nan)
        _, p_dqs = stats.wilcoxon(diffs_dqs) if len(diffs_dqs) >= 6 else (np.nan, np.nan)
        q_lo, q_hi = _mean_ci95(diffs_q)
        dq_lo, dq_hi = _mean_ci95(diffs_dq)
        rows.append({
            "stratum": cls_lbl,
            "era_matched": same_era,
            "n_dfaa": len(ev_s),
            "n_ord_wet_pool": len(or_s),
            "n_cross_month_fallback": n_fallback,
            "dfaa_Q_wet_mean": round(float(ev_s["Q_wet_z"].mean()), 3),
            "ord_pool_Q_wet_mean": round(float(or_s["Q_wet_z"].mean()), 3),
            "ord_matched_Q_wet_mean": round(float(np.mean(matched_ord_q)), 3),
            "paired_diff_Q_wet": round(float(np.mean(diffs_q)), 3),
            "Q_wet_ci95_lo": round(q_lo, 3) if np.isfinite(q_lo) else np.nan,
            "Q_wet_ci95_hi": round(q_hi, 3) if np.isfinite(q_hi) else np.nan,
            "paired_diff_Q_preTGP": round(float(np.mean(diffs_qp)), 3),
            "paired_diff_delta_Q": round(float(np.mean(diffs_dq)), 3) if diffs_dq else np.nan,
            "delta_Q_ci95_lo": round(dq_lo, 3) if np.isfinite(dq_lo) else np.nan,
            "delta_Q_ci95_hi": round(dq_hi, 3) if np.isfinite(dq_hi) else np.nan,
            "paired_diff_delta_Q_sym": round(float(np.mean(diffs_dqs)), 3) if diffs_dqs else np.nan,
            "mw_p_unpaired": round(float(p_mw), 4),
            "wilcoxon_p_matched": round(float(p_pair), 4) if np.isfinite(p_pair) else np.nan,
            "wilcoxon_p_delta_Q": round(float(p_dq), 4) if np.isfinite(p_dq) else np.nan,
            "wilcoxon_p_delta_Q_sym": round(float(p_dqs), 4) if np.isfinite(p_dqs) else np.nan,
        })
    return pd.DataFrame(rows)


def annual_frequency_and_trends(events_df, y0=1961, y1=2025):
    """Test annual transition count dispersion, Quasi-Poisson trend, and severity trends."""
    years = np.arange(y0, y1 + 1)
    counts = pd.Series(0, index=years)
    for yr in events_df["year"].astype(int):
        if yr in counts.index:
            counts[yr] += 1

    mu = float(counts.mean())
    var = float(counts.var(ddof=1))
    disp = var / mu
    df_chi = len(counts) - 1
    chi2_val = df_chi * disp
    p_up = 1.0 - float(stats.chi2.cdf(chi2_val, df=df_chi))
    p_lo = float(stats.chi2.cdf(chi2_val, df=df_chi))
    p_two = 2.0 * min(p_up, p_lo)

    df_reg = pd.DataFrame({
        "year": years,
        "y_center": years - 1993.0,
        "count": counts.to_numpy(),
    })
    glm_pois = smf.glm("count ~ y_center", data=df_reg, family=sm.families.Poisson()).fit()
    glm_qp = smf.glm("count ~ y_center", data=df_reg, family=sm.families.Poisson()).fit(scale="X2")

    mk_cnt = mk.original_test(counts.to_numpy())
    mk_sd = mk.original_test(events_df["S_d"].to_numpy())
    mk_fw = mk.original_test(events_df["F_w"].to_numpy())

    return {
        "n_years": len(years),
        "count_mean": round(mu, 4),
        "count_var": round(var, 4),
        "dispersion": round(disp, 4),
        "chi2_stat": round(chi2_val, 2),
        "chi2_df": df_chi,
        "dispersion_p_two_sided": round(p_two, 4),
        "poisson_slope_per_yr": round(float(glm_pois.params["y_center"]), 5),
        "poisson_p": round(float(glm_pois.pvalues["y_center"]), 4),
        "quasipoisson_scale": round(float(glm_qp.scale), 4),
        "quasipoisson_slope": round(float(glm_qp.params["y_center"]), 5),
        "quasipoisson_se": round(float(glm_qp.bse["y_center"]), 5),
        "quasipoisson_z": round(float(glm_qp.tvalues["y_center"]), 3),
        "quasipoisson_p": round(float(glm_qp.pvalues["y_center"]), 4),
        "mk_count_Z": round(float(mk_cnt.z), 3),
        "mk_count_p": round(float(mk_cnt.p), 4),
        "mk_Sd_Z": round(float(mk_sd.z), 3),
        "mk_Sd_p": round(float(mk_sd.p), 4),
        "mk_Fw_Z": round(float(mk_fw.z), 3),
        "mk_Fw_p": round(float(mk_fw.p), 4),
    }


def station_density_summary(stn_cube, frac, zone, lons, years=None):
    """Audit CRU TS station-count (`stn`) coverage across sectors and 1961-2019 vs 2020-2025."""
    domains = [
        ("Whole Basin", frac > 0),
        ("Upper West (<=102E)", (frac > 0) & (lons[None, :] <= 102.0)),
        ("Upper East (>102E, upper)", (frac > 0) & (zone == 1) & (lons[None, :] > 102.0)),
        ("Middle-Lower (>108E)", (frac > 0) & (lons[None, :] > 108.0)),
    ]
    time_slices = [("1961-2025 (Full)", slice(None))]
    if years is not None:
        time_slices.extend([
            ("1961-2019 (Historical)", years <= 2019),
            ("2020-2025 (Recent)", years >= 2020),
        ])

    rows = []
    for p_lbl, tmask in time_slices:
        sub_stn = stn_cube[tmask]
        for name, m in domains:
            vals = sub_stn[:, m]
            rows.append({
                "period": p_lbl,
                "domain": name,
                "n_cells": int(m.sum()),
                "stn_mean": round(float(np.nanmean(vals)), 2),
                "stn_median": round(float(np.nanmedian(vals)), 1),
                "stn_min": int(np.nanmin(vals)),
                "stn_max": int(np.nanmax(vals)),
                "pct_zero_stn": round(float(np.mean(vals == 0) * 100.0), 2),
            })
    return pd.DataFrame(rows)


def benchmark_cru_vs_gpcc(cru_pre, gpcc_nc_path, frac, zone, lats, lons, months, fit_spi_fn, detect_fn, years=None):
    """Compare CRU TS v4.10 precipitation and SPI-1 against GPCC v2025 across full and thinning epochs."""
    ds_g = xr.open_dataset(gpcc_nc_path).sortby("lat").sel(
        lat=slice(lats[0], lats[-1]),
        lon=slice(lons[0], lons[-1]),
        time=slice("1961-01-01", "2025-12-31"),
    )
    gpcc_pre = ds_g["precip"].values.astype(float)

    regions = [
        ("Whole Basin", frac, lats),
        ("Upper West (<=102E)", np.where(lons[None, :] <= 102.0, frac, 0.0), lats),
        ("Upper YRB", np.where(zone == 1, frac, 0.0), lats),
        ("Middle YRB", np.where(zone == 2, frac, 0.0), lats),
        ("Lower YRB", np.where(zone == 3, frac, 0.0), lats),
        ("Middle-Lower (>108E)", np.where(lons[None, :] > 108.0, frac, 0.0), lats),
    ]
    epoch_specs = [("1961-2025 (Full)", None)]
    if years is not None:
        epoch_specs.extend([
            ("1961-2019 (Historical)", years <= 2019),
            ("2020-2025 (Recent)", years >= 2020),
        ])

    rows = []
    for rname, rfrac, rlat in regions:
        p_cru = area_weighted_series(cru_pre, rfrac, zone=None, zone_id=0, lats=rlat)
        p_gpcc = area_weighted_series(gpcc_pre, rfrac, zone=None, zone_id=0, lats=rlat)
        s_cru = fit_spi_fn(p_cru, months, scale=1)
        s_gpcc = fit_spi_fn(p_gpcc, months, scale=1)
        evs_c, _ = detect_fn(s_cru)
        evs_g, _ = detect_fn(s_gpcc)

        for ep_lbl, emask in epoch_specs:
            if emask is None:
                pc, pg, sc, sg = p_cru, p_gpcc, s_cru, s_gpcc
                ec = evs_c
                eg = evs_g
            else:
                pc, pg, sc, sg = p_cru[emask], p_gpcc[emask], s_cru[emask], s_gpcc[emask]
                idx_set = set(np.where(emask)[0].tolist())
                ec = [e for e in evs_c if e["k"] in idx_set]
                eg = [e for e in evs_g if e["k"] in idx_set]

            r_mm, _ = stats.pearsonr(pc, pg)
            rmse_mm = float(np.sqrt(np.mean((pc - pg) ** 2)))
            bias_pct = float(100.0 * (np.mean(pc) - np.mean(pg)) / np.mean(pg))
            r_spi, _ = stats.pearsonr(sc, sg)
            n_match_pm1 = sum(1 for e in ec if any(abs(e["k"] - g["k"]) <= 1 for g in eg))
            ov_pct = float(100.0 * n_match_pm1 / len(ec)) if ec else np.nan
            if len(eg) >= 5:
                tau_g, p_tau_g = stats.kendalltau([e["S_d"] for e in eg], [e["F_w"] for e in eg])
            else:
                tau_g, p_tau_g = np.nan, np.nan
            rows.append({
                "period": ep_lbl,
                "domain": rname,
                "precip_r": round(float(r_mm), 4),
                "precip_rmse_mm": round(rmse_mm, 2),
                "precip_bias_pct": round(bias_pct, 2),
                "spi1_r": round(float(r_spi), 4),
                "n_events_cru": len(ec),
                "n_events_gpcc": len(eg),
                "n_matched_pm1mo": n_match_pm1,
                "event_overlap_pct": round(ov_pct, 1) if np.isfinite(ov_pct) else np.nan,
                "gpcc_tau": round(float(tau_g), 4) if np.isfinite(tau_g) else np.nan,
                "gpcc_tau_p": round(float(p_tau_g), 4) if np.isfinite(p_tau_g) else np.nan,
            })
    return pd.DataFrame(rows)
