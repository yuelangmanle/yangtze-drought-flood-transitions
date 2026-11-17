"""Runoff response stratification, count dispersion regression, and gauge density checks."""

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def load_datong_anomalies(xlsx_path, base_start=1961, base_end=2024):
    """Load Datong monthly discharge and compute monthly standardized anomalies."""
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


def attach_datong_to_events(events_df, z_lookup):
    """Attach dry-spell and wet-spell Datong anomalies to the basin transition catalog."""
    q_dry_list, q_wet_list, q_wet_max_list, dq_list = [], [], [], []
    for _, row in events_df.iterrows():
        dry_ps = _expand_months(row["dry_start"], row["dry_end"])
        wet_ps = _expand_months(row["wet_start"], row["wet_end"])
        if all(p in z_lookup for p in dry_ps):
            q_d = float(np.mean([z_lookup[p] for p in dry_ps]))
        else:
            q_d = np.nan
        if all(p in z_lookup for p in wet_ps):
            wvals = [z_lookup[p] for p in wet_ps]
            q_w = float(np.mean(wvals))
            q_wm = float(np.max(wvals))
        else:
            q_w, q_wm = np.nan, np.nan
        dq = (q_w - q_d) if (np.isfinite(q_w) and np.isfinite(q_d)) else np.nan
        q_dry_list.append(round(q_d, 3) if np.isfinite(q_d) else np.nan)
        q_wet_list.append(round(q_w, 3) if np.isfinite(q_w) else np.nan)
        q_wet_max_list.append(round(q_wm, 3) if np.isfinite(q_wm) else np.nan)
        dq_list.append(round(dq, 3) if np.isfinite(dq) else np.nan)

    out = events_df.copy()
    out["Q_dry_z"] = q_dry_list
    out["Q_wet_z"] = q_wet_list
    out["Q_wet_peak_z"] = q_wet_max_list
    out["delta_Q_z"] = dq_list
    return out


def stratify_datong_response(events_df):
    """Evaluate Datong wet-spell response across reservoir eras and wet-severity classes."""
    df = events_df[events_df["Q_wet_z"].notna()].copy()
    periods = [
        ("Full (1961-2024)", 1961, 2024),
        ("Pre-2003 (1961-2002)", 1961, 2002),
        ("Impoundment (2003-2012)", 2003, 2012),
        ("Mega-cascade (2013-2024)", 2013, 2024),
    ]
    rows = []
    for era_name, y0, y1 in periods:
        sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
        for cls_name, mask in [
            ("All", np.ones(len(sub), dtype=bool)),
            ("Fw < 1.0", sub["F_w"].to_numpy() < 1.0),
            ("Fw >= 1.0", sub["F_w"].to_numpy() >= 1.0),
        ]:
            vals = sub.loc[mask, "Q_wet_z"].to_numpy()
            if len(vals) == 0:
                continue
            p_w = float(stats.wilcoxon(vals).pvalue) if len(vals) >= 6 else np.nan
            rows.append({
                "era": era_name,
                "wet_class": cls_name,
                "n_events": len(vals),
                "Q_wet_mean_z": round(float(np.mean(vals)), 3),
                "Q_wet_median_z": round(float(np.median(vals)), 3),
                "Q_dry_mean_z": round(float(sub.loc[mask, "Q_dry_z"].mean()), 3),
                "delta_Q_mean_z": round(float(sub.loc[mask, "delta_Q_z"].mean()), 3),
                "wilcoxon_p": round(p_w, 4) if np.isfinite(p_w) else np.nan,
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


def station_density_summary(stn_cube, frac, lons):
    """Audit CRU TS station-count (`stn`) coverage across basin sub-domains."""
    masks = [
        ("Whole Basin", frac > 0),
        ("Upper West (<=102E)", (frac > 0) & (lons[None, :] <= 102.0)),
        ("Upper East (>102E, upper)", None),
        ("Middle-Lower (>108E)", (frac > 0) & (lons[None, :] > 108.0)),
    ]
    rows = []
    for name, m in masks:
        if m is None:
            continue
        vals = stn_cube[:, m]
        rows.append({
            "domain": name,
            "n_cells": int(m.sum()),
            "stn_mean": round(float(np.nanmean(vals)), 2),
            "stn_median": round(float(np.nanmedian(vals)), 1),
            "stn_min": int(np.nanmin(vals)),
            "stn_max": int(np.nanmax(vals)),
            "pct_zero_stn": round(float(np.mean(vals == 0) * 100.0), 2),
        })
    return pd.DataFrame(rows)
