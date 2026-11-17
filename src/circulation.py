"""Moisture transport divergence, dipole teleconnections, wave-activity flux, and reanalysis checks."""

import numpy as np
import pandas as pd
import xarray as xr
from src.dependence import bh_fdr


G_ACCEL = 9.80665
R_EARTH = 6.371e6
OMEGA_EARTH = 7.2921e-5


def detrend_monthly_std_1d(values, months):
    """Remove calendar-month linear trend and standardize each calendar month to unit variance."""
    out = np.full(len(values), np.nan, dtype=float)
    for mo in range(1, 13):
        idx = np.where((months == mo) & np.isfinite(values))[0]
        if len(idx) < 5:
            continue
        x = values[idx]
        t = np.arange(len(idx), dtype=float)
        trend = np.polyval(np.polyfit(t, x, 1), t)
        xd = x - trend
        sd = float(np.std(xd))
        out[idx] = xd / sd if sd > 0 else np.nan
    return out


def detrend_monthly_std_3d(field, months):
    """Standardize by calendar month and remove linear temporal trend at each grid point."""
    out = np.full_like(field, np.nan, dtype=float)
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        x = field[idx]
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd[sd == 0] = np.nan
        z = (x - mu) / sd
        t = np.arange(len(idx), dtype=float)
        tc = t - t.mean()
        zc = np.where(np.isfinite(z), z, 0.0)
        slope = np.nansum(tc[:, None, None] * zc, axis=0) / np.sum(tc**2)
        out[idx] = z - slope[None, :, :] * tc[:, None, None]
    return out


def vertical_vapor_flux_surface_bounded(q, u, v, levels_hpa, ps_hpa):
    """Integrate (q*u/g, q*v/g) from 300 hPa down to min(1000 hPa, p_s(t, lat, lon))."""
    order = np.argsort(levels_hpa)
    lv = levels_hpa[order]
    q = q[:, order, :, :].copy()
    u = u[:, order, :, :].copy()
    v = v[:, order, :, :].copy()
    ps_hpa = ps_hpa.copy()

    bad_t = (
        (~np.isfinite(q)).any(axis=(1, 2, 3))
        | (~np.isfinite(u)).any(axis=(1, 2, 3))
        | (~np.isfinite(v)).any(axis=(1, 2, 3))
        | (~np.isfinite(ps_hpa)).any(axis=(1, 2))
    )
    q[~np.isfinite(q)] = 0.0
    u[~np.isfinite(u)] = 0.0
    v[~np.isfinite(v)] = 0.0
    ps_hpa[~np.isfinite(ps_hpa)] = 1000.0

    if np.nanmax(q) > 0.2:
        q = q * 1e-3
    if np.nanmax(ps_hpa) > 2000.0:
        ps_hpa = ps_hpa / 100.0

    qu_lev = q * u / G_ACCEL
    qv_lev = q * v / G_ACCEL
    qu_int = np.zeros((q.shape[0], q.shape[2], q.shape[3]), dtype=float)
    qv_int = np.zeros_like(qu_int)

    for k in range(len(lv) - 1):
        p_top, p_bot = float(lv[k]), float(lv[k + 1])
        dp_full = (p_bot - p_top) * 100.0
        full = ps_hpa >= p_bot
        qu_int += np.where(full, 0.5 * (qu_lev[:, k] + qu_lev[:, k + 1]) * dp_full, 0.0)
        qv_int += np.where(full, 0.5 * (qv_lev[:, k] + qv_lev[:, k + 1]) * dp_full, 0.0)

        part = (ps_hpa > p_top) & (ps_hpa < p_bot)
        if np.any(part):
            w_bot = np.where(part, (ps_hpa - p_top) / (p_bot - p_top), 0.0)
            qu_sfc = (1.0 - w_bot) * qu_lev[:, k] + w_bot * qu_lev[:, k + 1]
            qv_sfc = (1.0 - w_bot) * qv_lev[:, k] + w_bot * qv_lev[:, k + 1]
            dp_part = np.where(part, (ps_hpa - p_top) * 100.0, 0.0)
            qu_int += np.where(part, 0.5 * (qu_lev[:, k] + qu_sfc) * dp_part, 0.0)
            qv_int += np.where(part, 0.5 * (qv_lev[:, k] + qv_sfc) * dp_part, 0.0)

    if bad_t.any():
        qu_int[bad_t] = np.nan
        qv_int[bad_t] = np.nan
    return qu_int, qv_int, bad_t


def spherical_divergence(qu, qv, lats, ddeg=2.5):
    """Compute horizontal spherical divergence [kg m^-2 s^-1]."""
    flip = lats[0] > lats[-1]
    if flip:
        la = lats[::-1]
        qu_a, qv_a = qu[:, ::-1, :], qv[:, ::-1, :]
    else:
        la = lats
        qu_a, qv_a = qu, qv

    cos_lat = np.cos(np.deg2rad(la))[None, :, None]
    dx = np.deg2rad(ddeg) * R_EARTH * cos_lat
    dy = np.deg2rad(ddeg) * R_EARTH
    div = np.gradient(qu_a, axis=2) / dx + np.gradient(qv_a * cos_lat, axis=1) / (dy * cos_lat)
    return div[:, ::-1, :] if flip else div


def build_reanalysis_basin_mask(lat_nc, lon_nc, basin_lats, basin_lons, basin_frac):
    """Map the 0.5-degree Yangtze basin mask onto the 2.5-degree reanalysis grid."""
    w = np.zeros((len(lat_nc), len(lon_nc)), dtype=float)
    for i, la in enumerate(lat_nc):
        for j, lo in enumerate(lon_nc):
            sub = (np.abs(basin_lats - la) <= 0.6)[:, None] & (np.abs(basin_lons - lo) <= 0.6)[None, :]
            if (basin_frac * sub).sum() > 0:
                w[i, j] = 1.0
    w /= w.sum()
    return w


def paired_sign_flip_test(diffs, n_perm=10000, rng=None):
    """Paired two-sided sign-flip permutation test plus leave-one-out bounds and 95% CI."""
    if rng is None:
        rng = np.random.default_rng(777)
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    obs = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    loo = np.array([np.delete(d, k).mean() for k in range(n)])
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = np.mean(signs * d[None, :], axis=1)
    p_val = float((np.sum(np.abs(null) >= abs(obs)) + 1.0) / (n_perm + 1.0))
    return {
        "n": n,
        "mean_diff": round(obs, 4),
        "ci95_lo": round(obs - 1.96 * se, 4),
        "ci95_hi": round(obs + 1.96 * se, 4),
        "loo_min": round(float(loo.min()), 4),
        "loo_max": round(float(loo.max()), 4),
        "p_value": round(p_val, 4),
    }


def _box_mean(Z, lats, lons, lat_min, lat_max, lon_min, lon_max):
    m = (lats[:, None] >= lat_min) & (lats[:, None] <= lat_max) & (lons[None, :] >= lon_min) & (lons[None, :] <= lon_max)
    return np.nanmean(Z[:, m], axis=1)


def _expand_months(start_str, end_str):
    p0 = pd.Period(start_str, freq="M")
    p1 = pd.Period(end_str, freq="M")
    return [p0 + k for k in range((p1 - p0).n + 1)]


def takaya_nakamura_flux_200(z200_c, u200_c, v200_c, lats):
    """Horizontal stationary wave-activity flux W [m^2 s^-2] at 200 hPa (Takaya & Nakamura 2001)."""
    flip = lats[0] > lats[-1]
    if flip:
        la = lats[::-1]
        phi = z200_c[::-1, :]
        uc = u200_c[::-1, :]
        vc = v200_c[::-1, :]
    else:
        la = lats
        phi, uc, vc = z200_c, u200_c, v200_c

    f_cor = 2.0 * OMEGA_EARTH * np.sin(np.deg2rad(la))[:, None]
    f_safe = np.where(np.abs(la[:, None]) >= 10.0, f_cor, np.nan)
    psi = phi / f_safe

    dx = np.deg2rad(2.5) * R_EARTH * np.cos(np.deg2rad(la))[:, None]
    dy = np.deg2rad(2.5) * R_EARTH
    psi_x = np.gradient(psi, axis=1) / dx
    psi_y = np.gradient(psi, axis=0) / dy
    psi_xx = np.gradient(psi_x, axis=1) / dx
    psi_xy = np.gradient(psi_x, axis=0) / dy
    psi_yy = np.gradient(psi_y, axis=0) / dy

    umag = np.hypot(uc, vc)
    umag = np.where(umag >= 3.0, umag, np.nan)
    cos_lat = np.cos(np.deg2rad(la))[:, None]
    pref = (200.0 / 1000.0) * cos_lat / (2.0 * umag)

    wx = pref * (uc * (psi_x**2 - psi * psi_xx) + vc * (psi_x * psi_y - psi * psi_xy))
    wy = pref * (uc * (psi_x * psi_y - psi * psi_xy) + vc * (psi_y**2 - psi * psi_yy))
    if flip:
        return wx[::-1, :], wy[::-1, :]
    return wx, wy


def run_circulation_diagnostics(raw_dir, events_df, spi_basin, spi_periods, mask_info, n_perm=10000, seed=777):
    """Execute full NCEP-1 and NCEP-DOE-2 circulation, window, control, and wave-flux diagnostics."""
    rng = np.random.default_rng(seed)
    hgt = xr.open_dataset(f"{raw_dir}/ncep_hgt.mon.mean.nc")["hgt"].sel(time=slice("1961-01-01", "2025-12-31"))
    shum = xr.open_dataset(f"{raw_dir}/ncep_shum.mon.mean.nc")["shum"].sel(time=slice("1961-01-01", "2025-12-31"))
    uwnd = xr.open_dataset(f"{raw_dir}/ncep_uwnd.mon.mean.nc")["uwnd"].sel(time=slice("1961-01-01", "2025-12-31"))
    vwnd = xr.open_dataset(f"{raw_dir}/ncep_vwnd.mon.mean.nc")["vwnd"].sel(time=slice("1961-01-01", "2025-12-31"))
    ps_ds = xr.open_dataset(f"{raw_dir}/ncep_pres_sfc_asia.nc")["pres"].sel(time=slice("1961-01-01", "2025-12-31"))

    lat_a = ps_ds.lat.values.astype(float)
    lon_a = ps_ds.lon.values.astype(float)
    hgt_s = hgt.sel(lat=lat_a, lon=lon_a)
    shum_s = shum.sel(lat=lat_a, lon=lon_a)
    uwnd_s = uwnd.sel(lat=lat_a, lon=lon_a)
    vwnd_s = vwnd.sel(lat=lat_a, lon=lon_a)

    tl_nc = pd.PeriodIndex(np.datetime_as_string(hgt_s.time.values, unit="M"), freq="M")
    tl_map = {p: i for i, p in enumerate(tl_nc)}
    mons_nc = np.array([p.month for p in tl_nc])

    lv = shum_s.level.values
    lv = lv[(lv >= 300) & (lv <= 1000)]
    qu_ps, qv_ps, bad_t = vertical_vapor_flux_surface_bounded(
        shum_s.sel(level=lv).values.astype(float),
        uwnd_s.sel(level=lv).values.astype(float),
        vwnd_s.sel(level=lv).values.astype(float),
        lv,
        ps_ds.values.astype(float),
    )

    # Fixed 1000 hPa reference integration for sensitivity comparison
    lv_ord = np.argsort(lv)
    lv_pa = lv[lv_ord] * 100.0
    qv_ref = shum_s.sel(level=lv[lv_ord]).values.astype(float)
    qv_ref[~np.isfinite(qv_ref)] = 0.0
    if np.nanmax(qv_ref) > 0.2:
        qv_ref *= 1e-3
    uv_ref = uwnd_s.sel(level=lv[lv_ord]).values.astype(float)
    vv_ref = vwnd_s.sel(level=lv[lv_ord]).values.astype(float)
    qu_10 = np.trapz(qv_ref * uv_ref / G_ACCEL, x=lv_pa, axis=1)
    qv_10 = np.trapz(qv_ref * vv_ref / G_ACCEL, x=lv_pa, axis=1)
    if bad_t.any():
        qu_10[bad_t] = np.nan
        qv_10[bad_t] = np.nan

    wgrid = build_reanalysis_basin_mask(lat_a, lon_a, mask_info["lat"], mask_info["lon"], mask_info["frac"])
    div_ps_3d = spherical_divergence(qu_ps, qv_ps, lat_a)
    div_10_3d = spherical_divergence(qu_10, qv_10, lat_a)

    div_ps_series = np.sum(div_ps_3d * wgrid[None, :, :], axis=(1, 2))
    div_10_series = np.sum(div_10_3d * wgrid[None, :, :], axis=(1, 2))
    if bad_t.any():
        div_ps_series[bad_t] = np.nan
        div_10_series[bad_t] = np.nan

    DIV_ps = detrend_monthly_std_1d(div_ps_series, mons_nc)
    DIV_10 = detrend_monthly_std_1d(div_10_series, mons_nc)

    h500 = hgt_s.sel(level=500, drop=True).values.astype(float)
    h200 = hgt_s.sel(level=200, drop=True).values.astype(float)
    h850 = hgt_s.sel(level=850, drop=True).values.astype(float)
    Z500 = detrend_monthly_std_3d(h500, mons_nc)
    Z200 = detrend_monthly_std_3d(h200, mons_nc)
    Z850 = detrend_monthly_std_3d(h850, mons_nc)

    DI500 = _box_mean(Z500, lat_a, lon_a, 35, 50, 125, 150) - _box_mean(Z500, lat_a, lon_a, 25, 42, 90, 110)
    DI200 = _box_mean(Z200, lat_a, lon_a, 35, 50, 125, 150) - _box_mean(Z200, lat_a, lon_a, 25, 42, 90, 110)
    EAP850 = (
        _box_mean(Z850, lat_a, lon_a, 15, 25, 110, 130)
        - _box_mean(Z850, lat_a, lon_a, 27.5, 40, 115, 135)
        + _box_mean(Z850, lat_a, lon_a, 42.5, 55, 115, 140)
    )

    strata = [
        ("All (1961-2025)", events_df),
        ("Flood (May-Sep)", events_df[events_df["trans_month"].between(5, 9)]),
        ("Winter (Dec-Feb)", events_df[events_df["trans_month"].isin([12, 1, 2])]),
        ("Shoulder (Mar-Apr, Oct-Nov)", events_df[~events_df["trans_month"].isin([5, 6, 7, 8, 9, 12, 1, 2])]),
        ("Post-1979 All", events_df[events_df["year"] >= 1979]),
        ("Post-1979 Flood", events_df[(events_df["year"] >= 1979) & events_df["trans_month"].between(5, 9)]),
    ]
    indices = [
        ("DIV_ps", DIV_ps),
        ("DIV_1000hPa_ref", DIV_10),
        ("DI500", DI500),
        ("DI200", DI200),
        ("EAP850", EAP850),
    ]

    summary_rows = []
    for iname, ser in indices:
        for sname, sub in strata:
            diffs = [
                ser[tl_map[pd.Period(r["wet_start"], freq="M")]]
                - float(np.mean([ser[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
                for _, r in sub.iterrows()
            ]
            res = paired_sign_flip_test(diffs, n_perm=n_perm, rng=rng)
            res.update({"index": iname, "stratum": sname})
            summary_rows.append(res)
    summary_df = pd.DataFrame(summary_rows)
    # Apply BH-FDR across the primary 16 seasonal/whole-basin tests (DIV_ps, DI500, DI200, EAP850 x 4 core strata)
    prim_mask = summary_df["index"].isin(["DIV_ps", "DI500", "DI200", "EAP850"]) & summary_df["stratum"].isin([
        "All (1961-2025)", "Flood (May-Sep)", "Winter (Dec-Feb)", "Shoulder (Mar-Apr, Oct-Nov)"
    ])
    summary_df["q_FDR"] = np.nan
    summary_df.loc[prim_mask, "q_FDR"] = np.round(bh_fdr(summary_df.loc[prim_mask, "p_value"].to_numpy()), 4)

    # Fixed event-window trajectories [t0-3 .. t0+1] for All and Flood strata
    mo_pools = {mo: np.where(mons_nc == mo)[0] for mo in range(1, 13)}
    win_rows = []
    for sname, sub in [("All", events_df), ("Flood", events_df[events_df["trans_month"].between(5, 9)])]:
        t0_idx = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in sub.iterrows()]
        t0_mos = [pd.Period(r["wet_start"], freq="M").month for _, r in sub.iterrows()]
        for iname, ser in [("DIV_ps", DIV_ps), ("DI500", DI500), ("DI200", DI200), ("EAP850", EAP850)]:
            for lag in (-3, -2, -1, 0, 1):
                vals = [ser[i0 + lag] for i0 in t0_idx if 0 <= i0 + lag < len(ser)]
                lag_mos = [((m0 - 1 + lag) % 12) + 1 for m0 in t0_mos]
                picks = np.column_stack([rng.choice(mo_pools[lm], size=1000) for lm in lag_mos])
                boot_band = np.nanmean(ser[picks], axis=1)
                win_rows.append({
                    "stratum": sname,
                    "index": iname,
                    "lag": lag,
                    "mean": round(float(np.mean(vals)), 3),
                    "se": round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 3),
                    "null_p05": round(float(np.percentile(boot_band, 5)), 3),
                    "null_p95": round(float(np.percentile(boot_band, 95)), 3),
                })
    windows_df = pd.DataFrame(win_rows)

    # Matched ordinary wet-month control (excluding all DFAA wet-spell months)
    spi_map = {p: i for i, p in enumerate(spi_periods)}
    all_wet = set()
    for _, r in events_df.iterrows():
        all_wet.update(_expand_months(r["wet_start"], r["wet_end"]))

    ctrl_rows = []
    for iname, ser in [("DIV_ps", DIV_ps), ("DI500", DI500), ("DI200", DI200), ("EAP850", EAP850)]:
        d_all, d_fl = [], []
        for idx_r, r in events_df.iterrows():
            t0 = pd.Period(r["wet_start"], freq="M")
            spi0 = spi_basin[spi_map[t0]]
            cands = [
                tl_map[p]
                for i, p in enumerate(spi_periods)
                if p.month == t0.month and p not in all_wet and abs(spi_basin[i] - spi0) <= 0.1 and p in tl_map
            ]
            if not cands:
                continue
            diff = float(ser[tl_map[t0]] - np.mean([ser[c] for c in cands]))
            d_all.append(diff)
            if 5 <= int(r["trans_month"]) <= 9:
                d_fl.append(diff)
        for sname, arr in [("All matched", d_all), ("Flood matched", d_fl)]:
            res = paired_sign_flip_test(arr, n_perm=n_perm, rng=rng)
            res.update({"index": iname, "stratum": sname})
            ctrl_rows.append(res)
    controls_df = pd.DataFrame(ctrl_rows)

    # Out-of-sample split-half and NCEP-DOE-2 cross-checks
    verif_rows = []
    h1 = events_df[events_df["year"] <= 1992]
    h2 = events_df[events_df["year"] >= 1993]
    for period_tag, sub in [("Fixed NE-PL 1961-1992", h1), ("Fixed NE-PL 1993-2025", h2)]:
        diffs = [
            DI500[tl_map[pd.Period(r["wet_start"], freq="M")]]
            - float(np.mean([DI500[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
            for _, r in sub.iterrows()
        ]
        res = paired_sign_flip_test(diffs, n_perm=n_perm, rng=rng)
        res.update({"check": period_tag, "dataset": "NCEP-1 500 hPa"})
        verif_rows.append(res)

    # Empirical 1961-1992 decile centers evaluated strictly out-of-sample on 1993-2025
    i_h1 = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in h1.iterrows()]
    comp_h1 = np.nanmean(Z500[i_h1], axis=0)
    ea_mask = (lat_a[:, None] >= 15) & (lat_a[:, None] <= 60) & (lon_a[None, :] >= 65) & (lon_a[None, :] <= 165)
    v_ea = comp_h1[ea_mask]
    hi_m = ea_mask & (comp_h1 >= np.percentile(v_ea, 90))
    lo_m = ea_mask & (comp_h1 <= np.percentile(v_ea, 10))
    DI_emp = np.nanmean(Z500[:, hi_m], axis=1) - np.nanmean(Z500[:, lo_m], axis=1)
    diffs_emp = [
        DI_emp[tl_map[pd.Period(r["wet_start"], freq="M")]]
        - float(np.mean([DI_emp[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
        for _, r in h2.iterrows()
    ]
    res_emp = paired_sign_flip_test(diffs_emp, n_perm=n_perm, rng=rng)
    res_emp.update({"check": "Empirical 1961-92 deciles OOS (1993-2025)", "dataset": "NCEP-1 500 hPa"})
    verif_rows.append(res_emp)

    # NCEP-DOE-2 500 hPa and 200 hPa independent verification (1979-2025)
    for lev_lbl, fname in [("NCEP-DOE-2 500 hPa", "ncep2_hgt500_asia.nc"), ("NCEP-DOE-2 200 hPa", "ncep2_hgt200_asia.nc")]:
        ds2 = xr.open_dataset(f"{raw_dir}/{fname}")
        z2_raw = ds2["hgt"].squeeze("level", drop=True).values.astype(float)
        la2, lo2 = ds2.lat.values.astype(float), ds2.lon.values.astype(float)
        tl2 = pd.PeriodIndex(np.datetime_as_string(ds2.time.values, unit="M"), freq="M")
        tl2_map = {p: i for i, p in enumerate(tl2)}
        m2 = np.array([p.month for p in tl2])
        z2_std = detrend_monthly_std_3d(z2_raw, m2)
        di2 = _box_mean(z2_std, la2, lo2, 35, 50, 125, 150) - _box_mean(z2_std, la2, lo2, 25, 42, 90, 110)
        for tag, sub in [("Post-1979 All", events_df[events_df["year"] >= 1979]),
                         ("Post-1979 Flood", events_df[(events_df["year"] >= 1979) & events_df["trans_month"].between(5, 9)])]:
            diffs = [
                di2[tl2_map[pd.Period(r["wet_start"], freq="M")]]
                - float(np.mean([di2[tl2_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
                for _, r in sub.iterrows()
            ]
            res2 = paired_sign_flip_test(diffs, n_perm=n_perm, rng=rng)
            res2.update({"check": tag, "dataset": lev_lbl})
            verif_rows.append(res2)

    verif_df = pd.DataFrame(verif_rows)

    # Spatial composites & wave-activity flux payload for figures
    b_fl = events_df[events_df["trans_month"].between(5, 9)]
    i_all = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in events_df.iterrows()]
    i_fl = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in b_fl.iterrows()]

    def _mean_q(arr, idx):
        return np.nanmean(arr[idx], axis=0)

    # May-Sep moisture flux anomaly and 5880 gpm ridge position
    qu_m, qv_m = np.zeros_like(qu_ps), np.zeros_like(qv_ps)
    u200 = uwnd_s.sel(level=200, drop=True).values.astype(float)
    v200 = vwnd_s.sel(level=200, drop=True).values.astype(float)
    phi200 = np.zeros_like(h200)
    for mo in range(1, 13):
        idx = np.where(mons_nc == mo)[0]
        qu_m[idx] = qu_ps[idx] - np.nanmean(qu_ps[idx], axis=0)
        qv_m[idx] = qv_ps[idx] - np.nanmean(qv_ps[idx], axis=0)
        phi200[idx] = (h200[idx] - np.nanmean(h200[idx], axis=0)) * G_ACCEL

    h500_fl = np.nanmean(h500[i_fl], axis=0)
    h500_fl_clim = np.nanmean(h500[np.isin(mons_nc, [5, 6, 7, 8, 9])], axis=0)
    u200_fl_clim = np.nanmean(u200[np.isin(mons_nc, [5, 6, 7, 8, 9])], axis=0)
    v200_fl_clim = np.nanmean(v200[np.isin(mons_nc, [5, 6, 7, 8, 9])], axis=0)
    wx_200, wy_200 = takaya_nakamura_flux_200(_mean_q(phi200, i_fl), u200_fl_clim, v200_fl_clim, lat_a)

    fields = {
        "lat": lat_a,
        "lon": lon_a,
        "Z500_all": np.nanmean(Z500[i_all], axis=0),
        "Z500_flood": np.nanmean(Z500[i_fl], axis=0),
        "Z200_flood": np.nanmean(Z200[i_fl], axis=0),
        "h500_flood": h500_fl,
        "h500_flood_clim": h500_fl_clim,
        "qu_anom_all": _mean_q(qu_m, i_all),
        "qv_anom_all": _mean_q(qv_m, i_all),
        "qu_anom_flood": _mean_q(qu_m, i_fl),
        "qv_anom_flood": _mean_q(qv_m, i_fl),
        "div_anom_flood": _mean_q(div_ps_3d, i_fl) - np.nanmean(div_ps_3d[np.isin(mons_nc, [5, 6, 7, 8, 9])], axis=0),
        "u200_clim_flood": u200_fl_clim,
        "wx_200_flood": wx_200,
        "wy_200_flood": wy_200,
        "qu_ps_series": np.nansum(qu_ps * wgrid[None, :, :], axis=(1, 2)),
        "qv_ps_series": np.nansum(qv_ps * wgrid[None, :, :], axis=(1, 2)),
        "DIV_ps": DIV_ps,
        "DI500": DI500,
        "DI200": DI200,
        "EAP850": EAP850,
    }
    return summary_df, windows_df, controls_df, verif_df, fields


def seasonal_teleconnection_skill(raw_dir, events_df):
    """Evaluate winter (DJF) and spring (MAM) ONI/SOI/NAO indices for transition-year prediction."""
    from pathlib import Path
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    raw_dir = Path(raw_dir)
    mon_map = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
               "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    oni = {}
    for line in (raw_dir / "cpc_oni.ascii.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in mon_map:
            oni[(int(parts[1]), mon_map[parts[0]])] = float(parts[3])

    def _parse_psl(path):
        d = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 13 and parts[0].lstrip("-").isdigit():
                yr = int(parts[0])
                for m in range(1, 13):
                    val = float(parts[m])
                    if val > -90.0:
                        d[(yr, m)] = val
        return d

    soi = _parse_psl(raw_dir / "psl_soi.data")
    nao = _parse_psl(raw_dir / "psl_nao.data")
    ev_yrs = set(events_df["year"].astype(int))

    rows = []
    for yr in range(1961, 2026):
        sub = events_df[events_df["year"] == yr]
        m0 = int(pd.Period(sub.iloc[0]["dry_start"], freq="M").month) if not sub.empty else 6
        djf_y = yr if m0 >= 3 else yr - 1
        mam_y = yr if m0 >= 6 else yr - 1
        rows.append({
            "year": yr,
            "is_event_year": 1 if yr in ev_yrs else 0,
            "ONI_DJF": oni.get((djf_y, 1), np.nan),
            "SOI_DJF": float(np.nanmean([soi.get((djf_y - 1, 12), np.nan), soi.get((djf_y, 1), np.nan), soi.get((djf_y, 2), np.nan)])),
            "NAO_DJF": float(np.nanmean([nao.get((djf_y - 1, 12), np.nan), nao.get((djf_y, 1), np.nan), nao.get((djf_y, 2), np.nan)])),
            "ONI_MAM": oni.get((mam_y, 4), np.nan),
            "SOI_MAM": float(np.nanmean([soi.get((mam_y, 3), np.nan), soi.get((mam_y, 4), np.nan), soi.get((mam_y, 5), np.nan)])),
            "NAO_MAM": float(np.nanmean([nao.get((mam_y, 3), np.nan), nao.get((mam_y, 4), np.nan), nao.get((mam_y, 5), np.nan)])),
        })

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    feats = ["ONI_DJF", "SOI_DJF", "NAO_DJF", "ONI_MAM", "SOI_MAM", "NAO_MAM"]
    x_mat = df[feats].to_numpy()
    y_vec = df["is_event_year"].to_numpy()
    probs = np.zeros(len(y_vec), dtype=float)
    for i in range(len(y_vec)):
        train = np.arange(len(y_vec)) != i
        scaler = StandardScaler().fit(x_mat[train])
        clf = LogisticRegression(C=0.5).fit(scaler.transform(x_mat[train]), y_vec[train])
        probs[i] = float(clf.predict_proba(scaler.transform(x_mat[i : i + 1]))[0, 1])

    loo_auc = round(float(roc_auc_score(y_vec, probs)), 3)
    out_rows = []
    pvals = []
    for col in feats:
        d_ev = df.loc[df["is_event_year"] == 1, col]
        d_no = df.loc[df["is_event_year"] == 0, col]
        _, p_t = stats.ttest_ind(d_ev, d_no)
        pvals.append(float(p_t))
        out_rows.append({
            "predictor": col,
            "event_mean": round(float(d_ev.mean()), 3),
            "nonevent_mean": round(float(d_no.mean()), 3),
            "diff": round(float(d_ev.mean() - d_no.mean()), 3),
            "p_value": round(float(p_t), 4),
            "loo_auc_6feat": loo_auc,
        })
    res_df = pd.DataFrame(out_rows)
    res_df["q_FDR"] = np.round(bh_fdr(pvals), 4)
    return res_df
