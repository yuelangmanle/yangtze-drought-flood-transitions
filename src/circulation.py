"""Moisture transport divergence, dipole teleconnections, wave-activity flux, and reanalysis checks."""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
from src.dependence import bh_fdr


G_ACCEL = 9.80665
R_EARTH = 6.371e6
OMEGA_EARTH = 7.2921e-5


def detrend_monthly_anom_1d(values, months):
    """Remove calendar-month linear trend and climatology while retaining physical units."""
    out = np.full(len(values), np.nan, dtype=float)
    for mo in range(1, 13):
        idx = np.where((months == mo) & np.isfinite(values))[0]
        if len(idx) < 5:
            continue
        x = values[idx]
        t = np.arange(len(idx), dtype=float)
        out[idx] = x - np.polyval(np.polyfit(t, x, 1), t)
    return out


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
        sd = float(np.std(xd, ddof=1))
        out[idx] = xd / sd if sd > 0 else np.nan
    return out


def detrend_monthly_std_3d(field, months):
    """Linearly detrend and standardize each grid point by calendar month."""
    out = np.full_like(field, np.nan, dtype=float)
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        x = field[idx].copy()
        t = np.arange(len(idx), dtype=float)
        tc = t - t.mean()
        ok = np.isfinite(x)
        x_fill = np.where(ok, x, 0.0)
        mean_x = np.nanmean(x, axis=0)
        slope = np.nansum(tc[:, None, None] * (x_fill - mean_x[None, :, :]), axis=0) / np.sum(tc**2)
        xd = x - (mean_x[None, :, :] + slope[None, :, :] * tc[:, None, None])
        sd = np.nanstd(xd, axis=0, ddof=1)
        sd[sd == 0] = np.nan
        out[idx] = xd / sd[None, :, :]
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


def specific_humidity_from_rh(air_k, rhum_pct, levels_hpa):
    """Convert air temperature [K] and relative humidity [%] on pressure levels to specific humidity [kg/kg]."""
    t_c = air_k - 273.15
    es_hpa = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e_hpa = np.clip(rhum_pct / 100.0, 0.0, 1.0) * es_hpa
    p_bcast = levels_hpa[None, :, None, None]
    return 0.622 * e_hpa / np.maximum(p_bcast - 0.378 * e_hpa, 1.0)


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
    """Map the 0.5-deg basin mask onto full 2.5-deg reanalysis cells with half-split boundary weights."""
    cos_b = np.cos(np.deg2rad(basin_lats))[:, None]
    w_fine = basin_frac * cos_b
    dlat_r = float(abs(lat_nc[1] - lat_nc[0])) * 0.5 if len(lat_nc) > 1 else 1.25
    dlon_r = float(abs(lon_nc[1] - lon_nc[0])) * 0.5 if len(lon_nc) > 1 else 1.25
    w = np.zeros((len(lat_nc), len(lon_nc)), dtype=float)
    for i, la in enumerate(lat_nc):
        dla = np.abs(basin_lats - la)
        wla = np.where(dla < dlat_r - 1e-9, 1.0, np.where(np.abs(dla - dlat_r) <= 1e-9, 0.5, 0.0))
        for j, lo in enumerate(lon_nc):
            dlo = np.abs(basin_lons - lo)
            wlo = np.where(dlo < dlon_r - 1e-9, 1.0, np.where(np.abs(dlo - dlon_r) <= 1e-9, 0.5, 0.0))
            val = float(np.sum(w_fine * wla[:, None] * wlo[None, :]))
            if val > 0.0:
                w[i, j] = val
    w /= w.sum()
    return w


def paired_sign_flip_test(diffs, n_perm=10000, rng=None):
    """Paired two-sided sign-flip permutation test plus LOO bounds and exact Student-t 95% CI."""
    if rng is None:
        rng = np.random.default_rng(777)
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    obs = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=max(n - 1, 1)))
    loo = np.array([np.delete(d, k).mean() for k in range(n)])
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = np.mean(signs * d[None, :], axis=1)
    p_val = float((np.sum(np.abs(null) >= abs(obs)) + 1.0) / (n_perm + 1.0))
    return {
        "n": n,
        "mean_diff": round(obs, 4),
        "ci95_lo": round(obs - t_crit * se, 4),
        "ci95_hi": round(obs + t_crit * se, 4),
        "loo_min": round(float(loo.min()), 4),
        "loo_max": round(float(loo.max()), 4),
        "p_value": round(p_val, 4),
    }


def _box_mean(Z, lats, lons, lat_min, lat_max, lon_min, lon_max, valid_2d=None):
    """Latitude-cosine-weighted box average over valid (non-NaN) grid cells."""
    m = (lats[:, None] >= lat_min) & (lats[:, None] <= lat_max) & (lons[None, :] >= lon_min) & (lons[None, :] <= lon_max)
    if valid_2d is not None:
        m = m & valid_2d
    w2 = np.where(m, np.cos(np.deg2rad(lats))[:, None], 0.0)
    vals = Z[:, m]
    w_vec = w2[m]
    finite = np.isfinite(vals)
    w_mat = np.where(finite, w_vec[None, :], 0.0)
    num = np.sum(np.where(finite, vals, 0.0) * w_mat, axis=1)
    den = np.sum(w_mat, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def _expand_months(start_str, end_str):
    p0 = pd.Period(start_str, freq="M")
    p1 = pd.Period(end_str, freq="M")
    return [p0 + k for k in range((p1 - p0).n + 1)]


def takaya_nakamura_flux_200(z200_c, u200_c, v200_c, lats, u_min=10.0, lat_min=25.0):
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
    valid_w = (np.abs(la[:, None]) >= lat_min) & (umag >= u_min)
    umag_safe = np.where(valid_w, umag, np.nan)
    cos_lat = np.cos(np.deg2rad(la))[:, None]
    pref = (200.0 / 1000.0) * cos_lat / (2.0 * umag_safe)

    wx = pref * (uc * (psi_x**2 - psi * psi_xx) + vc * (psi_x * psi_y - psi * psi_xy))
    wy = pref * (uc * (psi_x * psi_y - psi * psi_xy) + vc * (psi_y**2 - psi * psi_yy))
    if flip:
        return wx[::-1, :], wy[::-1, :]
    return wx, wy


def _ridge_western_edge(z_field, lats, lons, target=5880.0):
    """Return westernmost longitude [100E..180E] where Z500 >= target inside 15N..32.5N."""
    m_lat = (lats >= 15.0) & (lats <= 32.5)
    m_lon = lons >= 100.0
    sub = z_field[np.ix_(m_lat, m_lon)]
    lo_sub = lons[m_lon]
    hit_cols = np.where(np.nanmax(sub, axis=0) >= target)[0]
    if len(hit_cols) == 0:
        return np.nan
    return float(lo_sub[hit_cols[0]])


def run_circulation_diagnostics(raw_dir, events_df, spi_basin, spi_periods, mask_info, n_perm=10000, seed=777):
    """Execute NCEP-1 and NCEP-DOE-2 circulation, matched-control, OOS, and wave-flux diagnostics."""
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
    ps_vals_hpa = ps_ds.values.astype(float)
    if np.nanmax(ps_vals_hpa) > 2000.0:
        ps_vals_hpa = ps_vals_hpa / 100.0

    qu_ps, qv_ps, bad_t = vertical_vapor_flux_surface_bounded(
        shum_s.sel(level=lv).values.astype(float),
        uwnd_s.sel(level=lv).values.astype(float),
        vwnd_s.sel(level=lv).values.astype(float),
        lv,
        ps_vals_hpa,
    )

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

    # Deseasonalized stationary-flow moisture divergence [mm day^-1] alongside standardized shifts
    div_ps_mm_day = detrend_monthly_anom_1d(div_ps_series, mons_nc) * 86400.0
    DIV_ps = detrend_monthly_std_1d(div_ps_series, mons_nc)
    DIV_10 = detrend_monthly_std_1d(div_10_series, mons_nc)

    h500 = hgt_s.sel(level=500, drop=True).values.astype(float)
    h200 = hgt_s.sel(level=200, drop=True).values.astype(float)
    h850 = hgt_s.sel(level=850, drop=True).values.astype(float)
    Z500 = detrend_monthly_std_3d(h500, mons_nc)
    Z200 = detrend_monthly_std_3d(h200, mons_nc)
    Z850 = detrend_monthly_std_3d(h850, mons_nc)

    # Mask 850 hPa cells whose climatological surface pressure falls below 850 hPa (>1500 m)
    ps_clim_hpa = np.nanmean(ps_vals_hpa, axis=0)
    mask_850 = ps_clim_hpa >= 850.0

    # Standardize composite indices AFTER box differencing using exact Huang (2004) EAP850 over 115E-140E
    DI500_raw = _box_mean(Z500, lat_a, lon_a, 35, 50, 125, 150) - _box_mean(Z500, lat_a, lon_a, 25, 42, 90, 110)
    DI200_raw = _box_mean(Z200, lat_a, lon_a, 35, 50, 125, 150) - _box_mean(Z200, lat_a, lon_a, 25, 42, 90, 110)
    EAP850_S = _box_mean(Z850, lat_a, lon_a, 15, 25, 115, 140, valid_2d=mask_850)
    EAP850_M = _box_mean(Z850, lat_a, lon_a, 30, 40, 115, 140, valid_2d=mask_850)
    EAP850_N = _box_mean(Z850, lat_a, lon_a, 50, 60, 115, 140, valid_2d=mask_850)
    EAP850_raw = -0.25 * EAP850_S + 0.50 * EAP850_M - 0.25 * EAP850_N

    DI500 = detrend_monthly_std_1d(DI500_raw, mons_nc)
    DI200 = detrend_monthly_std_1d(DI200_raw, mons_nc)
    EAP850 = detrend_monthly_std_1d(EAP850_raw, mons_nc)
    NE500_pole = detrend_monthly_std_1d(_box_mean(Z500, lat_a, lon_a, 35, 50, 125, 150), mons_nc)
    PL500_pole = detrend_monthly_std_1d(_box_mean(Z500, lat_a, lon_a, 25, 42, 90, 110), mons_nc)
    PL500_west_pole = detrend_monthly_std_1d(_box_mean(Z500, lat_a, lon_a, 25, 42, 70, 90), mons_nc)
    EAP850_S_pole = detrend_monthly_std_1d(EAP850_S, mons_nc)
    EAP850_M_pole = detrend_monthly_std_1d(EAP850_M, mons_nc)
    EAP850_N_pole = detrend_monthly_std_1d(EAP850_N, mons_nc)

    flood_ov_mask = events_df.apply(
        lambda r: any(5 <= p.month <= 9 for p in _expand_months(r["wet_start"], r["wet_end"])),
        axis=1,
    )
    strata = [
        ("All (1961-2025)", events_df),
        ("Flood (May-Sep)", events_df[events_df["trans_month"].between(5, 9)]),
        ("JJA (Jun-Aug)", events_df[events_df["trans_month"].isin([6, 7, 8])]),
        ("Flood Overlap (May-Sep)", events_df[flood_ov_mask]),
        ("Spring (Mar-Apr)", events_df[events_df["trans_month"].isin([3, 4])]),
        ("Autumn (Oct-Nov)", events_df[events_df["trans_month"].isin([10, 11])]),
        ("Winter (Dec-Feb)", events_df[events_df["trans_month"].isin([12, 1, 2])]),
        ("Immediate g=1", events_df[events_df["gap"] == 1]),
        ("Short-gap g<=2", events_df[events_df["gap"] <= 2]),
        ("Post-1979 All", events_df[events_df["year"] >= 1979]),
        ("Post-1979 Flood", events_df[(events_df["year"] >= 1979) & events_df["trans_month"].between(5, 9)]),
    ]
    indices = [
        ("DIV_ps", DIV_ps),
        ("DIV_1000hPa_ref", DIV_10),
        ("DI500", DI500),
        ("DI200", DI200),
        ("EAP850", EAP850),
        ("NE500_pole", NE500_pole),
        ("PL500_pole", PL500_pole),
        ("PL500_west_pole", PL500_west_pole),
        ("EAP850_S_pole", EAP850_S_pole),
        ("EAP850_M_pole", EAP850_M_pole),
        ("EAP850_N_pole", EAP850_N_pole),
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
            if iname == "DIV_ps":
                phys_diffs = [
                    div_ps_mm_day[tl_map[pd.Period(r["wet_start"], freq="M")]]
                    - float(np.mean([div_ps_mm_day[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
                    for _, r in sub.iterrows()
                ]
                res["mean_diff_mm_day"] = round(float(np.nanmean(phys_diffs)), 2)
            else:
                res["mean_diff_mm_day"] = np.nan
            res.update({"index": iname, "stratum": sname})
            summary_rows.append(res)
    summary_df = pd.DataFrame(summary_rows)
    prim_mask = summary_df["index"].isin(["DIV_ps", "DI500", "DI200", "EAP850"]) & summary_df["stratum"].isin([
        "All (1961-2025)", "Flood (May-Sep)", "Spring (Mar-Apr)", "Autumn (Oct-Nov)", "Winter (Dec-Feb)"
    ])
    summary_df["q_FDR"] = np.nan
    summary_df.loc[prim_mask, "q_FDR"] = np.round(bh_fdr(summary_df.loc[prim_mask, "p_value"].to_numpy()), 4)

    # Event-window trajectories [t_w0-3 .. t_w0+1] across All, Flood, Flood Overlap, and gap strata
    mo_pools = {mo: np.where(mons_nc == mo)[0] for mo in range(1, 13)}
    win_rows = []
    for sname, sub in [
        ("All", events_df),
        ("Flood", events_df[events_df["trans_month"].between(5, 9)]),
        ("Flood Overlap", events_df[flood_ov_mask]),
        ("Immediate g=1", events_df[events_df["gap"] == 1]),
        ("Short-gap g<=2", events_df[events_df["gap"] <= 2]),
    ]:
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

    # Matched ordinary wet-month controls (all-wet & onset-only controls, strict >= +0.5 pool)
    spi_map = {p: i for i, p in enumerate(spi_periods)}
    all_wet = set()
    for _, r in events_df.iterrows():
        all_wet.update(_expand_months(r["wet_start"], r["wet_end"]))

    ctrl_schemes = [
        ("tol_0.1", 0.1, False, False),
        ("tol_0.2_nn_all39", 0.2, True, False),
        ("tol_0.1_onset", 0.1, False, True),
        ("tol_0.2_nn_onset", 0.2, True, True),
    ]
    ctrl_indices = [
        ("DIV_ps", DIV_ps),
        ("DI500", DI500),
        ("DI200", DI200),
        ("EAP850", EAP850),
        ("NE500_pole", NE500_pole),
        ("PL500_pole", PL500_pole),
    ]
    ctrl_rows = []
    ord_wet_flood_pools = {}
    for tol_lbl, tol, allow_nn, onset_only in ctrl_schemes:
        for iname, ser in ctrl_indices:
            ev_diff_list = []
            for ev_pos, (_, r) in enumerate(events_df.iterrows()):
                t0 = pd.Period(r["wet_start"], freq="M")
                spi0 = spi_basin[spi_map[t0]]
                cands = []
                for i, p in enumerate(spi_periods):
                    if p.month != t0.month or p in all_wet or p not in tl_map:
                        continue
                    if not np.isfinite(spi_basin[i]) or spi_basin[i] < 0.5:
                        continue
                    if onset_only and i > 0 and np.isfinite(spi_basin[i - 1]) and spi_basin[i - 1] >= 0.5:
                        continue
                    if abs(spi_basin[i] - spi0) <= tol:
                        cands.append(tl_map[p])
                if not cands and allow_nn:
                    mo_cands = []
                    for i, p in enumerate(spi_periods):
                        if p.month != t0.month or p in all_wet or p not in tl_map:
                            continue
                        if not np.isfinite(spi_basin[i]) or spi_basin[i] < 0.5:
                            continue
                        if onset_only and i > 0 and np.isfinite(spi_basin[i - 1]) and spi_basin[i - 1] >= 0.5:
                            continue
                        mo_cands.append((abs(spi_basin[i] - spi0), tl_map[p]))
                    if mo_cands:
                        mo_cands.sort(key=lambda x: x[0])
                        cands = [c[1] for c in mo_cands[:3]]
                if not cands:
                    continue
                if tol_lbl in ("tol_0.2_nn_all39", "tol_0.2_nn_onset") and iname == "DI500" and 5 <= int(r["trans_month"]) <= 9:
                    ord_wet_flood_pools.setdefault(tol_lbl, []).append(cands)
                diff = float(ser[tl_map[t0]] - np.nanmean([ser[c] for c in cands]))
                ev_diff_list.append({
                    "diff": diff,
                    "case_idx": tl_map[t0],
                    "cands": cands,
                    "year": int(r["year"]),
                    "trans_month": int(r["trans_month"]),
                    "flood_ov": bool(flood_ov_mask.iloc[ev_pos]),
                })

            ctrl_strata = [
                ("All matched", [d for d in ev_diff_list]),
                ("Flood matched", [d for d in ev_diff_list if 5 <= d["trans_month"] <= 9]),
                ("JJA matched", [d for d in ev_diff_list if d["trans_month"] in (6, 7, 8)]),
                ("Flood Overlap matched", [d for d in ev_diff_list if d["flood_ov"]]),
                ("1961-1992 matched", [d for d in ev_diff_list if d["year"] <= 1992]),
                ("1993-2025 matched", [d for d in ev_diff_list if d["year"] >= 1993]),
            ]
            for sname, slist in ctrl_strata:
                arr = [d["diff"] for d in slist]
                cases_i = [d["case_idx"] for d in slist]
                pools_i = [d["cands"] for d in slist]
                if len(arr) < 3:
                    continue
                res = paired_sign_flip_test(arr, n_perm=n_perm, rng=rng)
                obs_m = float(np.mean(arr))
                null_cc = np.empty(n_perm, dtype=float)
                for b_i in range(n_perm):
                    b_diffs = []
                    for c_idx, p_idxs in zip(cases_i, pools_i):
                        union_idxs = [c_idx] + list(p_idxs)
                        pick_case = union_idxs[int(rng.integers(0, len(union_idxs)))]
                        rem = [u_idx for u_idx in union_idxs if u_idx != pick_case]
                        b_diffs.append(ser[pick_case] - np.nanmean(ser[rem]))
                    null_cc[b_i] = float(np.nanmean(b_diffs))
                p_cc = float((np.sum(np.abs(null_cc) >= abs(obs_m)) + 1.0) / (n_perm + 1.0))
                res.update({
                    "scheme": tol_lbl,
                    "index": iname,
                    "stratum": sname,
                    "contrast": f"{iname}(t_w0) minus matched wet",
                    "pool_perm_p": round(p_cc, 4),
                })
                ctrl_rows.append(res)
    controls_df = pd.DataFrame(ctrl_rows)
    controls_df["q_FDR"] = np.nan
    for slbl, _, _, _ in ctrl_schemes:
        m_c = (
            (controls_df["scheme"] == slbl)
            & controls_df["index"].isin(["DIV_ps", "DI500", "DI200", "EAP850"])
            & controls_df["stratum"].isin(["All matched", "Flood matched"])
        )
        if m_c.any():
            controls_df.loc[m_c, "q_FDR"] = np.round(bh_fdr(controls_df.loc[m_c, "p_value"].to_numpy()), 4)

    # Split-half, seasonal/satellite splits, genuine OOS, non-YRB upstream boxes, & NCEP-DOE-2 verification
    verif_rows = []
    h1 = events_df[events_df["year"] <= 1992]
    h2 = events_df[events_df["year"] >= 1993]
    for period_tag, sub in [
        ("Fixed NE-PL split 1961-1992 (All)", h1),
        ("Fixed NE-PL split 1993-2025 (All)", h2),
        ("Fixed NE-PL split 1961-1992 (Flood)", h1[h1["trans_month"].between(5, 9)]),
        ("Fixed NE-PL split 1993-2025 (Flood)", h2[h2["trans_month"].between(5, 9)]),
        ("Fixed NE-PL satellite 1979-2001 (All)", events_df[(events_df["year"] >= 1979) & (events_df["year"] <= 2001)]),
        ("Fixed NE-PL satellite 2002-2025 (All)", events_df[events_df["year"] >= 2002]),
    ]:
        diffs = [
            DI500[tl_map[pd.Period(r["wet_start"], freq="M")]]
            - float(np.mean([DI500[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
            for _, r in sub.iterrows()
        ]
        res = paired_sign_flip_test(diffs, n_perm=n_perm, rng=rng)
        res.update({"check": period_tag, "dataset": "NCEP-1 500 hPa"})
        verif_rows.append(res)

    # Empirical 1961-1992 decile centers evaluated strictly out-of-sample on 1993-2025 (wet-dry and matched)
    i_h1 = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in h1.iterrows()]
    comp_h1 = np.nanmean(Z500[i_h1], axis=0)
    ea_mask = (lat_a[:, None] >= 15) & (lat_a[:, None] <= 60) & (lon_a[None, :] >= 65) & (lon_a[None, :] <= 165)
    v_ea = comp_h1[ea_mask]
    hi_m = ea_mask & (comp_h1 >= np.percentile(v_ea, 90))
    lo_m = ea_mask & (comp_h1 <= np.percentile(v_ea, 10))
    DI_emp_raw = _box_mean(Z500, lat_a, lon_a, 15, 60, 65, 165, valid_2d=hi_m) - _box_mean(Z500, lat_a, lon_a, 15, 60, 65, 165, valid_2d=lo_m)
    DI_emp = detrend_monthly_std_1d(DI_emp_raw, mons_nc)
    diffs_emp = [
        DI_emp[tl_map[pd.Period(r["wet_start"], freq="M")]]
        - float(np.mean([DI_emp[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
        for _, r in h2.iterrows()
    ]
    res_emp = paired_sign_flip_test(diffs_emp, n_perm=n_perm, rng=rng)
    res_emp.update({"check": "Empirical 1961-92 deciles OOS (1993-2025)", "dataset": "NCEP-1 500 hPa"})
    verif_rows.append(res_emp)

    # Flood-only empirical 1961-1992 decile OOS check on 1993-2025 Flood
    h1_fl = h1[h1["trans_month"].between(5, 9)]
    h2_fl = h2[h2["trans_month"].between(5, 9)]
    i_h1_fl = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in h1_fl.iterrows()]
    comp_h1_fl = np.nanmean(Z500[i_h1_fl], axis=0)
    v_fl = comp_h1_fl[ea_mask]
    hi_fl = ea_mask & (comp_h1_fl >= np.percentile(v_fl, 90))
    lo_fl = ea_mask & (comp_h1_fl <= np.percentile(v_fl, 10))
    DI_emp_fl = detrend_monthly_std_1d(
        _box_mean(Z500, lat_a, lon_a, 15, 60, 65, 165, valid_2d=hi_fl)
        - _box_mean(Z500, lat_a, lon_a, 15, 60, 65, 165, valid_2d=lo_fl),
        mons_nc,
    )
    diffs_emp_fl = [
        DI_emp_fl[tl_map[pd.Period(r["wet_start"], freq="M")]]
        - float(np.mean([DI_emp_fl[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
        for _, r in h2_fl.iterrows()
    ]
    res_emp_fl = paired_sign_flip_test(diffs_emp_fl, n_perm=n_perm, rng=rng)
    res_emp_fl.update({"check": "Empirical 1961-92 deciles OOS Flood (1993-2025)", "dataset": "NCEP-1 500 hPa"})
    verif_rows.append(res_emp_fl)

    # Matched ordinary-wet OOS check for empirical 1961-92 deciles on 1993-2025
    for oos_tag, oos_sub, oos_ser in [
        ("Empirical 1961-92 OOS matched (1993-2025 All)", h2, DI_emp),
        ("Empirical 1961-92 OOS matched Flood (1993-2025)", h2_fl, DI_emp_fl),
    ]:
        oos_diffs = []
        for _, r in oos_sub.iterrows():
            t0 = pd.Period(r["wet_start"], freq="M")
            spi0 = spi_basin[spi_map[t0]]
            cands = [
                tl_map[p]
                for i, p in enumerate(spi_periods)
                if p.month == t0.month and p not in all_wet and np.isfinite(spi_basin[i]) and spi_basin[i] >= 0.5 and abs(spi_basin[i] - spi0) <= 0.2 and p in tl_map
            ]
            if not cands:
                mo_cands = [
                    (abs(spi_basin[i] - spi0), tl_map[p])
                    for i, p in enumerate(spi_periods)
                    if p.month == t0.month and p not in all_wet and np.isfinite(spi_basin[i]) and spi_basin[i] >= 0.5 and p in tl_map
                ]
                mo_cands.sort(key=lambda x: x[0])
                cands = [c[1] for c in mo_cands[:3]]
            if cands:
                oos_diffs.append(float(oos_ser[tl_map[t0]] - np.nanmean([oos_ser[c] for c in cands])))
        res_om = paired_sign_flip_test(oos_diffs, n_perm=n_perm, rng=rng)
        res_om.update({"check": oos_tag, "dataset": "NCEP-1 500 hPa"})
        verif_rows.append(res_om)

    # Non-YRB upstream Plateau-West dipole (25-42N, 70-90E strictly west of YRB vs NE 35-50N, 125-150E)
    DI_non_yrb = detrend_monthly_std_1d(
        _box_mean(Z500, lat_a, lon_a, 35, 50, 125, 150) - _box_mean(Z500, lat_a, lon_a, 25, 42, 70, 90),
        mons_nc,
    )
    for tag, sub in [("Upstream non-YRB NE-PL_west (All)", events_df), ("Upstream non-YRB NE-PL_west (Flood)", events_df[events_df["trans_month"].between(5, 9)])]:
        diffs_w = [
            DI_non_yrb[tl_map[pd.Period(r["wet_start"], freq="M")]]
            - float(np.mean([DI_non_yrb[tl_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
            for _, r in sub.iterrows()
        ]
        rw = paired_sign_flip_test(diffs_w, n_perm=n_perm, rng=rng)
        rw.update({"check": tag, "dataset": "NCEP-1 500 hPa"})
        verif_rows.append(rw)

    # NCEP-DOE-2 500 hPa, 200 hPa, and warm-season Huang (2004) EAP850 verification (1979-2025)
    for lev_lbl, fname in [("NCEP-DOE-2 500 hPa", "ncep2_hgt500_asia.nc"), ("NCEP-DOE-2 200 hPa", "ncep2_hgt200_asia.nc")]:
        ds2 = xr.open_dataset(f"{raw_dir}/{fname}").sel(time=slice("1979-01-01", "2025-12-31"))
        z2_raw = ds2["hgt"].squeeze("level", drop=True).values.astype(float)
        la2, lo2 = ds2.lat.values.astype(float), ds2.lon.values.astype(float)
        tl2 = pd.PeriodIndex(np.datetime_as_string(ds2.time.values, unit="M"), freq="M")
        tl2_map = {p: i for i, p in enumerate(tl2)}
        m2 = np.array([p.month for p in tl2])
        z2_std = detrend_monthly_std_3d(z2_raw, m2)
        di2_raw = _box_mean(z2_std, la2, lo2, 35, 50, 125, 150) - _box_mean(z2_std, la2, lo2, 25, 42, 90, 110)
        di2 = detrend_monthly_std_1d(di2_raw, m2)
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

    # NCEP-DOE-2 surface-bounded DIV_ps and Huang (2004) EAP850 verification (1979-2025)
    air2 = xr.open_dataset(f"{raw_dir}/ncep2_air.mon.mean.nc")["air"].sel(time=slice("1979-01-01", "2025-12-31"), lat=lat_a, lon=lon_a)
    rh2 = xr.open_dataset(f"{raw_dir}/ncep2_rhum.mon.mean.nc")["rhum"].sel(time=slice("1979-01-01", "2025-12-31"), lat=lat_a, lon=lon_a)
    u2 = xr.open_dataset(f"{raw_dir}/ncep2_uwnd.mon.mean.nc")["uwnd"].sel(time=slice("1979-01-01", "2025-12-31"), lat=lat_a, lon=lon_a)
    v2 = xr.open_dataset(f"{raw_dir}/ncep2_vwnd.mon.mean.nc")["vwnd"].sel(time=slice("1979-01-01", "2025-12-31"), lat=lat_a, lon=lon_a)
    ps2 = xr.open_dataset(f"{raw_dir}/ncep2_pres.sfc.mon.mean.nc")["pres"].sel(time=slice("1979-01-01", "2025-12-31"))
    ps2_on_25 = ps2.sortby("lat").interp(lat=lat_a, lon=lon_a).values.astype(float) / 100.0
    from pathlib import Path
    raw_p = Path(raw_dir)
    h850_2_file = raw_p / "ncep2_hgt850_asia.nc"
    if not h850_2_file.exists():
        h850_2_file = raw_p / "ncep2_hgt.mon.mean.nc"
    if h850_2_file.exists():
        h850_2_da = xr.open_dataset(h850_2_file)["hgt"].sel(time=slice("1979-01-01", "2025-12-31"), lat=lat_a, lon=lon_a)
        if "level" in h850_2_da.dims:
            h850_2_da = h850_2_da.sel(level=850, drop=True)
        z850_2 = detrend_monthly_std_3d(h850_2_da.values.astype(float), np.array([p.month for p in pd.PeriodIndex(np.datetime_as_string(h850_2_da.time.values, unit="M"), freq="M")]))
        tl850_2 = pd.PeriodIndex(np.datetime_as_string(h850_2_da.time.values, unit="M"), freq="M")
        tl850_2_map = {p: i for i, p in enumerate(tl850_2)}
        m850_2 = np.array([p.month for p in tl850_2])
        mask_850_2 = np.nanmean(ps2_on_25, axis=0) >= 850.0
        eap2_raw = (
            -0.25 * _box_mean(z850_2, lat_a, lon_a, 15, 25, 115, 140, valid_2d=mask_850_2)
            + 0.50 * _box_mean(z850_2, lat_a, lon_a, 30, 40, 115, 140, valid_2d=mask_850_2)
            - 0.25 * _box_mean(z850_2, lat_a, lon_a, 50, 60, 115, 140, valid_2d=mask_850_2)
        )
        eap2 = detrend_monthly_std_1d(eap2_raw, m850_2)
        for tag, sub in [
            ("Post-1979 All EAP850", events_df[events_df["year"] >= 1979]),
            ("Post-1979 Flood EAP850", events_df[(events_df["year"] >= 1979) & events_df["trans_month"].between(5, 9)]),
            ("Post-1979 JJA EAP850", events_df[(events_df["year"] >= 1979) & events_df["trans_month"].isin([6, 7, 8])]),
        ]:
            diffs_e2 = [
                eap2[tl850_2_map[pd.Period(r["wet_start"], freq="M")]]
                - float(np.mean([eap2[tl850_2_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
                for _, r in sub.iterrows()
            ]
            re2 = paired_sign_flip_test(diffs_e2, n_perm=n_perm, rng=rng)
            re2.update({"check": tag, "dataset": "NCEP-DOE-2 850 hPa"})
            verif_rows.append(re2)
    lv2 = air2.level.values
    lv2 = lv2[(lv2 >= 300) & (lv2 <= 1000)]
    q2 = specific_humidity_from_rh(air2.sel(level=lv2).values.astype(float), rh2.sel(level=lv2).values.astype(float), lv2)
    qu2, qv2, bad2 = vertical_vapor_flux_surface_bounded(
        q2,
        u2.sel(level=lv2).values.astype(float),
        v2.sel(level=lv2).values.astype(float),
        lv2,
        ps2_on_25,
    )
    div2_3d = spherical_divergence(qu2, qv2, lat_a)
    div2_ser = np.sum(div2_3d * wgrid[None, :, :], axis=(1, 2))
    if bad2.any():
        div2_ser[bad2] = np.nan
    tl_div2 = pd.PeriodIndex(np.datetime_as_string(air2.time.values, unit="M"), freq="M")
    tl_div2_map = {p: i for i, p in enumerate(tl_div2)}
    m_div2 = np.array([p.month for p in tl_div2])
    DIV2_ps = detrend_monthly_std_1d(div2_ser, m_div2)
    div2_mm_day = detrend_monthly_anom_1d(div2_ser, m_div2) * 86400.0
    for tag, sub in [("Post-1979 All DIV_ps", events_df[events_df["year"] >= 1979]),
                     ("Post-1979 Flood DIV_ps", events_df[(events_df["year"] >= 1979) & events_df["trans_month"].between(5, 9)])]:
        diffs_d2 = [
            DIV2_ps[tl_div2_map[pd.Period(r["wet_start"], freq="M")]]
            - float(np.mean([DIV2_ps[tl_div2_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
            for _, r in sub.iterrows()
        ]
        res_d2 = paired_sign_flip_test(diffs_d2, n_perm=n_perm, rng=rng)
        phys_d2 = [
            div2_mm_day[tl_div2_map[pd.Period(r["wet_start"], freq="M")]]
            - float(np.mean([div2_mm_day[tl_div2_map[p]] for p in _expand_months(r["dry_start"], r["dry_end"])]))
            for _, r in sub.iterrows()
        ]
        res_d2["mean_diff_mm_day"] = round(float(np.nanmean(phys_d2)), 2)
        res_d2.update({"check": tag, "dataset": "NCEP-DOE-2 DIV_ps"})
        verif_rows.append(res_d2)

    verif_df = pd.DataFrame(verif_rows)

    # Spatial composites, detrended/month-matched/ordinary-wet 5880 gpm WNPSH, BH-FDR significance, & W-flux
    b_fl = events_df[events_df["trans_month"].between(5, 9)]
    i_all = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in events_df.iterrows()]
    i_fl = [tl_map[pd.Period(r["wet_start"], freq="M")] for _, r in b_fl.iterrows()]
    fl_mos = [int(pd.Period(r["wet_start"], freq="M").month) for _, r in b_fl.iterrows()]

    def _mean_q(arr, idx):
        return np.nanmean(arr[idx], axis=0)

    yrs_nc = np.array([p.year for p in tl_nc])
    h500_detrend = np.empty_like(h500)
    h500_clim_by_mo = {}
    for mo in range(1, 13):
        idx_mo = np.where(mons_nc == mo)[0]
        idx_ref = np.where((mons_nc == mo) & (yrs_nc >= 1991) & (yrs_nc <= 2020))[0]
        t_mo = np.arange(len(idx_mo), dtype=float)
        tc_mo = t_mo - t_mo.mean()
        slope_h = np.sum(tc_mo[:, None, None] * (h500[idx_mo] - np.mean(h500[idx_mo], axis=0)[None, :, :]), axis=0) / np.sum(tc_mo**2)
        h500_detrend[idx_mo] = h500[idx_mo] - slope_h[None, :, :] * (t_mo - np.mean(t_mo[(yrs_nc[idx_mo] >= 1991) & (yrs_nc[idx_mo] <= 2020)]))[:, None, None]
        h500_clim_by_mo[mo] = np.mean(h500_detrend[idx_ref], axis=0)

    h500_flood_matched = np.mean(h500_detrend[i_fl], axis=0)
    h500_clim_matched = np.mean([h500_clim_by_mo[m] for m in fl_mos], axis=0)
    ord_pools = ord_wet_flood_pools.get("tol_0.2_nn_onset") or ord_wet_flood_pools.get("tol_0.2_nn_all39", [])
    h500_ord_wet_flood = np.mean([np.mean(h500_detrend[p_list], axis=0) for p_list in ord_pools], axis=0) if ord_pools else np.full_like(h500_flood_matched, np.nan)
    edge_flood_lon = _ridge_western_edge(h500_flood_matched, lat_a, lon_a, 5880.0)
    edge_clim_lon = _ridge_western_edge(h500_clim_matched, lat_a, lon_a, 5880.0)
    edge_ord_wet_lon = _ridge_western_edge(h500_ord_wet_flood, lat_a, lon_a, 5880.0)

    # Pointwise t-test p-values plus Wilks (2016) BH-FDR across plotted East Asian domain
    _, p_z500_fl = stats.ttest_1samp(Z500[i_fl], 0.0, axis=0, nan_policy="omit")
    _, p_z500_all = stats.ttest_1samp(Z500[i_all], 0.0, axis=0, nan_policy="omit")
    _, p_z850_fl = stats.ttest_1samp(Z850[i_fl], 0.0, axis=0, nan_policy="omit")
    dom_m = (lat_a[:, None] >= 15.0) & (lat_a[:, None] <= 62.0) & (lon_a[None, :] >= 65.0) & (lon_a[None, :] <= 170.0)
    q_z500_fl = np.full_like(p_z500_fl, np.nan)
    q_z500_all = np.full_like(p_z500_all, np.nan)
    q_z850_fl = np.full_like(p_z850_fl, np.nan)
    m_fl_ok = dom_m & np.isfinite(p_z500_fl)
    m_all_ok = dom_m & np.isfinite(p_z500_all)
    m_850_ok = dom_m & mask_850 & np.isfinite(p_z850_fl)
    q_z500_fl[m_fl_ok] = bh_fdr(p_z500_fl[m_fl_ok])
    q_z500_all[m_all_ok] = bh_fdr(p_z500_all[m_all_ok])
    q_z850_fl[m_850_ok] = bh_fdr(p_z850_fl[m_850_ok])

    # Calendar-month-specific moisture flux anomalies & per-event calendar-month Takaya-Nakamura W-flux (>=25N, |U|>=10 m/s)
    qu_m, qv_m = np.zeros_like(qu_ps), np.zeros_like(qv_ps)
    u200 = uwnd_s.sel(level=200, drop=True).values.astype(float)
    v200 = vwnd_s.sel(level=200, drop=True).values.astype(float)
    phi200 = np.zeros_like(h200)
    u200_clim_mo, v200_clim_mo = {}, {}
    for mo in range(1, 13):
        idx = np.where(mons_nc == mo)[0]
        qu_m[idx] = qu_ps[idx] - np.nanmean(qu_ps[idx], axis=0)
        qv_m[idx] = qv_ps[idx] - np.nanmean(qv_ps[idx], axis=0)
        phi200[idx] = (h200[idx] - np.nanmean(h200[idx], axis=0)) * G_ACCEL
        u200_clim_mo[mo] = np.nanmean(u200[idx], axis=0)
        v200_clim_mo[mo] = np.nanmean(v200[idx], axis=0)

    wx_list, wy_list = [], []
    for i0, mo_e in zip(i_fl, fl_mos):
        wx_e, wy_e = takaya_nakamura_flux_200(phi200[i0], u200_clim_mo[mo_e], v200_clim_mo[mo_e], lat_a, u_min=10.0, lat_min=25.0)
        wx_list.append(wx_e)
        wy_list.append(wy_e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wx_200 = np.nanmean(np.stack(wx_list, axis=0), axis=0)
        wy_200 = np.nanmean(np.stack(wy_list, axis=0), axis=0)
    u200_fl_clim = np.nanmean([u200_clim_mo[m] for m in fl_mos], axis=0)
    z850_fl_mean = np.where(mask_850, np.nanmean(Z850[i_fl], axis=0), np.nan)

    fields = {
        "lat": lat_a,
        "lon": lon_a,
        "Z500_all": np.nanmean(Z500[i_all], axis=0),
        "Z500_flood": np.nanmean(Z500[i_fl], axis=0),
        "p_Z500_all": p_z500_all,
        "p_Z500_flood": p_z500_fl,
        "q_Z500_all": q_z500_all,
        "q_Z500_flood": q_z500_fl,
        "Z850_flood": z850_fl_mean,
        "p_Z850_flood": p_z850_fl,
        "q_Z850_flood": q_z850_fl,
        "Z200_flood": np.nanmean(Z200[i_fl], axis=0),
        "h500_flood": h500_flood_matched,
        "h500_flood_clim": h500_clim_matched,
        "h500_ord_wet_flood": h500_ord_wet_flood,
        "wnpsh_edge_flood_lon": edge_flood_lon,
        "wnpsh_edge_clim_lon": edge_clim_lon,
        "wnpsh_edge_ord_wet_lon": edge_ord_wet_lon,
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


def _run_logistic_eval(x_mat, y_vec, n_perm=500, seed=777):
    """Evaluate LOOCV AUC, 5-fold stratified CV AUC, and permutation AUC p-value."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    def _oof_auc(y_arr, folds):
        probs = np.zeros(len(y_arr), dtype=float)
        for tr, te in folds:
            scaler = StandardScaler().fit(x_mat[tr])
            clf = LogisticRegression(C=0.5, solver="lbfgs", tol=1e-3, max_iter=150).fit(
                scaler.transform(x_mat[tr]), y_arr[tr]
            )
            probs[te] = clf.predict_proba(scaler.transform(x_mat[te]))[:, 1]
        return float(roc_auc_score(y_arr, probs))

    n = len(y_vec)
    loo_probs = np.zeros(n, dtype=float)
    for i in range(n):
        train = np.arange(n) != i
        scaler = StandardScaler().fit(x_mat[train])
        clf = LogisticRegression(C=0.5, solver="lbfgs", tol=1e-3, max_iter=150).fit(
            scaler.transform(x_mat[train]), y_vec[train]
        )
        loo_probs[i] = float(clf.predict_proba(scaler.transform(x_mat[i : i + 1]))[0, 1])
    loo_auc = float(roc_auc_score(y_vec, loo_probs))

    n_splits = min(5, int((y_vec == 1).sum()), int((y_vec == 0).sum()))
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(x_mat, y_vec))
        cv5_auc = _oof_auc(y_vec, folds)
        rng_p = np.random.default_rng(seed)
        perm_aucs = np.empty(n_perm, dtype=float)
        skf_p = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for b in range(n_perm):
            y_perm = rng_p.permutation(y_vec)
            p_folds = list(skf_p.split(x_mat, y_perm))
            perm_aucs[b] = _oof_auc(y_perm, p_folds)
        perm_p = float((np.sum(perm_aucs >= cv5_auc) + 1.0) / (n_perm + 1.0))
    else:
        cv5_auc = loo_auc
        perm_p = np.nan

    return round(loo_auc, 3), round(cv5_auc, 3), round(perm_p, 4)


def seasonal_teleconnection_skill(raw_dir, events_df, dry_spells=None, periods=None, spi_basin=None):
    """Evaluate strictly causal pre-drought-onset ONI/SOI/NAO skill against collision-free drought and annual controls."""
    from pathlib import Path

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

    def _predictors_before(d_yr, d_mo):
        """Return strictly prior completed DJF and MAM indices before drought onset (d_yr, d_mo)."""
        djf_y = d_yr if d_mo >= 3 else d_yr - 1
        mam_y = d_yr if d_mo >= 6 else d_yr - 1
        assert pd.Period(f"{djf_y}-02", freq="M") < pd.Period(f"{d_yr}-{d_mo:02d}", freq="M")
        assert pd.Period(f"{mam_y}-05", freq="M") < pd.Period(f"{d_yr}-{d_mo:02d}", freq="M")
        return {
            "ONI_DJF": oni.get((djf_y, 1), np.nan),
            "SOI_DJF": float(np.nanmean([soi.get((djf_y - 1, 12), np.nan), soi.get((djf_y, 1), np.nan), soi.get((djf_y, 2), np.nan)])),
            "NAO_DJF": float(np.nanmean([nao.get((djf_y - 1, 12), np.nan), nao.get((djf_y, 1), np.nan), nao.get((djf_y, 2), np.nan)])),
            "ONI_MAM": oni.get((mam_y, 4), np.nan),
            "SOI_MAM": float(np.nanmean([soi.get((mam_y, 3), np.nan), soi.get((mam_y, 4), np.nan), soi.get((mam_y, 5), np.nan)])),
            "NAO_MAM": float(np.nanmean([nao.get((mam_y, 3), np.nan), nao.get((mam_y, 4), np.nan), nao.get((mam_y, 5), np.nan)])),
            "djf_y": djf_y,
            "mam_y": mam_y,
        }

    ev_rows = []
    ev_years_fl = set()
    ev_dry_starts = set()
    ev_all_months = set()
    for _, r in events_df.iterrows():
        ds_p = pd.Period(r["dry_start"], freq="M")
        ev_dry_starts.add(str(ds_p))
        ev_all_months.update(_expand_months(r["dry_start"], r["wet_end"]))
        preds = _predictors_before(ds_p.year, ds_p.month)
        preds.update({
            "label": 1,
            "event_id": int(r["event_id"]),
            "dry_year": int(ds_p.year),
            "wet_year": int(r["year"]),
            "is_flood": int(5 <= int(r["trans_month"]) <= 9),
        })
        ev_rows.append(preds)
        if 5 <= int(r["trans_month"]) <= 9:
            ev_years_fl.add(int(r["year"]))

    all_feats = ["ONI_DJF", "SOI_DJF", "NAO_DJF", "ONI_MAM", "SOI_MAM", "NAO_MAM"]
    noncol_feats = ["ONI_DJF", "NAO_DJF", "ONI_MAM", "NAO_MAM"]
    out_rows = []

    def _feat_key(row_d):
        return tuple(round(float(row_d[c]), 6) for c in all_feats)

    pos_keys = {_feat_key(r) for r in ev_rows}

    # Model A: Fixed-lag May-Sep Flood-Season Annual Prediction (1961-2025; DJF_y and MAM_y strictly < May_y)
    fl_yr_df = pd.DataFrame([
        {**_predictors_before(yr, 6), "label": 1 if yr in ev_years_fl else 0}
        for yr in range(1961, 2026)
    ]).dropna(subset=all_feats).reset_index(drop=True)

    # Model B: Fixed-lag Hydrological-Year Prediction ([Jun_y..May_{y+1}] from DJF_y/MAM_y)
    ev_jun_after = set()
    for _, r in events_df.iterrows():
        ws = pd.Period(r["wet_start"], freq="M")
        ev_jun_after.add(ws.year if ws.month >= 6 else ws.year - 1)
    hy_df = pd.DataFrame([
        {**_predictors_before(yr, 6), "label": 1 if yr in ev_jun_after else 0}
        for yr in range(1961, 2026)
    ]).dropna(subset=all_feats).reset_index(drop=True)

    setups = [
        ("Flood_MaySep_Annual_FixedLag", fl_yr_df),
        ("HydroYear_JunMay_FixedLag", hy_df),
    ]

    # Model C1: Genuine non-transitioning multi-month dry spells (D >= 2 without transition within G=4)
    if dry_spells is not None and periods is not None:
        neg_spells = []
        n_spell_coll = 0
        for s0, s1 in dry_spells:
            p0 = periods[s0]
            if str(p0) in ev_dry_starts:
                continue
            pr = _predictors_before(p0.year, p0.month)
            if _feat_key(pr) in pos_keys:
                n_spell_coll += 1
                continue
            pr.update({"label": 0, "dry_start": str(p0)})
            neg_spells.append(pr)
        spells_df = pd.DataFrame(ev_rows + neg_spells).dropna(subset=all_feats).reset_index(drop=True)
        setups.append(("AllDrySpells_Causal_PreDry", spells_df))

    # Model C2: Collision-free non-DFAA drought months (SPI <= -0.5 outside all 39 DFAA [dry_start..wet_end] runs)
    if spi_basin is not None and periods is not None:
        seen_neg_keys = set()
        neg_dry_months = []
        n_drymo_coll = 0
        for idx_t, p in enumerate(periods):
            if not np.isfinite(spi_basin[idx_t]) or spi_basin[idx_t] > -0.5:
                continue
            if p in ev_all_months:
                continue
            pr = _predictors_before(p.year, p.month)
            fk = _feat_key(pr)
            if fk in pos_keys:
                n_drymo_coll += 1
                continue
            if fk in seen_neg_keys:
                continue
            seen_neg_keys.add(fk)
            pr.update({"label": 0, "dry_month": str(p)})
            neg_dry_months.append(pr)
        drymo_df = pd.DataFrame(ev_rows + neg_dry_months).dropna(subset=all_feats).reset_index(drop=True)
        setups.append(("All39_Events_Causal_PreDry", drymo_df))

    for setup_name, sub_df in setups:
        pos_fset = {tuple(np.round(r, 6)) for r in sub_df.loc[sub_df["label"] == 1, all_feats].to_numpy()}
        neg_fset = {tuple(np.round(r, 6)) for r in sub_df.loc[sub_df["label"] == 0, all_feats].to_numpy()}
        assert len(pos_fset & neg_fset) == 0, f"Duplicate opposite-label features in {setup_name}"
        y_vec = sub_df["label"].to_numpy()
        loo6, cv5_6, perm_p6 = _run_logistic_eval(sub_df[all_feats].to_numpy(), y_vec)
        loo4, cv5_4, perm_p4 = _run_logistic_eval(sub_df[noncol_feats].to_numpy(), y_vec)
        pvals = []
        s_rows = []
        for col in all_feats:
            d_ev = sub_df.loc[sub_df["label"] == 1, col]
            d_no = sub_df.loc[sub_df["label"] == 0, col]
            _, p_t = stats.ttest_ind(d_ev, d_no)
            pvals.append(float(p_t))
            s_rows.append({
                "setup": setup_name,
                "n_pos": int((y_vec == 1).sum()),
                "n_neg": int((y_vec == 0).sum()),
                "predictor": col,
                "event_mean": round(float(d_ev.mean()), 3),
                "nonevent_mean": round(float(d_no.mean()), 3),
                "diff": round(float(d_ev.mean() - d_no.mean()), 3),
                "p_value": round(float(p_t), 4),
                "loo_auc_6feat": loo6,
                "cv5_auc_6feat": cv5_6,
                "perm_auc_p_6feat": perm_p6,
                "loo_auc_4feat_noncol": loo4,
                "cv5_auc_4feat_noncol": cv5_4,
                "perm_auc_p_4feat_noncol": perm_p4,
            })
        qvals = bh_fdr(pvals)
        for row_d, qv in zip(s_rows, qvals):
            row_d["q_FDR"] = round(float(qv), 4)
        out_rows.extend(s_rows)

    return pd.DataFrame(out_rows)
