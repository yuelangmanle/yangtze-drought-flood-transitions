"""Drought-to-flood indices (SPI, SPEI), event run theory, and null surrogates."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as gamma_fn


def fit_spi(series, months, scale=1):
    """Mixed-Gamma Standardized Precipitation Index (Thom 1966; McKee et al. 1993; Stagge et al. 2015)."""
    vals = pd.Series(series, dtype=float).rolling(scale).sum().to_numpy()
    out = np.full(len(vals), np.nan, dtype=float)
    for mo in range(1, 13):
        idx = np.where((months == mo) & np.isfinite(vals))[0]
        if len(idx) < 10:
            continue
        x = vals[idx]
        pos_mask = x > 0.0
        n_ok = len(x)
        n_zero = int((~pos_mask).sum())
        q0 = n_zero / float(n_ok + 1)

        cdf = np.empty(n_ok, dtype=float)
        if pos_mask.sum() >= 5:
            shape_hat, _, scale_hat = stats.gamma.fit(x[pos_mask], floc=0.0)
            g_cdf = stats.gamma.cdf(x[pos_mask], shape_hat, loc=0.0, scale=scale_hat)
            cdf[pos_mask] = q0 + (1.0 - q0) * g_cdf
        else:
            cdf[pos_mask] = 0.5
        if n_zero > 0:
            # Stagge et al. (2015) probability-mass midpoint inside [0, q0]
            cdf[~pos_mask] = 0.5 * q0

        out[idx] = stats.norm.ppf(np.clip(cdf, 1e-6, 1.0 - 1e-6))
    return out


def thornthwaite_pet(temp_c, years, months, mean_lat):
    """Thornthwaite (1948) monthly PET [mm] with astronomically corrected daylength."""
    mdays = np.array([31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
    mid_doy = np.array([15, 45, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344], dtype=float)
    decl = 0.409 * np.sin(2.0 * np.pi * mid_doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(np.deg2rad(mean_lat)) * np.tan(decl), -1.0, 1.0))
    daylight_hrs = (24.0 / np.pi) * ws
    k_factor = (daylight_hrs / 12.0) * (mdays / 30.0)

    annual_i = {}
    for yr in np.unique(years):
        t_pos = np.maximum(temp_c[years == yr], 0.0)
        annual_i[yr] = float(np.sum((t_pos / 5.0) ** 1.514))

    pet = np.zeros_like(temp_c, dtype=float)
    for i, (yr, mo, tc) in enumerate(zip(years, months, temp_c)):
        if tc <= 0.0 or not np.isfinite(tc):
            continue
        i_val = annual_i[yr]
        if i_val <= 0.0:
            continue
        a_exp = 6.75e-7 * i_val**3 - 7.71e-5 * i_val**2 + 1.792e-2 * i_val + 0.49239
        pet_raw = 16.0 * ((10.0 * tc / i_val) ** a_exp)
        pet[i] = pet_raw * k_factor[int(mo) - 1]
    return pet


def fit_spei(water_balance, months, scale=1):
    """Generalized logistic SPEI via unbiased L-moments (Vicente-Serrano et al. 2010; Begueria et al. 2014)."""
    vals = pd.Series(water_balance, dtype=float).rolling(scale).sum().to_numpy()
    out = np.full(len(vals), np.nan, dtype=float)
    for mo in range(1, 13):
        idx = np.where((months == mo) & np.isfinite(vals))[0]
        if len(idx) < 12:
            continue
        raw = vals[idx]
        xs = np.sort(raw)
        n = len(xs)
        r = np.arange(n, dtype=float)
        b0 = float(np.mean(xs))
        b1 = float(np.sum(xs * r / (n * (n - 1.0))))
        b2 = float(np.sum(xs * r * (r - 1.0) / (n * (n - 1.0) * (n - 2.0))))
        l1 = b0
        l2 = 2.0 * b1 - b0
        l3 = 6.0 * b2 - 6.0 * b1 + b0
        if l2 <= 0.0:
            continue
        k = -l3 / l2
        if abs(k) < 1e-4:
            alpha = l2
            xi = l1
            y = (raw - xi) / alpha
        else:
            sinc = np.pi * k / np.sin(np.pi * k)
            alpha = l2 / sinc
            xi = l1 - alpha * (1.0 - sinc) / k
            arg = 1.0 - k * (raw - xi) / alpha
            y = np.full(len(raw), -20.0 if k > 0 else 20.0, dtype=float)
            inside = arg > 1e-12
            y[inside] = -(1.0 / k) * np.log(arg[inside])

        cdf = 1.0 / (1.0 + np.exp(-np.clip(y, -30.0, 30.0)))
        out[idx] = stats.norm.ppf(np.clip(cdf, 1e-6, 1.0 - 1e-6))
    return out


def _season_label(mo):
    if 5 <= mo <= 9:
        return "Flood (May-Sep)"
    if mo in (12, 1, 2):
        return "Winter (Dec-Feb)"
    return "Shoulder (Mar-Apr, Oct-Nov)"


def detect_transitions(spi, periods=None, dry_th=-0.5, wet_th=0.5, min_dry=2, max_gap=4, min_fw=0.0):
    """Run-theory identification of sequential drought-to-wet transition events."""
    events = []
    dry_spells = []
    n = len(spi)
    i = 0
    while i < n:
        j = i
        while j < n and np.isfinite(spi[j]) and spi[j] <= dry_th:
            j += 1
        if j - i >= min_dry:
            s0, s1 = i, j - 1
            dry_spells.append((s0, s1))
            k = s1 + 1
            while k <= s1 + max_gap and k < n:
                if np.isfinite(spi[k]) and spi[k] >= wet_th:
                    j2 = k
                    while j2 < n and np.isfinite(spi[j2]) and spi[j2] >= wet_th:
                        j2 += 1
                    fw_val = float(spi[k:j2].sum())
                    if fw_val < min_fw:
                        break
                    sd_val = float(-spi[s0 : s1 + 1].sum())
                    d_len = s1 - s0 + 1
                    w_len = j2 - k
                    gap_len = k - s1
                    j_w = (fw_val / w_len) - (spi[s0 : s1 + 1].mean())
                    rec = {
                        "s0": s0,
                        "s1": s1,
                        "k": k,
                        "j2": j2 - 1,
                        "D": d_len,
                        "W": w_len,
                        "gap": gap_len,
                        "S_d": round(sd_val, 4),
                        "F_w": round(fw_val, 4),
                        "J_w": round(float(j_w), 4),
                    }
                    if periods is not None:
                        p_k = periods[k]
                        rec["dry_start"] = str(periods[s0])
                        rec["dry_end"] = str(periods[s1])
                        rec["wet_start"] = str(p_k)
                        rec["wet_end"] = str(periods[j2 - 1])
                        rec["year"] = int(p_k.year)
                        rec["trans_month"] = int(p_k.month)
                        rec["season"] = _season_label(int(p_k.month))
                    events.append(rec)
                    break
                k += 1
        i = max(j, i + 1)
    return events, dry_spells


def detect_with_bridge(spi, dry_th=-0.5, wet_th=0.5, min_dry=2, max_gap=4):
    """Sensitivity variant allowing a 1-month weak interruption inside dry or wet runs."""
    n = len(spi)
    dry_mask = np.isfinite(spi) & (spi <= dry_th)
    bridges = 0
    for t in range(1, n - 1):
        if np.all(np.isfinite(spi[t - 1 : t + 2])):
            if spi[t - 1] <= dry_th and dry_th < spi[t] <= 0.0 and spi[t + 1] <= dry_th:
                dry_mask[t] = True
                bridges += 1

    events = []
    i = 0
    while i < n:
        j = i
        while j < n and dry_mask[j]:
            j += 1
        if j - i >= min_dry:
            s0, s1 = i, j - 1
            k = s1 + 1
            while k <= s1 + max_gap and k < n:
                if np.isfinite(spi[k]) and spi[k] >= wet_th:
                    j2 = k
                    while j2 < n and np.isfinite(spi[j2]) and (
                        spi[j2] >= wet_th
                        or (j2 + 1 < n and 0.0 <= spi[j2] < wet_th and spi[j2 + 1] >= wet_th)
                    ):
                        j2 += 1
                    sd_val = float(-np.minimum(spi[s0 : s1 + 1], 0.0).sum())
                    fw_val = float(np.maximum(spi[k:j2], 0.0).sum())
                    events.append({"S_d": sd_val, "F_w": fw_val})
                    break
                k += 1
        i = max(j, i + 1)
    return events, bridges


def threshold_sensitivity_matrix(spi):
    """Evaluate 36 core threshold/duration/gap combinations plus G=1 and F_w>=1.0 extensions."""
    rows = []
    configs = [
        (dry_th, wet_th, min_dry, max_gap, 0.0, "core_36")
        for dry_th in (-0.4, -0.5, -0.6)
        for wet_th in (0.4, 0.5, 0.6)
        for min_dry in (1, 2)
        for max_gap in (3, 4)
    ]
    # Add G=1 immediate-transition and F_w>=1.0 moderate-to-severe rules
    configs.extend([
        (-0.5, 0.5, 2, 1, 0.0, "gap1_immediate"),
        (-0.5, 0.5, 1, 1, 0.0, "gap1_immediate"),
        (-0.5, 0.5, 2, 4, 1.0, "fw1_threshold"),
        (-0.5, 0.5, 2, 1, 1.0, "gap1_fw1"),
    ])
    for dry_th, wet_th, min_dry, max_gap, min_fw, rule_grp in configs:
        evs, _ = detect_transitions(spi, None, dry_th, wet_th, min_dry, max_gap, min_fw=min_fw)
        if len(evs) < 5:
            continue
        sd = [e["S_d"] for e in evs]
        fw = [e["F_w"] for e in evs]
        tau, p_val = stats.kendalltau(sd, fw)
        rows.append({
            "dry_th": dry_th,
            "wet_th": wet_th,
            "min_dry": min_dry,
            "max_gap": max_gap,
            "min_fw": min_fw,
            "rule_group": rule_grp,
            "n_events": len(evs),
            "tau": round(float(tau), 4),
            "p_value": round(float(p_val), 4),
        })
    return pd.DataFrame(rows)


def run_null_surrogates(spi, months, years, n_boot=1000, seed=2026, region="Basin"):
    """Compare observed transitions against Gaussian white-noise, seasonal AR(1)/AR(2), and block shuffles."""
    rng = np.random.default_rng(seed)
    obs_events, _ = detect_transitions(spi)
    obs_count = len(obs_events)
    obs_tau = float(stats.kendalltau([e["S_d"] for e in obs_events], [e["F_w"] for e in obs_events]).statistic)

    # Monthly lag-1 and lag-2 autoregressive parameters
    rhos = np.zeros(12, dtype=float)
    ar2_coef = np.zeros((12, 2), dtype=float)
    ar2_sig = np.ones(12, dtype=float)
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        idx1 = idx[idx > 0]
        r = np.corrcoef(spi[idx1 - 1], spi[idx1])[0, 1]
        rhos[mo - 1] = float(r) if np.isfinite(r) else 0.0
        idx2 = idx[idx > 1]
        X2 = np.column_stack([spi[idx2 - 1], spi[idx2 - 2]])
        beta, _, _, _ = np.linalg.lstsq(X2, spi[idx2], rcond=None)
        res2 = spi[idx2] - X2 @ beta
        ar2_coef[mo - 1] = beta
        ar2_sig[mo - 1] = float(np.std(res2, ddof=2))

    def _m2_rate(series):
        _, drys = detect_transitions(series)
        hits, total = 0, 0
        for s0, _ in drys:
            c = s0 + 1
            if c + 4 >= len(series):
                continue
            total += 1
            hit = np.where(series[c + 1 : c + 5] >= 0.5)[0]
            if len(hit):
                ka = c + 1 + int(hit[0])
                j2 = ka
                while j2 < len(series) and series[j2] >= 0.5:
                    j2 += 1
                if series[ka:j2].sum() >= 1.0:
                    hits += 1
        return hits / total if total else np.nan

    unique_years = np.unique(years)
    year_indices = [np.where(years == y)[0] for y in unique_years]

    records = []
    for mode in ("iid_gaussian", "seasonal_ar1", "seasonal_ar2", "year_block_shuffle"):
        cnts, taus, m2_rates = [], [], []
        for _ in range(n_boot):
            if mode == "iid_gaussian":
                sim = rng.normal(size=len(spi))
            elif mode == "seasonal_ar1":
                sim = np.empty(len(spi), dtype=float)
                sim[0] = rng.normal()
                for t in range(1, len(spi)):
                    r_m = rhos[int(months[t]) - 1]
                    sim[t] = r_m * sim[t - 1] + np.sqrt(max(1.0 - r_m**2, 1e-6)) * rng.normal()
            elif mode == "seasonal_ar2":
                sim = np.empty(len(spi), dtype=float)
                sim[0] = rng.normal()
                sim[1] = rhos[int(months[1]) - 1] * sim[0] + np.sqrt(max(1.0 - rhos[int(months[1]) - 1] ** 2, 1e-6)) * rng.normal()
                for t in range(2, len(spi)):
                    m_idx = int(months[t]) - 1
                    b1, b2 = ar2_coef[m_idx]
                    sim[t] = b1 * sim[t - 1] + b2 * sim[t - 2] + ar2_sig[m_idx] * rng.normal()
            else:
                perm = rng.permutation(len(year_indices))
                sim = np.concatenate([spi[year_indices[p]] for p in perm])

            evs, _ = detect_transitions(sim)
            cnts.append(len(evs))
            if len(evs) >= 5:
                taus.append(float(stats.kendalltau([e["S_d"] for e in evs], [e["F_w"] for e in evs]).statistic))
            m2_rates.append(_m2_rate(sim))

        cnt_arr = np.asarray(cnts, dtype=float)
        tau_arr = np.asarray(taus, dtype=float)
        m2_arr = np.asarray([m for m in m2_rates if np.isfinite(m)], dtype=float)
        p_cnt = float((np.sum(np.abs(cnt_arr - cnt_arr.mean()) >= abs(obs_count - cnt_arr.mean())) + 1) / (len(cnt_arr) + 1))
        records.append({
            "region": region,
            "surrogate": mode,
            "n_boot": n_boot,
            "obs_count": obs_count,
            "null_count_mean": round(float(cnt_arr.mean()), 2),
            "null_count_p05": int(np.percentile(cnt_arr, 5)),
            "null_count_p95": int(np.percentile(cnt_arr, 95)),
            "count_p_value": round(p_cnt, 4),
            "obs_tau": round(obs_tau, 4),
            "null_tau_mean": round(float(tau_arr.mean()), 4),
            "null_tau_p05": round(float(np.percentile(tau_arr, 5)), 4),
            "null_tau_p95": round(float(np.percentile(tau_arr, 95)), 4),
            "null_m2_fw1_mean": round(float(m2_arr.mean()), 4),
            "null_m2_fw1_p05": round(float(np.percentile(m2_arr, 5)), 4),
            "null_m2_fw1_p95": round(float(np.percentile(m2_arr, 95)), 4),
            "lag1_rho_min": round(float(rhos.min()), 3),
            "lag1_rho_max": round(float(rhos.max()), 3),
            "lag1_rho_mean": round(float(rhos.mean()), 3),
        })
    return pd.DataFrame(records), rhos


def tributary_amplification(
    events_df,
    pr_basin,
    pr_tribs,
    basin_spi,
    spi_tribs,
    months,
    periods,
    trib_areas_km2=None,
    basin_area_km2=1748000.0,
):
    """Compare basin-mean vs peak tributary SPI, depth excess (mm), and volumetric excess (km^3)."""
    p_map = {p: i for i, p in enumerate(periods)}
    clim_b = np.array([pr_basin[months == mo].mean() for mo in range(1, 13)])
    clim_t = {
        nm: np.array([pr_t[months == mo].mean() for mo in range(1, 13)])
        for nm, pr_t in pr_tribs.items()
    }

    rows = []
    for idx, row in events_df.iterrows():
        i0 = p_map[pd.Period(row["wet_start"], freq="M")]
        i1 = p_map[pd.Period(row["wet_end"], freq="M")]
        fw_b = float(row["F_w"])

        ex_b = float(sum(pr_basin[t] - clim_b[int(months[t]) - 1] for t in range(i0, i1 + 1)))
        cl_b_sum = float(sum(clim_b[int(months[t]) - 1] for t in range(i0, i1 + 1)))
        pct_b = 100.0 * ex_b / cl_b_sum if cl_b_sum > 0 else np.nan
        vol_b_km3 = ex_b * basin_area_km2 * 1e-6

        trib_fw, trib_ex, trib_pct, trib_vol = {}, {}, {}, {}
        for nm, s_t in spi_tribs.items():
            pos = np.maximum(s_t[i0 : i1 + 1], 0.0)
            trib_fw[nm] = float(pos.sum())
            ex_t = float(sum(pr_tribs[nm][t] - clim_t[nm][int(months[t]) - 1] for t in range(i0, i1 + 1)))
            cl_t = float(sum(clim_t[nm][int(months[t]) - 1] for t in range(i0, i1 + 1)))
            trib_ex[nm] = ex_t
            trib_pct[nm] = 100.0 * ex_t / cl_t if cl_t > 0 else np.nan
            if trib_areas_km2 and nm in trib_areas_km2:
                trib_vol[nm] = ex_t * float(trib_areas_km2[nm]) * 1e-6
            else:
                trib_vol[nm] = np.nan

        peak_nm = max(trib_fw, key=trib_fw.get)
        peak_fw = trib_fw[peak_nm]
        amp_spi = peak_fw / fw_b if fw_b > 0 else np.nan

        peak_mm_nm = max(trib_ex, key=trib_ex.get)
        amp_mm = trib_ex[peak_mm_nm] / ex_b if ex_b > 0 else np.nan
        amp_pct = trib_pct[peak_mm_nm] / pct_b if pct_b > 0 else np.nan

        peak_vol_nm = max(trib_vol, key=trib_vol.get) if trib_areas_km2 else peak_mm_nm
        peak_vol_km3 = trib_vol[peak_vol_nm] if trib_areas_km2 else np.nan
        vol_share_pct = 100.0 * peak_vol_km3 / vol_b_km3 if (vol_b_km3 > 0 and np.isfinite(peak_vol_km3)) else np.nan

        tmo = int(row["trans_month"])
        rec = {
            "event_id": int(idx) + 1,
            "dry_start": row["dry_start"],
            "wet_start": row["wet_start"],
            "wet_end": row["wet_end"],
            "year": int(row["year"]),
            "trans_month": tmo,
            "season": _season_label(tmo),
            "F_w_basin": round(fw_b, 3),
            "peak_tributary": peak_nm,
            "F_w_peak_trib": round(peak_fw, 3),
            "spi_amp_ratio": round(amp_spi, 3),
            "basin_precip_ex_mm": round(ex_b, 1),
            "basin_precip_ex_pct": round(pct_b, 1),
            "basin_precip_vol_km3": round(vol_b_km3, 2),
            "peak_mm_tributary": peak_mm_nm,
            "peak_trib_ex_mm": round(trib_ex[peak_mm_nm], 1),
            "peak_trib_ex_pct": round(trib_pct[peak_mm_nm], 1),
            "mm_excess_ratio": round(amp_mm, 3) if np.isfinite(amp_mm) else np.nan,
            "pct_excess_ratio": round(amp_pct, 3) if np.isfinite(amp_pct) else np.nan,
            "peak_vol_tributary": peak_vol_nm,
            "peak_trib_vol_km3": round(peak_vol_km3, 2) if np.isfinite(peak_vol_km3) else np.nan,
            "peak_vol_share_pct": round(vol_share_pct, 1) if np.isfinite(vol_share_pct) else np.nan,
            "spi_mm_basin_match": bool(peak_nm == peak_mm_nm),
        }
        for nm, val in trib_fw.items():
            rec[f"Fw_{nm}"] = round(val, 3)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_tributary_amplification(trib_df):
    """Compute season-stratified tributary amplification, volumetric shares, and uncoupled scaling tests."""
    rows = []
    strata = [
        ("All (1961-2025)", trib_df),
        ("Flood (May-Sep)", trib_df[trib_df["trans_month"].between(5, 9)]),
        ("Cool (Oct-Apr)", trib_df[~trib_df["trans_month"].between(5, 9)]),
        ("Winter (Dec-Feb)", trib_df[trib_df["trans_month"].isin([12, 1, 2])]),
        ("Shoulder (Mar-Apr, Oct-Nov)", trib_df[~trib_df["trans_month"].isin([5, 6, 7, 8, 9, 12, 1, 2])]),
        ("Mild (F_w < 1.0)", trib_df[trib_df["F_w_basin"] < 1.0]),
        ("Mod/Sev (F_w >= 1.0)", trib_df[trib_df["F_w_basin"] >= 1.0]),
    ]
    mild_df = trib_df[trib_df["F_w_basin"] < 1.0]
    sev_df = trib_df[trib_df["F_w_basin"] >= 1.0]
    _, p_mw_spi = stats.mannwhitneyu(mild_df["spi_amp_ratio"], sev_df["spi_amp_ratio"], alternative="two-sided")
    _, p_mw_mm = stats.mannwhitneyu(mild_df["mm_excess_ratio"].dropna(), sev_df["mm_excess_ratio"].dropna(), alternative="two-sided")
    rs_level, p_level = stats.spearmanr(trib_df["F_w_basin"], trib_df["F_w_peak_trib"])
    lr = stats.linregress(np.log(trib_df["F_w_basin"]), np.log(trib_df["F_w_peak_trib"]))
    t_sublinear = (lr.slope - 1.0) / lr.stderr
    p_sublinear = 2.0 * float(stats.t.cdf(-abs(t_sublinear), df=len(trib_df) - 2))

    for sname, sub in strata:
        if sub.empty:
            continue
        rows.append({
            "stratum": sname,
            "n_events": len(sub),
            "spi_ratio_median": round(float(sub["spi_amp_ratio"].median()), 3),
            "spi_ratio_iqr": f"{sub['spi_amp_ratio'].quantile(0.25):.2f}–{sub['spi_amp_ratio'].quantile(0.75):.2f}",
            "spi_ratio_mean": round(float(sub["spi_amp_ratio"].mean()), 3),
            "mm_ratio_median": round(float(sub["mm_excess_ratio"].median()), 3),
            "mm_ratio_iqr": f"{sub['mm_excess_ratio'].quantile(0.25):.2f}–{sub['mm_excess_ratio'].quantile(0.75):.2f}",
            "mm_ratio_mean": round(float(sub["mm_excess_ratio"].mean()), 3),
            "basin_ex_mm_median": round(float(sub["basin_precip_ex_mm"].median()), 1),
            "peak_trib_ex_mm_median": round(float(sub["peak_trib_ex_mm"].median()), 1),
            "peak_trib_ex_mm_mean": round(float(sub["peak_trib_ex_mm"].mean()), 1),
            "peak_vol_km3_median": round(float(sub["peak_trib_vol_km3"].median()), 2),
            "n_spi_mm_divergent": int((~sub["spi_mm_basin_match"]).sum()),
            "mw_spi_mild_vs_sev_p": round(float(p_mw_spi), 6) if sname.startswith("All") else np.nan,
            "mw_mm_mild_vs_sev_p": round(float(p_mw_mm), 4) if sname.startswith("All") else np.nan,
            "level_spearman_rs": round(float(rs_level), 4) if sname.startswith("All") else np.nan,
            "level_spearman_p": round(float(p_level), 8) if sname.startswith("All") else np.nan,
            "loglog_slope": round(float(lr.slope), 4) if sname.startswith("All") else np.nan,
            "loglog_slope_vs1_p": round(float(p_sublinear), 6) if sname.startswith("All") else np.nan,
        })
    return pd.DataFrame(rows)


def interannual_whiplash_overlap(pr_basin, years, events_df):
    """Compare annual precipitation whiplash I_w (Swain et al. 2018) with monthly DFAA event years."""
    yr_vals = np.arange(int(years.min()), int(years.max()) + 1)
    ann_p = pd.Series([float(pr_basin[years == y].sum()) for y in yr_vals], index=yr_vals)
    iw = (np.abs(ann_p.diff()) / (ann_p + ann_p.shift(1))).dropna()
    p90 = float(np.percentile(iw, 90))
    ext_yrs = sorted(iw[iw >= p90].index.astype(int).tolist())
    dfaa_yrs = sorted(events_df["year"].astype(int).unique().tolist())
    overlap = sorted(set(ext_yrs) & set(dfaa_yrs))
    union_n = len(set(ext_yrs) | set(dfaa_yrs))
    jaccard = len(overlap) / union_n if union_n else 0.0
    return {
        "n_years_eval": len(iw),
        "p90_threshold": round(p90, 4),
        "n_extreme_whiplash": len(ext_yrs),
        "n_dfaa_years": len(dfaa_yrs),
        "n_overlap": len(overlap),
        "jaccard": round(jaccard, 4),
        "extreme_years": ";".join(map(str, ext_yrs)),
        "overlap_years": ";".join(map(str, overlap)),
    }
