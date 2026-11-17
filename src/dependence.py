"""Marginal screening, bivariate copula selection, prospective onset skill, and spatial FDR."""

import numpy as np
import pandas as pd
from scipy import optimize, stats


MARGINAL_CANDIDATES = {
    "GEV": stats.genextreme,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Lognormal": stats.lognorm,
    "Exponential": stats.expon,
    "InvGamma": stats.invgamma,
}


def dependence_summary(sd, fw):
    """Compute Kendall's tau, Spearman's r_s, Pearson's r, and 80%-power MDE."""
    sd = np.asarray(sd, dtype=float)
    fw = np.asarray(fw, dtype=float)
    n = len(sd)
    tau, p_tau = stats.kendalltau(sd, fw)
    rs, p_rs = stats.spearmanr(sd, fw)
    rp, p_rp = stats.pearsonr(sd, fw)
    mde_r = float(np.tanh((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) / np.sqrt(n - 3)))
    return {
        "n": n,
        "S_d_mean": round(float(np.mean(sd)), 3),
        "S_d_median": round(float(np.median(sd)), 3),
        "F_w_mean": round(float(np.mean(fw)), 3),
        "F_w_median": round(float(np.median(fw)), 3),
        "kendall_tau": round(float(tau), 4),
        "kendall_p": round(float(p_tau), 4),
        "spearman_r": round(float(rs), 4),
        "spearman_p": round(float(p_rs), 4),
        "pearson_r": round(float(rp), 4),
        "pearson_p": round(float(p_rp), 4),
        "mde_r_80pct": round(mde_r, 3),
    }


def screen_marginals(sd, fw):
    """Fit six parametric candidate distributions with interior lower-bound support."""
    rows = []
    for var_name, values in (("S_d", sd), ("F_w", fw)):
        vals = np.asarray(values, dtype=float)
        n = len(vals)
        best_aicc = np.inf
        var_rows = []
        v_min = float(np.min(vals))
        for dist_name, dist in MARGINAL_CANDIDATES.items():
            if dist_name in ("GEV", "Exponential"):
                params = dist.fit(vals)
                k = len(params)
            else:
                def _neg_ll(loc_val):
                    try:
                        par = dist.fit(vals, floc=loc_val)
                        ll_val = float(np.sum(dist.logpdf(vals, *par)))
                        return -ll_val if np.isfinite(ll_val) else 1e9
                    except Exception:
                        return 1e9

                opt = optimize.minimize_scalar(
                    _neg_ll,
                    bounds=(max(0.0, v_min - 2.0) if v_min > 0 else v_min - 2.0, v_min - 1e-4),
                    method="bounded",
                )
                params = dist.fit(vals, floc=float(opt.x))
                k = len(params)
            ll = float(np.sum(dist.logpdf(vals, *params)))
            aicc = 2.0 * k - 2.0 * ll + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
            ks_d, ks_p = stats.kstest(vals, dist(*params).cdf)
            if aicc < best_aicc:
                best_aicc = aicc
            var_rows.append({
                "variable": var_name,
                "distribution": dist_name,
                "k": k,
                "loglik": round(ll, 2),
                "AICc": round(aicc, 2),
                "KS_D": round(float(ks_d), 3),
                "KS_p": round(float(ks_p), 3),
            })
        for r in var_rows:
            r["delta_AICc"] = round(r["AICc"] - best_aicc, 2)
        rows.extend(var_rows)
    return pd.DataFrame(rows)


def _cvm_cdfs(u, v, cdf_fn, n_boot=500, rng=None):
    """Genest et al. (2009) empirical vs parametric Cramer-von Mises test."""
    n = len(u)
    emp = np.array([
        np.mean((u <= u[i]) & (v <= v[i]))
        for i in range(n)
    ])
    mod = cdf_fn(u, v)
    obs = float(np.sum((emp - mod) ** 2))
    if rng is None:
        return obs, np.nan
    boots = []
    for _ in range(n_boot):
        ub = stats.rankdata(rng.uniform(size=n)) / (n + 1.0)
        vb = stats.rankdata(rng.uniform(size=n)) / (n + 1.0)
        eb = np.array([np.mean((ub <= ub[i]) & (vb <= vb[i])) for i in range(n)])
        boots.append(np.sum((eb - cdf_fn(ub, vb)) ** 2))
    return obs, float((np.sum(np.asarray(boots) >= obs) + 1.0) / (n_boot + 1.0))


def fit_copulas(region_name, sd, fw, seed=101):
    """Evaluate Independence, Gaussian, Clayton, Gumbel, and Frank copulas on pseudo-observations."""
    rng = np.random.default_rng(seed)
    n = len(sd)
    u = stats.rankdata(sd) / (n + 1.0)
    v = stats.rankdata(fw) / (n + 1.0)
    tau = float(stats.kendalltau(sd, fw).statistic)

    def aicc(ll, k):
        return 2.0 * k - 2.0 * ll + (2.0 * k * (k + 1.0) / (n - k - 1.0) if k > 0 else 0.0)

    # Independence (ll = 0, k = 0)
    rows = [{
        "region": region_name,
        "copula": "Independence",
        "k": 0,
        "param": np.nan,
        "loglik": 0.0,
        "AICc": 0.0,
        "CvM_p": round(_cvm_cdfs(u, v, lambda x, y: x * y, n_boot=400, rng=rng)[1], 3),
    }]

    # Gaussian
    zu, zv = stats.norm.ppf(u), stats.norm.ppf(v)
    rho = float(np.clip(np.corrcoef(zu, zv)[0, 1], -0.98, 0.98))
    ll_g = float(np.sum(-0.5 * np.log(1.0 - rho**2) - (rho**2 * (zu**2 + zv**2) - 2.0 * rho * zu * zv) / (2.0 * (1.0 - rho**2))))
    rows.append({
        "region": region_name,
        "copula": "Gaussian",
        "k": 1,
        "param": round(rho, 4),
        "loglik": round(ll_g, 3),
        "AICc": round(aicc(ll_g, 1), 3),
        "CvM_p": np.nan,
    })

    # Frank
    def _nll_frank(th_arr):
        th = float(th_arr[0])
        if abs(th) < 1e-4:
            return 0.0
        num = th * (1.0 - np.exp(-th)) * np.exp(-th * (u + v))
        den = ((1.0 - np.exp(-th)) - (1.0 - np.exp(-th * u)) * (1.0 - np.exp(-th * v))) ** 2
        return float(-np.sum(np.log(np.clip(num / np.clip(den, 1e-12, None), 1e-12, None))))

    res_f = optimize.minimize(_nll_frank, [np.clip(tau * 9.0, -8.0, 8.0)], bounds=[(-20.0, 20.0)])
    th_f = float(res_f.x[0])
    ll_f = float(-res_f.fun)
    def _frank_cdf(x, y):
        if abs(th_f) < 1e-4:
            return x * y
        return (-1.0 / th_f) * np.log(1.0 + ((np.exp(-th_f * x) - 1.0) * (np.exp(-th_f * y) - 1.0)) / (np.exp(-th_f) - 1.0))
    rows.append({
        "region": region_name,
        "copula": "Frank",
        "k": 1,
        "param": round(th_f, 4),
        "loglik": round(ll_f, 3),
        "AICc": round(aicc(ll_f, 1), 3),
        "CvM_p": round(_cvm_cdfs(u, v, _frank_cdf, n_boot=400, rng=rng)[1], 3),
    })

    # Clayton (lower tail)
    def _nll_clayton(th_arr):
        t = float(th_arr[0])
        if t <= 1e-5:
            return 0.0
        inner = np.clip(u ** (-t) + v ** (-t) - 1.0, 1e-12, None)
        return float(-np.sum(np.log(1.0 + t) - (1.0 + t) * (np.log(u) + np.log(v)) - (2.0 + 1.0 / t) * np.log(inner)))

    res_c = optimize.minimize(_nll_clayton, [0.5], bounds=[(1e-5, 20.0)])
    ll_c = float(-res_c.fun)
    rows.append({
        "region": region_name,
        "copula": "Clayton",
        "k": 1,
        "param": round(float(res_c.x[0]), 4),
        "loglik": round(ll_c, 3),
        "AICc": round(aicc(ll_c, 1), 3),
        "CvM_p": np.nan,
    })

    # Gumbel (upper tail)
    def _nll_gumbel(th_arr):
        t = float(th_arr[0])
        lu, lv = -np.log(u), -np.log(v)
        s = lu**t + lv**t
        c_val = np.exp(-(s ** (1.0 / t)))
        dens = c_val * (s ** (1.0 / t - 2.0)) * (lu ** (t - 1.0)) * (lv ** (t - 1.0)) * (s ** (1.0 / t) + t - 1.0) / (u * v)
        return float(-np.sum(np.log(np.clip(dens, 1e-12, None))))

    res_g = optimize.minimize(_nll_gumbel, [1.05], bounds=[(1.0, 20.0)])
    ll_gu = float(-res_g.fun)
    rows.append({
        "region": region_name,
        "copula": "Gumbel",
        "k": 1,
        "param": round(float(res_g.x[0]), 4),
        "loglik": round(ll_gu, 3),
        "AICc": round(aicc(ll_gu, 1), 3),
        "CvM_p": np.nan,
    })

    df = pd.DataFrame(rows)
    df["delta_AICc_vs_Indep"] = (df["AICc"] - 0.0).round(3)
    return df


def prospective_onset_skill(spi, dry_spells, dry_th=-0.5, wet_th=0.5, max_gap=4):
    """Evaluate wet-transition outcomes forward from the real-time month-2 confirmation origin."""
    o0 = o1 = o2 = o3 = n_unc = 0
    leads = []
    for s0, _ in dry_spells:
        origin = s0 + 1
        if origin + max_gap >= len(spi):
            continue
        n_unc += 1
        window = spi[origin + 1 : origin + max_gap + 1]
        hit = np.where(window >= wet_th)[0]
        if len(hit) == 0:
            o0 += 1
            continue
        k_abs = origin + 1 + int(hit[0])
        leads.append(k_abs - origin)
        j2 = k_abs
        while j2 < len(spi) and np.isfinite(spi[j2]) and spi[j2] >= wet_th:
            j2 += 1
        fw_val = float(spi[k_abs:j2].sum())
        if fw_val < 1.0:
            o1 += 1
        elif fw_val < 2.0:
            o2 += 1
        else:
            o3 += 1

    # Clopper-Pearson 95% binomial CI for moderate-to-severe transition probability (O2 + O3)
    k_mod_sev = o2 + o3
    ci_lo = float(stats.beta.ppf(0.025, k_mod_sev, n_unc - k_mod_sev + 1))
    ci_hi = float(stats.beta.ppf(0.975, k_mod_sev + 1, n_unc - k_mod_sev))

    # Empirical unconditional base rates across all 4-month windows in the record
    u_any = u_mod = u_tot = 0
    for t in range(len(spi) - max_gap):
        u_tot += 1
        w4 = spi[t + 1 : t + max_gap + 1]
        hit = np.where(w4 >= wet_th)[0]
        if len(hit) > 0:
            u_any += 1
            k_abs = t + 1 + int(hit[0])
            j2 = k_abs
            while j2 < len(spi) and np.isfinite(spi[j2]) and spi[j2] >= wet_th:
                j2 += 1
            if spi[k_abs:j2].sum() >= 1.0:
                u_mod += 1

    return {
        "n_triggers_total": len(dry_spells),
        "n_uncensored_m2": n_unc,
        "O0_no_wet": o0,
        "O1_mild": o1,
        "O2_moderate": o2,
        "O3_severe": o3,
        "O0_pct": round(100.0 * o0 / n_unc, 1),
        "O1_pct": round(100.0 * o1 / n_unc, 1),
        "O2_pct": round(100.0 * o2 / n_unc, 1),
        "O3_pct": round(100.0 * o3 / n_unc, 1),
        "mod_sev_pct": round(100.0 * k_mod_sev / n_unc, 1),
        "mod_sev_ci95_lo": round(100.0 * ci_lo, 1),
        "mod_sev_ci95_hi": round(100.0 * ci_hi, 1),
        "lead_median_months": float(np.median(leads)) if leads else np.nan,
        "uncond_any_pct": round(100.0 * u_any / u_tot, 1),
        "uncond_fw1_pct": round(100.0 * u_mod / u_tot, 1),
    }


def bh_fdr(p_values):
    """Benjamini-Hochberg false discovery rate adjusted q-values."""
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    order = np.argsort(p_arr)
    q_arr = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(prev, p_arr[idx] * m / (rank + 1.0))
        q_arr[idx] = val
        prev = val
    return q_arr


def grid_dependence_fdr(precip_grid, months, years, frac, lats, lons, spi_fn, det_fn, n_perm=100, seed=42):
    """Compute per-cell Kendall's tau across all 748 basin grid cells plus spatial-field permutation check."""
    rng = np.random.default_rng(seed)
    cells = np.argwhere(frac > 0)
    spi_cube = np.full_like(precip_grid, np.nan, dtype=float)

    rows = []
    valid_p = []
    for iy, ix in cells:
        s = spi_fn(precip_grid[:, iy, ix], months, scale=1)
        spi_cube[:, iy, ix] = s
        evs, _ = det_fn(s)
        n_ev = len(evs)
        if n_ev >= 8:
            tau, p_val = stats.kendalltau([e["S_d"] for e in evs], [e["F_w"] for e in evs])
            valid_p.append(p_val)
        else:
            tau, p_val = np.nan, np.nan
        rows.append({
            "iy": int(iy),
            "ix": int(ix),
            "lat": round(float(lats[iy]), 3),
            "lon": round(float(lons[ix]), 3),
            "n_events": n_ev,
            "tau": round(float(tau), 4) if np.isfinite(tau) else np.nan,
            "p_value": round(float(p_val), 4) if np.isfinite(p_val) else np.nan,
        })

    df = pd.DataFrame(rows)
    ok = df["p_value"].notna().to_numpy()
    df["q_fdr"] = np.nan
    df.loc[ok, "q_fdr"] = np.round(bh_fdr(df.loc[ok, "p_value"].to_numpy()), 4)
    df["sig_fdr_05"] = ok & (df["q_fdr"] < 0.05)

    # Joint calendar-year permutation across all cells preserving spatial covariance
    unique_years = np.unique(years)
    yr_idx = [np.where(years == y)[0] for y in unique_years]
    null_frac05 = []
    for _ in range(n_perm):
        perm = rng.permutation(len(yr_idx))
        idx_map = np.concatenate([yr_idx[p] for p in perm])
        ps = []
        for iy, ix in cells:
            evs, _ = det_fn(spi_cube[idx_map, iy, ix])
            if len(evs) >= 8:
                ps.append(stats.kendalltau([e["S_d"] for e in evs], [e["F_w"] for e in evs]).pvalue)
        null_frac05.append(float(np.mean(np.asarray(ps) < 0.05)))

    summary = {
        "n_cells_eval": int(ok.sum()),
        "tau_median": round(float(df.loc[ok, "tau"].median()), 4),
        "raw_sig_n": int((df.loc[ok, "p_value"] < 0.05).sum()),
        "raw_sig_frac": round(float((df.loc[ok, "p_value"] < 0.05).mean()), 4),
        "fdr_sig_n": int(df["sig_fdr_05"].sum()),
        "field_null_mean_frac": round(float(np.mean(null_frac05)), 4),
        "field_null_p95_frac": round(float(np.percentile(null_frac05, 95)), 4),
    }
    return df, summary, spi_cube
