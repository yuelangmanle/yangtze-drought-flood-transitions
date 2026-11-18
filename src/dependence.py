"""Marginal screening, bivariate copula selection, prospective onset skill, and spatial FDR."""

import warnings
import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


MARGINAL_CANDIDATES = {
    "GEV": stats.genextreme,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Lognormal": stats.lognorm,
    "Exponential": stats.expon,
    "InvGamma": stats.invgamma,
}


def dependence_summary(sd, fw):
    """Compute Kendall's tau, Spearman's r_s, Pearson's r, 80%-power MDEs, and quantile slopes."""
    sd = np.asarray(sd, dtype=float)
    fw = np.asarray(fw, dtype=float)
    n = len(sd)
    tau, p_tau = stats.kendalltau(sd, fw)
    rs, p_rs = stats.spearmanr(sd, fw)
    rp, p_rp = stats.pearsonr(sd, fw)
    z_crit = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    mde_r = float(np.tanh(z_crit / np.sqrt(max(n - 3, 1)))) if n > 3 else np.nan
    mde_tau_asymp = float(z_crit * np.sqrt(2.0 * (2 * n + 5) / (9.0 * n * max(n - 1, 1)))) if n > 3 else np.nan
    mde_tau_gauss = float((2.0 / np.pi) * np.arcsin(mde_r)) if np.isfinite(mde_r) else np.nan
    fisher_z25 = np.arctanh(0.25) * np.sqrt(max(n - 3, 1))
    power_r25 = float(stats.norm.cdf(fisher_z25 - stats.norm.ppf(0.975)) + stats.norm.cdf(-fisher_z25 - stats.norm.ppf(0.975))) if n > 3 else np.nan

    q50 = sm.QuantReg(fw, sm.add_constant(sd)).fit(q=0.50, max_iter=2000)
    x_mat = sm.add_constant(sd)
    q90 = sm.QuantReg(fw, x_mat).fit(q=0.90, max_iter=3000)
    q90_slope = float(q90.params[1])
    rng_q = np.random.default_rng(77)
    boot_slopes = np.empty(400, dtype=float)
    for b_i in range(400):
        idx_b = rng_q.integers(0, n, size=n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot_slopes[b_i] = float(sm.QuantReg(fw[idx_b], x_mat[idx_b]).fit(q=0.90, max_iter=3000).params[1])
        except Exception:
            boot_slopes[b_i] = np.nan
    valid_b = boot_slopes[np.isfinite(boot_slopes)]
    q90_se = float(np.std(valid_b, ddof=1)) if len(valid_b) >= 20 else np.nan
    q90_p = float(2.0 * stats.norm.cdf(-abs(q90_slope / q90_se))) if (np.isfinite(q90_se) and q90_se > 0) else np.nan
    q90_lo, q90_hi = (float(np.percentile(valid_b, 2.5)), float(np.percentile(valid_b, 97.5))) if len(valid_b) >= 20 else (np.nan, np.nan)

    rng_c = np.random.default_rng(78)
    b_tau, b_rs = np.empty(400, dtype=float), np.empty(400, dtype=float)
    for b_i in range(400):
        idx_c = rng_c.integers(0, n, size=n)
        b_tau[b_i] = float(stats.kendalltau(sd[idx_c], fw[idx_c]).statistic)
        b_rs[b_i] = float(stats.spearmanr(sd[idx_c], fw[idx_c]).statistic)
    t_lo, t_hi = np.nanpercentile(b_tau, [2.5, 97.5])
    r_lo, r_hi = np.nanpercentile(b_rs, [2.5, 97.5])

    loo_taus, loo_ps = [], []
    if n >= 6:
        for k in range(n):
            tk, pk = stats.kendalltau(np.delete(sd, k), np.delete(fw, k))
            loo_taus.append(float(tk))
            loo_ps.append(float(pk))
    return {
        "n": n,
        "S_d_mean": round(float(np.mean(sd)), 3),
        "S_d_median": round(float(np.median(sd)), 3),
        "F_w_mean": round(float(np.mean(fw)), 3),
        "F_w_median": round(float(np.median(fw)), 3),
        "kendall_tau": round(float(tau), 4),
        "tau_ci95_lo": round(float(t_lo), 4),
        "tau_ci95_hi": round(float(t_hi), 4),
        "kendall_p": round(float(p_tau), 4),
        "spearman_r": round(float(rs), 4),
        "rs_ci95_lo": round(float(r_lo), 4),
        "rs_ci95_hi": round(float(r_hi), 4),
        "spearman_p": round(float(p_rs), 4),
        "pearson_r": round(float(rp), 4),
        "pearson_p": round(float(p_rp), 4),
        "mde_r_80pct": round(mde_r, 3),
        "mde_tau_80pct": round(mde_tau_asymp, 3),
        "mde_tau_gauss": round(mde_tau_gauss, 3),
        "power_at_r25_pct": round(100.0 * power_r25, 1),
        "q50_slope": round(float(q50.params[1]), 4),
        "q50_se": round(float(q50.bse[1]), 4),
        "q50_p": round(float(q50.pvalues[1]), 4),
        "q90_slope": round(q90_slope, 4),
        "q90_se": round(q90_se, 4) if np.isfinite(q90_se) else np.nan,
        "q90_p": round(q90_p, 4) if np.isfinite(q90_p) else np.nan,
        "q90_ci95_lo": round(q90_lo, 4) if np.isfinite(q90_lo) else np.nan,
        "q90_ci95_hi": round(q90_hi, 4) if np.isfinite(q90_hi) else np.nan,
        "loo_tau_min": round(float(min(loo_taus)), 4) if loo_taus else np.nan,
        "loo_tau_max": round(float(max(loo_taus)), 4) if loo_taus else np.nan,
        "loo_tau_max_p": round(float(max(loo_ps)), 4) if loo_ps else np.nan,
    }


def _fit_marginal_params(vals, dist_name, dist, loc_floor):
    if dist_name == "GEV":
        return dist.fit(vals), 3
    if dist_name == "Exponential":
        return dist.fit(vals, floc=loc_floor), 1
    return dist.fit(vals, floc=loc_floor), 2


def screen_marginals(sd, fw, n_boot=400, seed=303):
    """Fit parametric marginals anchored at physical run thresholds with bootstrap Lilliefors KS p-values."""
    rng = np.random.default_rng(seed)
    rows = []
    for var_name, values, thr_floor in (("S_d", sd, 1.0), ("F_w", fw, 0.5)):
        vals = np.asarray(values, dtype=float)
        loc_floor = min(thr_floor - 1e-4, float(np.min(vals)) - 1e-4)
        n = len(vals)
        best_aicc = np.inf
        var_rows = []
        for dist_name, dist in MARGINAL_CANDIDATES.items():
            params, k = _fit_marginal_params(vals, dist_name, dist, loc_floor)
            ll = float(np.sum(dist.logpdf(vals, *params)))
            aicc = 2.0 * k - 2.0 * ll + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
            ks_d_obs, _ = stats.kstest(vals, dist(*params).cdf)

            # Parametric bootstrap KS p-value with in-loop parameter re-estimation
            exceed = 0
            valid_b = 0
            for _ in range(n_boot):
                sim = dist.rvs(*params, size=n, random_state=rng)
                if dist_name != "GEV":
                    sim = np.maximum(sim, loc_floor + 1e-5)
                try:
                    p_b, _ = _fit_marginal_params(sim, dist_name, dist, loc_floor)
                    d_b, _ = stats.kstest(sim, dist(*p_b).cdf)
                    if np.isfinite(d_b):
                        valid_b += 1
                        if d_b >= float(ks_d_obs):
                            exceed += 1
                except Exception:
                    continue
            ks_p_boot = (exceed + 1.0) / (valid_b + 1.0) if valid_b > 0 else np.nan
            if aicc < best_aicc:
                best_aicc = aicc
            var_rows.append({
                "variable": var_name,
                "distribution": dist_name,
                "k": k,
                "loc_bound": round(float(loc_floor), 4),
                "loglik": round(ll, 2),
                "AICc": round(aicc, 2),
                "KS_D": round(float(ks_d_obs), 3),
                "KS_p": round(float(ks_p_boot), 3),
            })
        for r in var_rows:
            r["delta_AICc"] = round(r["AICc"] - best_aicc, 2)
        rows.extend(var_rows)
    return pd.DataFrame(rows)


def _gauss_cdf_2d(u, v, rho):
    zu = stats.norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))
    zv = stats.norm.ppf(np.clip(v, 1e-6, 1.0 - 1e-6))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    return np.asarray([mvn.cdf([a, b]) for a, b in zip(zu, zv)], dtype=float)


def _frank_cdf(u, v, th):
    if abs(th) < 1e-4:
        return u * v
    return (-1.0 / th) * np.log(1.0 + ((np.exp(-th * u) - 1.0) * (np.exp(-th * v) - 1.0)) / (np.exp(-th) - 1.0))


def _clayton_cdf(u, v, th):
    if th <= 1e-5:
        return u * v
    return np.clip(u ** (-th) + v ** (-th) - 1.0, 1e-12, None) ** (-1.0 / th)


def _gumbel_cdf(u, v, th):
    if th <= 1.0001:
        return u * v
    lu = -np.log(np.clip(u, 1e-9, 1.0 - 1e-9))
    lv = -np.log(np.clip(v, 1e-9, 1.0 - 1e-9))
    return np.exp(-((lu**th + lv**th) ** (1.0 / th)))


def _fit_copula_param(name, u, v, tau):
    if name == "Independence":
        return np.nan, 0.0
    if name == "Gaussian":
        zu, zv = stats.norm.ppf(u), stats.norm.ppf(v)
        def _nll_gauss(r_arr):
            r = float(r_arr[0])
            return float(-np.sum(-0.5 * np.log(1.0 - r**2) - (r**2 * (zu**2 + zv**2) - 2.0 * r * zu * zv) / (2.0 * (1.0 - r**2))))
        res = optimize.minimize(_nll_gauss, [float(np.clip(np.corrcoef(zu, zv)[0, 1], -0.95, 0.95))], bounds=[(-0.98, 0.98)])
        return float(res.x[0]), float(-res.fun)
    if name == "Frank":
        def _nll(th_arr):
            th = float(th_arr[0])
            if abs(th) < 1e-4:
                return 0.0
            num = th * (1.0 - np.exp(-th)) * np.exp(-th * (u + v))
            den = ((1.0 - np.exp(-th)) - (1.0 - np.exp(-th * u)) * (1.0 - np.exp(-th * v))) ** 2
            return float(-np.sum(np.log(np.clip(num / np.clip(den, 1e-12, None), 1e-12, None))))
        res = optimize.minimize(_nll, [np.clip(tau * 9.0, -10.0, 10.0)], bounds=[(-20.0, 20.0)])
        return float(res.x[0]), float(-res.fun)

    if name.startswith("Clayton"):
        uu = 1.0 - u if name == "Clayton_rot90" else u
        vv = 1.0 - v if name == "Clayton_rot270" else v
        def _nll_c(th_arr):
            t = float(th_arr[0])
            if t <= 1e-5:
                return 0.0
            inner = np.clip(uu ** (-t) + vv ** (-t) - 1.0, 1e-12, None)
            return float(-np.sum(np.log(1.0 + t) - (1.0 + t) * (np.log(uu) + np.log(vv)) - (2.0 + 1.0 / t) * np.log(inner)))
        res = optimize.minimize(_nll_c, [0.35], bounds=[(1e-5, 20.0)])
        return float(res.x[0]), float(-res.fun)

    if name.startswith("Gumbel"):
        uu = 1.0 - u if name == "Gumbel_rot90" else u
        vv = 1.0 - v if name == "Gumbel_rot270" else v
        def _nll_g(th_arr):
            t = float(th_arr[0])
            lu, lv = -np.log(np.clip(uu, 1e-9, 1.0 - 1e-9)), -np.log(np.clip(vv, 1e-9, 1.0 - 1e-9))
            s = lu**t + lv**t
            c_val = np.exp(-(s ** (1.0 / t)))
            dens = c_val * (s ** (1.0 / t - 2.0)) * (lu ** (t - 1.0)) * (lv ** (t - 1.0)) * (s ** (1.0 / t) + t - 1.0) / (uu * vv)
            return float(-np.sum(np.log(np.clip(dens, 1e-12, None))))
        res = optimize.minimize(_nll_g, [1.08], bounds=[(1.0, 20.0)])
        return float(res.x[0]), float(-res.fun)
    raise ValueError(f"Unknown copula {name}")


def _eval_copula_cdf(name, u, v, param):
    if name == "Independence":
        return u * v
    if name == "Gaussian":
        return _gauss_cdf_2d(u, v, param)
    if name == "Frank":
        return _frank_cdf(u, v, param)
    if name == "Clayton":
        return _clayton_cdf(u, v, param)
    if name == "Clayton_rot90":
        return v - _clayton_cdf(1.0 - u, v, param)
    if name == "Clayton_rot270":
        return u - _clayton_cdf(u, 1.0 - v, param)
    if name == "Gumbel":
        return _gumbel_cdf(u, v, param)
    if name == "Gumbel_rot90":
        return v - _gumbel_cdf(1.0 - u, v, param)
    if name == "Gumbel_rot270":
        return u - _gumbel_cdf(u, 1.0 - v, param)
    raise ValueError(f"Unknown copula {name}")


def _sample_copula(name, param, n, rng):
    if name == "Independence":
        return rng.uniform(size=n), rng.uniform(size=n)
    if name == "Gaussian":
        z = rng.multivariate_normal([0.0, 0.0], [[1.0, param], [param, 1.0]], size=n)
        return stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])
    if name == "Frank":
        u = rng.uniform(1e-6, 1.0 - 1e-6, size=n)
        t = rng.uniform(1e-6, 1.0 - 1e-6, size=n)
        if abs(param) < 1e-4:
            return u, t
        v = (-1.0 / param) * np.log(1.0 + t * (np.exp(-param) - 1.0) / (t + (1.0 - t) * np.exp(-param * u)))
        return u, np.clip(v, 1e-6, 1.0 - 1e-6)
    if name.startswith("Clayton"):
        u = rng.uniform(1e-6, 1.0 - 1e-6, size=n)
        w = rng.uniform(1e-6, 1.0 - 1e-6, size=n)
        if param <= 1e-5:
            v = w
        else:
            v = ((u ** (-param)) * (w ** (-param / (1.0 + param)) - 1.0) + 1.0) ** (-1.0 / param)
        v = np.clip(v, 1e-6, 1.0 - 1e-6)
        if name == "Clayton_rot90":
            return 1.0 - u, v
        if name == "Clayton_rot270":
            return u, 1.0 - v
        return u, v
    if name.startswith("Gumbel"):
        if param <= 1.0001:
            u, v = rng.uniform(size=n), rng.uniform(size=n)
        else:
            alpha = 1.0 / param
            u_st = rng.uniform(1e-6, np.pi - 1e-6, size=n)
            w_st = rng.exponential(1.0, size=n)
            s_st = (np.sin(alpha * u_st) / (np.sin(u_st) ** (1.0 / alpha))) * (
                (np.sin((1.0 - alpha) * u_st) / w_st) ** ((1.0 - alpha) / alpha)
            )
            e1 = rng.exponential(1.0, size=n)
            e2 = rng.exponential(1.0, size=n)
            u = np.clip(np.exp(-((e1 / s_st) ** alpha)), 1e-6, 1.0 - 1e-6)
            v = np.clip(np.exp(-((e2 / s_st) ** alpha)), 1e-6, 1.0 - 1e-6)
        if name == "Gumbel_rot90":
            return 1.0 - u, v
        if name == "Gumbel_rot270":
            return u, 1.0 - v
        return u, v
    raise ValueError(f"Unknown copula {name}")


def _copula_cvm_bootstrap(name, u, v, param, n_boot=400, rng=None):
    """Genest et al. (2009) parametric bootstrap Cramer-von Mises test with parameter re-estimation."""
    n = len(u)
    emp = np.array([np.mean((u <= u[i]) & (v <= v[i])) for i in range(n)])
    mod = _eval_copula_cdf(name, u, v, param)
    obs = float(np.sum((emp - mod) ** 2))
    if rng is None:
        return obs, np.nan

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        us, vs = _sample_copula(name, param, n, rng)
        ub = stats.rankdata(us) / (n + 1.0)
        vb = stats.rankdata(vs) / (n + 1.0)
        tau_b = float(stats.kendalltau(ub, vb).statistic)
        par_b, _ = _fit_copula_param(name, ub, vb, tau_b)
        eb = np.array([np.mean((ub <= ub[i]) & (vb <= vb[i])) for i in range(n)])
        mb = _eval_copula_cdf(name, ub, vb, par_b)
        boots[b] = float(np.sum((eb - mb) ** 2))
    p_val = float((np.sum(boots >= obs) + 1.0) / (n_boot + 1.0))
    return obs, p_val


def fit_copulas(region_name, sd, fw, seed=101, n_boot=400):
    """Evaluate Independence, Gaussian, Frank, Clayton, Gumbel, and rotated negative-tail copulas."""
    rng = np.random.default_rng(seed)
    n = len(sd)
    u = stats.rankdata(sd) / (n + 1.0)
    v = stats.rankdata(fw) / (n + 1.0)
    tau = float(stats.kendalltau(sd, fw).statistic)

    def aicc(ll, k):
        return 2.0 * k - 2.0 * ll + (2.0 * k * (k + 1.0) / (n - k - 1.0) if k > 0 else 0.0)

    candidates = [
        ("Independence", 0),
        ("Gaussian", 1),
        ("Frank", 1),
        ("Clayton", 1),
        ("Clayton_rot90", 1),
        ("Clayton_rot270", 1),
        ("Gumbel", 1),
        ("Gumbel_rot90", 1),
        ("Gumbel_rot270", 1),
    ]
    rows = []
    for name, k in candidates:
        par, ll = _fit_copula_param(name, u, v, tau)
        ac = aicc(ll, k)
        _, cvm_p = _copula_cvm_bootstrap(name, u, v, par, n_boot=n_boot, rng=rng)
        rows.append({
            "region": region_name,
            "copula": name,
            "k": k,
            "param": round(float(par), 4) if np.isfinite(par) else np.nan,
            "loglik": round(float(ll), 3),
            "AICc": round(float(ac), 3),
            "CvM_p": round(float(cvm_p), 3),
        })

    df = pd.DataFrame(rows)
    df["delta_AICc_vs_Indep"] = (df["AICc"] - 0.0).round(3)
    best_idx = df["AICc"].idxmin()
    df["is_best_AICc"] = df.index == best_idx
    return df


def prospective_onset_skill(
    spi,
    dry_spells,
    dry_th=-0.5,
    wet_th=0.5,
    max_gap=4,
    periods=None,
    season_filter=None,
    rng_seed=404,
):
    """Evaluate real-time month-2 wet-transition probabilities, strict window censoring, and non-wet baselines."""
    rng = np.random.default_rng(rng_seed)
    o0 = o1 = o2 = o3 = n_unc = 0
    strict_o23 = 0
    n_after_horizon = 0
    leads = []
    trig_months = []
    trig_years = []
    trig_modsev = []

    def _season_match(mo_val):
        if season_filter is None:
            return True
        if season_filter == "Flood (May-Sep)":
            return 5 <= mo_val <= 9
        if season_filter == "Spring (Mar-Apr)":
            return mo_val in (3, 4)
        if season_filter == "Autumn (Oct-Nov)":
            return mo_val in (10, 11)
        if season_filter == "Winter (Dec-Feb)":
            return mo_val in (12, 1, 2)
        if season_filter == "Cool (Oct-Apr)":
            return not (5 <= mo_val <= 9)
        return True

    for s0, _ in dry_spells:
        origin = s0 + 1
        if origin + max_gap >= len(spi):
            continue
        if periods is not None and not _season_match(int(periods[origin].month)):
            continue

        n_unc += 1
        trig_months.append(int(periods[origin].month) if periods is not None else ((origin % 12) + 1))
        trig_years.append(int(periods[origin].year) if periods is not None else (1961 + origin // 12))

        window = spi[origin + 1 : origin + max_gap + 1]
        hit = np.where(window >= wet_th)[0]
        if len(hit) == 0:
            o0 += 1
            trig_modsev.append(0)
            # Check if a transition actually arrived at lead > max_gap within [s1+1, s1+max_gap]
            j_end = origin
            while j_end + 1 < len(spi) and np.isfinite(spi[j_end + 1]) and spi[j_end + 1] <= dry_th:
                j_end += 1
            ext = spi[origin + max_gap + 1 : min(j_end + max_gap + 1, len(spi))]
            if len(ext) > 0 and np.any(ext >= wet_th):
                n_after_horizon += 1
            continue

        k_abs = origin + 1 + int(hit[0])
        leads.append(k_abs - origin)
        j2 = k_abs
        while j2 < len(spi) and np.isfinite(spi[j2]) and spi[j2] >= wet_th:
            j2 += 1
        fw_val = float(spi[k_abs:j2].sum())
        fw_strict = float(spi[k_abs : min(j2, origin + max_gap + 1)].sum())
        if fw_strict >= 1.0:
            strict_o23 += 1

        if fw_val < 1.0:
            o1 += 1
            trig_modsev.append(0)
        elif fw_val < 2.0:
            o2 += 1
            trig_modsev.append(1)
        else:
            o3 += 1
            trig_modsev.append(1)

    k_mod_sev = o2 + o3
    ci_lo = float(stats.beta.ppf(0.025, k_mod_sev, n_unc - k_mod_sev + 1)) if n_unc > 0 else np.nan
    ci_hi = float(stats.beta.ppf(0.975, k_mod_sev + 1, n_unc - k_mod_sev)) if n_unc > 0 else np.nan

    # Year-clustered bootstrap 95% CI across calendar years
    if n_unc > 0:
        yr_arr = np.asarray(trig_years)
        ms_arr = np.asarray(trig_modsev)
        uniq_yrs = np.unique(yr_arr)
        yr_groups = [ms_arr[yr_arr == y] for y in uniq_yrs]
        b_rates = []
        for _ in range(1000):
            draw_idx = rng.integers(0, len(yr_groups), size=len(yr_groups))
            samp = np.concatenate([yr_groups[g] for g in draw_idx])
            b_rates.append(float(np.mean(samp)))
        c_lo, c_hi = np.percentile(b_rates, [2.5, 97.5])
    else:
        c_lo, c_hi = np.nan, np.nan

    # All-month rolling baseline plus seasonally matched non-wet-origin (SPI < wet_th) & disjoint neutral (-0.5 < SPI < wet_th) baselines
    u_any = u_mod = u_tot = 0
    nw_by_mo = {m: [] for m in range(1, 13)}
    neu_by_mo = {m: [] for m in range(1, 13)}
    nw_yr_mo = {}
    for t in range(len(spi) - max_gap):
        mo_t = int(periods[t].month) if periods is not None else ((t % 12) + 1)
        yr_t = int(periods[t].year) if periods is not None else (1961 + t // 12)
        if not _season_match(mo_t):
            continue
        u_tot += 1
        w4 = spi[t + 1 : t + max_gap + 1]
        hit = np.where(w4 >= wet_th)[0]
        hit_fw1 = 0
        if len(hit) > 0:
            u_any += 1
            k_abs = t + 1 + int(hit[0])
            j2 = k_abs
            while j2 < len(spi) and np.isfinite(spi[j2]) and spi[j2] >= wet_th:
                j2 += 1
            if spi[k_abs:j2].sum() >= 1.0:
                u_mod += 1
                hit_fw1 = 1
        if np.isfinite(spi[t]) and spi[t] < wet_th:
            nw_by_mo[mo_t].append(hit_fw1)
            nw_yr_mo.setdefault(yr_t, []).append((mo_t, hit_fw1))
            if spi[t] > dry_th:
                neu_by_mo[mo_t].append(hit_fw1)

    if trig_months:
        mo_weights = {m: trig_months.count(m) / float(len(trig_months)) for m in range(1, 13)}
        nonwet_rate = sum(
            mo_weights[m] * (float(np.mean(nw_by_mo[m])) if nw_by_mo[m] else 0.0)
            for m in range(1, 13)
        )
        neutral_rate = sum(
            mo_weights[m] * (float(np.mean(neu_by_mo[m])) if neu_by_mo[m] else 0.0)
            for m in range(1, 13)
        )
        # Year-block bootstrap CI for the weighted non-wet baseline preserving intra-year overlap
        base_yrs = sorted(nw_yr_mo.keys())
        nw_boots = []
        for _ in range(1000):
            sampled_yrs = rng.choice(base_yrs, size=len(base_yrs), replace=True)
            b_sums = {m: 0.0 for m in range(1, 13)}
            b_cnts = {m: 0 for m in range(1, 13)}
            for yy in sampled_yrs:
                for mo_v, hv in nw_yr_mo[int(yy)]:
                    b_sums[mo_v] += hv
                    b_cnts[mo_v] += 1
            r_b = sum(
                mo_weights[m] * (b_sums[m] / b_cnts[m] if b_cnts[m] > 0 else float(np.mean(nw_by_mo[m])))
                for m in range(1, 13) if mo_weights[m] > 0
            )
            nw_boots.append(r_b)
        nw_lo, nw_hi = np.percentile(nw_boots, [2.5, 97.5])
    else:
        nonwet_rate, neutral_rate, nw_lo, nw_hi = np.nan, np.nan, np.nan, np.nan

    return {
        "season_filter": season_filter or "All",
        "n_triggers_total": len(dry_spells),
        "n_uncensored_m2": n_unc,
        "O0_no_wet": o0,
        "O1_mild": o1,
        "O2_moderate": o2,
        "O3_severe": o3,
        "n_after_4mo_horizon": n_after_horizon,
        "O0_pct": round(100.0 * o0 / n_unc, 1) if n_unc else np.nan,
        "O1_pct": round(100.0 * o1 / n_unc, 1) if n_unc else np.nan,
        "O2_pct": round(100.0 * o2 / n_unc, 1) if n_unc else np.nan,
        "O3_pct": round(100.0 * o3 / n_unc, 1) if n_unc else np.nan,
        "mod_sev_pct": round(100.0 * k_mod_sev / n_unc, 1) if n_unc else np.nan,
        "mod_sev_ci95_lo": round(100.0 * ci_lo, 1) if np.isfinite(ci_lo) else np.nan,
        "mod_sev_ci95_hi": round(100.0 * ci_hi, 1) if np.isfinite(ci_hi) else np.nan,
        "cluster_ci95_lo": round(100.0 * c_lo, 1) if np.isfinite(c_lo) else np.nan,
        "cluster_ci95_hi": round(100.0 * c_hi, 1) if np.isfinite(c_hi) else np.nan,
        "strict_4mo_mod_sev_pct": round(100.0 * strict_o23 / n_unc, 1) if n_unc else np.nan,
        "lead_median_months": float(np.median(leads)) if leads else np.nan,
        "uncond_any_pct": round(100.0 * u_any / u_tot, 1) if u_tot else np.nan,
        "uncond_fw1_pct": round(100.0 * u_mod / u_tot, 1) if u_tot else np.nan,
        "nonwet_base_pct": round(100.0 * nonwet_rate, 1) if np.isfinite(nonwet_rate) else np.nan,
        "nonwet_base_ci95_lo": round(100.0 * nw_lo, 1) if np.isfinite(nw_lo) else np.nan,
        "nonwet_base_ci95_hi": round(100.0 * nw_hi, 1) if np.isfinite(nw_hi) else np.nan,
        "neutral_base_pct": round(100.0 * neutral_rate, 1) if np.isfinite(neutral_rate) else np.nan,
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


if _HAS_NUMBA:
    @njit
    def _extract_sd_fw_nb(x):
        n = len(x)
        sd_out = np.empty(128, dtype=np.float64)
        fw_out = np.empty(128, dtype=np.float64)
        cnt = 0
        i = 0
        while i < n:
            j = i
            while j < n and not np.isnan(x[j]) and x[j] <= -0.5:
                j += 1
            if j - i >= 2:
                s0 = i
                s1 = j - 1
                k = s1 + 1
                limit = s1 + 4
                if limit >= n:
                    limit = n - 1
                while k <= limit:
                    if not np.isnan(x[k]) and x[k] >= 0.5:
                        j2 = k
                        while j2 < n and not np.isnan(x[j2]) and x[j2] >= 0.5:
                            j2 += 1
                        sd_v = 0.0
                        for t in range(s0, s1 + 1):
                            sd_v -= x[t]
                        fw_v = 0.0
                        for t in range(k, j2):
                            fw_v += x[t]
                        sd_out[cnt] = sd_v
                        fw_out[cnt] = fw_v
                        cnt += 1
                        break
                    k += 1
            i = j if j > i else i + 1
        return sd_out[:cnt], fw_out[:cnt]

    @njit
    def _kendall_tau_fast_nb(a, b):
        n = len(a)
        con = 0
        dis = 0
        t_a = 0
        t_b = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                da = a[i] - a[j]
                db = b[i] - b[j]
                if da == 0.0 and db == 0.0:
                    continue
                if da == 0.0:
                    t_a += 1
                elif db == 0.0:
                    t_b += 1
                elif da * db > 0.0:
                    con += 1
                else:
                    dis += 1
        tot = n * (n - 1) // 2
        den = np.sqrt(float(tot - t_a) * float(tot - t_b))
        if den <= 0.0:
            return 0.0, 1.0
        tau = (con - dis) / den
        # Normal approximation matches scipy exact at n >= 8 smoothly for field null screening
        var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
        z = (con - dis) / np.sqrt(var_s)
        return tau, z
else:
    def _extract_sd_fw_nb(x):
        return None, None


def grid_dependence_fdr(precip_grid, months, years, frac, lats, lons, spi_fn, det_fn, n_perm=1000, seed=42):
    """Compute per-cell Kendall's tau with BH-FDR and a spatially coherent seasonal AR(1) field null."""
    rng = np.random.default_rng(seed)
    cells = np.argwhere(frac > 0)
    n_cells = len(cells)
    n_time = precip_grid.shape[0]
    spi_cube = np.full((n_time, n_cells), np.nan, dtype=float)

    rows = []
    for c_i, (iy, ix) in enumerate(cells):
        s = spi_fn(precip_grid[:, iy, ix], months, scale=1)
        spi_cube[:, c_i] = s
        evs, _ = det_fn(s)
        n_ev = len(evs)
        if n_ev >= 8:
            tau, p_val = stats.kendalltau([e["S_d"] for e in evs], [e["F_w"] for e in evs])
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

    # Fit per-cell calendar-month lag-1 AR(1) parameters and spatial innovation correlation matrix
    r_mat = np.zeros((12, n_cells), dtype=float)
    innov = np.zeros((n_time - 1, n_cells), dtype=float)
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        idx = idx[idx > 0]
        x0 = spi_cube[idx - 1]
        x1 = spi_cube[idx]
        x0_c = x0 - np.mean(x0, axis=0, keepdims=True)
        x1_c = x1 - np.mean(x1, axis=0, keepdims=True)
        num = np.sum(x0_c * x1_c, axis=0)
        den = np.sqrt(np.sum(x0_c**2, axis=0) * np.sum(x1_c**2, axis=0))
        r_m = np.where(den > 1e-12, np.clip(num / den, -0.8, 0.8), 0.0)
        r_mat[mo - 1] = r_m
        innov[idx - 1] = (x1 - r_m[None, :] * x0) / np.sqrt(np.maximum(1.0 - r_m[None, :] ** 2, 1e-6))

    spat_corr = np.corrcoef(innov, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(spat_corr)
    keep = eigvals > 1e-10
    chol_spat = np.ascontiguousarray(eigvecs[:, keep] * np.sqrt(eigvals[keep])[None, :], dtype=float)
    z_crit = 1.959963984540054

    null_frac05 = np.empty(n_perm, dtype=float)
    null_mean_taus = np.empty(n_perm, dtype=float)
    for p_i in range(n_perm):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            eps = rng.normal(size=(n_time, chol_spat.shape[1])) @ chol_spat.T
        if not np.all(np.isfinite(eps)):
            raise FloatingPointError("Non-finite Gaussian field draw in spatial null simulation")
        sim_cube = np.empty((n_time, n_cells), dtype=float)
        sim_cube[0] = eps[0]
        for t in range(1, n_time):
            rm = r_mat[int(months[t]) - 1]
            sim_cube[t] = rm * sim_cube[t - 1] + np.sqrt(np.maximum(1.0 - rm**2, 1e-6)) * eps[t]

        n_sig = 0
        n_eval_p = 0
        tau_acc = []
        if _HAS_NUMBA:
            for c_i in range(n_cells):
                sd_a, fw_a = _extract_sd_fw_nb(sim_cube[:, c_i])
                if len(sd_a) >= 8:
                    tau_v, z_v = _kendall_tau_fast_nb(sd_a, fw_a)
                    n_eval_p += 1
                    tau_acc.append(tau_v)
                    if abs(z_v) >= z_crit:
                        n_sig += 1
        else:
            for c_i in range(n_cells):
                evs, _ = det_fn(sim_cube[:, c_i])
                if len(evs) >= 8:
                    tv, pv = stats.kendalltau([e["S_d"] for e in evs], [e["F_w"] for e in evs])
                    n_eval_p += 1
                    tau_acc.append(float(tv))
                    if pv < 0.05:
                        n_sig += 1
        null_frac05[p_i] = float(n_sig / max(n_eval_p, 1))
        null_mean_taus[p_i] = float(np.mean(tau_acc)) if tau_acc else 0.0

    obs_raw_frac = float((df.loc[ok, "p_value"] < 0.05).mean())
    field_p_val = float((np.sum(null_frac05 >= obs_raw_frac) + 1.0) / (n_perm + 1.0))

    summary = {
        "n_cells_eval": int(ok.sum()),
        "n_perm": int(n_perm),
        "field_null_type": "spatial_seasonal_ar1_h0",
        "tau_median": round(float(df.loc[ok, "tau"].median()), 4),
        "raw_sig_n": int((df.loc[ok, "p_value"] < 0.05).sum()),
        "raw_sig_frac": round(obs_raw_frac, 4),
        "fdr_sig_n": int(df["sig_fdr_05"].sum()),
        "min_q_fdr": round(float(df.loc[ok, "q_fdr"].min()), 4),
        "field_null_mean_tau": round(float(np.mean(null_mean_taus)), 4),
        "field_null_mean_frac": round(float(np.mean(null_frac05)), 4),
        "field_null_p95_frac": round(float(np.percentile(null_frac05, 95)), 4),
        "field_p_value": round(field_p_val, 4),
    }
    return df, summary, spi_cube
