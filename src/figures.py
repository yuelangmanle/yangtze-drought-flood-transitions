"""Publication-grade figure suite for Yangtze drought-to-flood transition diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy import stats


PALETTE = {
    "navy": "#1D3557",
    "blue": "#2A6F97",
    "sky": "#61A5C2",
    "red": "#B22222",
    "amber": "#D97706",
    "teal": "#0F766E",
    "slate": "#475569",
    "light": "#E2E8F0",
    "upper": "#DCE8F2",
    "middle": "#DCEFE6",
    "lower": "#F7E6D4",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _save_fig(fig, out_dir, stem, formats=("png",), dpi=600):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_coastlines(ax, segments, xlim, ylim, color="#64748B", lw=0.65, zorder=2):
    for seg in segments:
        if (
            np.nanmax(seg[:, 0]) < xlim[0]
            or np.nanmin(seg[:, 0]) > xlim[1]
            or np.nanmax(seg[:, 1]) < ylim[0]
            or np.nanmin(seg[:, 1]) > ylim[1]
        ):
            continue
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, zorder=zorder)


def _draw_basin_perimeter(ax, lats, lons, frac, color="#111827", lw=1.2, zorder=5):
    ax.contour(lons, lats, (frac > 0).astype(float), levels=[0.5], colors=[color], linewidths=[lw], zorder=zorder)


def _draw_subbasin_dividers(ax, lats, lons, zone, frac, lw=0.8, color="#475569", zorder=4):
    ax.contour(lons, lats, (zone == 1).astype(float), levels=[0.5], colors=[color], linewidths=[lw], linestyles=["--"], zorder=zorder)
    ax.contour(lons, lats, (zone == 3).astype(float), levels=[0.5], colors=[color], linewidths=[lw], linestyles=["--"], zorder=zorder)


def plot_fig1_study_area(mask, pr_basin, months, segments, stn_grid, out_dir, formats=("png",)):
    """Basin geographic division, tributary locations, CRU station count, and rainfall seasonality."""
    lats, lons, frac, zone = mask["lat"], mask["lon"], mask["frac"], mask["zone"]
    fig = plt.figure(figsize=(11.2, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    z_plot = np.where(frac > 0, zone.astype(float), np.nan)
    cmap_sub = matplotlib.colors.ListedColormap([PALETTE["upper"], PALETTE["middle"], PALETTE["lower"]])
    ax.pcolormesh(lons, lats, z_plot, cmap=cmap_sub, vmin=0.5, vmax=3.5, shading="nearest", zorder=1)
    _draw_coastlines(ax, segments, (89, 123.5), (23.5, 36.5), color="#94A3B8", lw=0.6, zorder=2)
    _draw_basin_perimeter(ax, lats, lons, frac, color="#0F172A", lw=1.35, zorder=3)
    _draw_subbasin_dividers(ax, lats, lons, zone, frac, lw=0.85, color="#334155", zorder=3)

    # Key gauge & reservoir markers
    ax.scatter([111.00, 117.62], [30.82, 30.77], c=[PALETTE["red"], PALETTE["navy"]], s=[65, 65],
               marker="^", edgecolors="white", linewidths=0.8, zorder=6)
    ax.annotate("Three Gorges\n(Yichang)", (111.00, 30.82), xytext=(-22, -26),
                textcoords="offset points", fontsize=7.8, fontweight="bold", color=PALETTE["red"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["red"], lw=0.7), zorder=7)
    ax.annotate("Datong\nStation", (117.62, 30.77), xytext=(8, -22),
                textcoords="offset points", fontsize=7.8, fontweight="bold", color=PALETTE["navy"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["navy"], lw=0.7), zorder=7)

    # Subbasin & tributary labels
    ax.text(101.5, 31.3, "Upper Basin\n(TGP upstream)", fontsize=8.4, fontweight="bold", color="#1E3A5F", ha="center", zorder=6)
    ax.text(112.0, 27.0, "Middle\nBasin", fontsize=8.4, fontweight="bold", color="#0F5132", ha="center", zorder=6)
    ax.text(119.6, 32.1, "Lower", fontsize=8.4, fontweight="bold", color="#7C2D12", ha="center", zorder=6)

    for lbl, lo, la in [
        ("Yalong", 101.2, 28.4),
        ("Min-Tuo", 103.6, 30.6),
        ("Jialing", 106.1, 31.8),
        ("Wujiang", 107.6, 27.6),
        ("Hanjiang", 111.0, 32.8),
        ("Dongting", 111.4, 28.7),
        ("Poyang", 115.8, 27.6),
    ]:
        ax.text(lo, la, lbl, fontsize=7.3, color="#334155", style="italic", ha="center", zorder=6)

    handles = [
        mpatches.Patch(facecolor=PALETTE["upper"], edgecolor="#334155", lw=0.6, label="Upper YRB (22.6–35.8°N)"),
        mpatches.Patch(facecolor=PALETTE["middle"], edgecolor="#334155", lw=0.6, label="Middle YRB (Yichang–Hukou)"),
        mpatches.Patch(facecolor=PALETTE["lower"], edgecolor="#334155", lw=0.6, label="Lower YRB (Hukou–Estuary)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.5)
    ax.set_xlim(89.5, 123.0)
    ax.set_ylim(23.8, 36.2)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(a) Subregions and hydrometric stations", loc="left")

    ax2 = fig.add_subplot(gs[0, 1])
    mo_means = [float(np.mean(pr_basin[months == m])) for m in range(1, 13)]
    mo_iqr25 = [float(np.percentile(pr_basin[months == m], 25)) for m in range(1, 13)]
    mo_iqr75 = [float(np.percentile(pr_basin[months == m], 75)) for m in range(1, 13)]
    x_mo = np.arange(1, 13)
    colors = [
        PALETTE["blue"] if 5 <= m <= 9 else (PALETTE["sky"] if m in (3, 4, 10, 11) else PALETTE["slate"])
        for m in x_mo
    ]
    ax2.bar(x_mo, mo_means, color=colors, width=0.68, edgecolor="#1E293B", linewidth=0.5, alpha=0.88, zorder=3)
    ax2.errorbar(x_mo, mo_means, yerr=[np.array(mo_means) - np.array(mo_iqr25), np.array(mo_iqr75) - np.array(mo_means)],
                 fmt="none", ecolor="#0F172A", elinewidth=1.0, capsize=2.5, zorder=4)
    ax2.axvspan(4.5, 9.5, color=PALETTE["blue"], alpha=0.07, zorder=1)
    ax2.text(7.0, max(mo_iqr75) * 1.18, "Flood season (May–Sep)\n41.0% of events (n = 16)", ha="center",
             va="top", fontsize=8.0, color=PALETTE["navy"], fontweight="bold")
    ax2.set_xticks(x_mo)
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.set_xlabel("Calendar Month (1961–2025)")
    ax2.set_ylabel("Basin-Mean Precipitation (mm month$^{-1}$)")
    ax2.set_ylim(0, max(mo_iqr75) * 1.24)
    ax2.set_title("(b) Monthly rainfall regime (1961–2025)", loc="left")

    _save_fig(fig, out_dir, "Fig1", formats=formats)


def plot_fig2_tributary_amplification(trib_df, out_dir, formats=("png",)):
    """Basin-average masking and tributary-scale amplification across the 39 events."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.26})

    fw_b = trib_df["F_w_basin"].to_numpy()
    fw_t = trib_df["F_w_peak_trib"].to_numpy()
    ratio = trib_df["spi_amp_ratio"].to_numpy()
    is_mild = fw_b < 1.0

    # Scatter of basin Fw vs peak tributary Fw
    lim = max(fw_b.max(), fw_t.max()) * 1.1
    ax1.plot([0, lim], [0, lim], ls="--", color="#94A3B8", lw=1.0, label="1:1 (No spatial amplification)", zorder=1)
    ax1.axvline(1.0, ls=":", color=PALETTE["slate"], lw=1.0, zorder=1)

    ax1.scatter(fw_b[is_mild], fw_t[is_mild], c=PALETTE["amber"], s=58, edgecolors="#1E293B", lw=0.7,
                label=f"Mild basin $F_w$ < 1.0 (median ratio {np.median(ratio[is_mild]):.2f}×)", zorder=4)
    ax1.scatter(fw_b[~is_mild], fw_t[~is_mild], c=PALETTE["teal"], s=58, marker="D", edgecolors="#1E293B", lw=0.7,
                label=f"Mod/Sev basin $F_w$ ≥ 1.0 (median ratio {np.median(ratio[~is_mild]):.2f}×)", zorder=4)

    # Annotate top 4 illustrative transitions cleanly without overlap
    highlights = trib_df.sort_values("F_w_peak_trib", ascending=False).head(2).index.tolist()
    mild_top = trib_df[is_mild].sort_values("spi_amp_ratio", ascending=False).head(2).index.tolist()
    offsets = [(10, -14), (-14, 16), (10, 6), (20, -6)]
    for idx_pos, idx in enumerate(highlights + mild_top):
        row = trib_df.loc[idx]
        dx, dy = offsets[idx_pos % len(offsets)]
        ax1.annotate(
            f"{row['wet_start']} ({row['peak_tributary']})",
            (row["F_w_basin"], row["F_w_peak_trib"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=7.6, color="#1E293B",
            arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6), zorder=6,
        )

    ax1.set_xlim(0.3, fw_b.max() * 1.18)
    ax1.set_ylim(0.5, fw_t.max() * 1.22)
    ax1.set_xlabel("Basin-Mean Wet Severity $F_w$ (SPI-1 sum)")
    ax1.set_ylabel("Peak Tributary Wet Severity $F_w$ (SPI-1 sum)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CBD5E1")
    ax1.text(0.04, 0.94, f"All events median SPI ratio: {np.median(ratio):.2f}×\n"
                         f"Physical precip. excess median: {trib_df['mm_excess_ratio'].median():.2f}× (mm)",
             transform=ax1.transAxes, va="top", fontsize=8.0,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax1.set_title("(a) Subbasin vs. basin-mean $F_w$", loc="left")

    # Right: Tributary peak count + amplification ratio boxplot by basin stratum
    trib_counts = trib_df["peak_tributary"].value_counts()
    order = ["Yalong", "Min-Tuo", "Jialing", "Wujiang", "Hanjiang", "Dongting", "Poyang"]
    vals = [int(trib_counts.get(k, 0)) for k in order]
    bars = ax2.barh(order[::-1], vals[::-1], color=PALETTE["blue"], edgecolor="#1E293B", lw=0.6, alpha=0.86, zorder=3)
    for bar, v in zip(bars, vals[::-1]):
        ax2.text(v + 0.18, bar.get_y() + bar.get_height() / 2.0, str(v), va="center", fontsize=8.5, color="#0F172A")
    ax2.set_xlim(0, max(vals) + 2.5)
    ax2.set_xlabel("Number of Events Peaking in Tributary (Total = 39)")
    ax2.set_title("(b) Peak tributary frequency", loc="left")

    _save_fig(fig, out_dir, "Fig2", formats=formats)


def plot_fig3_chronology_and_seasonality(events_df, out_dir, formats=("png",)):
    """Full 39-event chronology (1961–2025) and seasonal distribution split by flood/winter/shoulder."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.22})

    x_yr = events_df["year"].to_numpy() + (events_df["trans_month"].to_numpy() - 0.5) / 12.0
    sd = events_df["S_d"].to_numpy()
    fw = events_df["F_w"].to_numpy()

    ax1.axhline(0, color="#0F172A", lw=0.8)
    ax1.bar(x_yr, -sd, width=0.55, color=PALETTE["amber"], alpha=0.88, edgecolor="#78350F", lw=0.4, label="Preceding drought severity (−S_d)")
    ax1.bar(x_yr, fw, width=0.55, color=PALETTE["blue"], alpha=0.88, edgecolor="#1E3A5F", lw=0.4, label="Subsequent wet severity (+F_w)")

    ax1.axvline(2003, ls="--", color=PALETTE["red"], lw=0.9, alpha=0.75)
    ax1.axvline(2013, ls=":", color=PALETTE["red"], lw=0.9, alpha=0.75)
    ax1.text(2002.4, max(fw) * 1.24, "2003 TGP", fontsize=7.5, color=PALETTE["red"], ha="right")
    ax1.text(2012.4, max(fw) * 1.24, "2013 Cascade", fontsize=7.5, color=PALETTE["red"], ha="right")

    # Dynamically identify extreme markers from the actual catalog
    i_sd_max = events_df["S_d"].idxmax()
    i_fw_max = events_df["F_w"].idxmax()
    i_iw_max = events_df["I_w"].idxmax()
    marks = [
        (i_sd_max, f"{events_df.loc[i_sd_max, 'wet_start']} (max $S_d$={events_df.loc[i_sd_max, 'S_d']:.2f})", (-24, -14), "neg"),
        (i_fw_max, f"{events_df.loc[i_fw_max, 'wet_start']} (max $F_w$={events_df.loc[i_fw_max, 'F_w']:.2f})", (10, 6), "pos"),
        (i_iw_max, f"{events_df.loc[i_iw_max, 'wet_start']} (max $I_w$={events_df.loc[i_iw_max, 'I_w']:.2f})", (-18, 12), "pos"),
    ]
    used_idx = set()
    for idx_e, label, (dx, dy), side in marks:
        if idx_e in used_idx and side == "pos":
            continue
        used_idx.add(idx_e)
        xv = x_yr[idx_e]
        yv = fw[idx_e] if side == "pos" else -sd[idx_e]
        ax1.annotate(label, (xv, yv), xytext=(dx, dy), textcoords="offset points",
                     fontsize=7.5, fontweight="bold", color="#1E293B",
                     arrowprops=dict(arrowstyle="-", color="#475569", lw=0.65))

    ax1.set_xlim(1960, 2028)
    ax1.set_ylim(-max(sd) * 1.24, max(fw) * 1.34)
    ax1.set_xlabel("Transition Onset Year")
    ax1.set_ylabel("Cumulative SPI-1 Magnitude")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1")
    ax1.set_title("(a) Event chronology (1961–2025)", loc="left")

    # Panel b: Monthly bars color-coded by hydrological season
    m_counts = [int((events_df["trans_month"] == m).sum()) for m in range(1, 13)]
    m_cols = [
        PALETTE["blue"] if 5 <= m <= 9 else (PALETTE["slate"] if m in (12, 1, 2) else PALETTE["sky"])
        for m in range(1, 13)
    ]
    bars = ax2.bar(np.arange(1, 13), m_counts, color=m_cols, width=0.68, edgecolor="#1E293B", lw=0.5, alpha=0.9)
    for bar, v in zip(bars, m_counts):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, v + 0.14, str(v), ha="center", va="bottom", fontsize=8.0)

    leg_b = [
        mpatches.Patch(facecolor=PALETTE["blue"], edgecolor="#1E293B", lw=0.5, label="Flood May–Sep (n = 16, 41.0%)"),
        mpatches.Patch(facecolor=PALETTE["slate"], edgecolor="#1E293B", lw=0.5, label="Winter Dec–Feb (n = 13, 33.3%)"),
        mpatches.Patch(facecolor=PALETTE["sky"], edgecolor="#1E293B", lw=0.5, label="Shoulder Mar–Apr/Oct–Nov (n = 10)"),
    ]
    ax2.legend(handles=leg_b, loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.6)
    ax2.set_xticks(np.arange(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.set_ylim(0, max(m_counts) + 2.5)
    ax2.set_xlabel("Transition Onset Month")
    ax2.set_ylabel("Event Count (1961–2025)")
    ax2.set_title("(b) Seasonal onset distribution", loc="left")

    _save_fig(fig, out_dir, "Fig3", formats=formats)


def plot_fig4_dependence_and_spatial(events_df, dep_stats, grid_df, grid_summary, mask, out_dir, formats=("png",)):
    """Scatter and power bounds for basin S_d vs F_w plus 748-grid BH-FDR spatial map."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.22})

    sd = events_df["S_d"].to_numpy()
    fw = events_df["F_w"].to_numpy()
    fl = events_df["trans_month"].between(5, 9).to_numpy()

    ax1.scatter(sd[fl], fw[fl], c=PALETTE["blue"], s=58, edgecolors="#1E293B", lw=0.7, label="Flood May–Sep (n = 16)", zorder=4)
    ax1.scatter(sd[~fl], fw[~fl], c=PALETTE["slate"], s=54, marker="s", alpha=0.82, edgecolors="#1E293B", lw=0.7,
                label="Cool / Shoulder (n = 23)", zorder=4)

    # Lowess-like linear fit line + 95% confidence band just to show flatness
    xs = np.linspace(sd.min(), sd.max(), 100)
    slope, intercept, *_ = stats.linregress(sd, fw)
    ax1.plot(xs, intercept + slope * xs, color=PALETTE["red"], lw=1.3, ls="--", label="Linear fit (flat within MDE)", zorder=3)

    ax1.text(0.96, 0.96,
             f"$\\tau$ = {dep_stats['kendall_tau']:+.3f} (p = {dep_stats['kendall_p']:.3f})\n"
             f"$r_s$ = {dep_stats['spearman_r']:+.3f} (p = {dep_stats['spearman_p']:.3f})\n"
             f"80%-power MDE |r| = {dep_stats['mde_r_80pct']:.3f}\n"
             f"|$\\Delta$AICc| $\\leq$ 0.44 vs. Independence",
             transform=ax1.transAxes, ha="right", va="top", fontsize=7.9,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax1.set_ylim(0.3, 5.2)
    ax1.set_xlabel("Preceding Drought Severity $S_d$ (SPI-1 sum)")
    ax1.set_ylabel("Subsequent Wet Severity $F_w$ (SPI-1 sum)")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.6)
    ax1.set_title("(a) Basin-scale $S_d$ vs. $F_w$ (n = 39)", loc="left")

    # Panel b: 748-grid Kendall tau map with strict zero-truncation Gamma
    lats, lons, frac, zone = mask["lat"], mask["lon"], mask["frac"], mask["zone"]
    tau_grid = np.full(frac.shape, np.nan, dtype=float)
    for _, r in grid_df.iterrows():
        tau_grid[int(r["iy"]), int(r["ix"])] = r["tau"]

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    mesh = ax2.pcolormesh(lons, lats, tau_grid, cmap="RdBu_r", norm=norm, shading="nearest", zorder=1)
    _draw_basin_perimeter(ax2, lats, lons, frac, color="#0F172A", lw=1.25, zorder=3)
    _draw_subbasin_dividers(ax2, lats, lons, zone, frac, lw=0.75, color="#334155", zorder=3)

    cb = plt.colorbar(mesh, ax=ax2, fraction=0.042, pad=0.03)
    cb.set_label("Grid-Cell Kendall $\\tau$ ($S_d$ vs. $F_w$)")

    ax2.text(0.03, 0.05,
             f"748 cells median $\\tau$ = {grid_summary['tau_median']:+.3f}\n"
             f"Nominal p < 0.05: {int(round(grid_summary['raw_sig_n']))}/748 ({100*grid_summary['raw_sig_frac']:.1f}%)\n"
             f"Field-null P95: {100*grid_summary['field_null_p95_frac']:.1f}% | BH-FDR q < 0.05: {int(round(grid_summary['fdr_sig_n']))}/748",
             transform=ax2.transAxes, va="bottom", fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFFFF", alpha=0.92, edgecolor="#CBD5E1"), zorder=6)
    ax2.set_xlim(89.5, 123.0)
    ax2.set_ylim(23.8, 36.2)
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.set_title("(b) Grid-cell $\\tau$ with BH-FDR control", loc="left")

    _save_fig(fig, out_dir, "Fig4", formats=formats)


def plot_fig5_sensitivity_and_runoff(schemes_df, idx_rob_df, events_df, datong_strata_df, out_dir, formats=("png",)):
    """Sensitivity matrix across 36 rules & alternative indices plus stratified Datong runoff response."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.25})

    # Left: 36-scheme tau distribution + SPI-3 / SPEI-1 / SPEI-3 / bridge markers
    taus = schemes_df["tau"].to_numpy()
    pvals = schemes_df["p_value"].to_numpy()
    y_pos = np.arange(len(taus))
    colors = [PALETTE["red"] if p < 0.05 else PALETTE["sky"] for p in pvals]
    order = np.argsort(taus)
    ax1.axvline(0.0, color="#0F172A", lw=0.9)
    ax1.barh(y_pos, taus[order], color=np.array(colors)[order], height=0.72, edgecolor="#334155", lw=0.35, alpha=0.9)
    ax1.set_yticks([])
    ax1.set_xlabel("Kendall τ (S_d vs. F_w)")
    ax1.set_ylabel("36 Threshold / Gap Schemes Ranked by τ")
    ax1.set_xlim(-0.32, 0.32)

    rob_lines = [
        f"{r['index']}: n={int(r['n_events'])}, τ={r['tau']:+.3f} (p={r['p_value']:.2f})"
        for _, r in idx_rob_df.iterrows()
    ]
    n_sig = int((pvals < 0.05).sum())
    ax1.text(0.97, 0.05,
             f"36 schemes: {n_sig}/36 nominal p < 0.05 ({100*n_sig/len(taus):.1f}%)\n"
             f"$\\tau$ range: [{taus.min():+.3f}, {taus.max():+.3f}]\n" + "\n".join(rob_lines),
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.3,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax1.set_title("(a) Scheme and index sensitivity", loc="left")

    # Right: Basin Fw vs Datong wet-spell runoff anomaly stratified by dam era & Fw=1.0
    sub = events_df[events_df["Q_wet_z"].notna()].copy()
    era_pre = sub["year"] <= 2002
    era_imp = (sub["year"] >= 2003) & (sub["year"] <= 2012)
    era_meg = sub["year"] >= 2013

    ax2.axhline(0.0, color="#64748B", ls="--", lw=0.8)
    ax2.axvline(1.0, color="#64748B", ls=":", lw=1.0)
    ax2.scatter(sub.loc[era_pre, "F_w"], sub.loc[era_pre, "Q_wet_z"], c=PALETTE["blue"], s=52,
                edgecolors="#0F172A", lw=0.6, label="1961–2002 Pre-TGP (n = 27, +0.13$\\sigma$)", zorder=4)
    ax2.scatter(sub.loc[era_imp, "F_w"], sub.loc[era_imp, "Q_wet_z"], c=PALETTE["amber"], s=65, marker="v",
                edgecolors="#0F172A", lw=0.6, label="2003–2012 Impoundment (n = 5, −0.79$\\sigma$)", zorder=5)
    ax2.scatter(sub.loc[era_meg, "F_w"], sub.loc[era_meg, "Q_wet_z"], c=PALETTE["teal"], s=65, marker="D",
                edgecolors="#0F172A", lw=0.6, label="2013–2024 Mega-cascade (n = 6, +0.32$\\sigma$)", zorder=5)

    # Trend line on log(Fw) or Fw
    rs_q, p_q = stats.spearmanr(sub["F_w"], sub["Q_wet_z"])
    p_str = "p < 0.001" if p_q < 0.001 else f"p = {p_q:.3f}"
    xs = np.linspace(sub["F_w"].min(), sub["F_w"].max(), 100)
    b1, b0 = np.polyfit(np.log1p(sub["F_w"]), sub["Q_wet_z"], 1)
    ax2.plot(xs, b0 + b1 * np.log1p(xs), color=PALETTE["red"], lw=1.3, label=f"Log fit ($r_s$ = {rs_q:+.2f}, {p_str})", zorder=3)

    # Call out informative physical outliers cleanly
    for wet_s, lbl, (dx, dy) in [("2011-06", "2011-06", (8, -10)), ("1992-12", "1992-12", (-42, -12))]:
        m_row = sub[sub["wet_start"] == wet_s]
        if not m_row.empty:
            r0 = m_row.iloc[0]
            ax2.annotate(lbl, (r0["F_w"], r0["Q_wet_z"]), xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.6, color="#334155", arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6))

    ax2.set_xlabel("Basin Wet Severity $F_w$ (SPI-1 sum)")
    ax2.set_ylabel("Datong Wet-Spell Discharge Anomaly ($\\sigma$)")
    ax2.set_ylim(-1.85, 2.55)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.5)
    ax2.set_title("(b) Datong runoff across $F_w$ and dam eras", loc="left")

    _save_fig(fig, out_dir, "Fig5", formats=formats)


def plot_fig6_circulation_composites(fields, windows_df, segments, mask, circ_summary, controls_df, out_dir, formats=("png",)):
    """500 hPa composites (Flood n=16 vs All n=39), 5880 gpm ridge, moisture flux, and windows."""
    lats_nc, lons_nc = fields["lat"], fields["lon"]
    fig = plt.figure(figsize=(11.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18, height_ratios=[1.15, 0.9])

    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2)
    xlim, ylim = (65, 165), (15, 62)

    for col_idx, (title, z_fld, qu_fld, qv_fld, draw_5880) in enumerate([
        ("(a) Flood transitions (May–Sep, n = 16)", fields["Z500_flood"], fields["qu_anom_flood"], fields["qv_anom_flood"], True),
        ("(b) All transitions (Jan–Dec, n = 39)", fields["Z500_all"], fields["qu_anom_all"], fields["qv_anom_all"], False),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        cf = ax.contourf(lons_nc, lats_nc, z_fld, levels=np.linspace(-1.2, 1.2, 13), cmap="RdBu_r", norm=norm, extend="both", zorder=1)
        _draw_coastlines(ax, segments, xlim, ylim, color="#475569", lw=0.65, zorder=2)
        _draw_basin_perimeter(ax, mask["lat"], mask["lon"], mask["frac"], color="#111827", lw=1.1, zorder=4)

        # NE and PL dipole boxes
        ax.add_patch(mpatches.Rectangle((125, 35), 25, 15, fill=False, edgecolor="#7F1D1D", lw=1.3, ls="-", zorder=5))
        ax.add_patch(mpatches.Rectangle((90, 25), 20, 17, fill=False, edgecolor="#1E3A8A", lw=1.3, ls="-", zorder=5))
        ax.text(137.5, 42.5, "NE", fontsize=8.5, fontweight="bold", color="#7F1D1D", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.75, edgecolor="none"), zorder=6)
        ax.text(100.0, 36.8, "PL", fontsize=8.5, fontweight="bold", color="#1E3A8A", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.80, edgecolor="none"), zorder=6)

        if draw_5880:
            ax.contour(lons_nc, lats_nc, fields["h500_flood_clim"], levels=[5880], colors=["#0F766E"], linewidths=[1.3], linestyles=["--"], zorder=5)
            ax.contour(lons_nc, lats_nc, fields["h500_flood"], levels=[5880], colors=["#991B1B"], linewidths=[1.5], linestyles=["-"], zorder=5)
            ax.text(138.0, 17.0, "5880 gpm:\nClim (142.5°E)\nTrans (165.0°E)", fontsize=7.0, color="#7F1D1D",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", alpha=0.86, edgecolor="#CBD5E1"), ha="left", zorder=6)

        q_step = 2
        qv_plot = ax.quiver(lons_nc[::q_step], lats_nc[::q_step], qu_fld[::q_step, ::q_step], qv_fld[::q_step, ::q_step],
                            scale=360, width=0.0028, color="#1E293B", alpha=0.82, zorder=5)
        if col_idx == 1:
            ax.quiverkey(qv_plot, 0.86, 1.04, 25, "25 kg m$^{-1}$ s$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 7.8})

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Longitude (°E)")
        if col_idx == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.set_title(title, loc="left")

        if col_idx == 1:
            cax = fig.add_axes([0.92, 0.52, 0.015, 0.35])
            cb = fig.colorbar(cf, cax=cax)
            cb.set_label("Z500 Standardized Anomaly ($\\sigma$)")

    # Bottom row: lag trajectories t0-3..t0+1 comparing Flood (n=16) vs All (n=39)
    lags = np.array([-3, -2, -1, 0, 1])
    for col_idx, (idx_code, title, ylabel, box_key) in enumerate([
        ("DIV_ps", "(c) Moisture divergence ($\\mathrm{DIV}_{ps}^*$) window", "Standardized Divergence ($\\sigma$)", "DIV_ps"),
        ("DI500", "(d) Dipole index (DI500) window", "DI500 Anomaly ($\\sigma$)", "DI500"),
    ]):
        ax = fig.add_subplot(gs[1, col_idx])
        sub_fl = windows_df[(windows_df["index"] == idx_code) & (windows_df["stratum"] == "Flood")].sort_values("lag")
        sub_all = windows_df[(windows_df["index"] == idx_code) & (windows_df["stratum"] == "All")].sort_values("lag")

        ax.axhline(0, color="#64748B", lw=0.8)
        ax.axvline(0, color="#94A3B8", ls=":", lw=1.0)
        ax.fill_between(lags, sub_all["null_p05"], sub_all["null_p95"], color="#CBD5E1", alpha=0.55, label="Climatological 5–95% band")

        ax.errorbar(lags - 0.05, sub_fl["mean"], yerr=sub_fl["se"], color=PALETTE["red"], marker="o", lw=1.8, capsize=3,
                    label="Flood May–Sep (n = 16)", zorder=4)
        ax.errorbar(lags + 0.05, sub_all["mean"], yerr=sub_all["se"], color=PALETTE["navy"], marker="s", ls="--", lw=1.5, capsize=3,
                    label="All transitions (n = 39)", zorder=4)

        row_all = circ_summary[(circ_summary["index"] == idx_code) & (circ_summary["stratum"] == "All (1961-2025)")].iloc[0]
        row_fl = circ_summary[(circ_summary["index"] == idx_code) & (circ_summary["stratum"] == "Flood (May-Sep)")].iloc[0]
        ctrl_row = controls_df[(controls_df["index"] == idx_code) & (controls_df["stratum"] == "All matched")].iloc[0]
        ax.text(0.03, 0.95,
                f"Dry→wet shift (All): {row_all['mean_diff']:+.2f}σ (p={row_all['p_value']:.4f})\n"
                f"Dry→wet shift (Flood): {row_fl['mean_diff']:+.2f}σ (p={row_fl['p_value']:.4f})\n"
                f"Matched wet ctrl (n={int(ctrl_row['n'])}): {ctrl_row['mean_diff']:+.2f}σ (p={ctrl_row['p_value']:.3f})",
                transform=ax.transAxes, va="top", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFFFF", alpha=0.92, edgecolor="#CBD5E1"))
        ax.set_xticks(lags)
        ax.set_xticklabels(["$t_0-3$", "$t_0-2$", "$t_0-1$", "$t_0$\n(Onset)", "$t_0+1$"])
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("Lag Relative to Wet-Spell Onset Month $t_0$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.6)
        ax.set_title(title, loc="left")

    _save_fig(fig, out_dir, "Fig6", formats=formats)


def plot_esm_fig1_wave_flux(fields, segments, mask, out_dir, formats=("png",)):
    """200 hPa geopotential anomalies, May–Sep jet core contours, and Takaya–Nakamura wave-activity flux."""
    lats_nc, lons_nc = fields["lat"], fields["lon"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2)
    cf = ax.contourf(lons_nc, lats_nc, fields["Z200_flood"], levels=np.linspace(-1.2, 1.2, 13),
                     cmap="RdBu_r", norm=norm, extend="both", zorder=1)
    _draw_coastlines(ax, segments, (65, 165), (15, 62), color="#475569", lw=0.65, zorder=2)
    _draw_basin_perimeter(ax, mask["lat"], mask["lon"], mask["frac"], color="#111827", lw=1.2, zorder=4)

    # Jet core contours (25 and 30 m/s)
    cs = ax.contour(lons_nc, lats_nc, fields["u200_clim_flood"], levels=[25, 30], colors=["#7E22CE"],
                    linewidths=[1.2, 1.8], linestyles=["--", "-"], zorder=4)
    ax.clabel(cs, fmt="%d m s$^{-1}$", inline=True, inline_spacing=4, fontsize=7.6, colors="#581C87")

    # NE / PL boxes
    ax.add_patch(mpatches.Rectangle((125, 35), 25, 15, fill=False, edgecolor="#7F1D1D", lw=1.3, zorder=5))
    ax.add_patch(mpatches.Rectangle((90, 25), 20, 17, fill=False, edgecolor="#1E3A8A", lw=1.3, zorder=5))
    ax.text(137.5, 42.5, "NE", fontsize=8.5, fontweight="bold", color="#7F1D1D", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.80, edgecolor="none"), zorder=6)
    ax.text(100.0, 36.8, "PL", fontsize=8.5, fontweight="bold", color="#1E3A8A", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.82, edgecolor="none"), zorder=6)

    # Mask weak wave-activity vectors (< 0.08 m^2/s^2) and equatorial singularities (< 17.5 N)
    wx = fields["wx_200_flood"].copy()
    wy = fields["wy_200_flood"].copy()
    lat_mask = np.abs(lats_nc)[:, None] < 17.5
    mag = np.hypot(wx, wy)
    wx = np.where(lat_mask | (mag < 0.08), np.nan, wx)
    wy = np.where(lat_mask | (mag < 0.08), np.nan, wy)

    step = 2
    qv = ax.quiver(lons_nc[::step], lats_nc[::step], wx[::step, ::step], wy[::step, ::step],
                   scale=6.0, width=0.0032, color="#0F172A", zorder=6)
    ax.quiverkey(qv, 0.85, 1.03, 0.5, "0.5 m$^2$ s$^{-2}$ ($\\mathbf{W}$ flux)", labelpos="E", coordinates="axes", fontproperties={"size": 8.0})

    cb = plt.colorbar(cf, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("200 hPa Geopotential Height Anomaly ($\\sigma$)")
    ax.set_xlim(65, 165)
    ax.set_ylim(15, 62)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("200 hPa wave-activity flux and jet core (n = 16)", loc="left")

    _save_fig(fig, out_dir, "ESM_Fig1", formats=formats)
