"""Publication-grade visualization suite for Yangtze River Basin transition diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from scipy import optimize, stats
import statsmodels.api as sm

from src.basin import TRIBUTARY_OUTLETS
from src.dependence import MARGINAL_CANDIDATES, _fit_marginal_params


PALETTE = {
    "navy": "#1D3557",
    "blue": "#2A6F97",
    "sky": "#61A5C2",
    "red": "#B22222",
    "crimson": "#991B1B",
    "amber": "#D97706",
    "teal": "#0F766E",
    "slate": "#475569",
    "muted": "#94A3B8",
    "light": "#E2E8F0",
    "upper": "#DCE8F2",
    "middle": "#DCF0E6",
    "lower": "#F9E6D2",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#475569",
    "axes.linewidth": 0.75,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HALO = [pe.withStroke(linewidth=2.2, foreground="white", alpha=0.92)]


def _save_fig(fig, out_dir, stem, formats=("png",), dpi=600):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        if fmt.lower() == "png":
            with Image.open(p) as im:
                if im.mode != "RGB":
                    im.convert("RGB").save(p, format="PNG", dpi=(dpi, dpi))
    plt.close(fig)


def _draw_rings(ax, rings, facecolor=None, edgecolor="#0F172A", lw=1.0, ls="-", alpha=1.0, zorder=2):
    for coords in rings:
        if len(coords) < 3:
            continue
        if facecolor is not None:
            patch = mpatches.Polygon(
                coords, closed=True, facecolor=facecolor, edgecolor="none", alpha=alpha, zorder=zorder
            )
            ax.add_patch(patch)
        if edgecolor is not None and lw > 0:
            ax.plot(coords[:, 0], coords[:, 1], color=edgecolor, lw=lw, ls=ls, zorder=zorder + 0.5)


def _draw_coastlines(ax, segments, xlim, ylim, color="#94A3B8", lw=0.6, zorder=1):
    for seg in segments:
        if (
            np.nanmax(seg[:, 0]) < xlim[0]
            or np.nanmin(seg[:, 0]) > xlim[1]
            or np.nanmax(seg[:, 1]) < ylim[0]
            or np.nanmin(seg[:, 1]) > ylim[1]
        ):
            continue
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, zorder=zorder)


def plot_fig1_study_area(mask, clim_df, vectors, out_dir, formats=("png",)):
    """Smooth HydroBASINS subregions, tributary catchments, labeled isohyets, and monthly hydroclimate."""
    lats, lons, frac = mask["lat"], mask["lon"], mask["frac"]
    fig = plt.figure(figsize=(12.2, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.36, 1.0], wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    _draw_coastlines(ax, vectors["coast"], (89.5, 123.5), (23.5, 36.5), color="#94A3B8", lw=0.65, zorder=1)

    _draw_rings(ax, vectors["upper"], facecolor=PALETTE["upper"], edgecolor="#334155", lw=0.55, ls="--", alpha=0.92, zorder=2)
    _draw_rings(ax, vectors["middle"], facecolor=PALETTE["middle"], edgecolor="#334155", lw=0.55, ls="--", alpha=0.92, zorder=2)
    _draw_rings(ax, vectors["lower"], facecolor=PALETTE["lower"], edgecolor="#334155", lw=0.55, ls="--", alpha=0.92, zorder=2)

    for tname in TRIBUTARY_OUTLETS:
        key = f"trib_{tname}"
        if key in vectors:
            _draw_rings(ax, vectors[key], facecolor=None, edgecolor="#64748B", lw=0.65, ls=":", zorder=3)

    if "pr_annual_clim" in vectors:
        pr_clim = np.where(frac > 0, vectors["pr_annual_clim"], np.nan)
        cs = ax.contour(
            lons, lats, pr_clim, levels=[600, 1000, 1400], colors=["#1D4ED8"], linewidths=0.95, alpha=0.80, zorder=3.6
        )
        for lv, segs in zip(cs.levels, cs.allsegs):
            if not segs:
                continue
            seg_main = max(segs, key=len)
            mid_idx = int(len(seg_main) * (0.45 if lv == 600 else (0.55 if lv == 1000 else 0.72)))
            lx, ly = seg_main[min(mid_idx, len(seg_main) - 1)]
            ax.text(lx, ly, f"{int(lv)}", fontsize=7.6, fontweight="bold", color="#1E3A8A",
                    ha="center", va="center", path_effects=HALO, zorder=6.2)

    _draw_rings(ax, vectors["yrb"], facecolor=None, edgecolor="#0F172A", lw=1.4, ls="-", zorder=4.5)

    for _, (lo0, la0) in TRIBUTARY_OUTLETS.items():
        ax.scatter([lo0], [la0], c="#0F172A", s=20, marker="o", edgecolors="white", linewidths=0.6, zorder=5.5)

    ax.scatter([111.00, 117.62], [30.82, 30.77], c=[PALETTE["red"], PALETTE["navy"]], s=[72, 72],
               marker="^", edgecolors="white", linewidths=0.85, zorder=6)
    ax.annotate("Three Gorges\n(Yichang)", (111.00, 30.82), xytext=(-26, -28),
                textcoords="offset points", fontsize=7.6, fontweight="bold", color=PALETTE["red"],
                path_effects=HALO, arrowprops=dict(arrowstyle="-", color=PALETTE["red"], lw=0.75), zorder=7)
    ax.annotate("Datong\nStation", (117.62, 30.77), xytext=(10, -24),
                textcoords="offset points", fontsize=7.6, fontweight="bold", color=PALETTE["navy"],
                path_effects=HALO, arrowprops=dict(arrowstyle="-", color=PALETTE["navy"], lw=0.75), zorder=7)

    ax.text(101.4, 31.3, "Upper Basin\n(TGP upstream)", fontsize=8.3, fontweight="bold",
            color="#1E3A5F", ha="center", path_effects=HALO, zorder=6)
    ax.text(112.1, 27.0, "Middle\nBasin", fontsize=8.3, fontweight="bold",
            color="#0F5132", ha="center", path_effects=HALO, zorder=6)
    ax.text(119.5, 32.1, "Lower", fontsize=8.3, fontweight="bold",
            color="#7C2D12", ha="center", path_effects=HALO, zorder=6)

    for lbl, lo, la in [
        ("Yalong", 101.0, 28.5),
        ("Min-Tuo", 103.5, 30.5),
        ("Jialing", 106.0, 31.9),
        ("Wujiang", 107.6, 27.6),
        ("Hanjiang", 110.8, 32.9),
        ("Dongting", 111.3, 28.6),
        ("Poyang", 115.8, 27.6),
    ]:
        ax.text(lo, la, lbl, fontsize=7.2, color="#334155", style="italic", ha="center", path_effects=HALO, zorder=6)

    handles = [
        mpatches.Patch(facecolor=PALETTE["upper"], edgecolor="#334155", lw=0.6, label="Upper YRB (Upstream Yichang)"),
        mpatches.Patch(facecolor=PALETTE["middle"], edgecolor="#334155", lw=0.6, label="Middle YRB (Yichang–Datong)"),
        mpatches.Patch(facecolor=PALETTE["lower"], edgecolor="#334155", lw=0.6, label="Lower YRB (Datong–Estuary)"),
        mlines.Line2D([], [], color="white", marker="o", markerfacecolor="#0F172A", markeredgecolor="white",
                      markersize=5.5, label="Tributary confluence outlet (7)"),
        mlines.Line2D([], [], color="#1D4ED8", lw=0.9, label="Mean precip. isohyets (mm yr$^{-1}$)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1", fontsize=6.9)

    ax_in = ax.inset_axes([0.015, 0.03, 0.27, 0.34])
    _draw_coastlines(ax_in, vectors["coast"], (72, 136), (17, 54), color="#64748B", lw=0.45, zorder=1)
    _draw_rings(ax_in, vectors["yrb"], facecolor=PALETTE["blue"], edgecolor="#0F172A", lw=0.55, alpha=0.88, zorder=2)
    ax_in.text(0.5, 0.10, "Yangtze Basin", transform=ax_in.transAxes, ha="center", fontsize=6.5, fontweight="bold", color="#0F172A")
    ax_in.set_xlim(73, 135)
    ax_in.set_ylim(18, 53)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for sp in ax_in.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#94A3B8")
        sp.set_linewidth(0.75)

    ax.set_xlim(89.5, 123.0)
    ax.set_ylim(23.6, 36.4)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    # Panel (b): Non-parametric monthly precipitation median + IQR paired with median temperature
    ax2 = fig.add_subplot(gs[0, 1])
    x_mo = clim_df["month"].to_numpy()
    mo_med = clim_df["precip_median_mm"].to_numpy() if "precip_median_mm" in clim_df.columns else clim_df["precip_mean_mm"].to_numpy()
    mo_iqr25 = clim_df["precip_q25_mm"].to_numpy()
    mo_iqr75 = clim_df["precip_q75_mm"].to_numpy()
    t_med = clim_df["temp_median_c"].to_numpy() if "temp_median_c" in clim_df.columns else clim_df["temp_mean_c"].to_numpy()

    colors = [
        PALETTE["blue"] if 5 <= m <= 9 else (PALETTE["teal"] if m in (3, 4) else (PALETTE["sky"] if m in (10, 11) else PALETTE["slate"]))
        for m in x_mo
    ]
    ax2.bar(x_mo, mo_med, color=colors, width=0.66, edgecolor="#1E293B", linewidth=0.5, alpha=0.86, zorder=3)
    ax2.errorbar(x_mo, mo_med, yerr=[np.maximum(mo_med - mo_iqr25, 0.0), np.maximum(mo_iqr75 - mo_med, 0.0)],
                 fmt="none", ecolor="#0F172A", elinewidth=0.95, capsize=2.4, zorder=4)
    ax2.axvspan(4.5, 9.5, color=PALETTE["blue"], alpha=0.07, zorder=1)
    ax2.text(0.67, 0.96, "Flood season (May–Sep)\n41.0% of events (n = 16)", transform=ax2.transAxes,
             ha="center", va="top", fontsize=7.4, color=PALETTE["navy"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", alpha=0.88, edgecolor="#CBD5E1"))

    ax2_t = ax2.twinx()
    ax2_t.spines["top"].set_visible(False)
    ax2_t.spines["right"].set_visible(True)
    ax2_t.spines["right"].set_color(PALETTE["crimson"])
    t_line, = ax2_t.plot(x_mo, t_med, color=PALETTE["crimson"], marker="o", ms=4.2, mfc="white", mew=1.2,
                         lw=1.6, zorder=5, label="Temp. (median °C)")
    ax2_t.set_ylabel("Basin-Mean Temperature (°C)", color=PALETTE["crimson"])
    ax2_t.tick_params(axis="y", colors=PALETTE["crimson"])
    ax2_t.set_ylim(-2.0, 29.0)

    ax2.set_xticks(x_mo)
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.set_xlabel("Calendar Month (1961–2025)")
    ax2.set_ylabel("Basin-Mean Precipitation (mm month$^{-1}$)")
    ax2.set_ylim(0, max(mo_iqr75) * 1.44)
    ax2.grid(axis="y", ls=":", alpha=0.25, zorder=0)

    bar_handles = [
        mpatches.Patch(facecolor=PALETTE["blue"], edgecolor="#1E293B", lw=0.5, label="Flood (May–Sep)"),
        mpatches.Patch(facecolor=PALETTE["teal"], edgecolor="#1E293B", lw=0.5, label="Spring (Mar–Apr)"),
        mpatches.Patch(facecolor=PALETTE["sky"], edgecolor="#1E293B", lw=0.5, label="Autumn (Oct–Nov)"),
        mpatches.Patch(facecolor=PALETTE["slate"], edgecolor="#1E293B", lw=0.5, label="Winter (Dec–Feb)"),
        t_line,
    ]
    ax2.legend(handles=bar_handles, loc="upper left", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=6.5)
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig1", formats=formats)


def plot_fig5_tributary_amplification(trib_df, out_dir, formats=("png",), trib_sum_df=None):
    """Subbasin masking rays with duration control, ord-wet/null benchmarks, TGP zoning, and spread."""
    fig = plt.figure(figsize=(12.6, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.68, 0.72], wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax2)

    fw_b = trib_df["F_w_basin"].to_numpy()
    fw_t = trib_df["F_w_peak_trib_th05"].to_numpy() if "F_w_peak_trib_th05" in trib_df.columns else trib_df["F_w_peak_trib"].to_numpy()
    ratio = trib_df["spi_amp_ratio_th05"].to_numpy() if "spi_amp_ratio_th05" in trib_df.columns else trib_df["spi_amp_ratio"].to_numpy()
    is_mild = fw_b < 1.0
    is_flood = trib_df["trans_month"].between(5, 9).to_numpy()

    ax1.axvspan(0.3, 1.0, color=PALETTE["amber"], alpha=0.055, zorder=0)
    ax1.axvline(1.0, ls=":", color=PALETTE["slate"], lw=1.0, zorder=1)
    xs_ray = np.linspace(0.3, 5.2, 100)
    for mult, col, lbl_x, angle in [
        (1.0, "#94A3B8", 3.30, 36),
        (2.0, "#CBD5E1", 2.08, 55),
        (3.0, "#CBD5E1", 1.25, 65),
        (4.0, "#E2E8F0", 0.90, 71),
    ]:
        ax1.plot(xs_ray, mult * xs_ray, ls="--", color=col, lw=0.9, zorder=1)
        ax1.text(lbl_x, mult * lbl_x + 0.08, f"{int(mult)}$\\times$", fontsize=7.2,
                 color="#64748B", rotation=angle, ha="center", va="bottom", path_effects=HALO, zorder=2)

    # Encode season via marker shape (circle = Flood May-Sep; square = Non-Flood Oct-Apr) and severity via color
    for m_cond, f_cond, col, mk, lbl in [
        (is_mild, is_flood, PALETTE["amber"], "o", f"Mild Flood May–Sep (n={(is_mild & is_flood).sum()})"),
        (is_mild, ~is_flood, PALETTE["amber"], "s", f"Mild Oct–Apr (n={(is_mild & ~is_flood).sum()})"),
        (~is_mild, is_flood, PALETTE["teal"], "o", f"Mod/Sev Flood May–Sep (n={(~is_mild & is_flood).sum()})"),
        (~is_mild, ~is_flood, PALETTE["teal"], "s", f"Mod/Sev Oct–Apr (n={(~is_mild & ~is_flood).sum()})"),
    ]:
        sel = m_cond & f_cond
        ax1.scatter(fw_b[sel], fw_t[sel], c=col, s=54, marker=mk, edgecolors="#1E293B", lw=0.7, label=lbl, zorder=4)

    fw_col = "F_w_peak_trib_th05" if "F_w_peak_trib_th05" in trib_df.columns else "F_w_peak_trib"
    rat_col = "spi_amp_ratio_th05" if "spi_amp_ratio_th05" in trib_df.columns else "spi_amp_ratio"
    highlights = trib_df.sort_values(fw_col, ascending=False).head(2).index.tolist()
    mild_top = trib_df[is_mild].sort_values(rat_col, ascending=False).head(2).index.tolist()
    offsets = [(10, -14), (-14, 16), (10, 6), (20, -6)]
    for idx_pos, idx in enumerate(highlights + mild_top):
        row = trib_df.loc[idx]
        dx, dy = offsets[idx_pos % len(offsets)]
        ax1.annotate(
            f"{row['wet_start']} ({row['peak_tributary']})",
            (row["F_w_basin"], row[fw_col]),
            xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color="#1E293B",
            arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6), zorder=6,
        )

    _, p_mw = stats.mannwhitneyu(ratio[is_mild], ratio[~is_mild], alternative="two-sided")
    is_w1 = trib_df["W"].to_numpy() == 1 if "W" in trib_df.columns else np.ones(len(trib_df), dtype=bool)
    rs_w1, p_w1 = stats.spearmanr(fw_b[is_w1], fw_t[is_w1]) if is_w1.sum() >= 6 else (np.nan, np.nan)
    rs_lvl, p_lvl = stats.spearmanr(fw_b, fw_t)

    if trib_sum_df is not None and not trib_sum_df.empty:
        all_row = trib_sum_df[trib_sum_df["stratum"] == "All (1961-2025)"].iloc[0]
        m_row = trib_sum_df[trib_sum_df["stratum"] == "Mild (F_w < 1.0)"].iloc[0]
        s_row = trib_sum_df[trib_sum_df["stratum"] == "Mod/Sev (F_w >= 1.0)"].iloc[0]
        ord_r = trib_sum_df[trib_sum_df["stratum"] == "Benchmark: OrdWet All"]
        mvn_r = trib_sum_df[trib_sum_df["stratum"].str.startswith("Benchmark: 8x8 MVN Null")]
        b_dfaa = float(all_row["loglog_slope"])
        b_int = float(all_row["intensity_loglog_slope"])
        b_ord = float(ord_r["loglog_slope"].iloc[0]) if not ord_r.empty and pd.notna(ord_r["loglog_slope"].iloc[0]) else np.nan
        b_mvn = float(mvn_r["loglog_slope"].iloc[0]) if not mvn_r.empty and pd.notna(mvn_r["loglog_slope"].iloc[0]) else np.nan
        bench_lines = [
            f"Uncoupled SPI ratio: {float(m_row['spi_ratio_median']):.2f}$\\times$ vs. {float(s_row['spi_ratio_median']):.2f}$\\times$ (MW $p$={float(all_row['mw_spi_mild_vs_sev_p']):.2e})",
            f"Fixed $W=1$ ($n$={int(is_w1.sum())}): $r_s$={float(all_row['w1_level_rs']):+.2f} ($p$={float(all_row['w1_level_p']):.2f}) | All $r_s$={float(all_row['level_spearman_rs']):+.2f} ($p$={float(all_row['level_spearman_p']):.2e})",
            f"Slope $\\beta$: DFAA {b_dfaa:.2f} (int. {b_int:.2f}) | OrdWet {b_ord:.2f} | MVN {b_mvn:.2f}",
        ]
    else:
        bench_lines = [
            f"Uncoupled SPI ratio: {np.median(ratio[is_mild]):.2f}$\\times$ vs. {np.median(ratio[~is_mild]):.2f}$\\times$ (MW $p$={p_mw:.2e})",
            f"Fixed $W=1$ ($n$={is_w1.sum()}): $r_s$={rs_w1:+.2f} ($p$={p_w1:.2f}) | All $r_s$={rs_lvl:+.2f} ($p$={p_lvl:.2e})",
        ]

    ax1.set_xlim(0.3, fw_b.max() * 1.16)
    ax1.set_ylim(0.5, fw_t.max() * 1.22)
    ax1.set_xlabel("Basin-Mean Wet Severity $F_w$ (SPI-1 sum)")
    ax1.set_ylabel("Peak Tributary Wet Severity $F_w$ (SPI-1 sum)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1", fontsize=6.5)
    ax1.text(0.03, 0.96, "\n".join(bench_lines),
             transform=ax1.transAxes, va="top", fontsize=6.45,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax1.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    # Panel (b): Stacked event counts (Mild vs Mod/Sev) split upstream vs downstream of TGP
    order = ["Poyang", "Dongting", "Hanjiang", "Wujiang", "Jialing", "Min-Tuo", "Yalong"]
    mild_counts = [int(((trib_df["peak_tributary"] == t) & (trib_df["F_w_basin"] < 1.0)).sum()) for t in order]
    sev_counts = [int(((trib_df["peak_tributary"] == t) & (trib_df["F_w_basin"] >= 1.0)).sum()) for t in order]
    y_idx = np.arange(len(order))

    ax2.barh(y_idx, mild_counts, color=PALETTE["amber"], height=0.64, edgecolor="#1E293B", lw=0.55,
             label="Mild ($F_w < 1.0$)", alpha=0.9, zorder=3)
    ax2.barh(y_idx, sev_counts, left=mild_counts, color=PALETTE["teal"], height=0.64, edgecolor="#1E293B", lw=0.55,
             label="Mod/Sev ($F_w \\geq 1.0$)", alpha=0.88, zorder=3)
    for y0, mc, sc in zip(y_idx, mild_counts, sev_counts):
        tot = mc + sc
        ax2.text(tot + 0.25, y0, str(tot), va="center", fontsize=8.3, fontweight="bold", color="#0F172A")

    ax2.axhline(2.5, color="#64748B", ls="--", lw=0.95, zorder=2)
    xmax_b = max(m + s for m, s in zip(mild_counts, sev_counts)) + 3.2
    ax2.text(xmax_b - 0.2, 2.38, "TGP Downstream ↓", ha="right", va="top", fontsize=6.7, color="#334155", style="italic")
    ax2.text(xmax_b - 0.2, 2.62, "TGP Upstream ↑", ha="right", va="bottom", fontsize=6.7, color="#334155", style="italic")

    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(order, fontweight="medium")
    ax2.set_xlim(0, xmax_b)
    ax2.grid(axis="x", ls=":", alpha=0.25, zorder=0)
    ax2.set_xlabel("Peak Event Count (Total = 39)")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.0)
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    # Panel (c): Amplification ratio distribution per peak tributary
    rng_j = np.random.default_rng(42)
    ax3.axhline(2.5, color="#64748B", ls="--", lw=0.95, zorder=2)
    ax3.axvline(1.0, color="#64748B", ls="--", lw=1.0, zorder=1)
    ax3.axvline(np.median(ratio), color=PALETTE["navy"], ls=":", lw=1.0, zorder=1)
    for y0, tname in enumerate(order):
        sub_t = trib_df[trib_df["peak_tributary"] == tname]
        vals = sub_t[rat_col].to_numpy()
        if len(vals) >= 3:
            ax3.boxplot(
                [vals], positions=[y0], vert=False, widths=0.42, patch_artist=True, showfliers=False,
                manage_ticks=False,
                boxprops=dict(facecolor="#F1F5F9", edgecolor="#64748B", linewidth=0.85),
                medianprops=dict(color="#0F172A", linewidth=1.4),
                whiskerprops=dict(color="#64748B", linewidth=0.8),
                capprops=dict(color="#64748B", linewidth=0.8),
            )
        jit = rng_j.uniform(-0.14, 0.14, size=len(vals))
        pt_cols = [PALETTE["amber"] if f < 1.0 else PALETTE["teal"] for f in sub_t["F_w_basin"]]
        ax3.scatter(vals, y0 + jit, c=pt_cols, s=28, edgecolors="#0F172A", linewidths=0.45, alpha=0.92, zorder=4)

    ax3.text(np.median(ratio) + 0.08, 6.45, f"Med. {np.median(ratio):.2f}$\\times$", fontsize=7.2, color=PALETTE["navy"])
    ax3.tick_params(axis="y", labelleft=False)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(order, fontweight="medium")
    ax3.set_xlim(0.8, max(ratio) * 1.10)
    ax3.grid(axis="x", ls=":", alpha=0.25, zorder=0)
    ax3.set_xlabel("SPI Amplification ($F_{w,\\mathrm{trib}} / F_{w,\\mathrm{basin}}$)")
    ax3.set_title("c", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig5", formats=formats)


def plot_fig2_chronology_and_seasonality(events_df, out_dir, formats=("png",)):
    """Bidirectional event chronology with 11-yr rolling frequency ribbon and 12-month polar rose."""
    fig = plt.figure(figsize=(12.2, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.48, 1.0], wspace=0.20)

    gs_l = gs[0, 0].subgridspec(2, 1, height_ratios=[0.34, 1.0], hspace=0.09)
    ax_top = fig.add_subplot(gs_l[0, 0])
    ax1 = fig.add_subplot(gs_l[1, 0], sharex=ax_top)

    years_all = np.arange(1961, 2026)
    yr_counts = pd.Series(0, index=years_all, dtype=float)
    vc = events_df["year"].value_counts()
    for y, c in vc.items():
        if int(y) in yr_counts.index:
            yr_counts.loc[int(y)] = float(c)
    roll_rate = yr_counts.rolling(11, center=True, min_periods=5).mean()
    mean_rate = len(events_df) / 65.0

    ax_top.axvspan(2003, 2013, color=PALETTE["amber"], alpha=0.06, zorder=0)
    ax_top.axvspan(2013, 2026, color=PALETTE["teal"], alpha=0.05, zorder=0)
    ax_top.axhline(mean_rate, color="#64748B", ls="--", lw=0.85, zorder=1)
    ax_top.plot(roll_rate.index, roll_rate.values, color=PALETTE["teal"], lw=1.6, zorder=3, label="11-yr rolling rate")
    ax_top.fill_between(roll_rate.index, mean_rate, roll_rate.values,
                        where=roll_rate.values >= mean_rate, color=PALETTE["teal"], alpha=0.14, interpolate=True)
    ax_top.axvline(2003, ls="--", color=PALETTE["red"], lw=0.85, alpha=0.75)
    ax_top.axvline(2013, ls=":", color=PALETTE["red"], lw=0.85, alpha=0.75)
    ax_top.set_ylabel("Rate\n(yr$^{-1}$)", fontsize=8.2)
    ax_top.set_ylim(0.0, 1.35)
    ax_top.set_yticks([0.0, 0.6, 1.2])
    ax_top.text(2025.8, mean_rate + 0.06, f"Mean {mean_rate:.2f} yr$^{{-1}}$", fontsize=7.1,
                color="#475569", ha="right", va="bottom")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    x_yr = events_df["year"].to_numpy() + (events_df["trans_month"].to_numpy() - 0.5) / 12.0
    sd = events_df["S_d"].to_numpy()
    fw = events_df["F_w"].to_numpy()
    j_col = "J_w" if "J_w" in events_df.columns else "I_w"

    ax1.axvspan(2003, 2013, color=PALETTE["amber"], alpha=0.06, zorder=0)
    ax1.axvspan(2013, 2026, color=PALETTE["teal"], alpha=0.05, zorder=0)
    ax1.axhline(0, color="#0F172A", lw=0.85)
    ax1.bar(x_yr, -sd, width=0.55, color=PALETTE["amber"], alpha=0.88, edgecolor="#78350F", lw=0.4,
            label="Preceding drought ($-S_d$)")
    ax1.bar(x_yr, fw, width=0.55, color=PALETTE["blue"], alpha=0.88, edgecolor="#1E3A5F", lw=0.4,
            label="Subsequent wet ($+F_w$)")
    ax1.axvline(2003, ls="--", color=PALETTE["red"], lw=0.85, alpha=0.75)
    ax1.axvline(2013, ls=":", color=PALETTE["red"], lw=0.85, alpha=0.75)
    ax1.text(2002.4, max(fw) * 1.24, "2003 TGP", fontsize=7.3, color=PALETTE["red"], ha="right")
    ax1.text(2012.6, max(fw) * 1.24, "2013 Cascade", fontsize=7.3, color=PALETTE["red"], ha="right")

    i_sd_max = events_df["S_d"].idxmax()
    i_fw_max = events_df["F_w"].idxmax()
    i_jw_max = events_df[j_col].idxmax()
    marks = [
        (i_sd_max, f"{events_df.loc[i_sd_max, 'wet_start']} (max $S_d$={events_df.loc[i_sd_max, 'S_d']:.2f})", (-24, -14), "neg"),
        (i_fw_max, f"{events_df.loc[i_fw_max, 'wet_start']} (max $F_w$={events_df.loc[i_fw_max, 'F_w']:.2f})", (10, -4), "pos"),
        (i_jw_max, f"{events_df.loc[i_jw_max, 'wet_start']} (max $J_w$={events_df.loc[i_jw_max, j_col]:.2f})", (-18, 12), "pos"),
    ]
    used_idx = set()
    for idx_e, label, (dx, dy), side in marks:
        if idx_e in used_idx and side == "pos":
            continue
        used_idx.add(idx_e)
        xv = x_yr[idx_e]
        yv = fw[idx_e] if side == "pos" else -sd[idx_e]
        ax1.annotate(label, (xv, yv), xytext=(dx, dy), textcoords="offset points",
                     fontsize=7.3, fontweight="bold", color="#1E293B",
                     arrowprops=dict(arrowstyle="-", color="#475569", lw=0.6))

    ax1.set_xlim(1960, 2027)
    ax1.set_ylim(-max(sd) * 1.22, max(fw) * 1.32)
    ax1.set_xlabel("Transition Onset Year")
    ax1.set_ylabel("Cumulative SPI-1 Magnitude")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.5)

    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.spines["polar"].set_color("#CBD5E1")
    ax2.spines["polar"].set_linewidth(0.8)

    m_counts = np.array([int((events_df["trans_month"] == m).sum()) for m in range(1, 13)], dtype=float)
    thetas = np.deg2rad(np.arange(12) * 30.0 + 15.0)
    width = np.deg2rad(25.0)
    m_cols = [
        PALETTE["blue"] if 5 <= m <= 9 else (PALETTE["teal"] if m in (3, 4) else (PALETTE["sky"] if m in (10, 11) else PALETTE["slate"]))
        for m in range(1, 13)
    ]

    ax2.bar(thetas, m_counts, width=width, color=m_cols, edgecolor="#0F172A", linewidth=0.65, alpha=0.90, zorder=3)
    for th, cnt in zip(thetas, m_counts):
        if cnt > 0:
            ax2.text(th, cnt + 0.52, f"{int(cnt)}", ha="center", va="center", fontsize=8.0,
                     fontweight="bold", color="#0F172A", zorder=5)

    arc_cfg = [
        (4.0 * 30.0, 9.0 * 30.0, PALETTE["blue"]),
        (2.0 * 30.0, 4.0 * 30.0, PALETTE["teal"]),
        (9.0 * 30.0, 11.0 * 30.0, PALETTE["sky"]),
        (11.0 * 30.0, 14.0 * 30.0, PALETTE["slate"]),
    ]
    for d0, d1, col in arc_cfg:
        arc_th = np.deg2rad(np.linspace(d0 + 2, d1 - 2, 40))
        ax2.plot(arc_th, np.full_like(arc_th, 6.85), color=col, lw=3.0, alpha=0.85, solid_capstyle="round", zorder=2)

    ax2.set_ylim(0, 7.6)
    ax2.set_yticks([2, 4, 6])
    ax2.set_yticklabels(["2", "4", "6"], fontsize=7.0, color="#64748B")
    ax2.set_rlabel_position(62)
    ax2.set_xticks(thetas)
    ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                        fontsize=8.1, fontweight="medium", color="#1E293B")
    ax2.grid(color="#CBD5E1", ls=":", lw=0.65)
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold", pad=16)

    leg_b = [
        mpatches.Patch(facecolor=PALETTE["blue"], edgecolor="#1E293B", lw=0.5, label=f"Flood May–Sep (n = {int(events_df['trans_month'].between(5,9).sum())}, {100*events_df['trans_month'].between(5,9).mean():.1f}%)"),
        mpatches.Patch(facecolor=PALETTE["teal"], edgecolor="#1E293B", lw=0.5, label=f"Spring Mar–Apr (n = {int(events_df['trans_month'].isin([3,4]).sum())}, {100*events_df['trans_month'].isin([3,4]).mean():.1f}%)"),
        mpatches.Patch(facecolor=PALETTE["sky"], edgecolor="#1E293B", lw=0.5, label=f"Autumn Oct–Nov (n = {int(events_df['trans_month'].isin([10,11]).sum())}, {100*events_df['trans_month'].isin([10,11]).mean():.1f}%)"),
        mpatches.Patch(facecolor=PALETTE["slate"], edgecolor="#1E293B", lw=0.5, label=f"Winter Dec–Feb (n = {int(events_df['trans_month'].isin([12,1,2]).sum())}, {100*events_df['trans_month'].isin([12,1,2]).mean():.1f}%)"),
    ]
    ax2.legend(handles=leg_b, loc="upper center", bbox_to_anchor=(0.5, -0.09),
               frameon=True, framealpha=0.94, edgecolor="#CBD5E1", fontsize=6.9, ncol=2)

    _save_fig(fig, out_dir, "Fig2", formats=formats)


def plot_fig3_dependence_and_spatial(events_df, dep_stats, marg_df, grid_df, grid_summary, mask, vectors, out_dir, formats=("png",)):
    """Joint marginal-density/quantile plot for S_d vs F_w plus 748-grid spatial tau with p < 0.05 markers."""
    fig = plt.figure(figsize=(12.4, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.32], wspace=0.22)

    gs_j = gs[0, 0].subgridspec(2, 2, height_ratios=[0.24, 1.0], width_ratios=[1.0, 0.24], hspace=0.06, wspace=0.06)
    ax_top = fig.add_subplot(gs_j[0, 0])
    ax_main = fig.add_subplot(gs_j[1, 0], sharex=ax_top)
    ax_rgt = fig.add_subplot(gs_j[1, 1], sharey=ax_main)

    sd = events_df["S_d"].to_numpy()
    fw = events_df["F_w"].to_numpy()
    fl = events_df["trans_month"].between(5, 9).to_numpy()

    ax_main.scatter(sd[fl], fw[fl], c=PALETTE["blue"], s=52, edgecolors="#1E293B", lw=0.65, label=f"Flood May–Sep (n = {fl.sum()})", zorder=4)
    ax_main.scatter(sd[~fl], fw[~fl], c=PALETTE["slate"], s=48, marker="s", alpha=0.84, edgecolors="#1E293B", lw=0.65,
                    label=f"Non-Flood Oct–Apr (n = {(~fl).sum()})", zorder=4)

    xs = np.linspace(sd.min() * 0.95, sd.max() * 1.03, 120)
    X_fit = sm.add_constant(sd)
    X_pred = sm.add_constant(xs)
    ols = sm.OLS(fw, X_fit).fit()
    pred_ci = ols.get_prediction(X_pred).summary_frame(alpha=0.05)
    ax_main.fill_between(xs, pred_ci["mean_ci_lower"], pred_ci["mean_ci_upper"], color=PALETTE["red"], alpha=0.12, zorder=2)
    ax_main.plot(xs, pred_ci["mean"], color=PALETTE["red"], lw=1.35, ls="--", label="OLS fit ± 95% CI", zorder=3)

    q_lines = []
    for q_val, ls_q, col_q, lbl_q, p_key in [
        (0.50, "-.", PALETTE["teal"], "$Q_{50}$", "q50_p"),
        (0.90, ":", PALETTE["amber"], "$Q_{90}$", "q90_p"),
    ]:
        q_mod = sm.QuantReg(fw, X_fit).fit(q=q_val, max_iter=3000)
        yq = q_mod.params[0] + q_mod.params[1] * xs
        ax_main.plot(xs, yq, color=col_q, lw=1.25, ls=ls_q, zorder=3.5)
        ax_main.text(xs[-1], yq[-1], f" {lbl_q}", color=col_q, fontsize=7.4, fontweight="bold", va="center")
        pv_q = dep_stats.get(p_key, q_mod.pvalues[1])
        p_txt = f"{pv_q:.2f}" if np.isfinite(pv_q) else "n/a"
        q_lines.append(f"{lbl_q} slope {q_mod.params[1]:+.3f} (p={p_txt})")

    mde_tau = dep_stats.get("mde_tau_80pct", 0.305)
    ax_main.text(0.98, 0.97,
                 f"$\\tau$ = {dep_stats['kendall_tau']:+.3f} (p = {dep_stats['kendall_p']:.3f})\n"
                 f"$r_s$ = {dep_stats['spearman_r']:+.3f} (p = {dep_stats['spearman_p']:.3f})\n"
                 f"MDE |$\\tau$| = {mde_tau:.3f} (|r| = {dep_stats['mde_r_80pct']:.3f})\n"
                 + "\n".join(q_lines),
                 transform=ax_main.transAxes, ha="right", va="top", fontsize=6.8,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax_main.set_xlim(0.7, sd.max() * 1.14)
    ax_main.set_ylim(0.2, 5.75)
    ax_main.set_xlabel("Preceding Drought Severity $S_d$ (SPI-1 sum)")
    ax_main.set_ylabel("Subsequent Wet Severity $F_w$ (SPI-1 sum)")
    ax_main.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=6.8)

    # Top marginal: S_d histogram + best threshold-anchored marginal curve read dynamically from marg_df
    ax_top.hist(sd, bins=12, density=True, color="#CBD5E1", edgecolor="#475569", lw=0.5, alpha=0.75)
    sd_best = marg_df[(marg_df["variable"] == "S_d") & (marg_df["delta_AICc"] == 0.0)].iloc[0]
    dist_sd = MARGINAL_CANDIDATES[sd_best["distribution"]]
    par_sd, _ = _fit_marginal_params(sd, sd_best["distribution"], dist_sd, float(sd_best["loc_bound"]))
    ax_top.plot(xs, dist_sd.pdf(xs, *par_sd), color=PALETTE["amber"], lw=1.45)
    ax_top.text(0.98, 0.78, f"{sd_best['distribution']}\n(AICc={sd_best['AICc']:.1f})", transform=ax_top.transAxes, ha="right", va="top",
                fontsize=7.0, color="#92400E")
    ax_top.set_yticks([])
    ax_top.spines["left"].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    # Right marginal: F_w horizontal histogram + best threshold-anchored marginal curve read dynamically from marg_df
    ys = np.linspace(0.2, 5.5, 120)
    ax_rgt.hist(fw, bins=11, density=True, orientation="horizontal", color="#CBD5E1", edgecolor="#475569", lw=0.5, alpha=0.75)
    fw_best = marg_df[(marg_df["variable"] == "F_w") & (marg_df["delta_AICc"] == 0.0)].iloc[0]
    dist_fw = MARGINAL_CANDIDATES[fw_best["distribution"]]
    par_fw, _ = _fit_marginal_params(fw, fw_best["distribution"], dist_fw, float(fw_best["loc_bound"]))
    ax_rgt.plot(dist_fw.pdf(ys, *par_fw), ys, color=PALETTE["blue"], lw=1.45)
    ax_rgt.text(0.95, 0.94, f"{fw_best['distribution']}\n({fw_best['AICc']:.1f})", transform=ax_rgt.transAxes, ha="right", va="top", fontsize=7.0, color=PALETTE["navy"])
    ax_rgt.set_xticks([])
    ax_rgt.spines["bottom"].set_visible(False)
    plt.setp(ax_rgt.get_yticklabels(), visible=False)

    ax2 = fig.add_subplot(gs[0, 1])
    lats, lons, frac = mask["lat"], mask["lon"], mask["frac"]
    tau_grid = np.full(frac.shape, np.nan, dtype=float)
    for _, r in grid_df.iterrows():
        tau_grid[int(r["iy"]), int(r["ix"])] = r["tau"]

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    mesh = ax2.pcolormesh(lons, lats, tau_grid, cmap="RdBu_r", norm=norm, shading="nearest", zorder=1)
    _draw_rings(ax2, vectors["upper"], facecolor=None, edgecolor="#475569", lw=0.65, ls="--", zorder=3)
    _draw_rings(ax2, vectors["middle"], facecolor=None, edgecolor="#475569", lw=0.65, ls="--", zorder=3)
    _draw_rings(ax2, vectors["lower"], facecolor=None, edgecolor="#475569", lw=0.65, ls="--", zorder=3)
    _draw_rings(ax2, vectors["yrb"], facecolor=None, edgecolor="#0F172A", lw=1.35, ls="-", zorder=4)

    sig_pts = grid_df[grid_df["p_value"] < 0.05]
    ax2.scatter(sig_pts["lon"], sig_pts["lat"], c="none", edgecolors="#0F172A", s=13, linewidths=0.75,
                label=f"Nominal p < 0.05 ({len(sig_pts)}/748; {int(grid_summary['fdr_sig_n'])} FDR)", zorder=5)

    cb = plt.colorbar(mesh, ax=ax2, fraction=0.042, pad=0.03)
    cb.set_label("Grid-Cell Kendall $\\tau$ ($S_d$ vs. $F_w$)")

    ax2.legend(loc="upper right", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.3)
    ax2.text(0.03, 0.04,
             f"748 cells median $\\tau$ = {grid_summary['tau_median']:+.3f}\n"
             f"Nominal p < 0.05: {int(round(grid_summary['raw_sig_n']))}/748 ({100*grid_summary['raw_sig_frac']:.1f}%)\n"
             f"Field null (B={int(grid_summary.get('n_perm', 1000))}): P95 = {100*grid_summary['field_null_p95_frac']:.1f}%, $p$ = {grid_summary.get('field_p_value', np.nan):.3f}\n"
             f"BH-FDR q < 0.05: {int(round(grid_summary['fdr_sig_n']))}/748 (min $q$ = {grid_summary.get('min_q_fdr', np.nan):.2f})",
             transform=ax2.transAxes, va="bottom", fontsize=7.1,
             bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFFFFF", alpha=0.93, edgecolor="#CBD5E1"), zorder=6)
    ax2.set_xlim(89.5, 123.0)
    ax2.set_ylim(23.6, 36.4)
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig3", formats=formats)


def plot_fig4_sensitivity_and_runoff(schemes_df, idx_rob_df, events_df, datong_df, out_dir, formats=("png",)):
    """36-scheme matrix, index-bridge forest plot with Kendall-tau MDE, and seasonally marked Datong runoff."""
    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.12], wspace=0.26)

    gs_l = gs[0, 0].subgridspec(2, 1, height_ratios=[1.15, 0.95], hspace=0.52)
    ax_hm = fig.add_subplot(gs_l[0, 0])
    ax_fp = fig.add_subplot(gs_l[1, 0])

    core_df = schemes_df[schemes_df["rule_group"] == "core_36"] if "rule_group" in schemes_df.columns else schemes_df
    row_keys = [(1, 3), (1, 4), (2, 3), (2, 4)]
    col_keys = [
        (d, w)
        for d in sorted(core_df["dry_th"].unique(), reverse=True)
        for w in sorted(core_df["wet_th"].unique())
    ]
    mat_tau = np.zeros((4, 9), dtype=float)
    mat_p = np.ones((4, 9), dtype=float)
    base_pos = None
    for r_i, (md, mg) in enumerate(row_keys):
        for c_i, (dt, wt) in enumerate(col_keys):
            hit = core_df[
                (core_df["min_dry"] == md)
                & (core_df["max_gap"] == mg)
                & (np.isclose(core_df["dry_th"], dt))
                & (np.isclose(core_df["wet_th"], wt))
            ].iloc[0]
            mat_tau[r_i, c_i] = hit["tau"]
            mat_p[r_i, c_i] = hit["p_value"]
            if md == 2 and mg == 4 and np.isclose(dt, -0.5) and np.isclose(wt, 0.5):
                base_pos = (c_i, r_i)

    norm = TwoSlopeNorm(vmin=-0.26, vcenter=0.0, vmax=0.26)
    im = ax_hm.imshow(mat_tau, cmap="RdBu_r", norm=norm, aspect="auto")
    for r_i in range(4):
        for c_i in range(9):
            tv = mat_tau[r_i, c_i]
            pv = mat_p[r_i, c_i]
            txt = f"{tv:+.2f}" + ("*" if pv < 0.05 else "")
            tcol = "white" if abs(tv) > 0.16 else "#0F172A"
            ax_hm.text(c_i, r_i, txt, ha="center", va="center", fontsize=7.0,
                       fontweight="bold" if pv < 0.05 else "normal", color=tcol)

    if base_pos is not None:
        bc, br = base_pos
        ax_hm.add_patch(mpatches.Rectangle((bc - 0.5, br - 0.5), 1.0, 1.0, fill=False,
                                           edgecolor="#0F172A", lw=2.0, zorder=5))

    ax_hm.set_xticks(np.arange(9))
    ax_hm.set_xticklabels([f"{d:+.1f}\n{w:+.1f}" for d, w in col_keys], fontsize=7.1)
    ax_hm.set_yticks(np.arange(4))
    ax_hm.set_yticklabels([f"$D\\geq${md}, $g\\leq${mg}" for md, mg in row_keys], fontsize=7.6)
    ax_hm.set_xlabel("Drought / Wet Threshold Pair ($\\theta_d$ / $\\theta_w$)", fontsize=8.3)
    cb_hm = plt.colorbar(im, ax=ax_hm, fraction=0.032, pad=0.02)
    cb_hm.set_label("$\\tau$", fontsize=8.0)
    cb_hm.ax.tick_params(labelsize=7.2)
    ax_hm.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    # Forest plot with true Kendall-tau 80%-power MDE band per index sample size n
    y_fp = np.arange(len(idx_rob_df))[::-1]
    ax_fp.axvline(0.0, color="#0F172A", lw=0.95, zorder=1)
    for y_pos, (_, row) in zip(y_fp, idx_rob_df.iterrows()):
        n_ev = int(row["n_events"])
        mde_t = float(row["mde_tau_80pct"]) if "mde_tau_80pct" in row and pd.notna(row["mde_tau_80pct"]) else (2.8016 * np.sqrt(2.0 * (2 * n_ev + 5) / (9.0 * n_ev * max(n_ev - 1, 1))))
        ax_fp.barh(y_pos, 2.0 * mde_t, left=-mde_t, height=0.62, color="#E2E8F0", alpha=0.75, zorder=0)
        tv = float(row["tau"])
        lo = float(row["ci95_lo"])
        hi = float(row["ci95_hi"])
        is_base = row["index"] == "SPI-1"
        col_pt = PALETTE["red"] if is_base else PALETTE["navy"]
        ax_fp.errorbar(tv, y_pos, xerr=[[tv - lo], [hi - tv]], fmt="o" if not is_base else "D",
                       color=col_pt, ecolor=col_pt, elinewidth=1.4, capsize=3.0, ms=5.6, zorder=4)
        ax_fp.text(0.46, y_pos, f"$\\tau$={tv:+.3f} (p={row['p_value']:.2f}, MDE$\\pm${mde_t:.2f})",
                   va="center", ha="left", fontsize=6.9, color="#1E293B")

    ax_fp.set_yticks(y_fp)
    ax_fp.set_yticklabels([f"{r['index']} (n={int(r['n_events'])})" for _, r in idx_rob_df.iterrows()], fontsize=7.5)
    ax_fp.set_xlim(-0.42, 0.99)
    ax_fp.set_ylim(-0.65, len(idx_rob_df) + 0.15)
    ax_fp.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax_fp.set_xlabel("Kendall $\\tau$ ($S_d$ vs. $F_w$, 95% Bootstrap CI; gray = $\\tau$ 80%-MDE)")
    ax_fp.grid(axis="x", ls=":", alpha=0.25)

    # Panel (b): Datong discharge stratified by era AND Flood (filled) vs Cool (open) markers
    ax2 = fig.add_subplot(gs[0, 1])
    sub = events_df[events_df["Q_wet_z"].notna()].copy()
    is_fl = sub["trans_month"].between(5, 9)
    era_pre = sub["year"] <= 2002
    era_imp = (sub["year"] >= 2003) & (sub["year"] <= 2012)
    era_meg = sub["year"] >= 2013

    ax2.axvspan(0.3, 1.0, color=PALETTE["amber"], alpha=0.055, zorder=0)
    ax2.axvspan(1.0, 4.85, color=PALETTE["teal"], alpha=0.045, zorder=0)
    ax2.axhline(0.0, color="#64748B", ls="--", lw=0.8)
    ax2.axvline(1.0, color="#64748B", ls=":", lw=1.0)

    def _era_scatter(cond, color, marker, label):
        fl_c = cond & is_fl
        co_c = cond & (~is_fl)
        ax2.scatter(sub.loc[fl_c, "F_w"], sub.loc[fl_c, "Q_wet_z"], c=color, s=56, marker=marker,
                    edgecolors="#0F172A", lw=0.75, label=label + " [filled=Flood]", zorder=4)
        ax2.scatter(sub.loc[co_c, "F_w"], sub.loc[co_c, "Q_wet_z"], facecolors="white", edgecolors=color,
                    s=54, marker=marker, lw=1.4, zorder=4)

    pre_m = sub.loc[era_pre, "Q_wet_z"].mean()
    imp_m = sub.loc[era_imp, "Q_wet_z"].mean()
    meg_m = sub.loc[era_meg, "Q_wet_z"].mean()
    _era_scatter(era_pre, PALETTE["blue"], "o", f"1961–2002 Pre-TGP (n={era_pre.sum()}, {pre_m:+.2f}$\\sigma$)")
    _era_scatter(era_imp, PALETTE["amber"], "v", f"2003–2012 Impound. caveat (n={era_imp.sum()}, {imp_m:+.2f}$\\sigma$)")
    _era_scatter(era_meg, PALETTE["teal"], "D", f"2013–2024 Cascade caveat (n={era_meg.sum()}, {meg_m:+.2f}$\\sigma$)")

    rs_q, p_q = stats.spearmanr(sub["F_w"], sub["Q_wet_z"])
    p_str = "p < 0.001" if p_q < 0.001 else f"p = {p_q:.3f}"
    xs = np.linspace(sub["F_w"].min(), sub["F_w"].max(), 100)
    X_q = sm.add_constant(np.log1p(sub["F_w"].to_numpy()))
    q_ols = sm.OLS(sub["Q_wet_z"].to_numpy(), X_q).fit()
    ci_q = q_ols.get_prediction(sm.add_constant(np.log1p(xs))).summary_frame(alpha=0.05)
    ax2.fill_between(xs, ci_q["mean_ci_lower"], ci_q["mean_ci_upper"], color=PALETTE["red"], alpha=0.12, zorder=2)
    ax2.plot(xs, ci_q["mean"], color=PALETTE["red"], lw=1.35, label=f"OLS log fit ($r_s$={rs_q:+.2f}, {p_str})", zorder=3)

    for wet_s, lbl, (dx, dy) in [("2011-06", "2011-06", (8, -10)), ("1992-12", "1992-12", (-44, -12))]:
        m_row = sub[sub["wet_start"] == wet_s]
        if not m_row.empty:
            r0 = m_row.iloc[0]
            ax2.annotate(lbl, (r0["F_w"], r0["Q_wet_z"]), xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.5, color="#334155", arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6))

    mild_row = datong_df[(datong_df["era"] == "Full (1961-2024)") & (datong_df["wet_class"] == "Fw < 1.0")].iloc[0]
    sev_row = datong_df[(datong_df["era"] == "Full (1961-2024)") & (datong_df["wet_class"] == "Fw >= 1.0")].iloc[0]
    meg_sev = datong_df[(datong_df["era"].str.startswith("Mega-cascade")) & (datong_df["wet_class"] == "Fw >= 1.0")].iloc[0]

    ax2.text(0.35, -2.08, f"Mild $F_w<1$: {mild_row['Q_wet_mean_z']:+.2f}$\\sigma$ ($p$={mild_row['wilcoxon_p']:.3f})",
             ha="left", va="bottom", fontsize=6.9, color="#92400E")
    ax2.text(2.85, -2.08,
             f"Mod/Sev $F_w\\geq 1$: {sev_row['Q_wet_mean_z']:+.2f}$\\sigma$ ($p$={sev_row['wilcoxon_p']:.3f})\n"
             f"2013–24 ($n$=4): $Q_w$={meg_sev['Q_wet_mean_z']:+.2f}$\\sigma$, $Q_d$={meg_sev['Q_dry_mean_z']:+.2f}$\\sigma$",
             ha="center", va="bottom", fontsize=6.9, color="#115E59")

    ax2.set_xlim(0.32, 4.78)
    ax2.set_ylim(-2.18, 2.55)
    ax2.set_xlabel("Basin Wet Severity $F_w$ (SPI-1 sum)")
    ax2.set_ylabel("Datong Wet-Spell Discharge Anomaly ($\\sigma$)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=6.7)
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig4", formats=formats)


def plot_fig6_circulation_composites(fields, windows_df, vectors, mask, circ_summary, controls_df, out_dir, formats=("png",)):
    """Detrended 500 hPa composites, pointwise vs Wilks BH-FDR markers, WNPSH contours, and lag trajectories."""
    lats_nc, lons_nc = fields["lat"], fields["lon"]
    fig = plt.figure(figsize=(12.0, 8.3))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18, height_ratios=[1.15, 0.9])

    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2)
    xlim, ylim = (65, 170), (15, 62)

    edge_c = fields.get("wnpsh_edge_clim_lon", 142.5)
    edge_f = fields.get("wnpsh_edge_flood_lon", 165.0)
    edge_o = fields.get("wnpsh_edge_ord_wet_lon", np.nan)
    for arr_name in ("edge_c", "edge_f", "edge_o"):
        val_e = locals()[arr_name]
        if isinstance(val_e, np.ndarray):
            if arr_name == "edge_c":
                edge_c = float(val_e.item())
            elif arr_name == "edge_f":
                edge_f = float(val_e.item())
            else:
                edge_o = float(val_e.item())

    for col_idx, (panel_lbl, badge_txt, z_fld, p_fld, q_fld, qu_fld, qv_fld, draw_5880) in enumerate([
        ("a", "May–Sep flood (n = 16)", fields["Z500_flood"], fields.get("p_Z500_flood"), fields.get("q_Z500_flood"), fields["qu_anom_flood"], fields["qv_anom_flood"], True),
        ("b", "All events (n = 39)", fields["Z500_all"], fields.get("p_Z500_all"), fields.get("q_Z500_all"), fields["qu_anom_all"], fields["qv_anom_all"], False),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        cf = ax.contourf(lons_nc, lats_nc, z_fld, levels=np.linspace(-1.2, 1.2, 13), cmap="RdBu_r", norm=norm, extend="both", zorder=1)
        dom_ok = (lats_nc[:, None] >= ylim[0]) & (lats_nc[:, None] <= ylim[1]) & (lons_nc[None, :] >= xlim[0]) & (lons_nc[None, :] <= xlim[1])
        if p_fld is not None:
            sig_iy, sig_ix = np.where((p_fld < 0.05) & dom_ok)
            ax.scatter(lons_nc[sig_ix], lats_nc[sig_iy], s=4.5, c="#334155", marker=".", alpha=0.48, zorder=2.4)
        if q_fld is not None:
            fdr_iy, fdr_ix = np.where((q_fld < 0.05) & dom_ok)
            if len(fdr_iy) > 0:
                ax.scatter(lons_nc[fdr_ix], lats_nc[fdr_iy], s=18, c="#0F172A", marker="x", linewidths=0.95, zorder=2.8)

        _draw_coastlines(ax, vectors["coast"], xlim, ylim, color="#475569", lw=0.65, zorder=2)
        _draw_rings(ax, vectors["yrb"], facecolor=None, edgecolor="#111827", lw=1.15, zorder=4)
        n_raw_sig = int(np.sum((p_fld < 0.05) & dom_ok)) if p_fld is not None else 0
        n_fdr_sig = int(np.sum((q_fld < 0.05) & dom_ok)) if q_fld is not None else 0
        n_dom = int(np.sum(dom_ok))
        ax.text(0.025, 0.96, f"{badge_txt}\npt p<0.05 (·): {n_raw_sig}/{n_dom}; FDR q<0.05 (×): {n_fdr_sig}/{n_dom}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.7,
                fontweight="bold", color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", alpha=0.88, edgecolor="#CBD5E1"), zorder=6)

        ax.add_patch(mpatches.Rectangle((125, 35), 25, 15, fill=False, edgecolor="#7F1D1D", lw=1.3, ls="-", zorder=5))
        ax.add_patch(mpatches.Rectangle((90, 25), 20, 17, fill=False, edgecolor="#1E3A8A", lw=1.3, ls="-", zorder=5))
        ax.text(137.5, 42.5, "NE", fontsize=8.4, fontweight="bold", color="#7F1D1D", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.80, edgecolor="none"), zorder=6)
        ax.text(100.0, 38.8, "PL", fontsize=8.4, fontweight="bold", color="#1E3A8A", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.82, edgecolor="none"), zorder=6)

        if draw_5880:
            ax.contour(lons_nc, lats_nc, fields["h500_flood_clim"], levels=[5880], colors=["#0F766E"], linewidths=[1.35], linestyles=["--"], zorder=5)
            if "h500_ord_wet_flood" in fields and np.isfinite(fields["h500_ord_wet_flood"]).any():
                ax.contour(lons_nc, lats_nc, fields["h500_ord_wet_flood"], levels=[5880], colors=["#D97706"], linewidths=[1.45], linestyles=["-."], zorder=5)
            ax.contour(lons_nc, lats_nc, fields["h500_flood"], levels=[5880], colors=["#991B1B"], linewidths=[1.6], linestyles=["-"], zorder=5)
            c_lbl = f"{edge_c:.1f}°E" if np.isfinite(edge_c) else "east"
            o_lbl = f"{edge_o:.1f}°E" if np.isfinite(edge_o) else "n/a"
            f_lbl = f"{edge_f:.1f}°E" if np.isfinite(edge_f) else ">170°E"
            ax.text(66.8, 16.5,
                    f"5880 gpm (detrended):\n— DFAA Flood ({f_lbl})\n-·- OrdWet Ctrl ({o_lbl})\n-- Clim 91–20 ({c_lbl})",
                    fontsize=6.5, color="#1E293B",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", alpha=0.90, edgecolor="#CBD5E1"),
                    ha="left", va="bottom", zorder=6)

        q_step = 2
        qv_plot = ax.quiver(lons_nc[::q_step], lats_nc[::q_step], qu_fld[::q_step, ::q_step], qv_fld[::q_step, ::q_step],
                            scale=360, width=0.0028, color="#1E293B", alpha=0.82, zorder=5)
        if col_idx == 1:
            ax.quiverkey(qv_plot, 0.85, 1.05, 25, "25 kg m$^{-1}$ s$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 7.6})

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Longitude (°E)")
        if col_idx == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.set_title(panel_lbl, loc="left", fontsize=11.5, fontweight="bold")

        if col_idx == 1:
            cax = fig.add_axes([0.92, 0.52, 0.015, 0.35])
            cb = fig.colorbar(cf, cax=cax)
            cb.set_label("Z500 Standardized Anomaly ($\\sigma$)")

    lags = np.array([-3, -2, -1, 0, 1])
    for col_idx, (idx_code, panel_lbl, ylabel) in enumerate([
        ("DIV_ps", "c", "$\\mathrm{DIV}_{ps}$ Divergence ($\\sigma$)"),
        ("DI500", "d", "DI500 Anomaly ($\\sigma$)"),
    ]):
        ax = fig.add_subplot(gs[1, col_idx])
        sub_fl = windows_df[(windows_df["index"] == idx_code) & (windows_df["stratum"] == "Flood")].sort_values("lag")
        sub_all = windows_df[(windows_df["index"] == idx_code) & (windows_df["stratum"] == "All")].sort_values("lag")

        ax.axhline(0, color="#64748B", lw=0.8)
        ax.axvline(0, color="#94A3B8", ls=":", lw=1.0)
        # Plot separate null 5-95% bands for n=16 Flood and n=39 All strata
        ax.fill_between(lags, sub_fl["null_p05"], sub_fl["null_p95"], color="#FECACA", alpha=0.40, label="Flood 5–95% band (n=16)")
        ax.fill_between(lags, sub_all["null_p05"], sub_all["null_p95"], color="#CBD5E1", alpha=0.55, label="All 5–95% band (n=39)")

        ax.errorbar(lags - 0.05, sub_fl["mean"], yerr=sub_fl["se"], color=PALETTE["red"], marker="o", lw=1.8, capsize=3,
                    label="Flood May–Sep (n = 16)", zorder=4)
        ax.errorbar(lags + 0.05, sub_all["mean"], yerr=sub_all["se"], color=PALETTE["navy"], marker="s", ls="--", lw=1.5, capsize=3,
                    label="All transitions (n = 39)", zorder=4)

        row_all = circ_summary[(circ_summary["index"] == idx_code) & (circ_summary["stratum"] == "All (1961-2025)")].iloc[0]
        row_fl = circ_summary[(circ_summary["index"] == idx_code) & (circ_summary["stratum"] == "Flood (May-Sep)")].iloc[0]
        ctrl_sub = controls_df[(controls_df["index"] == idx_code) & (controls_df["stratum"] == "All matched")]
        if "scheme" in ctrl_sub.columns:
            ctrl_sub = ctrl_sub[ctrl_sub["scheme"] == "tol_0.1"]
        ctrl_row = ctrl_sub.iloc[0]
        ax.text(0.03, 0.95,
                f"$\\Delta$ (All): {row_all['mean_diff']:+.2f}$\\sigma$ (p={row_all['p_value']:.4f})\n"
                f"$\\Delta$ (Flood): {row_fl['mean_diff']:+.2f}$\\sigma$ (p={row_fl['p_value']:.4f})\n"
                f"Matched $t_{{w0}}$ (n={int(ctrl_row['n'])}): {ctrl_row['mean_diff']:+.2f}$\\sigma$ (p={ctrl_row['p_value']:.3f})",
                transform=ax.transAxes, va="top", fontsize=7.2,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFFFFF", alpha=0.93, edgecolor="#CBD5E1"))
        ax.set_xticks(lags)
        ax.set_xticklabels(["$t_{w0}-3$", "$t_{w0}-2$", "$t_{w0}-1$", "$t_{w0}$\n(Onset)", "$t_{w0}+1$"])
        ax.set_ylim(-1.20, 1.20)
        ax.set_xlabel("Lag Relative to Wet-Spell Onset Month $t_{w0}$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=6.6, ncol=2)
        ax.set_title(panel_lbl, loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig6", formats=formats)


def plot_esm_fig1_wave_flux(fields, vectors, mask, out_dir, formats=("png",)):
    """200 hPa geopotential anomalies, May–Sep jet core contours, and month-matched Takaya–Nakamura flux."""
    lats_nc, lons_nc = fields["lat"], fields["lon"]
    fig, ax = plt.subplots(figsize=(8.8, 4.9))

    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2)
    cf = ax.contourf(lons_nc, lats_nc, fields["Z200_flood"], levels=np.linspace(-1.2, 1.2, 13),
                     cmap="RdBu_r", norm=norm, extend="both", zorder=1)
    _draw_coastlines(ax, vectors["coast"], (65, 170), (15, 62), color="#475569", lw=0.65, zorder=2)
    _draw_rings(ax, vectors["yrb"], facecolor=None, edgecolor="#111827", lw=1.2, zorder=4)

    cs = ax.contour(lons_nc, lats_nc, fields["u200_clim_flood"], levels=[25, 30], colors=["#7E22CE"],
                    linewidths=[1.2, 1.8], linestyles=["--", "-"], zorder=4)
    ax.clabel(cs, fmt="%d m s$^{-1}$", inline=True, inline_spacing=4, fontsize=7.6, colors="#581C87")

    ax.add_patch(mpatches.Rectangle((125, 35), 25, 15, fill=False, edgecolor="#7F1D1D", lw=1.3, zorder=5))
    ax.add_patch(mpatches.Rectangle((90, 25), 20, 17, fill=False, edgecolor="#1E3A8A", lw=1.3, zorder=5))
    ax.text(137.5, 42.5, "NE", fontsize=8.5, fontweight="bold", color="#7F1D1D", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.80, edgecolor="none"), zorder=6)
    ax.text(100.0, 38.8, "PL", fontsize=8.5, fontweight="bold", color="#1E3A8A", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.82, edgecolor="none"), zorder=6)

    wx = fields["wx_200_flood"].copy()
    wy = fields["wy_200_flood"].copy()
    lat_mask = np.abs(lats_nc)[:, None] < 25.0
    u_mask = np.hypot(fields["u200_clim_flood"], 0.0) < 10.0
    mag = np.hypot(wx, wy)
    bad_w = lat_mask | u_mask | (mag < 0.08)
    wx = np.where(bad_w, np.nan, wx)
    wy = np.where(bad_w, np.nan, wy)

    step = 2
    qv = ax.quiver(lons_nc[::step], lats_nc[::step], wx[::step, ::step], wy[::step, ::step],
                   scale=22.0, width=0.0030, headwidth=3.6, color="#0F172A", zorder=6)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.71, 0.90), 0.275, 0.075, transform=ax.transAxes,
        boxstyle="round,pad=0.01", facecolor="#FFFFFF", alpha=0.92, edgecolor="#CBD5E1", lw=0.75, zorder=6.5,
    ))
    ax.quiverkey(qv, 0.77, 0.938, 1.0, "1.0 m$^2$ s$^{-2}$ ($\\mathbf{W}$ flux)",
                 labelpos="E", coordinates="axes", fontproperties={"size": 7.8}).set_zorder(7)

    cb = plt.colorbar(cf, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("200 hPa Geopotential Height Anomaly ($\\sigma$)")
    ax.set_xlim(65, 170)
    ax.set_ylim(15, 62)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    _save_fig(fig, out_dir, "ESM_Fig1", formats=formats)
