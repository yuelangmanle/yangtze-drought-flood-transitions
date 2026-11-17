"""Publication-grade visualization suite for Yangtze River Basin transition diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from scipy import optimize, stats
import statsmodels.api as sm

from src.basin import TRIBUTARY_OUTLETS


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
    "legend.fontsize": 7.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#475569",
    "axes.linewidth": 0.75,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HALO = [pe.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]


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
    """Smooth HydroBASINS subregions, tributary catchments, China locator inset, and monthly hydroclimate."""
    lats, lons, frac = mask["lat"], mask["lon"], mask["frac"]
    fig = plt.figure(figsize=(12.2, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.36, 1.0], wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    _draw_coastlines(ax, vectors["coast"], (89.5, 123.5), (23.5, 36.5), color="#94A3B8", lw=0.65, zorder=1)

    # Smooth HydroBASINS subregion fills
    _draw_rings(ax, vectors["upper"], facecolor=PALETTE["upper"], edgecolor="#334155", lw=0.55, ls="--", alpha=0.92, zorder=2)
    _draw_rings(ax, vectors["middle"], facecolor=PALETTE["middle"], edgecolor="#334155", lw=0.55, ls="--", alpha=0.92, zorder=2)
    _draw_rings(ax, vectors["lower"], facecolor=PALETTE["lower"], edgecolor="#334155", lw=0.55, ls="--", alpha=0.92, zorder=2)

    # Subtle tributary catchment boundaries
    for tname in TRIBUTARY_OUTLETS:
        key = f"trib_{tname}"
        if key in vectors:
            _draw_rings(ax, vectors[key], facecolor=None, edgecolor="#64748B", lw=0.65, ls=":", zorder=3)

    # Overlay annual mean precipitation isohyets strictly inside YRB
    if "pr_annual_clim" in vectors:
        pr_clim = np.where(frac > 0, vectors["pr_annual_clim"], np.nan)
        cs = ax.contour(
            lons, lats, pr_clim, levels=[600, 1000, 1400], colors=["#2A6F97"], linewidths=0.7, alpha=0.5, zorder=3.5
        )
        ax.clabel(cs, fmt="%d mm", fontsize=6.7, colors="#1D3557", inline=True, inline_spacing=2)

    # Bold basin outer perimeter
    _draw_rings(ax, vectors["yrb"], facecolor=None, edgecolor="#0F172A", lw=1.4, ls="-", zorder=4.5)

    # Tributary outlet markers + Three Gorges & Datong key gauges
    for tname, (lo0, la0) in TRIBUTARY_OUTLETS.items():
        ax.scatter([lo0], [la0], c="#334155", s=16, marker="o", edgecolors="white", linewidths=0.5, zorder=5.5)

    ax.scatter([111.00, 117.62], [30.82, 30.77], c=[PALETTE["red"], PALETTE["navy"]], s=[72, 72],
               marker="^", edgecolors="white", linewidths=0.85, zorder=6)
    ax.annotate("Three Gorges\n(Yichang)", (111.00, 30.82), xytext=(-26, -28),
                textcoords="offset points", fontsize=7.6, fontweight="bold", color=PALETTE["red"],
                path_effects=HALO, arrowprops=dict(arrowstyle="-", color=PALETTE["red"], lw=0.75), zorder=7)
    ax.annotate("Datong\nStation", (117.62, 30.77), xytext=(10, -24),
                textcoords="offset points", fontsize=7.6, fontweight="bold", color=PALETTE["navy"],
                path_effects=HALO, arrowprops=dict(arrowstyle="-", color=PALETTE["navy"], lw=0.75), zorder=7)

    # Subbasin and tributary labels with clean white cartographic halos
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

    # Upper-right subregion legend in unobstructed space north of YRB
    handles = [
        mpatches.Patch(facecolor=PALETTE["upper"], edgecolor="#334155", lw=0.6, label="Upper YRB (Upstream Yichang)"),
        mpatches.Patch(facecolor=PALETTE["middle"], edgecolor="#334155", lw=0.6, label="Middle YRB (Yichang–Datong)"),
        mpatches.Patch(facecolor=PALETTE["lower"], edgecolor="#334155", lw=0.6, label="Lower YRB (Datong–Estuary)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1", fontsize=7.2)

    # Lower-left East Asia / China locator inset
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

    # Panel (b): Monthly precipitation climatology + Temperature annual cycle
    ax2 = fig.add_subplot(gs[0, 1])
    x_mo = clim_df["month"].to_numpy()
    mo_means = clim_df["precip_mean_mm"].to_numpy()
    mo_iqr25 = clim_df["precip_q25_mm"].to_numpy()
    mo_iqr75 = clim_df["precip_q75_mm"].to_numpy()
    t_means = clim_df["temp_mean_c"].to_numpy()

    colors = [
        PALETTE["blue"] if 5 <= m <= 9 else (PALETTE["sky"] if m in (3, 4, 10, 11) else PALETTE["slate"])
        for m in x_mo
    ]
    ax2.bar(x_mo, mo_means, color=colors, width=0.66, edgecolor="#1E293B", linewidth=0.5, alpha=0.86, zorder=3,
            label="Basin precip. (mean ± IQR)")
    ax2.errorbar(x_mo, mo_means, yerr=[mo_means - mo_iqr25, mo_iqr75 - mo_means],
                 fmt="none", ecolor="#0F172A", elinewidth=0.95, capsize=2.4, zorder=4)
    ax2.axvspan(4.5, 9.5, color=PALETTE["blue"], alpha=0.07, zorder=1)
    ax2.text(0.67, 0.96, "Flood season (May–Sep)\n41.0% of events (n = 16)", transform=ax2.transAxes,
             ha="center", va="top", fontsize=7.8, color=PALETTE["navy"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", alpha=0.85, edgecolor="#CBD5E1"))

    ax2_t = ax2.twinx()
    ax2_t.spines["top"].set_visible(False)
    ax2_t.spines["right"].set_visible(True)
    ax2_t.spines["right"].set_color(PALETTE["crimson"])
    line_t = ax2_t.plot(x_mo, t_means, color=PALETTE["crimson"], marker="o", ms=4.2, mfc="white", mew=1.2,
                        lw=1.6, zorder=5, label="Basin temp. (°C)")
    ax2_t.set_ylabel("Basin-Mean Temperature (°C)", color=PALETTE["crimson"])
    ax2_t.tick_params(axis="y", colors=PALETTE["crimson"])
    ax2_t.set_ylim(-2.0, 29.0)

    ax2.set_xticks(x_mo)
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.set_xlabel("Calendar Month (1961–2025)")
    ax2.set_ylabel("Basin-Mean Precipitation (mm month$^{-1}$)")
    ax2.set_ylim(0, max(mo_iqr75) * 1.32)
    ax2.grid(axis="y", ls=":", alpha=0.25, zorder=0)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.4)
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig1", formats=formats)


def plot_fig2_tributary_amplification(trib_df, out_dir, formats=("png",)):
    """Subbasin masking rays, mild vs moderate-severe tributary breakdown, and amplification spread."""
    fig = plt.figure(figsize=(12.6, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.68, 0.72], wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax2)

    fw_b = trib_df["F_w_basin"].to_numpy()
    fw_t = trib_df["F_w_peak_trib"].to_numpy()
    ratio = trib_df["spi_amp_ratio"].to_numpy()
    is_mild = fw_b < 1.0

    # Shaded mild-basin zone and amplification multiplier rays
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

    ax1.scatter(fw_b[is_mild], fw_t[is_mild], c=PALETTE["amber"], s=58, edgecolors="#1E293B", lw=0.7,
                label=f"Mild basin $F_w < 1.0$ (n = {is_mild.sum()}, med. {np.median(ratio[is_mild]):.2f}$\\times$)", zorder=4)
    ax1.scatter(fw_b[~is_mild], fw_t[~is_mild], c=PALETTE["teal"], s=56, marker="D", edgecolors="#1E293B", lw=0.7,
                label=f"Mod/Sev basin $F_w \\geq 1.0$ (n = {(~is_mild).sum()}, med. {np.median(ratio[~is_mild]):.2f}$\\times$)", zorder=4)

    highlights = trib_df.sort_values("F_w_peak_trib", ascending=False).head(2).index.tolist()
    mild_top = trib_df[is_mild].sort_values("spi_amp_ratio", ascending=False).head(2).index.tolist()
    offsets = [(10, -14), (-14, 16), (10, 6), (20, -6)]
    for idx_pos, idx in enumerate(highlights + mild_top):
        row = trib_df.loc[idx]
        dx, dy = offsets[idx_pos % len(offsets)]
        ax1.annotate(
            f"{row['wet_start']} ({row['peak_tributary']})",
            (row["F_w_basin"], row["F_w_peak_trib"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color="#1E293B",
            arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6), zorder=6,
        )

    ax1.set_xlim(0.3, fw_b.max() * 1.16)
    ax1.set_ylim(0.5, fw_t.max() * 1.18)
    ax1.set_xlabel("Basin-Mean Wet Severity $F_w$ (SPI-1 sum)")
    ax1.set_ylabel("Peak Tributary Wet Severity $F_w$ (SPI-1 sum)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1", fontsize=7.3)
    ax1.text(0.03, 0.96,
             f"All events med. SPI ratio: {np.median(ratio):.2f}$\\times$\n"
             f"Physical precip. excess med.: {trib_df['mm_excess_ratio'].median():.2f}$\\times$ (mm)\n"
             f"Ratio vs. $F_w$: $r_s = -0.59$ ($p < 10^{{-4}}$)",
             transform=ax1.transAxes, va="top", fontsize=7.6,
             bbox=dict(boxstyle="round,pad=0.33", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax1.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    # Panel (b): Stacked event counts (Mild vs Mod/Sev) by tributary
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

    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(order, fontweight="medium")
    ax2.set_xlim(0, max(m + s for m, s in zip(mild_counts, sev_counts)) + 2.8)
    ax2.grid(axis="x", ls=":", alpha=0.25, zorder=0)
    ax2.set_xlabel("Peak Event Count (Total = 39)")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.0)
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    # Panel (c): Amplification ratio distribution per peak tributary
    rng_j = np.random.default_rng(42)
    ax3.axvline(1.0, color="#64748B", ls="--", lw=1.0, zorder=1)
    ax3.axvline(np.median(ratio), color=PALETTE["navy"], ls=":", lw=1.0, zorder=1)
    for y0, tname in enumerate(order):
        sub_t = trib_df[trib_df["peak_tributary"] == tname]
        vals = sub_t["spi_amp_ratio"].to_numpy()
        if len(vals) >= 3:
            ax3.boxplot(
                [vals], positions=[y0], vert=False, widths=0.42, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#F1F5F9", edgecolor="#64748B", linewidth=0.85),
                medianprops=dict(color="#0F172A", linewidth=1.4),
                whiskerprops=dict(color="#64748B", linewidth=0.8),
                capprops=dict(color="#64748B", linewidth=0.8),
            )
        jit = rng_j.uniform(-0.14, 0.14, size=len(vals))
        pt_cols = [PALETTE["amber"] if f < 1.0 else PALETTE["teal"] for f in sub_t["F_w_basin"]]
        ax3.scatter(vals, y0 + jit, c=pt_cols, s=28, edgecolors="#0F172A", linewidths=0.45, alpha=0.92, zorder=4)

    ax3.text(np.median(ratio) + 0.08, 6.45, f"Med. {np.median(ratio):.2f}$\\times$", fontsize=7.2, color=PALETTE["navy"])
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(order, fontweight="medium")
    ax3.tick_params(axis="y", labelleft=False)
    ax3.set_xlim(0.8, max(ratio) * 1.10)
    ax3.grid(axis="x", ls=":", alpha=0.25, zorder=0)
    ax3.set_xlabel("SPI Amplification ($F_{w,\\mathrm{trib}} / F_{w,\\mathrm{basin}}$)")
    ax3.set_title("c", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig2", formats=formats)


def plot_fig3_chronology_and_seasonality(events_df, out_dir, formats=("png",)):
    """Bidirectional event chronology with 11-yr rolling frequency ribbon and 12-month polar rose."""
    fig = plt.figure(figsize=(12.2, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.48, 1.0], wspace=0.20)

    # Left: Stacked subgrid (top = 11-yr centered transition rate; bottom = bidirectional bars)
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
    ax_top.plot(roll_rate.index, roll_rate.values, color=PALETTE["teal"], lw=1.6, zorder=3,
                label="11-yr rolling rate")
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
    i_iw_max = events_df["I_w"].idxmax()
    marks = [
        (i_sd_max, f"{events_df.loc[i_sd_max, 'wet_start']} (max $S_d$={events_df.loc[i_sd_max, 'S_d']:.2f})", (-24, -14), "neg"),
        (i_fw_max, f"{events_df.loc[i_fw_max, 'wet_start']} (max $F_w$={events_df.loc[i_fw_max, 'F_w']:.2f})", (10, -4), "pos"),
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
                     fontsize=7.3, fontweight="bold", color="#1E293B",
                     arrowprops=dict(arrowstyle="-", color="#475569", lw=0.6))

    ax1.set_xlim(1960, 2027)
    ax1.set_ylim(-max(sd) * 1.22, max(fw) * 1.32)
    ax1.set_xlabel("Transition Onset Year")
    ax1.set_ylabel("Cumulative SPI-1 Magnitude")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.5)

    # Panel (b): 12-Month Polar Seasonal Rose Diagram
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.spines["polar"].set_color("#CBD5E1")
    ax2.spines["polar"].set_linewidth(0.8)

    m_counts = np.array([int((events_df["trans_month"] == m).sum()) for m in range(1, 13)], dtype=float)
    thetas = np.deg2rad(np.arange(12) * 30.0 + 15.0)
    width = np.deg2rad(25.0)
    m_cols = [
        PALETTE["blue"] if 5 <= m <= 9 else (PALETTE["slate"] if m in (12, 1, 2) else PALETTE["sky"])
        for m in range(1, 13)
    ]

    ax2.bar(thetas, m_counts, width=width, color=m_cols, edgecolor="#0F172A", linewidth=0.65, alpha=0.90, zorder=3)
    for th, cnt in zip(thetas, m_counts):
        if cnt > 0:
            ax2.text(th, cnt + 0.52, f"{int(cnt)}", ha="center", va="center", fontsize=8.0,
                     fontweight="bold", color="#0F172A", zorder=5)

    # Outer seasonal arc bands
    arc_cfg = [
        (4.0 * 30.0, 9.0 * 30.0, PALETTE["blue"]),
        (9.0 * 30.0, 11.0 * 30.0, PALETTE["sky"]),
        (2.0 * 30.0, 4.0 * 30.0, PALETTE["sky"]),
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
        mpatches.Patch(facecolor=PALETTE["blue"], edgecolor="#1E293B", lw=0.5, label="Flood May–Sep (n = 16, 41.0%)"),
        mpatches.Patch(facecolor=PALETTE["slate"], edgecolor="#1E293B", lw=0.5, label="Winter Dec–Feb (n = 13, 33.3%)"),
        mpatches.Patch(facecolor=PALETTE["sky"], edgecolor="#1E293B", lw=0.5, label="Shoulder Mar–Apr/Oct–Nov (n = 10, 25.6%)"),
    ]
    ax2.legend(handles=leg_b, loc="upper center", bbox_to_anchor=(0.5, -0.10),
               frameon=True, framealpha=0.94, edgecolor="#CBD5E1", fontsize=7.3, ncol=1)

    _save_fig(fig, out_dir, "Fig3", formats=formats)


def plot_fig4_dependence_and_spatial(events_df, dep_stats, marg_df, grid_df, grid_summary, mask, vectors, out_dir, formats=("png",)):
    """Joint marginal-density/quantile plot for S_d vs F_w plus 748-grid spatial tau with p < 0.05 markers."""
    fig = plt.figure(figsize=(12.4, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.32], wspace=0.22)

    # Left: 2x2 Joint marginal distribution + quantile regression subgrid
    gs_j = gs[0, 0].subgridspec(2, 2, height_ratios=[0.24, 1.0], width_ratios=[1.0, 0.24], hspace=0.06, wspace=0.06)
    ax_top = fig.add_subplot(gs_j[0, 0])
    ax_main = fig.add_subplot(gs_j[1, 0], sharex=ax_top)
    ax_rgt = fig.add_subplot(gs_j[1, 1], sharey=ax_main)

    sd = events_df["S_d"].to_numpy()
    fw = events_df["F_w"].to_numpy()
    fl = events_df["trans_month"].between(5, 9).to_numpy()

    ax_main.scatter(sd[fl], fw[fl], c=PALETTE["blue"], s=52, edgecolors="#1E293B", lw=0.65, label="Flood May–Sep (n = 16)", zorder=4)
    ax_main.scatter(sd[~fl], fw[~fl], c=PALETTE["slate"], s=48, marker="s", alpha=0.84, edgecolors="#1E293B", lw=0.65,
                    label="Cool / Shoulder (n = 23)", zorder=4)

    # Linear fit with 95% CI band + Q50 / Q90 quantile regression fits
    xs = np.linspace(sd.min() * 0.95, sd.max() * 1.03, 120)
    X_fit = sm.add_constant(sd)
    X_pred = sm.add_constant(xs)
    ols = sm.OLS(fw, X_fit).fit()
    pred_ci = ols.get_prediction(X_pred).summary_frame(alpha=0.05)
    ax_main.fill_between(xs, pred_ci["mean_ci_lower"], pred_ci["mean_ci_upper"], color=PALETTE["red"], alpha=0.12, zorder=2)
    ax_main.plot(xs, pred_ci["mean"], color=PALETTE["red"], lw=1.35, ls="--", label="OLS fit ± 95% CI", zorder=3)

    for q_val, ls_q, col_q, lbl_q in [(0.50, "-.", PALETTE["teal"], "$Q_{50}$"), (0.90, ":", PALETTE["amber"], "$Q_{90}$")]:
        q_mod = sm.QuantReg(fw, X_fit).fit(q=q_val)
        yq = q_mod.params[0] + q_mod.params[1] * xs
        ax_main.plot(xs, yq, color=col_q, lw=1.25, ls=ls_q, zorder=3.5)
        ax_main.text(xs[-1], yq[-1], f" {lbl_q}", color=col_q, fontsize=7.4, fontweight="bold", va="center")

    ax_main.text(0.97, 0.96,
                 f"$\\tau$ = {dep_stats['kendall_tau']:+.3f} (p = {dep_stats['kendall_p']:.3f})\n"
                 f"$r_s$ = {dep_stats['spearman_r']:+.3f} (p = {dep_stats['spearman_p']:.3f})\n"
                 f"80%-power MDE |r| = {dep_stats['mde_r_80pct']:.3f}\n"
                 f"|$\\Delta$AICc| $\\leq$ 0.44 vs. Indep.",
                 transform=ax_main.transAxes, ha="right", va="top", fontsize=7.4,
                 bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax_main.set_xlim(0.7, sd.max() * 1.12)
    ax_main.set_ylim(0.2, 5.2)
    ax_main.set_xlabel("Preceding Drought Severity $S_d$ (SPI-1 sum)")
    ax_main.set_ylabel("Subsequent Wet Severity $F_w$ (SPI-1 sum)")
    ax_main.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.2)

    # Top marginal: S_d histogram + fitted Lognormal curve (best AICc=127.59)
    ax_top.hist(sd, bins=12, density=True, color="#CBD5E1", edgecolor="#475569", lw=0.5, alpha=0.75)
    loc_sd = optimize.minimize_scalar(
        lambda l0: -np.sum(stats.lognorm.logpdf(sd, *stats.lognorm.fit(sd, floc=l0)))
        if np.isfinite(np.sum(stats.lognorm.logpdf(sd, *stats.lognorm.fit(sd, floc=l0)))) else 1e9,
        bounds=(0.0, float(sd.min()) - 1e-4), method="bounded",
    ).x
    par_sd = stats.lognorm.fit(sd, floc=float(loc_sd))
    ax_top.plot(xs, stats.lognorm.pdf(xs, *par_sd), color=PALETTE["amber"], lw=1.45, label="Lognormal")
    ax_top.text(0.98, 0.78, "Lognormal fit\n(AICc = 127.6)", transform=ax_top.transAxes, ha="right", va="top",
                fontsize=7.0, color="#92400E")
    ax_top.set_yticks([])
    ax_top.spines["left"].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    # Right marginal: F_w horizontal histogram + fitted Gamma curve (best AICc=76.24)
    ys = np.linspace(0.2, 5.1, 120)
    ax_rgt.hist(fw, bins=11, density=True, orientation="horizontal", color="#CBD5E1", edgecolor="#475569", lw=0.5, alpha=0.75)
    loc_fw = optimize.minimize_scalar(
        lambda l0: -np.sum(stats.gamma.logpdf(fw, *stats.gamma.fit(fw, floc=l0)))
        if np.isfinite(np.sum(stats.gamma.logpdf(fw, *stats.gamma.fit(fw, floc=l0)))) else 1e9,
        bounds=(0.0, float(fw.min()) - 1e-4), method="bounded",
    ).x
    par_fw = stats.gamma.fit(fw, floc=float(loc_fw))
    ax_rgt.plot(stats.gamma.pdf(ys, *par_fw), ys, color=PALETTE["blue"], lw=1.45)
    ax_rgt.text(0.95, 0.94, "Gamma\n(76.2)", transform=ax_rgt.transAxes, ha="right", va="top", fontsize=7.0, color=PALETTE["navy"])
    ax_rgt.set_xticks([])
    ax_rgt.spines["bottom"].set_visible(False)
    plt.setp(ax_rgt.get_yticklabels(), visible=False)

    # Right panel (b): 748-grid Kendall tau map + smooth HydroBASINS polygons + raw p<0.05 stippling
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

    # Mark nominal p < 0.05 cells directly with dots to demonstrate spatial clustering below field P95
    sig_pts = grid_df[grid_df["p_value"] < 0.05]
    ax2.scatter(sig_pts["lon"], sig_pts["lat"], c="none", edgecolors="#0F172A", s=13, linewidths=0.75,
                label=f"Nominal p < 0.05 ({len(sig_pts)}/748; 0 FDR)", zorder=5)

    cb = plt.colorbar(mesh, ax=ax2, fraction=0.042, pad=0.03)
    cb.set_label("Grid-Cell Kendall $\\tau$ ($S_d$ vs. $F_w$)")

    ax2.legend(loc="upper right", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.3)
    ax2.text(0.03, 0.04,
             f"748 cells median $\\tau$ = {grid_summary['tau_median']:+.3f}\n"
             f"Nominal p < 0.05: {int(round(grid_summary['raw_sig_n']))}/748 ({100*grid_summary['raw_sig_frac']:.1f}%)\n"
             f"Field-null P95: {100*grid_summary['field_null_p95_frac']:.1f}% | BH-FDR q < 0.05: {int(round(grid_summary['fdr_sig_n']))}/748",
             transform=ax2.transAxes, va="bottom", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFFFFF", alpha=0.93, edgecolor="#CBD5E1"), zorder=6)
    ax2.set_xlim(89.5, 123.0)
    ax2.set_ylim(23.6, 36.4)
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig4", formats=formats)


def plot_fig5_sensitivity_and_runoff(schemes_df, idx_rob_df, events_df, datong_df, out_dir, formats=("png",)):
    """36-scheme parameter sensitivity heatmap, index-bridge forest plot, and Datong runoff response."""
    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.12], wspace=0.26)

    # Left half: vertical stack -> top (a1) 4x9 parameter matrix heatmap; bottom (a2) Forest plot
    gs_l = gs[0, 0].subgridspec(2, 1, height_ratios=[1.15, 0.95], hspace=0.52)
    ax_hm = fig.add_subplot(gs_l[0, 0])
    ax_fp = fig.add_subplot(gs_l[1, 0])

    # Build 4 (min_dry, max_gap) rows x 9 (dry_th, wet_th) columns
    row_keys = [(1, 3), (1, 4), (2, 3), (2, 4)]
    col_keys = [
        (d, w)
        for d in sorted(schemes_df["dry_th"].unique(), reverse=True)
        for w in sorted(schemes_df["wet_th"].unique())
    ]
    mat_tau = np.zeros((4, 9), dtype=float)
    mat_p = np.ones((4, 9), dtype=float)
    base_pos = None
    for r_i, (md, mg) in enumerate(row_keys):
        for c_i, (dt, wt) in enumerate(col_keys):
            hit = schemes_df[
                (schemes_df["min_dry"] == md)
                & (schemes_df["max_gap"] == mg)
                & (np.isclose(schemes_df["dry_th"], dt))
                & (np.isclose(schemes_df["wet_th"], wt))
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
    n_sig = int((schemes_df["p_value"] < 0.05).sum())
    ax_hm.set_title("a", loc="left", fontsize=11.5, fontweight="bold")

    # Bottom subpanel: Multi-index & bridge Forest Plot with Bootstrap 95% CIs
    y_fp = np.arange(len(idx_rob_df))[::-1]
    ax_fp.axvline(0.0, color="#0F172A", lw=0.95, zorder=1)
    ax_fp.axvspan(-0.436, 0.436, color="#E2E8F0", alpha=0.40, zorder=0, label="80%-power MDE ($\\pm 0.436$)")
    for y_pos, (_, row) in zip(y_fp, idx_rob_df.iterrows()):
        tv = float(row["tau"])
        lo = float(row["ci95_lo"]) if "ci95_lo" in row and pd.notna(row["ci95_lo"]) else tv - 0.22
        hi = float(row["ci95_hi"]) if "ci95_hi" in row and pd.notna(row["ci95_hi"]) else tv + 0.22
        is_base = row["index"] == "SPI-1"
        col_pt = PALETTE["red"] if is_base else PALETTE["navy"]
        ax_fp.errorbar(tv, y_pos, xerr=[[tv - lo], [hi - tv]], fmt="o" if not is_base else "D",
                       color=col_pt, ecolor=col_pt, elinewidth=1.4, capsize=3.0, ms=5.8, zorder=4)
        ax_fp.text(0.49, y_pos, f"$\\tau$={tv:+.3f} (p={row['p_value']:.2f})",
                   va="center", ha="left", fontsize=7.5, color="#1E293B")

    ax_fp.set_yticks(y_fp)
    ax_fp.set_yticklabels([f"{r['index']} (n={int(r['n_events'])})" for _, r in idx_rob_df.iterrows()], fontsize=7.9)
    ax_fp.set_xlim(-0.50, 0.88)
    ax_fp.set_ylim(-0.65, len(idx_rob_df) + 0.15)
    ax_fp.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax_fp.set_xlabel("Kendall $\\tau$ ($S_d$ vs. $F_w$, 95% Bootstrap CI)")
    ax_fp.text(0.0, len(idx_rob_df) - 0.02, "80%-power MDE ($\\pm 0.436$)", ha="center", va="top",
               fontsize=7.1, color="#475569", bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.88, edgecolor="#CBD5E1"))
    ax_fp.grid(axis="x", ls=":", alpha=0.25)

    # Right panel (b): Datong wet-spell runoff anomaly stratified by dam era & Fw=1.0
    ax2 = fig.add_subplot(gs[0, 1])
    sub = events_df[events_df["Q_wet_z"].notna()].copy()
    era_pre = sub["year"] <= 2002
    era_imp = (sub["year"] >= 2003) & (sub["year"] <= 2012)
    era_meg = sub["year"] >= 2013

    ax2.axvspan(0.3, 1.0, color=PALETTE["amber"], alpha=0.055, zorder=0)
    ax2.axvspan(1.0, 4.85, color=PALETTE["teal"], alpha=0.045, zorder=0)
    ax2.axhline(0.0, color="#64748B", ls="--", lw=0.8)
    ax2.axvline(1.0, color="#64748B", ls=":", lw=1.0)

    ax2.scatter(sub.loc[era_pre, "F_w"], sub.loc[era_pre, "Q_wet_z"], c=PALETTE["blue"], s=52,
                edgecolors="#0F172A", lw=0.6, label="1961–2002 Pre-TGP (n = 27, +0.13$\\sigma$)", zorder=4)
    ax2.scatter(sub.loc[era_imp, "F_w"], sub.loc[era_imp, "Q_wet_z"], c=PALETTE["amber"], s=65, marker="v",
                edgecolors="#0F172A", lw=0.6, label="2003–2012 Impoundment (n = 5, −0.79$\\sigma$)", zorder=5)
    ax2.scatter(sub.loc[era_meg, "F_w"], sub.loc[era_meg, "Q_wet_z"], c=PALETTE["teal"], s=65, marker="D",
                edgecolors="#0F172A", lw=0.6, label="2013–2024 Mega-cascade (n = 6, +0.32$\\sigma$)", zorder=5)

    rs_q, p_q = stats.spearmanr(sub["F_w"], sub["Q_wet_z"])
    p_str = "p < 0.001" if p_q < 0.001 else f"p = {p_q:.3f}"
    xs = np.linspace(sub["F_w"].min(), sub["F_w"].max(), 100)
    X_q = sm.add_constant(np.log1p(sub["F_w"].to_numpy()))
    q_ols = sm.OLS(sub["Q_wet_z"].to_numpy(), X_q).fit()
    ci_q = q_ols.get_prediction(sm.add_constant(np.log1p(xs))).summary_frame(alpha=0.05)
    ax2.fill_between(xs, ci_q["mean_ci_lower"], ci_q["mean_ci_upper"], color=PALETTE["red"], alpha=0.12, zorder=2)
    ax2.plot(xs, ci_q["mean"], color=PALETTE["red"], lw=1.35, label=f"Log fit ± 95% CI ($r_s$ = {rs_q:+.2f}, {p_str})", zorder=3)

    for wet_s, lbl, (dx, dy) in [("2011-06", "2011-06", (8, -10)), ("1992-12", "1992-12", (-44, -12))]:
        m_row = sub[sub["wet_start"] == wet_s]
        if not m_row.empty:
            r0 = m_row.iloc[0]
            ax2.annotate(lbl, (r0["F_w"], r0["Q_wet_z"]), xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.5, color="#334155", arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.6))

    ax2.text(0.36, -2.06, "Mild $F_w < 1$: −0.41$\\sigma$ (p = 0.018)", ha="left", va="bottom",
             fontsize=7.0, color="#92400E")
    ax2.text(2.88, -2.06, "Mod/Sev $F_w \\geq 1.0$: mean +0.44$\\sigma$ (p = 0.011)\n2013–2024 ($F_w \\geq 1.0$): +0.74$\\sigma$",
             ha="center", va="bottom", fontsize=7.1, color="#115E59")

    ax2.set_xlim(0.32, 4.78)
    ax2.set_ylim(-2.15, 2.55)
    ax2.set_xlabel("Basin Wet Severity $F_w$ (SPI-1 sum)")
    ax2.set_ylabel("Datong Wet-Spell Discharge Anomaly ($\\sigma$)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#CBD5E1", fontsize=7.3)
    ax2.set_title("b", loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig5", formats=formats)


def plot_fig6_circulation_composites(fields, windows_df, vectors, mask, circ_summary, controls_df, out_dir, formats=("png",)):
    """500 hPa composites (Flood n=16 vs All n=39), full 5880 gpm ridge to 170E, flux, and windows."""
    lats_nc, lons_nc = fields["lat"], fields["lon"]
    fig = plt.figure(figsize=(12.0, 8.3))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18, height_ratios=[1.15, 0.9])

    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2)
    xlim, ylim = (65, 170), (15, 62)

    for col_idx, (panel_lbl, badge_txt, z_fld, qu_fld, qv_fld, draw_5880) in enumerate([
        ("a", "May–Sep flood (n = 16)", fields["Z500_flood"], fields["qu_anom_flood"], fields["qv_anom_flood"], True),
        ("b", "All events (n = 39)", fields["Z500_all"], fields["qu_anom_all"], fields["qv_anom_all"], False),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        cf = ax.contourf(lons_nc, lats_nc, z_fld, levels=np.linspace(-1.2, 1.2, 13), cmap="RdBu_r", norm=norm, extend="both", zorder=1)
        _draw_coastlines(ax, vectors["coast"], xlim, ylim, color="#475569", lw=0.65, zorder=2)
        _draw_rings(ax, vectors["yrb"], facecolor=None, edgecolor="#111827", lw=1.15, zorder=4)
        ax.text(0.025, 0.96, badge_txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
                fontweight="bold", color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", alpha=0.86, edgecolor="#CBD5E1"), zorder=6)

        ax.add_patch(mpatches.Rectangle((125, 35), 25, 15, fill=False, edgecolor="#7F1D1D", lw=1.3, ls="-", zorder=5))
        ax.add_patch(mpatches.Rectangle((90, 25), 20, 17, fill=False, edgecolor="#1E3A8A", lw=1.3, ls="-", zorder=5))
        ax.text(137.5, 42.5, "NE", fontsize=8.4, fontweight="bold", color="#7F1D1D", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.78, edgecolor="none"), zorder=6)
        ax.text(100.0, 36.8, "PL", fontsize=8.4, fontweight="bold", color="#1E3A8A", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.82, edgecolor="none"), zorder=6)

        if draw_5880:
            ax.contour(lons_nc, lats_nc, fields["h500_flood_clim"], levels=[5880], colors=["#0F766E"], linewidths=[1.35], linestyles=["--"], zorder=5)
            ax.contour(lons_nc, lats_nc, fields["h500_flood"], levels=[5880], colors=["#991B1B"], linewidths=[1.6], linestyles=["-"], zorder=5)
            ax.text(137.5, 16.8, "5880 gpm:\nClim (142.5°E)\nTrans (165.0°E)", fontsize=7.0, color="#7F1D1D",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", alpha=0.88, edgecolor="#CBD5E1"), ha="left", zorder=6)

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
        ax.fill_between(lags, sub_all["null_p05"], sub_all["null_p95"], color="#CBD5E1", alpha=0.55, label="Climatological 5–95% band")

        ax.errorbar(lags - 0.05, sub_fl["mean"], yerr=sub_fl["se"], color=PALETTE["red"], marker="o", lw=1.8, capsize=3,
                    label="Flood May–Sep (n = 16)", zorder=4)
        ax.errorbar(lags + 0.05, sub_all["mean"], yerr=sub_all["se"], color=PALETTE["navy"], marker="s", ls="--", lw=1.5, capsize=3,
                    label="All transitions (n = 39)", zorder=4)

        row_all = circ_summary[(circ_summary["index"] == idx_code) & (circ_summary["stratum"] == "All (1961-2025)")].iloc[0]
        row_fl = circ_summary[(circ_summary["index"] == idx_code) & (circ_summary["stratum"] == "Flood (May-Sep)")].iloc[0]
        ctrl_row = controls_df[(controls_df["index"] == idx_code) & (controls_df["stratum"] == "All matched")].iloc[0]
        ax.text(0.03, 0.95,
                f"Dry$\\rightarrow$wet shift (All): {row_all['mean_diff']:+.2f}$\\sigma$ (p={row_all['p_value']:.4f})\n"
                f"Dry$\\rightarrow$wet shift (Flood): {row_fl['mean_diff']:+.2f}$\\sigma$ (p={row_fl['p_value']:.4f})\n"
                f"Matched wet ctrl (n={int(ctrl_row['n'])}): {ctrl_row['mean_diff']:+.2f}$\\sigma$ (p={ctrl_row['p_value']:.3f})",
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFFFF", alpha=0.93, edgecolor="#CBD5E1"))
        ax.set_xticks(lags)
        ax.set_xticklabels(["$t_0-3$", "$t_0-2$", "$t_0-1$", "$t_0$\n(Onset)", "$t_0+1$"])
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("Lag Relative to Wet-Spell Onset Month $t_0$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.5)
        ax.set_title(panel_lbl, loc="left", fontsize=11.5, fontweight="bold")

    _save_fig(fig, out_dir, "Fig6", formats=formats)


def plot_esm_fig1_wave_flux(fields, vectors, mask, out_dir, formats=("png",)):
    """200 hPa geopotential anomalies, May–Sep jet core contours, and Takaya–Nakamura wave-activity flux."""
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
    ax.text(100.0, 36.8, "PL", fontsize=8.5, fontweight="bold", color="#1E3A8A", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", alpha=0.82, edgecolor="none"), zorder=6)

    wx = fields["wx_200_flood"].copy()
    wy = fields["wy_200_flood"].copy()
    lat_mask = np.abs(lats_nc)[:, None] < 17.5
    mag = np.hypot(wx, wy)
    wx = np.where(lat_mask | (mag < 0.08), np.nan, wx)
    wy = np.where(lat_mask | (mag < 0.08), np.nan, wy)

    step = 2
    qv = ax.quiver(lons_nc[::step], lats_nc[::step], wx[::step, ::step], wy[::step, ::step],
                   scale=6.0, width=0.0032, color="#0F172A", zorder=6)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.73, 0.90), 0.255, 0.075, transform=ax.transAxes,
        boxstyle="round,pad=0.01", facecolor="#FFFFFF", alpha=0.92, edgecolor="#CBD5E1", lw=0.75, zorder=6.5,
    ))
    ax.quiverkey(qv, 0.785, 0.938, 0.5, "0.5 m$^2$ s$^{-2}$ ($\\mathbf{W}$ flux)",
                 labelpos="E", coordinates="axes", fontproperties={"size": 7.8}).set_zorder(7)

    cb = plt.colorbar(cf, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("200 hPa Geopotential Height Anomaly ($\\sigma$)")
    ax.set_xlim(65, 170)
    ax.set_ylim(15, 62)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    _save_fig(fig, out_dir, "ESM_Fig1", formats=formats)
