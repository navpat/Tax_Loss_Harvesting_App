"""
charts.py
---------
All plotting logic. Takes simulation output, returns matplotlib figures.
No Streamlit imports — fully reusable outside the app.
"""

import numpy as np
import matplotlib.pyplot as plt

from model import LEVERAGE_LEVELS, LEVERAGE_LABELS

COLORS = {
    0.0: "#888780",
    0.5: "#378ADD",
    1.0: "#1D9E75",
    1.5: "#D85A30",
}


# ─────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("#F5F4F0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D3D1C7")
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.tick_params(colors="#5F5E5A", labelsize=9)
    ax.yaxis.label.set_color("#5F5E5A")
    ax.xaxis.label.set_color("#5F5E5A")
    ax.title.set_color("#2C2C2A")
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.4, zorder=0)


def _make_fig(ncols=2, figsize=(12, 4.5)):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    fig.patch.set_facecolor("#FAFAF9")
    return fig, axes


# ─────────────────────────────────────────────
# Generic line plotter
# ─────────────────────────────────────────────

def _plot_lines(ax, results, t_axis, key, title, ylabel,
                hline=None, skip_zero_fill=False):
    """
    Plot median + 10th/90th percentile band for each leverage level.
    L=0.0 always dashed (baseline).
    """
    for L in LEVERAGE_LEVELS:
        data = results[L][key]
        med  = np.median(data, axis=0)
        p10  = np.percentile(data, 10, axis=0)
        p90  = np.percentile(data, 90, axis=0)
        c    = COLORS[L]
        ls   = "--" if L == 0.0 else "-"
        ax.plot(t_axis, med, color=c, lw=1.8, linestyle=ls,
                label=LEVERAGE_LABELS[L], zorder=4)
        if L != 0.0 and not skip_zero_fill:
            ax.fill_between(t_axis, p10, p90, color=c, alpha=0.10)

    if hline is not None:
        ax.axhline(hline, color="#2C2C2A", lw=0.7, linestyle=":", alpha=0.5)

    ax.set_title(title, fontsize=10, fontweight="normal", pad=8)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, framealpha=0)
    _style_ax(ax)


# ─────────────────────────────────────────────
# Public chart functions
# ─────────────────────────────────────────────

def fig_tax_and_cncl(results, t_axis):
    """
    Left:  tax bill over time (with vs without overlay)
    Right: CNCL as % of overlay capital
    """
    fig, (ax1, ax2) = _make_fig()

    _plot_lines(ax1, results, t_axis, "tax_with",
                "Tax bill if stock sold today ($M)", "$M")

    # add no-overlay baseline as dotted black line on ax1
    med_baseline = np.median(results[0.0]["tax_no"], axis=0)
    ax1.plot(t_axis, med_baseline, color="#2C2C2A", lw=1.0,
             linestyle=":", alpha=0.5, label="No overlay (tax_no)")
    ax1.legend(fontsize=8, framealpha=0)

    _plot_lines(ax2, results, t_axis, "cncl_pct",
                "CNCL as % of overlay capital", "%", hline=100)

    plt.tight_layout(pad=1.5)
    return fig


def fig_net_benefit(results, t_axis):
    """
    Left:  net benefit (tax saved minus costs) over time
    Right: cumulative costs over time
    """
    fig, (ax1, ax2) = _make_fig()

    _plot_lines(ax1, results, t_axis, "net_benefit",
                "Net benefit: tax saved minus costs ($M)", "$M", hline=0)

    _plot_lines(ax2, results, t_axis, "cum_costs",
                "Cumulative financing + transaction costs ($M)", "$M")

    plt.tight_layout(pad=1.5)
    return fig


def fig_leverage_sensitivity(results, years):
    """
    Left:  year-N net benefit by leverage (bar + error bars)
    Right: year-N tax saved by leverage (bar)
    """
    fig, (ax1, ax2) = _make_fig()
    yr_idx    = years
    x         = np.arange(len(LEVERAGE_LEVELS))
    bar_cols  = [COLORS[L] for L in LEVERAGE_LEVELS]
    xlbls     = [LEVERAGE_LABELS[L] for L in LEVERAGE_LEVELS]

    def _bar_with_errors(ax, key, title):
        meds = [np.median(results[L][key][:, yr_idx]) for L in LEVERAGE_LEVELS]
        p10s = [np.percentile(results[L][key][:, yr_idx], 10) for L in LEVERAGE_LEVELS]
        p90s = [np.percentile(results[L][key][:, yr_idx], 90) for L in LEVERAGE_LEVELS]
        bars = ax.bar(x, meds, color=bar_cols, width=0.5, zorder=3)
        ax.errorbar(
            x, meds,
            yerr=[np.array(meds) - np.array(p10s),
                  np.array(p90s) - np.array(meds)],
            fmt="none", color="#444441", capsize=4, lw=1, zorder=4
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="normal", pad=8)
        ax.set_ylabel("$M", fontsize=9)
        ax.axhline(0, color="#2C2C2A", lw=0.7, linestyle=":", alpha=0.5)
        for bar, val in zip(bars, meds):
            offset = 0.01 * max(abs(v) for v in meds) if meds else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"${val:.2f}M",
                ha="center", va="bottom", fontsize=8, color="#2C2C2A"
            )
        _style_ax(ax)

    _bar_with_errors(ax1, "net_benefit",
                     f"Year-{years} net benefit by leverage ($M)")

    # tax saved = tax_no - tax_with (computed inline)
    saveds = [
        np.median(results[L]["tax_no"][:, yr_idx])
        - np.median(results[L]["tax_with"][:, yr_idx])
        for L in LEVERAGE_LEVELS
    ]
    bars2 = ax2.bar(x, saveds, color=bar_cols, width=0.5, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlbls, fontsize=8)
    ax2.set_title(f"Year-{years} tax saved by leverage ($M)",
                  fontsize=10, fontweight="normal", pad=8)
    ax2.set_ylabel("$M", fontsize=9)
    for bar, val in zip(bars2, saveds):
        offset = 0.01 * max(saveds) if saveds else 0
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"${val:.2f}M",
            ha="center", va="bottom", fontsize=8, color="#2C2C2A"
        )
    _style_ax(ax2)

    plt.tight_layout(pad=1.5)
    return fig