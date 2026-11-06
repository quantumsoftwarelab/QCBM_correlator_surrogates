"""Figure 14: RMPS subsystem purity (published entropy panel)."""
from born.common import set_style
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from born.tensor_network import purity_moment


def make_figure():
    set_style()
    n = 25
    k_list = range(0, 13, 2)
    D_list = np.logspace(0, 5, 30)
    variances = {
        k: np.array([purity_moment(["i"] * (n - k) + ["f"] * k, D) for D in D_list]) for k in k_list
    }
    colors = sns.color_palette("mako_r", len(k_list))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k_idx, k in enumerate(k_list):
        ax.axhline(
            1 / 2**k + 2 ** (k - n),
            linestyle="dashed",
            linewidth=2,
            alpha=0.7,
            color=colors[k_idx],
        )
        if k != n - k:
            ax.plot(
                D_list,
                variances[k],
                label=f"$k$ = {k}, {n - k}",
                color=colors[k_idx],
                marker="o",
                markeredgecolor="gray",
            )
        else:
            ax.plot(
                D_list,
                variances[k],
                label=f"$k$ = {k}",
                color=colors[k_idx],
                marker="o",
                markeredgecolor="gray",
            )
    legend1 = ax.legend(
        title="\\textbf{$\\mathbf{k}$-order}",
        loc="center left",
        bbox_to_anchor=(1, 0.35),
        frameon=False,
    )
    ax.add_artist(legend1)
    asymptote_line = mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="$\\frac{1}{2^{k}} + \\frac{2^{k}}{2^n}$",
    )
    ax.legend(
        [asymptote_line],
        ["$\\frac{1}{2^{k}} +\\frac{2^{k}}{2^n}$"],
        title="\\textbf{$\\mathbf{\\chi \\to \\infty}$}",
        loc="upper left",
        bbox_to_anchor=(1.025, 0.975),
        frameon=False,
    )
    for spine in ax.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="both", color="lightgray", size=8)
    ax.set_yscale("log")
    ax.set_xlabel("Bond Dimension $\\chi$")
    ax.set_xscale("log")
    ax.set_ylabel("$S_2\\left(\\rho_{k,n}\\right)_{max}$")
    ax.grid(True)
    fig.tight_layout(pad=0)
    return fig
