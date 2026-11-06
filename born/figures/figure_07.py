"""Figure 7: random matrix-product-state moments."""
from born.common import set_style
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from born.tensor_network import observable_moment, order_moments
from matplotlib.ticker import LogLocator


def make_figure():
    set_style()
    TITLE_SIZE, LEGEND_SIZE = (20, 16)
    n = 25
    k_list = range(1, n + 1, 4)
    D_list = range(0, 30)
    variances = [
        [observable_moment(["proj"] * k + ["id"] * (n - k), D) for D in D_list] for k in k_list
    ]
    variances_dict = {n: np.cumsum(order_moments(n, 1000)) / 2**n for n in range(10, 21, 2)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=False)
    plt.subplots_adjust(wspace=1.4, left=0.08, right=0.98, top=0.96, bottom=0.18)
    colors_left = sns.color_palette("YlOrBr_r", len(k_list) * 2)[: len(k_list) + 1][::-1]
    for k_idx, k in enumerate(k_list):
        ax1.axhline(
            1 / 2 ** (2 * k) * (1 + 2**k / 2 ** len(D_list)),
            color=colors_left[k_idx],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        ax1.plot(
            D_list,
            np.asarray(variances[k_idx]),
            label=f"$m \\!=\\! {k}$",
            marker="o",
            color=colors_left[k_idx],
            markeredgecolor="gray",
        )
    asymptote_line = mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="$\\frac{1}{2^{2m}}\\left(1+\\frac{2^m}{2^n} \\right)$",
    )
    legend1 = ax1.legend(
        [asymptote_line],
        ["$\\frac{1}{2^{2m}}\\left(1+\\frac{2^m}{2^n} \\right)$"],
        title="\\textbf{$\\mathbf{\\chi \\to \\infty}$}",
        loc="upper left",
        bbox_to_anchor=(0.98, 1.04),
        frameon=False,
        handlelength=1.2,
        handleheight=0.8,
        fontsize=TITLE_SIZE,
        title_fontsize=TITLE_SIZE,
        labelspacing=0.2,
    )
    ax1.add_artist(legend1)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles,
        labels,
        title="\\textbf{Marginal Size}",
        loc="upper left",
        bbox_to_anchor=(1.035, 0.68),
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
        labelspacing=0.3,
    )
    for spine in ax1.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    ax1.tick_params(axis="both", which="both", color="lightgray", size=8)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="-", linewidth=0.5)
    ax1.set_ylabel("$\\Var( \\Pi^{(m)}_{\\boldsymbol{x}})_{\\max}$", fontsize=TITLE_SIZE)
    ax1.set_xlabel("Bond Dimension $\\chi$", fontsize=TITLE_SIZE)
    ax1.text(
        -0.2,
        0.93,
        "a)",
        transform=ax1.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    colors_right = sns.color_palette("YlGn", 30)[7::3]
    any_n = next(iter(variances_dict))
    len(variances_dict[any_n])
    for idx, n in enumerate(sorted(variances_dict.keys())):
        y_vals = np.asarray(variances_dict[n])
        k_vals = np.arange(len(y_vals))
        ax2.plot(
            k_vals,
            y_vals,
            label=f"$n={n}$",
            color=colors_right[idx % len(colors_right)],
            marker="o",
            markeredgecolor="gray",
        )
    ax2.set_yscale("log")
    ax2.legend(
        title="\\textbf{System Size}",
        loc="best",
        bbox_to_anchor=(1.0, 0.75),
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
        labelspacing=0.21,
    )
    ax2.set_xlabel("$k$-order", fontsize=TITLE_SIZE)
    ax2.set_ylabel(
        "$\\Var(\\operatorname{Pr}^{(k)}_{\\boldsymbol{\\theta}}(\\boldsymbol{x}=\\boldsymbol{0}))$",
        fontsize=TITLE_SIZE,
    )
    try:
        ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
    except Exception:
        pass
    ax2.grid(True, linestyle="-", linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    ax2.tick_params(axis="both", which="both", color="lightgray", size=8)
    ax2.text(
        -0.18,
        0.93,
        "b)",
        transform=ax2.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax2.set_xticks(np.arange(0, len(k_vals), 2))
    fig.tight_layout(pad=3)
    return fig
