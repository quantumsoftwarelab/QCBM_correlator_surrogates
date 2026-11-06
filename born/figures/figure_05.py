"""Figure 5: gene-activation correlators."""
from born.common import set_style
import matplotlib.pyplot as plt
from scipy.special import binom
from born.common import correlators, target_probabilities
from matplotlib.ticker import ScalarFormatter


def make_figure():
    set_style()
    bio_corr = correlators(target_probabilities("bio"))
    n_qubits = 6
    fig, ax = plt.subplots(figsize=(7, 2.7))
    ax.bar(range(1, 2**n_qubits), bio_corr[1:], color="#ffa630")
    v_line = 1
    ax.axvline(x=v_line - 0.5, color="k", linestyle="-", linewidth=2.5)
    for p in range(1, n_qubits + 1):
        v_line += binom(n_qubits, p)
        ax.axvline(x=v_line - 0.5, color="k", linestyle="-", linewidth=2.5)
    ax.set_ylabel("$\\langle S_{\\boldsymbol{i}}\\rangle$", fontsize=17)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5)
    ax.text(0.47, 0.85, "Genes", ha="center", va="center", transform=ax.transAxes, fontsize=17)
    for spine in ax.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="both", color="lightgray", size=8)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    ax.ticklabel_format(style="plain", axis="x")
    ax.set_xticks(range(1, 2**n_qubits))
    ax.set_xticklabels([])
    ax.set_xlim(0.2, 2**n_qubits)
    ax.set_xlabel("Sites index vector $\\boldsymbol{i}$", labelpad=40, fontsize=17)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    ax.ticklabel_format(style="plain", axis="y")
    plt.tight_layout(pad=0.6)
    # Order-group labels replace the paper's separately overlaid TikZ braces.
    axis = ax
    left = 0.5
    from math import comb

    for order in range(1, 7):
        width = comb(6, order)
        axis.text(
            left + width / 2,
            -0.25 if order == 6 else -0.13,
            f"$k={order}$",
            transform=axis.get_xaxis_transform(),
            ha="center",
            fontsize=10,
        )
        left += width
    return fig
