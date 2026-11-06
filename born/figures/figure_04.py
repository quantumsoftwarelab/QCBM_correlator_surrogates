"""Figure 4: ground-state correlators."""
from born.common import set_style
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from born.common import target_probabilities, correlators


def make_figure():
    set_style()
    n_qubits = 6
    experiment_name_list = ["1d_ising", "1d_heisenberg", "1d_haldane", "2d_haldane"]
    colors = ["#a4b391", "#386d48", "#2e5077", "#4da1a9"]
    labels = [
        "Transverse \n Ising Eq.\\!\\! 10",
        "Alternating \n Heisenberg Eq.\\!\\! 11",
        "Haldane \n Eq.\\!\\! 12",
        "Haldane $3\\!\\times\\!2$ \n Eq.\\!\\! 13",
    ]
    xpositions = [0.49, 0.485, 0.49, 0.49]
    letter = ["a)", "b)", "c)", "d)"]
    fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=False)
    for i, (df, color, label) in enumerate(zip(experiment_name_list, colors, labels)):
        exp_vals = np.abs(correlators(target_probabilities(df)))
        axes[i].bar(np.arange(2**n_qubits)[1:], exp_vals[1:], color=color)
        v_line = 0
        axes[i].axvline(x=v_line + 0.5, color="k", linestyle="-", linewidth=2.5)
        for p in range(1, n_qubits + 1):
            v_line += binom(n_qubits, p)
            axes[i].axvline(x=v_line + 0.5, color="k", linestyle="-", linewidth=2.5)
        axes[i].set_ylabel("$|\\langle Z^*_{\\boldsymbol{i}}\\rangle |$")
        axes[i].grid(True, axis="y", linestyle="-", linewidth=0.5)
        for spine in axes[i].spines.values():
            spine.set_color("lightgray")
            spine.set_linewidth(1.5)
        axes[i].tick_params(axis="both", which="both", color="lightgray", size=8)
        axes[i].text(-0.18, 0.97, letter[i], transform=axes[i].transAxes, fontsize=17)
        axes[i].text(
            xpositions[i],
            0.81,
            label,
            ha="center",
            va="center",
            transform=axes[i].transAxes,
            fontsize=17,
        )
    for ax in axes[:-1]:
        ax.set_xticks([])
        ax.set_xlim(1 - 0.8, int(2**6))
    axes[-1].set_xticklabels([])
    axes[-1].set_xticks(np.arange(2**n_qubits)[1:])
    axes[-1].set_xlim(1 - 0.8, int(2**6))
    axes[-1].set_xlabel("Sites index vector $\\boldsymbol{i}$", labelpad=40)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    common_ymin = 1e-18
    common_ymax = 1.1
    for ax in axes[1:]:
        ax.set_ylim(common_ymin, common_ymax)
    plt.tight_layout(pad=0.6)
    # Order-group labels replace the paper's separately overlaid TikZ braces.
    axis = axes[-1]
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
