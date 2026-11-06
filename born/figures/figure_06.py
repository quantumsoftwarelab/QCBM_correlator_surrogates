"""Figure 6: matchcircuit variances."""
from born.common import set_style
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import binom


def make_figure():
    set_style()
    colors = sns.color_palette("blend:#7AB,#EDA", 6)
    sns.color_palette("blend:#0b1364,#EDA", 6)
    n_values = range(10, 21, 2)
    variances_1 = {}
    for n in range(10, 21, 2):
        variances_1[n] = []
        for p in range(1, n):
            variance = 0
            norm = sum([binom(n, p_prime) for p_prime in range(0, p + 1)])
            norm = 2**n
            for kappa in range(2, 2 * n, 2):
                if kappa // 2 <= p:
                    variance += (
                        1
                        / norm**2
                        * binom(n, kappa // 2)
                        * binom(n, kappa // 2)
                        / binom(2 * n, kappa)
                    )
                if n - kappa // 2 <= p and kappa // 2 <= p:
                    variance += (
                        1
                        / norm**2
                        * binom(n, kappa // 2)
                        * binom(n, kappa // 2)
                        / binom(2 * n, kappa)
                    )
            variances_1[n].append(variance)
    variances_2 = {}
    for n in n_values:
        variances_2[n] = []
        for m in range(2, 2 * n, 2):
            variances_2[n].append(binom(n, m // 2) / binom(2 * n, m))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), sharey=False)
    ax1 = axes[1]
    for idx, n in enumerate(n_values):
        ax1.plot(
            range(1, len(variances_1[n]) + 1),
            variances_1[n],
            label=f"$n = {n}$",
            marker="o",
            color=colors[idx],
            markeredgecolor="gray",
        )
    ax1.set_yscale("log")
    ax1.set_xticks(range(0, 21, 2))
    ax1.set_xlabel("Truncation $k$-order", fontsize=23)
    ax1.set_ylabel(
        "$\\Var_{\\boldsymbol \\theta}^{\\rm MG}\\left(\\operatorname{Pr}_{\\boldsymbol{\\theta}}^{(k)}(\\boldsymbol{x})\\right)$",
        fontsize=23,
    )
    ax1.text(-0.2, 1.06, "b)", transform=ax1.transAxes, fontsize=26)
    ax1.grid(True)
    for spine in ax1.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    ax1.tick_params(axis="both", which="both", color="lightgray", size=8)
    ax2 = axes[0]
    for idx, n in enumerate(n_values):
        ax2.plot(
            range(1, len(variances_2[n]) + 1),
            variances_2[n],
            marker="o",
            color=colors[idx],
            markeredgecolor="gray",
        )
    ax2.set_yscale("log")
    ax2.set_xticks(range(0, 21, 2))
    ax2.set_xlabel("Correlator Order $k=|\\boldsymbol{i}|$", fontsize=23)
    ax2.set_ylabel(
        "$\\Var_{\\boldsymbol \\theta}^{\\rm MG}(\\langle Z_{\\boldsymbol{i}}\\rangle_{\\boldsymbol \\theta})$",
        fontsize=23,
    )
    ax2.grid(True)
    ax2.text(
        0.98,
        1.08,
        "Recreated from [72]",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=20,
    )
    ax2.text(-0.1, 1.07, "a)", ha="right", va="top", transform=ax2.transAxes, fontsize=26)
    for spine in ax2.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    ax2.tick_params(axis="both", which="both", color="lightgray", size=8)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="System Size $n$",
        loc="upper center",
        bbox_to_anchor=(0.53, 1.05),
        ncol=len(n_values),
        frameon=False,
        title_fontsize=26,
        fontsize=25,
        columnspacing=0.98,
        handletextpad=0.4,
    )
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.8)
    return fig
