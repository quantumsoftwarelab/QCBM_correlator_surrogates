"""Figure 15: RMPS observable-placement dependence."""
from born.common import set_style
from born.tensor_network import observable_moment as var
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import Normalize
import matplotlib.lines as mlines


def make_figure():
    set_style()
    LABEL_SIZE = 18

    def plot_variances_for_ns(n_values, D, step=1):
        fig, axes = plt.subplots(
            1, len(n_values), figsize=(5 * len(n_values) + 2, 5.5), sharey=True, facecolor="white"
        )
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=step, vmax=max(n_values))
        count_i = 0
        for ax, n in zip(axes, n_values):
            ax.set_facecolor("white")
            obs = ["proj", "id"]
            O_list = list(itertools.product(obs, repeat=n))
            O_dict = {}
            for O in O_list:
                k = O.count("proj")
                O_dict.setdefault(k, []).append(O)
            variances = [[var(O, D) for O in O_dict[k]] for k in range(1, n + 1, step)]
            for k_idx, k in enumerate(range(1, n + 1, step)):
                if count_i == 0:
                    ax.scatter(
                        np.arange(len(variances[k_idx])),
                        variances[k_idx],
                        label=f"$k$={k}",
                        color=cmap(norm(k)),
                    )
                elif k_idx != len(variances) - 1:
                    ax.plot(
                        np.arange(len(variances[k_idx])),
                        variances[k_idx],
                        label=f"$k$={k}",
                        color=cmap(norm(k)),
                    )
                else:
                    ax.scatter(
                        np.arange(len(variances[k_idx])),
                        variances[k_idx],
                        color=cmap(norm(k)),
                        s=10,
                    )
            count_i += 1
            ax.set_yscale("log")
            ax.set_xlabel("Bitstring $\\boldsymbol{x}$")
            if ax is axes[0]:
                ax.set_ylabel("$\\Var(\\boldsymbol{x})$")
            for spine in ax.spines.values():
                spine.set_color("lightgrey")
            ax.tick_params(axis="both", which="both", color="lightgray", labelsize=LABEL_SIZE)
            ax.grid(True, which="both", ls="-", lw=0.5, alpha=0.5)
            ax.text(0.95, 0.12, f"$n={n}$", transform=ax.transAxes, ha="right", va="top")
        k_values = list(range(1, max(n_values) + 1, step))
        handles = [
            mlines.Line2D(
                [], [], color=cmap(norm(k)), marker="o", linestyle="None", label=f"$k$={k}"
            )
            for k in k_values
        ]
        leg = fig.legend(
            handles,
            [f"$k$={k}" for k in k_values],
            title="$k$-order",
            loc="center left",
            bbox_to_anchor=(0.87, 0.54),
            ncol=1,
            frameon=False,
            edgecolor="lightgray",
            labelspacing=0.2,
            handletextpad=0.4,
            title_fontsize=LABEL_SIZE + 2,
        )
        leg.get_title().set_position((0, 11))
        plt.tight_layout(rect=[0, 0, 0.89, 1])
        return fig

    fig = plot_variances_for_ns([5, 10, 15], 7)
    return fig
