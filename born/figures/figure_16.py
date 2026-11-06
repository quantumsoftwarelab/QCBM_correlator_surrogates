"""Figure 16: ANOVA kernels."""
from born.common import data_path, load_results, set_style
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def make_figure():
    set_style()
    experiment = "1d_ising"
    anova_degrees = list(range(1, 7))
    history_template = "diff_losses/histories_" + experiment + "_mmd_anova_{:d}.npz"
    kernel_template = "kernels_6/anova_{:d}.npy"
    max_x = 800
    sample_history = load_results(data_path(history_template.format(anova_degrees[0])))
    k_orders = sorted(sample_history.keys())
    n_k = len(k_orders)
    colors = sns.color_palette("magma", n_colors=2 * n_k)[::-2][:n_k]
    all_kernels = []
    for d in anova_degrees:
        K = np.load(data_path(kernel_template.format(d)))
        all_kernels.append(K)
    vmin = min((K.min() for K in all_kernels))
    vmax = max((K.max() for K in all_kernels))
    np.load(data_path("gammas.npy"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharex=True)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    axes = axes.flatten()
    x = np.arange(max_x)
    axins_x = [0.83, 0.83, 0.83, 0.7, 0.7, 0.52]
    axtext_x = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    axtext_y = [0.7, 0.7, 0.7, 0.57, 0.57, 0.4]
    letter = ["a)", "b)", "c)", "d)", "e)", "f)"]
    for idx, degree in enumerate(anova_degrees):
        ax = axes[idx]
        ax.patch.set_facecolor("none")
        ax.minorticks_on()
        ax.grid(which="major", linestyle="-", linewidth=0.5, color="lightgray", alpha=0.7)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=19,
            direction="out",
            length=6,
            width=2,
            color="lightgray",
            pad=2,
        )
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=19,
            direction="out",
            length=6,
            width=2,
            color="lightgray",
            pad=2,
        )
        ax.tick_params(axis="y", which="minor", direction="out", color="lightgray", pad=2)
        ax.tick_params(
            axis="x",
            which="minor",
            labelsize=19,
            direction="out",
            length=0,
            color="lightgray",
            pad=2,
        )
        for spine in ax.spines.values():
            spine.set_color("lightgray")
            spine.set_linewidth(1.5)
        if idx in [0, 1, 2]:
            ax.tick_params(
                axis="x",
                which="major",
                labelsize=19,
                direction="out",
                length=0,
                color="lightgray",
                pad=2,
            )
        history_max_k = load_results(data_path(history_template.format(degree)))
        for i, k in enumerate(k_orders):
            arr = np.array(history_max_k[k])[:max_x]
            log_arr = np.log(arr + 1e-12)
            mean_log = log_arr.mean(axis=0)
            std_log = log_arr.std(axis=0)
            median = np.exp(mean_log)
            lower_eps = np.exp(mean_log - std_log)
            upper_eps = np.exp(mean_log + std_log)
            ax.plot(x, median, label=f"$k$={k}", color=colors[i])
            ax.fill_between(x, lower_eps, upper_eps, color=colors[i], alpha=0.3)
        ax.text(
            axtext_x[idx],
            axtext_y[idx],
            letter[idx] + f" v={degree}",
            transform=ax.transAxes,
            fontsize=24,
        )
        ax.set_yscale("log")
        if idx % 3 == 0:
            ax.set_ylabel("MMD Loss", fontsize=23, color="black")
        if idx // 3 == 1:
            ax.set_xlabel("Iteration", fontsize=23, color="black")
        axins = inset_axes(
            ax,
            width=1.2,
            height=1.2,
            loc="upper right",
            bbox_to_anchor=(1, axins_x[idx], 0, 0),
            bbox_transform=ax.transAxes,
        )
        im = axins.imshow(
            all_kernels[idx],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
        )
        for spine in axins.spines.values():
            spine.set_edgecolor("lightgrey")
        axins.set_xticks([])
        axins.set_yticks([])
        ax.set_xlim(0, 800)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        title="Maximum $k$-order",
        title_fontsize=26,
        fontsize=25,
        ncol=n_k,
        loc="upper center",
        bbox_to_anchor=(0.47, 1.02),
        frameon=True,
    )
    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_alpha(0)
    leg.get_title().set_color("black")
    cbar = fig.colorbar(im, ax=axes.tolist(), orientation="vertical", fraction=0.02, pad=0.04)
    cbar.ax.patch.set_facecolor("none")
    cbar.outline.set_edgecolor("lightgrey")
    cbar.set_label(
        "Bitstring Distance $d_{\\boldsymbol{x}}$", fontsize=27, color="black", labelpad=21
    )
    cbar.ax.tick_params(
        axis="both",
        which="major",
        labelsize=19,
        colors="black",
        direction="out",
        length=6,
        width=2,
        color="lightgray",
        pad=2,
    )
    fig.tight_layout(pad=1.3, rect=[0, 0, 0.89, 0.92])
    return fig
