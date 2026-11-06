"""Figure 9: deployment under order and random-subset truncation."""
from born.common import data_path, set_style
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def make_figure():
    set_style()
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 18
    LEGEND_SIZE = 20
    experiments = ["1d_ising", "1d_heisenberg", "1d_haldane", "2d_haldane", "bio"]
    legends = ["1D Ising", "1D Heisenberg", "1D Haldane", "2D Haldane", "Genes"]
    colors = ["#a4b391", "#386d48", "#2e5077", "#4da1a9", "#ffa630"]
    algebras = ["iqp", "matchgate", "heisenberg", "haldane"]
    col_titles = [
        "IQP",
        "$\\mathfrak{g}_{\\text{Matchcircuit}}$",
        "$\\mathfrak{g}_{\\text{Heisenberg}}$",
        "$\\mathfrak{g}_{\\text{Haldane}}$",
    ]
    n_qubits = 6
    depth = 40
    n_exp = 10
    xvals = np.arange(1, n_qubits + 1)
    Y_MIN = -0.03
    Y_MIN_RAISED = Y_MIN - 0.07
    TOP_SPEC = ("G_info", "k")
    BOTTOM_SPEC = ("G_info_rfc", "ss")

    def style_axes(ax):
        for spine in ax.spines.values():
            spine.set_color("lightgray")
            spine.set_linewidth(1.5)
        ax.tick_params(axis="both", which="both", colors="lightgray", labelcolor="black")
        ax.grid(True)

    def base_path(folder, algebra, experiment, key_name, n_qubits, depth, n_exp):
        return os.path.join(
            folder,
            f"tgt_{experiment}_alg_{algebra}_{key_name}_{n_qubits}_n_{n_qubits}_d_{depth}",
            f"n_exp_{n_exp}",
        )

    def k_path(folder, algebra, experiment, key_name, max_k, n_qubits, depth, n_exp):
        return os.path.join(
            folder,
            f"tgt_{experiment}_alg_{algebra}_{key_name}_{max_k}_n_{n_qubits}_d_{depth}",
            f"n_exp_{n_exp}",
        )

    def plot_algebra_on_axis(ax, folder, key_name, algebra):
        curves = []
        for exp_idx, experiment in enumerate(experiments):
            p_base = base_path(folder, algebra, experiment, key_name, n_qubits, depth, n_exp)
            probs_n_order = np.load(data_path(os.path.join(p_base, "all_probs.npy")))
            kl_divs_all = []
            for max_k in range(1, n_qubits + 1):
                p_k = k_path(folder, algebra, experiment, key_name, max_k, n_qubits, depth, n_exp)
                probabilities = np.load(data_path(os.path.join(p_k, "all_probs.npy")))
                kl_vals = []
                for i in range(n_exp):
                    p = np.clip(probabilities[i], 1e-15, 1)
                    q = np.clip(probs_n_order[i], 1e-15, 1)
                    kl_vals.append(np.sum(p * np.log(p / q)))
                kl_divs_all.append(kl_vals)
            kl_divs_all = np.array(kl_divs_all)
            mean_kl = np.clip(np.mean(kl_divs_all, axis=1), 0, None)
            std_kl = np.std(kl_divs_all, axis=1)
            ax.plot(
                xvals,
                mean_kl,
                marker="o",
                linestyle="-",
                color=colors[exp_idx],
                label=legends[exp_idx],
                markeredgecolor="gray",
            )
            ax.fill_between(
                xvals,
                np.maximum(mean_kl - std_kl, 0),
                mean_kl + std_kl,
                color=colors[exp_idx],
                alpha=0.2,
            )
            curves.append(
                {
                    "x": xvals,
                    "mean": mean_kl,
                    "std": std_kl,
                    "color": colors[exp_idx],
                    "label": legends[exp_idx],
                }
            )
        return curves

    def add_inset(ax, curves, y_range=(0.0, 1.0), width="58%", height="46%", loc=(0.4, 0.05)):
        inset = inset_axes(
            ax,
            width=width,
            height=height,
            bbox_to_anchor=(loc[0], loc[1], 1, 0.9),
            bbox_transform=ax.transAxes,
            loc="upper left",
            borderpad=0,
        )
        for c in curves:
            x = c["x"]
            m = c["mean"]
            s = c["std"]
            col = c["color"]
            inset.plot(x, m, marker="o", linestyle="-", color=col, markeredgecolor="gray")
            inset.fill_between(x, np.maximum(m - s, 0), m + s, color=col, alpha=0.2)
        inset.set_xlim(xvals[0] - 0.2, xvals[-1] + 0.2)
        inset.set_ylim(y_range)
        inset.set_yticks([0, 0.5])
        inset.set_xticks([2, 4, 6])
        inset.tick_params(
            axis="both",
            which="both",
            colors="lightgray",
            labelcolor="black",
            labelsize=TICK_SIZE * 0.6,
        )
        for spine in inset.spines.values():
            spine.set_color("lightgray")
            spine.set_linewidth(1.0)
        inset.grid(True, alpha=0.6)
        return inset

    fig, axes = plt.subplots(
        nrows=2, ncols=4, figsize=(16, 9.2), sharex=True, sharey="col", constrained_layout=False
    )
    plt.subplots_adjust(top=0.83, left=0.07, right=0.99, bottom=0.14, hspace=0.35, wspace=0.22)
    inset_cols = {1, 2, 3}
    extra_headroom_cols = {1, 2, 3}
    for col, (algebra, col_title) in enumerate(zip(algebras, col_titles)):
        ax_top = axes[0, col]
        style_axes(ax_top)
        curves_top = plot_algebra_on_axis(ax_top, TOP_SPEC[0], TOP_SPEC[1], algebra)
        ax_top.set_xlim(0.85, n_qubits + 0.15)
        ax_top.set_xticks([1, 2, 3, 4, 5, 6])
        Y_MAX = 1.3
        Y_MAX_RAISED = 4
        ax_top.set_title(col_title, fontsize=TITLE_SIZE, color="black", pad=8)
        ax_bot = axes[1, col]
        style_axes(ax_bot)
        curves_bot = plot_algebra_on_axis(ax_bot, BOTTOM_SPEC[0], BOTTOM_SPEC[1], algebra)
        ax_bot.set_xlim(0.85, n_qubits + 0.15)
        ax_bot.set_xticks([1, 2, 3, 4, 5, 6])
        ax_bot.set_ylim(bottom=Y_MIN if col == 0 else Y_MIN_RAISED)
        if col in inset_cols:
            add_inset(ax_top, curves_top, y_range=(-0.05, 0.6))
            add_inset(ax_bot, curves_bot, y_range=(-0.05, 0.6))
        if col in extra_headroom_cols:
            col_max = max([np.max(c["mean"] + c["std"]) for c in curves_top + curves_bot])
            for ax in (ax_top, ax_bot):
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, max(ymax, col_max * 1.12))
        ax_top.set_ylim(
            bottom=Y_MIN if col == 0 else Y_MIN_RAISED, top=Y_MAX if col == 0 else Y_MAX_RAISED
        )
    fig.supxlabel(
        "Number of constrained correlators index $k$: $\\displaystyle |\\Omega_{\\mathcal{C}\\!\\ell}(k)| = \\sum_{p=0}^{k} \\binom{n}{p}$",
        fontsize=LABEL_SIZE,
        x=0.52,
        y=-0.0,
    )
    fig.supylabel(
        "Average Deployed KL Divergence $\\mathcal{L}(\\operatorname{Pr}_{\\mathcal{Q}}(\\theta^*_{\\mathcal{C}\\!\\ell}), \\operatorname{Pr}^*)$",
        fontsize=LABEL_SIZE,
        x=0.02,
    )
    fig.text(0.4, 0.89, "a) $k$-order truncation approximation", fontsize=LABEL_SIZE, color="black")
    fig.text(
        0.332,
        0.46,
        "b) Random Fourier Correlator truncation approximation",
        fontsize=LABEL_SIZE,
        color="black",
    )
    exp_handles = [
        mlines.Line2D([], [], color=colors[i], linestyle="-", label=legends[i])
        for i in range(len(experiments))
    ]
    fig.legend(
        title="Datasets",
        handles=exp_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.52, 1.05),
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=LABEL_SIZE,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=0.8,
    )
    return fig
