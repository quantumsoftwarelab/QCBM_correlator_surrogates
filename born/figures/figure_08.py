"""Figure 8: loss functions under correlator truncation."""
from born.common import data_path, load_results, set_style
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_figure():
    set_style()
    experiment = "1d_ising"
    file_names = [
        "diff_losses/histories_1d_ising_mmd_hamming_0.0001_20.npz",
        "diff_losses/histories_1d_ising_emd.npz",
        "diff_losses/histories_2d_haldane_mmd_hamming_0.0001_20.npz",
        "diff_losses/histories_1d_ising_mmd_hamming_0.01_20.npz",
        "diff_losses/histories_1d_ising_tv.npz",
        "diff_losses/histories_bio_mmd_hamming_0.0001_20.npz",
    ]
    sample_history = load_results(data_path(file_names[0]))
    k_orders = sorted(sample_history.keys())
    n_k = len(k_orders)
    colors = sns.color_palette("cool_r", n_colors=2 * n_k)[::-2][:n_k]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    axes = axes.flatten()
    axtext_x = -0.2
    axtext_y = 0.98
    letter = ["a)", "b)", "e)", "c)", "d)", "f)"]
    loss_text = [
        "MMD $\\sigma=10^{-3}$",
        "EMD",
        "MMD $\\sigma=10^{-3}$",
        "MMD $\\sigma=10^{-1}$",
        "SQE",
        "MMD $\\sigma=10^{-3}$",
        "MMD $\\sigma=10^{-3}$",
    ]
    data_text = ["1D TFIM", "1D TFIM", "2D Haldane", "1D TFIM", "1D TFIM", "Genes"]
    loss_text_x = [0.45, 0.7, 0.45, 0.45, 0.7, 0.45]
    data_text_x = 0.03
    loss_text_y = [0.4, 0.87, 0.87, 0.4, 0.45, 0.45]
    data_text_y = [0.03] * 6
    for idx, experiment in enumerate(file_names):
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
        history_max_k = load_results(data_path(experiment))
        for i, k in enumerate(k_orders):
            arr = np.array(history_max_k[k])
            log_arr = np.log(arr + 1e-12)
            mean_log = log_arr.mean(axis=0)
            std_log = log_arr.std(axis=0)
            median = np.exp(mean_log)
            lower_eps = np.exp(mean_log - std_log)
            upper_eps = np.exp(mean_log + std_log)
            n_it_max = len(median)
            if idx == 2:
                n_it_max = 300
            elif idx == 4:
                n_it_max = 400
                ax.set_ylim(bottom=0.013, top=1.5)
            x = np.arange(n_it_max)
            ax.plot(x, median[:n_it_max], label=f"$k$={k}", color=colors[i])
            ax.fill_between(
                x, lower_eps[:n_it_max], upper_eps[:n_it_max], color=colors[i], alpha=0.3
            )
        ax.text(axtext_x, axtext_y, letter[idx], transform=ax.transAxes, fontsize=24)
        ax.text(
            loss_text_x[idx],
            loss_text_y[idx],
            loss_text[idx],
            transform=ax.transAxes,
            fontsize=24,
            color="black",
        )
        ax.text(
            data_text_x,
            data_text_y[idx],
            data_text[idx],
            transform=ax.transAxes,
            fontsize=24,
            color="black",
        )
        ax.set_yscale("log")
        if idx % 3 == 0:
            ax.set_ylabel("Training Loss Value $\\mathcal{L}$", fontsize=23, color="black")
        if idx // 3 == 1:
            ax.set_xlabel("Iteration", fontsize=23, color="black")
        ax.set_xlim(0, n_it_max)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        title="Maximum $k$-order",
        title_fontsize=26,
        fontsize=25,
        ncol=n_k,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
    )
    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_alpha(0)
    leg.get_title().set_color("black")
    bbox0 = axes[0].get_position()
    bbox1 = axes[1].get_position()
    bbox0.x1 + bbox1.x0
    fig.lines.append(
        plt.Line2D(
            [0.665, 0.665],
            [0.04, 0.9],
            transform=fig.transFigure,
            color="lightgray",
            linestyle="-",
            linewidth=4,
        )
    )
    fig.tight_layout(pad=1, rect=[0, 0, 1, 0.92])
    return fig
