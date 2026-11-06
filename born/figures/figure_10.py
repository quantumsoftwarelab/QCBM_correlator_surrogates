"""Figure 10: batch-size deployment."""
from born.common import data_path, load_results, set_style
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def make_figure():
    set_style()
    LABEL_SIZE = 20
    colors = ["#03FD88", "#33C58E", "#276174"][::-1]
    border_color = "lightgray"

    def load_kl_dicts(n: int):
        base = f"results/ising_{n}_1"
        shots = [128, 256, 512]
        loaded = {}
        for S in shots:
            path = os.path.join(base, f"xb_{S}", f"avg_kl_per_zb_xb_xb_{S}.npz")
            if os.path.isfile(data_path(path)):
                data = load_results(data_path(path))
                loaded[S] = (list(data.keys()), list(data.values()))
        return loaded

    def keys_to_percent(keys, n: int):
        keys = np.asarray(keys, dtype=float)
        return keys / 2.0**n * 100.0

    percent_formatter = FuncFormatter(lambda x, pos: f"{x:.1f}\\%")
    data_12 = load_kl_dicts(12)
    data_20 = load_kl_dicts(20)
    if len(data_12) == 0:
        raise FileNotFoundError("No 12-qubit pickle files were found. Check the paths for n=12.")
    if len(data_20) == 0:
        raise FileNotFoundError("No 20-qubit pickle files were found. Check the paths for n=20.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.text(
        -0.024,
        0.5,
        "Average Deployed KL Divergence\n$\\mathcal{L}(\\operatorname{Pr}_{\\mathcal{Q}}(\\theta^*_{\\mathcal{C}\\!\\ell}), \\operatorname{Pr}^*)$",
        ha="center",
        va="center",
        rotation="vertical",
    )

    def style_axis(ax):
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.5)
        ax.tick_params(axis="both", which="both", color=border_color)
        ax.grid(True, which="both", axis="both", alpha=0.2)

    axL = axes[0]
    for idx, S in enumerate(sorted(data_12.keys())):
        x_keys, y_vals = data_12[S]
        x_pct = keys_to_percent(x_keys, n=12)
        axL.plot(
            x_pct,
            y_vals,
            marker="o",
            linestyle="-",
            label=f"$2^{{{int(np.log2(S))}}}$",
            c=colors[idx % len(colors)],
            markeredgecolor="gray",
        )
    axL.xaxis.set_major_formatter(percent_formatter)
    axL.text(
        0.08,
        0.98,
        "a) 12 qubits",
        transform=axL.transAxes,
        ha="left",
        va="top",
        fontsize=LABEL_SIZE,
    )
    style_axis(axL)
    axR = axes[1]
    for idx, S in enumerate(sorted(data_20.keys())):
        x_keys, y_vals = data_20[S]
        x_pct = keys_to_percent(x_keys, n=20)
        axR.plot(
            x_pct,
            y_vals,
            marker="o",
            linestyle="-",
            label=f"$2^{{{int(np.log2(S))}}}$",
            c=colors[idx % len(colors)],
            markeredgecolor="gray",
        )
    axR.xaxis.set_major_formatter(percent_formatter)
    axR.text(
        0.02,
        0.75,
        "b) 20 qubits",
        transform=axR.transAxes,
        ha="left",
        va="top",
        fontsize=LABEL_SIZE,
    )
    style_axis(axR)
    handles, labels = axR.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="$x$-batch size",
        title_fontsize=LABEL_SIZE,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    def snug_xlim(ax):
        lines = ax.get_lines()
        if not lines:
            return
        xs = np.concatenate([ln.get_xdata(orig=False) for ln in lines])
        if xs.size:
            span = xs.max() - xs.min()
            pad = 0.03 * (span if span > 0 else max(xs.max(), 1.0))
            ax.set_xlim(xs.min() - pad, xs.max() + pad)

    snug_xlim(axL)
    snug_xlim(axR)
    fig.text(
        0.5,
        -0.01,
        "Number of constrained correlators (\\% of $2^{n}$)",
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    return fig
