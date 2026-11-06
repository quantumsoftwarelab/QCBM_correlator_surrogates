"""Figure 12: surrogate discrepancy heatmaps."""
from born.common import data_path, load_results, set_style
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def make_figure():
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    tn = load_results(data_path("tn_mse_dict.npz"))
    matrices = [
        np.array([[sum(tn[k][b].values()) for b in sorted(tn[k])] for k in sorted(tn)]),
        np.load(data_path("mse_h_p.npy")),
    ]
    for target in ["1d_ising", "2d_haldane"]:
        values = load_results(data_path(f"mse_dict_{target}.npz"))
        matrices.append(
            np.array(
                [
                    [
                        sum(values[k][h].values())
                        if isinstance(values[k][h], dict)
                        else np.sum(values[k][h])
                        for h in sorted(values[k])
                    ]
                    for k in sorted(values)
                ]
            )
        )
    titles = [
        "a) TN / TFIM / strongly entangling",
        "b) PPS / random IQP / IQP",
        "c) PPS / TFIM / IQP",
        "d) PPS / 2D Haldane / IQP",
    ]
    for i, (ax, matrix, title) in enumerate(zip(axes.flat, matrices, titles)):
        im = ax.imshow(matrix, origin="lower", norm=LogNorm(), cmap="YlGn" if i == 0 else "GnBu")
        ax.set_title(title)
        ax.set_xlabel("Bond dimension $\\chi$" if i == 0 else "Number of flips $h$")
        ax.set_ylabel("$k$-order")
        ax.set_xticks(np.arange(matrix.shape[1]), np.arange(1, matrix.shape[1] + 1))
        ax.set_yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
        fig.colorbar(im, ax=ax, label="$\\operatorname{MSE}_k$")
    fig.tight_layout()
    return fig
