"""Figure 17: two-qubit parameter dependence."""
from born.common import set_style
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, Colormap


def make_figure():
    set_style()
    LABEL_SIZE = 18

    def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=256):
        """Accepts either a Matplotlib colormap object or a registered name."""
        base = plt.get_cmap(cmap) if not isinstance(cmap, Colormap) else cmap
        new_colors = base(np.linspace(minval, maxval, n))
        return LinearSegmentedColormap.from_list(
            f'trunc({getattr(base, "name", "cmap")},{minval:.2f},{maxval:.2f})', new_colors
        )

    cmap_full = sns.color_palette("Spectral", as_cmap=True)
    cmap_poshalf = truncate_cmap(cmap_full, 0.5, 1.0)
    theta1 = np.linspace(0, 2 * np.pi, 200)
    theta2 = np.linspace(0, 2 * np.pi, 200)
    T1, T2 = np.meshgrid(theta1, theta2)
    Z1 = np.cos(T1)
    Z12 = np.cos(T2)
    Z2 = Z1 * Z12
    P00 = (1 + Z1 + Z2 + Z12) / 4.0
    P01 = (1 + Z1 - Z2 - Z12) / 4.0
    P10 = (1 - Z1 + Z2 - Z12) / 4.0
    P11 = (1 - Z1 - Z2 + Z12) / 4.0
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    gs = fig.add_gridspec(2, 8, wspace=1, hspace=0.0)
    ax_t0 = fig.add_subplot(gs[0, 1:3])
    ax_t1 = fig.add_subplot(gs[0, 3:5], sharex=ax_t0, sharey=ax_t0)
    ax_t2 = fig.add_subplot(gs[0, 5:7], sharex=ax_t0, sharey=ax_t0)
    ax_b0 = fig.add_subplot(gs[1, 0:2])
    ax_b1 = fig.add_subplot(gs[1, 2:4], sharex=ax_b0, sharey=ax_b0)
    ax_b2 = fig.add_subplot(gs[1, 4:6], sharex=ax_b0, sharey=ax_b0)
    ax_b3 = fig.add_subplot(gs[1, 6:8], sharex=ax_b0, sharey=ax_b0)
    extent = [0, 2 * np.pi, 0, 2 * np.pi]
    tick_positions = [0, np.pi, 2 * np.pi]
    tick_labels = ["$0$", "$\\pi$", "$2\\pi$"]
    ax_t0.imshow(Z1, origin="lower", extent=extent, cmap=cmap_full, vmin=-1, vmax=1)
    ax_t1.imshow(Z2, origin="lower", extent=extent, cmap=cmap_full, vmin=-1, vmax=1)
    im_t2 = ax_t2.imshow(Z12, origin="lower", extent=extent, cmap=cmap_full, vmin=-1, vmax=1)
    ax_t0.set_title("$\\langle Z_1\\rangle$", pad=10)
    ax_t1.set_title("$\\langle Z_2\\rangle$", pad=10)
    ax_t2.set_title("$\\langle Z_1 Z_2\\rangle$", pad=10)
    for ax in (ax_t0, ax_t1, ax_t2):
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_yticks(tick_positions, tick_labels)
        ax.set_xlabel("$\\theta_1$")
        ax.set_ylabel("$\\theta_2$")
    cbar_top = fig.colorbar(im_t2, ax=[ax_t0, ax_t1, ax_t2], pad=0.02, shrink=0.5)
    for spine in cbar_top.ax.spines.values():
        spine.set_color("lightgrey")
    cbar_top.ax.tick_params(color="lightgray", labelsize=LABEL_SIZE)
    norm01 = Normalize(vmin=0.0, vmax=1.0)
    ax_b0.imshow(P00, origin="lower", extent=extent, cmap=cmap_poshalf, norm=norm01)
    ax_b1.imshow(P01, origin="lower", extent=extent, cmap=cmap_poshalf, norm=norm01)
    ax_b2.imshow(P10, origin="lower", extent=extent, cmap=cmap_poshalf, norm=norm01)
    im_b3 = ax_b3.imshow(P11, origin="lower", extent=extent, cmap=cmap_poshalf, norm=norm01)
    ax_b0.set_title("$\\mathrm{Pr}(00)$", pad=10)
    ax_b1.set_title("$\\mathrm{Pr}(01)$", pad=10)
    ax_b2.set_title("$\\mathrm{Pr}(10)$", pad=10)
    ax_b3.set_title("$\\mathrm{Pr}(11)$", pad=10)
    for ax in (ax_b0, ax_b1, ax_b2, ax_b3):
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_yticks(tick_positions, tick_labels)
        ax.set_xlabel("$\\theta_1$")
        ax.set_ylabel("$\\theta_2$")
        for spine in ax.spines.values():
            spine.set_color("lightgrey")
        ax.tick_params(axis="both", which="both", color="lightgray", labelsize=LABEL_SIZE)
    for ax in (ax_t0, ax_t1, ax_t2):
        for spine in ax.spines.values():
            spine.set_color("lightgrey")
        ax.tick_params(axis="both", which="both", color="lightgray", labelsize=LABEL_SIZE)
    cbar_bottom = fig.colorbar(im_b3, ax=[ax_b0, ax_b1, ax_b2, ax_b3], pad=0.02, shrink=0.5)
    cbar_bottom.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    for spine in cbar_bottom.ax.spines.values():
        spine.set_color("lightgrey")
    cbar_bottom.ax.tick_params(color="lightgray", labelsize=LABEL_SIZE)
    ax_t1.set_ylabel("")
    ax_t2.set_ylabel("")
    for ax in (ax_t1, ax_t2, ax_b1, ax_b2, ax_b3):
        ax.label_outer()
    for ax in (ax_t0, ax_t1, ax_t2):
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("$\\theta_1$")
    return fig
