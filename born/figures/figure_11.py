"""Figure 11: surrogate training curves."""
from born.common import data_path, set_style
import os, re
import numpy as np
import matplotlib.pyplot as plt


def make_figure():
    set_style()
    LABEL_SIZE = 18
    bond_dims = [4, 5, 6, 7, 8]
    tn_colors = plt.cm.YlGn(np.linspace(0.2, 1, len(bond_dims) + 1))[::-1]
    loss_histories, loss_means = ({}, {})
    for bond_dim in bond_dims:
        loss_histories[bond_dim] = np.load(
            data_path(f"TN_data/bond_dim_{bond_dim}_loss_histories.npy")
        )
        loss_means[bond_dim] = np.mean(loss_histories[bond_dim], axis=0)
    in_dir = "runs_histories"
    npz_path = os.path.join(in_dir, "histories_summary.npz")
    data = np.load(data_path(npz_path), allow_pickle=False)
    kh_pairs = []
    for key in data.files:
        m = re.match("H_k(\\d+)_h(\\d+)$", key)
        if m:
            kh_pairs.append((int(m.group(1)), int(m.group(2))))
    if not kh_pairs:
        raise ValueError("No keys of the form 'H_k{K}_h{H}' found in the NPZ.")
    max_k_list = sorted({k for k, _ in kh_pairs})
    h_by_k = {k: sorted([h for kk, h in kh_pairs if kk == k]) for k in max_k_list}
    k_to_show = max_k_list[-1]
    h_list = h_by_k[k_to_show]

    def get_log_stats(k, h):
        mu_key = f"MUlog_k{k}_h{h}"
        sd_key = f"SDlog_k{k}_h{h}"
        if mu_key in data.files and sd_key in data.files:
            mu_log = data[mu_key]
            sd_log = data[sd_key]
            return (mu_log, sd_log)
        H_key = f"H_k{k}_h{h}"
        if H_key not in data.files:
            raise KeyError(f"Missing {mu_key}/{sd_key} and {H_key}; cannot reconstruct stats.")
        H = data[H_key]
        if np.any(H <= 0):
            eps = float(data["eps_loss"]) if "eps_loss" in data.files else 1e-30
            H = np.maximum(H, eps)
        logH = np.log(H)
        mu_log = logH.mean(axis=0)
        sd_log = logH.std(axis=0, ddof=0)
        return (mu_log, sd_log)

    kh_colors = plt.cm.GnBu(np.linspace(0.2, 1, max(len(h_list), 5) + 1))[::-1]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 5), sharey=True, gridspec_kw={"wspace": 0.3})
    axL.grid(True)
    for i, bond_dim in enumerate(bond_dims):
        mean = loss_means[bond_dim]
        std = np.std(loss_histories[bond_dim], axis=0)
        T = np.arange(len(mean))
        axL.plot(T, mean, label=f"$\\chi\\!=\\!{bond_dim}$", color=tn_colors[i])
        axL.fill_between(T, mean - std, mean + std, color=tn_colors[i], alpha=0.3)
    axL.set_xlim(0, 200)
    axL.set_yscale("log")
    axL.set_ylim(1e-05, 1)
    axL.set_xlabel("Iteration")
    axL.set_ylabel(
        "Averaged Training MMD Loss, $\\mathcal{L}(\\operatorname{Pr}_{\\mathcal{C}\\!\\ell}(\\boldsymbol\\theta),\\operatorname{Pr}^*)$"
    )
    for spine in axL.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    axL.tick_params(axis="both", which="both", color="lightgray", length=4)
    axL.legend(
        title="Bond dimension $\\chi$",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.21),
        ncol=len(bond_dims),
        frameon=False,
        title_fontsize=LABEL_SIZE,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=0.8,
    )
    axL.text(
        -0.12,
        1.26,
        "a) TN: TFIM dataset, strongly entangling ansatz",
        transform=axL.transAxes,
        va="top",
        ha="left",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    axR.grid(True)
    T_right = None
    for i, h in enumerate(h_list):
        mu_log, sd_log = get_log_stats(k_to_show, h)
        geom_mean = np.exp(mu_log)
        lower_band = np.exp(mu_log - sd_log)
        upper_band = np.exp(mu_log + sd_log)
        if T_right is None:
            T_right = np.arange(len(geom_mean))
        axR.plot(T_right, geom_mean, label=f"$h\\!=\\!{h}$", color=kh_colors[i % len(kh_colors)])
        axR.fill_between(
            T_right, lower_band, upper_band, color=kh_colors[i % len(kh_colors)], alpha=0.3
        )
    axR.set_xlim(0, 200)
    axR.set_yscale("log")
    axR.set_ylim(1e-05, 1)
    axR.set_xlabel("Iteration")
    for spine in axR.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)
    axR.tick_params(axis="both", which="both", color="lightgray", length=4)
    axR.legend(
        title="Number of flips $h$",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.21),
        ncol=min(5, max(1, len(h_list))),
        frameon=False,
        title_fontsize=LABEL_SIZE,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=0.8,
    )
    axR.text(
        -0.08,
        1.26,
        "b) PPS: TFIM dataset, IQP ansatz",
        transform=axR.transAxes,
        va="top",
        ha="left",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    axR.text(180, 0.0012, "$h=4,6$", fontsize=LABEL_SIZE, va="center", ha="right", color="black")
    axR.text(175, 2e-05, "$h=8,10$", fontsize=LABEL_SIZE, va="center", ha="right", color="black")
    fig.tight_layout()
    return fig
