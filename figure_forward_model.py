import argparse
import sys
import optax
import torch
import numpy as np
import time
import jax
import jax.numpy as jnp
import matplotlib as mp
import haiku as hk
import dill as pickle

try:
    mp.use("Qt5Agg")
    mp.rc('text', usetex=True)
    mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm

except ImportError:
    pass

import deep_lagrangian_networks.jax_HNN_model as hnn
import deep_lagrangian_networks.jax_DeLaN_model as delan
import deep_lagrangian_networks.jax_Black_Box_model as black_box
from deep_lagrangian_networks.utils import load_dataset, init_env, activations
from deep_lagrangian_networks.jax_integrator import symplectic_euler, explicit_euler, runge_kutta_4


def running_mean(x, n):
  cumsum = np.cumsum(np.concatenate([x[0] * np.ones((n,)), x]))
  return (cumsum[n:] - cumsum[:-n]) / n


if __name__ == "__main__":
    n_plot = 5
    dataset = "uniform"
    model_id = ["structured", "black_box", "structured", "black_box", "black_box"]
    module_key = ["DeLaN", "DeLaN", "HNN", "HNN", "Network"]

    colors = {
        "DeLaN structured": cm.get_cmap(cm.Set1)(0),
        "DeLaN black_box": cm.get_cmap(cm.Set1)(1),
        "HNN structured": cm.get_cmap(cm.Set1)(2),
        "HNN black_box": cm.get_cmap(cm.Set1)(3),
        "Network black_box": cm.get_cmap(cm.Set1)(4),
    }

    results = {}
    for i in range(n_plot):
        with open(f"data/results/{module_key[i]}_{model_id[i]}_{dataset}.pickle", "rb") as file:
            results[module_key[i] + " " + model_id[i]] = pickle.load(file)

    if dataset == "char":
        train_data, test_data, divider, dt = load_dataset(
            filename="data/character_data.pickle",
            test_label=["e", "q", "v"])

    elif dataset == "uniform":
        train_data, test_data, divider, dt = load_dataset(
            filename="data/uniform_data.pickle",
            test_label=["Test 0", "Test 1", "Test 2"])

    else:
        raise ValueError

    vpt_th = 1.e-2
    for i in range(n_plot):
        key = f"{module_key[i]} {model_id[i]}"
        n_seeds = results[key]['forward_model']['q_error'].shape[0]
        xd_error = np.mean(results[key]['forward_model']['xd_error']), 2. * np.std(results[key]['forward_model']['xd_error'])

        n_test = 2
        vpt = np.zeros((0, n_test))
        for i in range(n_seeds):
            vpt_i = []
            for j in range(n_test):
                traj = np.concatenate([
                    results[key]['forward_model']['q_error'][i, divider[j]:divider[j+1]],
                    results[key]['forward_model']['q_error'][i, -1:] * 0.0 + 1.])
                vpt_i = vpt_i + [np.argwhere(traj >= vpt_th)[0, 0]]
            vpt = np.concatenate([vpt, np.array([vpt_i])])

        vpt = np.mean(vpt), np.std(vpt)

        unit = r"\text{s}"
        string = f"${xd_error[0]:.1e}{'}'} \pm {xd_error[1]:.1e}{'}'}$    &    ${vpt[0]*dt:.2f}{unit} \pm {vpt[1]*dt:.2f}{unit}$ \\\\".replace("e-", r"\mathrm{e}{-").replace("e+", r"\mathrm{e}{+")
        print(f"{key:20} -     " + string)

    test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g = test_data
    tau_g, tau_c, tau_m, tau = jnp.array(test_g), jnp.array(test_c), jnp.array(test_m), jnp.array(test_tau)
    q, qd, qdd = jnp.array(test_qp), jnp.array(test_qv), jnp.array(test_qa)
    p, pd = jnp.array(test_p), jnp.array(test_pd)
    dHdt = jax.vmap(jnp.dot, [0, 0])(qd, tau)
    H = jnp.concatenate([dt * jnp.cumsum(dHdt[divider[i]: divider[i+1]]) for i in range(3)])

    def smoothing(x):
        return np.concatenate([running_mean(x[divider[i]:divider[i + 1]], 10) for i in range(3)])

    print("\n################################################")
    print("Plotting Performance:")

    # Alpha of the graphs:
    plot_alpha = 0.8
    y_offset = -0.15
    n_test = 2

    # Plot the performance:
    q_low = np.clip(1.5 * np.min(np.array(q), axis=0), -np.inf, -0.01)
    q_max = np.clip(1.5 * np.max(np.array(q), axis=0), 0.01, np.inf)

    if dataset == "char":
        q_max = np.array([0.25, 3.])
        q_low = np.array([-1.25, 1.])

    qd_low = np.clip(1.5 * np.min(qd, axis=0), -np.inf, -0.01)
    qd_max = np.clip(1.5 * np.max(qd, axis=0), 0.01, np.inf)

    p_low = np.clip(1.2 * np.min(p, axis=0), -np.inf, -0.01)
    p_max = np.clip(1.2 * np.max(p, axis=0), 0.01, np.inf)

    H_lim = [-0.01, +0.01] if dataset == "uniform" else [-2.75, +2.75]
    err_min, err_max = 1.e-5, 1.e3

    plt.rc('text', usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = np.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    fig = plt.figure(figsize=(24.0 / 1.54, 8.0 / 1.54), dpi=100)
    fig.subplots_adjust(left=0.06, bottom=0.12, right=0.98, top=0.95, wspace=0.24, hspace=0.2)
    fig.canvas.set_window_title('')

    legend = [
        mp.patches.Patch(color=colors["DeLaN structured"], label="DeLaN - Structured Lagrangian"),
        mp.patches.Patch(color=colors["DeLaN black_box"], label="DeLaN - Black-Box Lagrangian"),
        mp.patches.Patch(color=colors["HNN structured"], label="HNN - Structured Hamiltonian"),
        mp.patches.Patch(color=colors["HNN black_box"], label="HNN - Black-Box Hamiltonian"),
        mp.patches.Patch(color=colors["Network black_box"], label="Feed-Forward Network"),
        mp.patches.Patch(color="k", label="Ground Truth")]

    ax0 = fig.add_subplot(3, 4, 1)
    ax0.set_title(r"Generalized Position $\mathbf{q}$")
    ax0.text(s=r"\textbf{Joint 0}", x=-0.25, y=.5, fontsize=12, fontweight="bold", rotation=90,
             horizontalalignment="center", verticalalignment="center", transform=ax0.transAxes)
    ax0.set_ylabel(r"$\mathbf{q}_0$ [Rad]")
    ax0.get_yaxis().set_label_coords(-0.2, 0.5)
    ax0.set_ylim(q_low[0], q_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[n_test])
    ax0.yaxis.set_label_coords(y_offset, 0.5)

    ax1 = fig.add_subplot(3, 4, 5)
    ax1.text(s=r"\textbf{Joint 1}", x=-.25, y=0.5, fontsize=12, fontweight="bold", rotation=90,
             horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel(r"$\mathbf{q}_1$ [Rad]")
    ax1.get_yaxis().set_label_coords(-0.2, 0.5)
    ax1.set_ylim(q_low[1], q_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[n_test])
    ax1.yaxis.set_label_coords(y_offset, 0.5)

    ax2 = fig.add_subplot(3, 4, 9)
    ax2.text(s=r"\textbf{Error}", x=-.25, y=0.5, fontsize=12, fontweight="bold", rotation=90,
             horizontalalignment="center", verticalalignment="center", transform=ax2.transAxes)

    ax2.text(s=r"\textbf{(a)}", x=.5, y=-0.35, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.get_yaxis().set_label_coords(-0.2, 0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(test_labels)
    [ax2.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax2.set_xlim(divider[0], divider[n_test])
    ax2.set_ylim(err_min, err_max)
    ax2.set_yscale('log')
    ax2.set_ylabel(r"Position Error")
    ax2.yaxis.set_label_coords(y_offset, 0.5)
    ax2.axhline(vpt_th, color="k", linestyle="--")

    # Plot Ground Truth Torque:
    ax0.plot(q[:, 0], color="k")
    ax1.plot(q[:, 1], color="k")

    # Plot DeLaN Torque:
    for key in results.keys():
        color = colors[key]

        q_pred = results[key]["forward_model"]["q_pred"]
        q_error = results[key]["forward_model"]["q_error"]

        q_pred_min, q_pred_mean, q_pred_max = np.min(q_pred, axis=0), np.median(q_pred, axis=0), np.max(q_pred, axis=0)
        q_error_min, q_error_mean, q_error_max = np.min(q_error, axis=0), np.median(q_error, axis=0), np.max(q_error, axis=0)

        q_error_min = smoothing(q_error_min)
        q_error_mean = smoothing(q_error_mean)
        q_error_max = smoothing(q_error_max)

        x = np.arange(q_pred_max.shape[0])

        ax0.plot(q_pred_mean[:, 0], color=color, alpha=plot_alpha)
        ax0.fill_between(x, q_pred_min[:, 0], q_pred_max[:, 0], color=color, alpha=plot_alpha/8.)

        ax1.plot(q_pred_mean[:, 1], color=color, alpha=plot_alpha)
        ax1.fill_between(x, q_pred_min[:, 1], q_pred_max[:, 1], color=color, alpha=plot_alpha/8.)

        ax2.plot(q_error_mean, color=color, alpha=plot_alpha)
        ax2.fill_between(x, q_error_min, q_error_max, color=color, alpha=plot_alpha/8.)

    # Plot Mass Torque
    ax0 = fig.add_subplot(3, 4, 2)
    ax0.set_title(r"Generalized Velocity $\dot{\mathbf{q}}$")
    ax0.set_ylabel(r"$\dot{\mathbf{q}}_0$ [Rad/s]")
    ax0.set_ylim(qd_low[0], qd_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[n_test])
    ax0.yaxis.set_label_coords(y_offset, 0.5)

    ax1 = fig.add_subplot(3, 4, 6)
    ax1.set_ylabel(r"$\dot{\mathbf{q}}_{1}$ [Rad/s]")
    ax1.set_ylim(qd_low[1], qd_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[n_test])
    ax1.yaxis.set_label_coords(y_offset, 0.5)

    ax2 = fig.add_subplot(3, 4, 10)
    ax2.text(s=r"\textbf{(b)}", x=.5, y=-0.35, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.get_yaxis().set_label_coords(-0.2, 0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(test_labels)
    [ax2.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax2.set_xlim(divider[0], divider[n_test])
    ax2.set_ylim(err_min, err_max)
    ax2.set_yscale('log')
    ax2.set_ylabel(r"Velocity Error")
    ax2.yaxis.set_label_coords(y_offset, 0.5)

    # Plot Ground Truth Inertial Torque:
    ax0.plot(qd[:, 0], color="k")
    ax1.plot(qd[:, 1], color="k")

    # Plot DeLaN Inertial Torque:
    for key in results.keys():
        color = colors[key]
        qd_pred = results[key]["forward_model"]["qd_pred"]
        qd_error = results[key]["forward_model"]["qd_error"]

        qd_pred_min, qd_pred_mean, qd_pred_max = np.min(qd_pred, axis=0), np.median(qd_pred, axis=0), np.max(qd_pred, axis=0)
        qd_error_min, qd_error_mean, qd_error_max = np.min(qd_error, axis=0), np.median(qd_error, axis=0), np.max(qd_error, axis=0)
        x = np.arange(qd_pred_max.shape[0])

        qd_error_min = smoothing(qd_error_min)
        qd_error_mean = smoothing(qd_error_mean)
        qd_error_max = smoothing(qd_error_max)

        ax0.plot(qd_pred_mean[:, 0], color=color, alpha=plot_alpha)
        ax0.fill_between(x, qd_pred_min[:, 0], qd_pred_max[:, 0], color=color, alpha=plot_alpha/8.)

        ax1.plot(qd_pred_mean[:, 1], color=color, alpha=plot_alpha)
        ax1.fill_between(x, qd_pred_min[:, 1], qd_pred_max[:, 1], color=color, alpha=plot_alpha/8.)

        ax2.plot(qd_error_mean, color=color, alpha=plot_alpha)
        ax2.fill_between(x, qd_error_min, qd_error_max, color=color, alpha=plot_alpha/8.)

    # Plot Coriolis Torque
    ax0 = fig.add_subplot(3, 4, 3)
    ax0.set_title(r"Generalized Momentum $\mathbf{p}$")
    ax0.set_ylabel(r"$\mathbf{p}_0$")
    ax0.set_ylim(p_low[0], p_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[n_test])
    ax0.yaxis.set_label_coords(y_offset, 0.5)

    ax1 = fig.add_subplot(3, 4, 7)
    ax1.set_ylabel(r"$\mathbf{p}_1$")
    ax1.set_ylim(p_low[1], p_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[n_test])
    ax1.yaxis.set_label_coords(y_offset, 0.5)

    ax2 = fig.add_subplot(3, 4, 11)
    ax2.text(s=r"\textbf{(c)}", x=.5, y=-0.35, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.get_yaxis().set_label_coords(-0.2, 0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(test_labels)
    [ax2.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax2.set_xlim(divider[0], divider[n_test])
    ax2.set_ylim(err_min, err_max)
    ax2.set_yscale('log')
    ax2.set_ylabel(r"Impulse Error")
    ax2.yaxis.set_label_coords(y_offset, 0.5)

    # Plot Ground Truth Coriolis & Centrifugal Torque:
    ax0.plot(p[:, 0], color="k")
    ax1.plot(p[:, 1], color="k")

    for key in results.keys():
        color = colors[key]
        p_pred = results[key]["forward_model"]["p_pred"]
        p_error = results[key]["forward_model"]["p_error"]

        p_pred_min, p_pred_mean, p_pred_max = np.min(p_pred, axis=0), np.median(p_pred, axis=0), np.max(p_pred, axis=0)
        p_error_min, p_error_mean, p_error_max = np.min(p_error, axis=0), np.median(p_error, axis=0), np.max(p_error, axis=0)
        x = np.arange(p_pred_max.shape[0])

        p_error_min = smoothing(p_error_min)
        p_error_mean = smoothing(p_error_mean)
        p_error_max = smoothing(p_error_max)

        ax0.plot(p_pred_mean[:, 0], color=color, alpha=plot_alpha)
        ax0.fill_between(x, p_pred_min[:, 0], p_pred_max[:, 0], color=color, alpha=plot_alpha/8.)

        ax1.plot(p_pred_mean[:, 1], color=color, alpha=plot_alpha)
        ax1.fill_between(x, p_pred_min[:, 1], p_pred_max[:, 1], color=color, alpha=plot_alpha/8.)

        ax2.plot(p_error_mean, color=color, alpha=plot_alpha)
        ax2.fill_between(x, p_error_min, p_error_max, color=color, alpha=plot_alpha/8.)


    # Plot Gravity
    ax0 = fig.add_subplot(3, 4, 4)
    ax0.set_title(r"Normalized Energy $\mathcal{H}$")
    ax0.set_ylabel("$\mathcal{H}$")
    ax0.yaxis.set_label_coords(y_offset, 0.5)

    ax0.set_ylim(H_lim[0], H_lim[1])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[n_test])

    ax0.plot(H[:], color="k")

    for key in results.keys():
        if key == "Network black_box":
            continue

        color = colors[key]
        H_pred = results[key]["forward_model"]["H_pred"]
        H_pred_min, H_pred_mean, H_pred_max = np.min(H_pred, axis=0), np.median(H_pred, axis=0), np.max(H_pred, axis=0)
        x = np.arange(H_pred_max.shape[0])

        ax0.plot(H_pred_mean[:], color=color, alpha=plot_alpha)
        ax0.fill_between(x, H_pred_min[:], H_pred_max[:], color=color, alpha=plot_alpha/8.)

    ax2 = fig.add_subplot(3, 4, 12)
    ax2.text(s=r"\textbf{(d)}", x=.5, y=-0.35, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.set_frame_on(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(handles=legend, bbox_to_anchor=(-0.0375, 2.1), loc='upper left', ncol=1, framealpha=0., labelspacing=1.0)

    # fig.savefig(f"figures/forward_model_{module_key}_{model_id}_Performance.pdf", format="pdf")
    # fig.savefig(f"figures/forward_model_{module_key}_{model_id}_Performance.png", format="png")
    print("\n################################################\n\n\n")
    plt.show()

