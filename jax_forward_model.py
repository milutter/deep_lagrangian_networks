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

except ImportError:
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import deep_lagrangian_networks.jax_DeLaN_model as delan
from deep_lagrangian_networks.replay_memory import ReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env


if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[42, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the DeLaN model")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    rng_key = jax.random.PRNGKey(seed)

    # Construct Hyperparameters:
    hyper = {'n_width': 64,
             'n_depth': 2,
             'n_minibatch': 512,
             'diagonal_epsilon': 0.01,
             'activation': jax.nn.softplus,
             'learning_rate': 5.e-04,
             'weight_decay': 1.e-5,
             'max_epoch': 10000,
             'lagrangian_type': delan.structured_lagrangian_fn,
             # 'lagrangian_type': delan.blackbox_lagrangian_fn,
             }

    model_id = "black_box"
    if hyper['lagrangian_type'].__name__ == 'structured_lagrangian_fn':
        model_id = "structured"

    # Read the dataset:
    n_dof, dt = 2, 1./50.
    train_data, test_data, divider = load_dataset()
    train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g = test_data

    print("\n\n################################################")
    print("Characters:")
    print("   Test Characters = {0}".format(test_labels))
    print("  Train Characters = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    print("\n################################################")
    print("Training Deep Lagrangian Networks (DeLaN):\n")

    # Load existing model parameters:
    t0 = time.perf_counter()

    # Construct Model:
    rng_key, init_key = jax.random.split(rng_key)

    if load_model:
        with open(f"data/delan_{model_id}_model.jax", 'rb') as f:
            data = pickle.load(f)

        hyper = data["hyper"]
        params = data["params"]

    else:
        params = None

    # lagrangian_fn = hk.transform(jax.partial(
    #     hyper['lagrangian_type'],
    #     n_dof=n_dof,
    #     shape=(hyper['n_width'],) * hyper['n_depth'],
    #     activation=hyper['activation'],
    #     epsilon=hyper['diagonal_epsilon'],
    # ))
    #
    # # Initialize Parameters:
    # if params is None:
    #     params = lagrangian_fn.init(init_key, q[0], qd[0])
    #
    # # Trace Model:
    # lagrangian = lagrangian_fn.apply
    # delan_model = jax.jit(jax.partial(delan.forward_model, lagrangian=lagrangian))
    # _ = delan_model(params, None, q[:1], qd[:1], tau[:1])
    # t_build = time.perf_counter() - t0
    # print(f"DeLaN Build Time     = {t_build:.2f}s")

    print("\n################################################")
    print("Evaluating Forward Model:")

    # Convert NumPy samples to torch:
    q, qd, qdd, tau = jnp.array(test_qp), jnp.array(test_qv), jnp.array(test_qa), jnp.array(test_tau)
    p, pd = jnp.array(test_p), jnp.array(test_pd)
    zeros = jnp.zeros_like(q)

    dHdt = jax.vmap(jnp.dot, [0, 0])(qd, tau)

    print("\n################################################")
    print("Plotting Performance:")

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    # y_t_low = np.clip(1.2 * np.min(np.vstack((test_tau, delan_tau)), axis=0), -np.inf, -0.01)
    # y_t_max = np.clip(1.5 * np.max(np.vstack((test_tau, delan_tau)), axis=0), 0.01, np.inf)
    #
    # y_m_low = np.clip(1.2 * np.min(np.vstack((test_m, delan_m)), axis=0), -np.inf, -0.01)
    # y_m_max = np.clip(1.2 * np.max(np.vstack((test_m, delan_m)), axis=0), 0.01, np.inf)
    #
    # y_c_low = np.clip(1.2 * np.min(np.vstack((test_c, delan_c)), axis=0), -np.inf, -0.01)
    # y_c_max = np.clip(1.2 * np.max(np.vstack((test_c, delan_c)), axis=0), 0.01, np.inf)
    #
    # y_g_low = np.clip(1.2 * np.min(np.vstack((test_g, delan_g)), axis=0), -np.inf, -0.01)
    # y_g_max = np.clip(1.2 * np.max(np.vstack((test_g, delan_g)), axis=0), 0.01, np.inf)

    plt.rc('text', usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = np.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    fig = plt.figure(figsize=(24.0/1.54, 8.0/1.54), dpi=100)
    fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
    fig.canvas.set_window_title('Seed = {0}'.format(seed))

    legend = [mp.patches.Patch(color=color_i[0], label="DeLaN"),
              mp.patches.Patch(color="k", label="Ground Truth")]

    # Plot Torque
    ax0 = fig.add_subplot(2, 4, 1)
    ax0.set_title(r"Generalized Position $\mathbf{q}$")
    ax0.text(s=r"\textbf{Joint 0}", x=-0.35, y=.5, fontsize=12, fontweight="bold", rotation=90, horizontalalignment="center", verticalalignment="center", transform=ax0.transAxes)
    ax0.set_ylabel(r"$\mathbf{q}_0$ [Rad]")
    ax0.get_yaxis().set_label_coords(-0.2, 0.5)
    # ax0.set_ylim(y_t_low[0], y_t_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 5)
    ax1.text(s=r"\textbf{Joint 1}", x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
             horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax1.text(s=r"\textbf{(a)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel(r"$\mathbf{q}_1$ [Rad]")
    ax1.get_yaxis().set_label_coords(-0.2, 0.5)
    # ax1.set_ylim(y_t_low[1], y_t_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[-1])

    # ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=1.)

    # Plot Ground Truth Torque:
    ax0.plot(q[:, 0], color="k")
    ax1.plot(q[:, 1], color="k")

    # Plot DeLaN Torque:
    # ax0.plot(delan_tau[:, 0], color=color_i[0], alpha=plot_alpha)
    # ax1.plot(delan_tau[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Mass Torque
    ax0 = fig.add_subplot(2, 4, 2)
    ax0.set_title(r"Generalized Velocity $\dot{\mathbf{q}}$")
    ax0.set_ylabel(r"$\dot{\mathbf{q}}_0$ [Rad/s]")
    # ax0.set_ylim(y_m_low[0], y_m_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 6)
    ax1.text(s=r"\textbf{(b)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel(r"$\dot{\mathbf{q}}_{1}$ [Rad/s]")
    # ax1.set_ylim(y_m_low[1], y_m_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Inertial Torque:
    ax0.plot(qd[:, 0], color="k")
    ax1.plot(qd[:, 1], color="k")

    # Plot DeLaN Inertial Torque:
    # ax0.plot(delan_m[:, 0], color=color_i[0], alpha=plot_alpha)
    # ax1.plot(delan_m[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Coriolis Torque
    ax0 = fig.add_subplot(2, 4, 3)
    ax0.set_title(r"Generalized Momentum $\mathbf{p}$")
    ax0.set_ylabel(r"$\mathbf{p}_0$")
    # ax0.set_ylim(y_c_low[0], y_c_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 7)
    ax1.text(s=r"\textbf{(c)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel(r"$\mathbf{p}_1$")
    # ax1.set_ylim(y_c_low[1], y_c_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Coriolis & Centrifugal Torque:
    ax0.plot(p[:, 0], color="k")
    ax1.plot(p[:, 1], color="k")

    # Plot Gravity
    ax0 = fig.add_subplot(2, 4, 4)
    ax0.set_title(r"Change of Energy $d\mathcal{H}/dt$")
    ax0.set_ylabel("$d\mathcal{H}/dt$")
    # ax0.set_ylim(y_g_low[0], y_g_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax0.plot(dHdt[:], color="k")

    ax1 = fig.add_subplot(2, 4, 8)
    ax1.text(s=r"\textbf{(d)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_frame_on(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=0.)

    # ax1.set_ylabel("Torque [Nm]")
    # # ax1.set_ylim(y_g_low[1], y_g_max[1])
    # ax1.set_xticks(ticks)
    # ax1.set_xticklabels(test_labels)
    # [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    # ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Gravity Torque:
    # ax0.plot(test_g[:, 0], color="k")
    # ax1.plot(test_g[:, 1], color="k")

    # Plot DeLaN Gravity Torque:
    # ax0.plot(delan_g[:, 0], color=color_i[0], alpha=plot_alpha)
    # ax1.plot(delan_g[:, 1], color=color_i[0], alpha=plot_alpha)

    # fig.savefig(f"figures/forward_model_DeLaN_{model_id}_Performance.pdf", format="pdf")
    # fig.savefig(f"figures/forward_model_DeLaN_{model_id}_Performance.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")

