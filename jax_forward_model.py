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

except ImportError:
    pass

import deep_lagrangian_networks.jax_HNN_model as hnn
import deep_lagrangian_networks.jax_DeLaN_model as delan
import deep_lagrangian_networks.jax_Black_Box_model as black_box
from deep_lagrangian_networks.utils import load_dataset, init_env, activations
from deep_lagrangian_networks.jax_integrator import symplectic_euler, explicit_euler, runge_kutta_4

if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[1, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the DeLaN model")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    rng_key = jax.random.PRNGKey(seed)

    time_scaling = 20
    dataset = "uniform"
    integrator_fn = runge_kutta_4

    module = delan
    key = "lagrangian"
    module_key = "DeLaN"
    # model_type = delan.structured_lagrangian_fn
    model_type = delan.blackbox_lagrangian_fn

    # module = hnn
    # key = "hamiltonian"
    # module_key = "HNN"
    # model_type = hnn.structured_hamiltonian_fn
    # model_type = hnn.blackbox_hamiltonian_fn

    # module = black_box
    # key = "black_box_model"
    # module_key = "Network"
    # model_type = black_box.dynamics_network

    model_id = "black_box"
    if model_type.__name__.split("_")[0] == 'structured':
        model_id = "structured"

    # Read the dataset:
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

    train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g = test_data
    n_dof = test_qp.shape[-1]

    # Convert NumPy samples to torch:
    q, qd, qdd, tau = jnp.array(test_qp), jnp.array(test_qv), jnp.array(test_qa), jnp.array(test_tau)
    p, pd = jnp.array(test_p), jnp.array(test_pd)
    dHdt = jax.vmap(jnp.dot, [0, 0])(qd, tau)
    H = dt * jnp.cumsum(dHdt)

    print("\n\n################################################")
    print("Characters:")
    print(f"   Test Characters = {test_labels}")
    print(f"  Train Characters = {train_labels}")
    print(f"# Training Samples = {int(train_qp.shape[0]):05d}")
    print(f"                dt = {dt:.2f}s / {1. / dt:.1f}Hz")
    print("")

    # Training Parameters:
    print("\n################################################")
    print("Run the Forward Model:\n")

    # Load existing model parameters:
    t0 = time.perf_counter()

    # Construct Model:
    rng_key, init_key = jax.random.split(rng_key)

    with open(f"data/{module_key.lower()}_models/{module_key.lower()}_{model_id}_{dataset}_seed_{seed}.jax", 'rb') as f:
        data = pickle.load(f)

    hyper = data["hyper"]
    params = data["params"]

    dynamics_fn = hk.transform(jax.partial(
        hyper[key + '_type'],
        n_dof=n_dof,
        shape=(hyper['n_width'],) * hyper['n_depth'],
        activation=activations[hyper['activation']],
        epsilon=hyper['diagonal_epsilon'],
        shift=hyper['diagonal_shift'],
    ))

    # Initialize Parameters:
    if params is None:
        params = dynamics_fn.init(init_key, q[0], qd[0])

    # Trace Model:
    dynamics_fn = dynamics_fn.apply
    forward_model = jax.jit(jax.partial(module.forward_model, **{key:dynamics_fn, 'n_dof': n_dof}))
    _ = forward_model(params, None, q[:1], qd[:1], tau[:1])
    t_build = time.perf_counter() - t0
    print(f"Model Build Time     = {t_build:.2f}s")

    rollout = jax.jit(jax.partial(
        module.rollout,
        **{key: dynamics_fn,
        "forward_model": forward_model,
        "integrator": integrator_fn,
        "dt": dt / time_scaling}))

    n_steps = np.array(divider[1:]) - np.array(divider[:-1])
    q_pred, qd_pred, p_pred = jnp.zeros((0, 2)), jnp.zeros((0, 2)), jnp.zeros((0, 2))
    # q_error, qd_error, p_error = jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros((0,))
    H_pred = jnp.zeros((0,))

    for i, char in enumerate(test_labels):
        print(f"\nCharacter = {char} - # Steps = {n_steps[i]:03d}")
        q_i, qd_i, p_i = q[divider[i]:divider[i+1]], qd[divider[i]:divider[i+1]], p[divider[i]:divider[i+1]]
        u_i = tau[divider[i]:divider[i+1]]

        # Unroll Trajectory:
        q_i_pred, qd_i_pred, p_i_pred, H_i_pred = rollout(
            params, None, q_i[0:1], qd_i[0:1], p_i[0:1], jnp.repeat(u_i, int(time_scaling), axis=0))

        # Normalize Hamiltonian to start with H(0) = 0
        H_i_pred = H_i_pred - H_i_pred[0]

        # Stack Predictions & Errors:
        (q_pred, qd_pred, p_pred, H_pred) = jax.tree_map(
            lambda x, xi: jnp.concatenate([x, xi[::int(time_scaling)]], axis=0),
            (q_pred, qd_pred, p_pred, H_pred),
            (q_i_pred, qd_i_pred, p_i_pred, H_i_pred)
        )

    # Compute Error:
    (q_error, qd_error, p_error) = jax.tree_map(
        lambda x_pred, x: jnp.sum((x - x_pred) ** 2, axis=-1),
        (q_pred, qd_pred, p_pred),
        (q, qd, p),
    )

    print("\n################################################")
    print("Plotting Performance:")

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    q_low = np.clip(1.5 * np.min(q, axis=0), -np.inf, -0.01)
    q_max = np.clip(1.5 * np.max(q, axis=0), 0.01, np.inf)

    qd_low = np.clip(1.5 * np.min(qd, axis=0), -np.inf, -0.01)
    qd_max = np.clip(1.5 * np.max(qd, axis=0), 0.01, np.inf)

    p_low = np.clip(1.2 * np.min(p, axis=0), -np.inf, -0.01)
    p_max = np.clip(1.2 * np.max(p, axis=0), 0.01, np.inf)

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
    ax0 = fig.add_subplot(3, 4, 1)
    ax0.set_title(r"Generalized Position $\mathbf{q}$")
    ax0.text(s=r"\textbf{Joint 0}", x=-0.35, y=.5, fontsize=12, fontweight="bold", rotation=90, horizontalalignment="center", verticalalignment="center", transform=ax0.transAxes)
    ax0.set_ylabel(r"$\mathbf{q}_0$ [Rad]")
    ax0.get_yaxis().set_label_coords(-0.2, 0.5)
    ax0.set_ylim(q_low[0], q_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(3, 4, 5)
    ax1.text(s=r"\textbf{Joint 1}", x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
             horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel(r"$\mathbf{q}_1$ [Rad]")
    ax1.get_yaxis().set_label_coords(-0.2, 0.5)
    ax1.set_ylim(q_low[1], q_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[-1])

    ax2 = fig.add_subplot(3, 4, 9)
    ax2.text(s=r"\textbf{Error}", x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
             horizontalalignment="center", verticalalignment="center", transform=ax2.transAxes)

    ax2.text(s=r"\textbf{(a)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.get_yaxis().set_label_coords(-0.2, 0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(test_labels)
    [ax2.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax2.set_xlim(divider[0], divider[-1])
    ax2.set_ylim(1e-6, 1e0)
    ax2.set_yscale('log')

    # Plot Ground Truth Torque:
    ax0.plot(q[:, 0], color="k")
    ax1.plot(q[:, 1], color="k")

    # Plot DeLaN Torque:
    ax0.plot(q_pred[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(q_pred[:, 1], color=color_i[0], alpha=plot_alpha)
    ax2.plot(q_error, color=color_i[0], alpha=plot_alpha)

    # Plot Mass Torque
    ax0 = fig.add_subplot(3, 4, 2)
    ax0.set_title(r"Generalized Velocity $\dot{\mathbf{q}}$")
    ax0.set_ylabel(r"$\dot{\mathbf{q}}_0$ [Rad/s]")
    ax0.set_ylim(qd_low[0], qd_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(3, 4, 6)
    ax1.set_ylabel(r"$\dot{\mathbf{q}}_{1}$ [Rad/s]")
    ax1.set_ylim(qd_low[1], qd_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[-1])

    ax2 = fig.add_subplot(3, 4, 10)
    ax2.text(s=r"\textbf{(b)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.get_yaxis().set_label_coords(-0.2, 0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(test_labels)
    [ax2.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax2.set_xlim(divider[0], divider[-1])
    ax2.set_ylim(1e-6, 1e0)
    ax2.set_yscale('log')

    # Plot Ground Truth Inertial Torque:
    ax0.plot(qd[:, 0], color="k")
    ax1.plot(qd[:, 1], color="k")

    # Plot DeLaN Inertial Torque:
    ax0.plot(qd_pred[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(qd_pred[:, 1], color=color_i[0], alpha=plot_alpha)
    ax2.plot(qd_error, color=color_i[0], alpha=plot_alpha)

    # Plot Coriolis Torque
    ax0 = fig.add_subplot(3, 4, 3)
    ax0.set_title(r"Generalized Momentum $\mathbf{p}$")
    ax0.set_ylabel(r"$\mathbf{p}_0$")
    ax0.set_ylim(p_low[0], p_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(3, 4, 7)
    ax1.set_ylabel(r"$\mathbf{p}_1$")
    ax1.set_ylim(p_low[1], p_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    [ax1.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax1.set_xlim(divider[0], divider[-1])

    ax2 = fig.add_subplot(3, 4, 11)
    ax2.text(s=r"\textbf{(c)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.get_yaxis().set_label_coords(-0.2, 0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(test_labels)
    [ax2.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax2.set_xlim(divider[0], divider[-1])
    ax2.set_ylim(1e-6, 1e0)
    ax2.set_yscale('log')

    # Plot Ground Truth Coriolis & Centrifugal Torque:
    ax0.plot(p[:, 0], color="k")
    ax1.plot(p[:, 1], color="k")

    ax0.plot(p_pred[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(p_pred[:, 1], color=color_i[0], alpha=plot_alpha)

    ax2.plot(p_error, color=color_i[0], alpha=plot_alpha)

    # Plot Gravity
    ax0 = fig.add_subplot(3, 4, 4)
    ax0.set_title(r"Normalized Energy $\mathcal{H}$")
    ax0.set_ylabel("$\mathcal{H}$")
    # ax0.set_ylim(y_g_low[0], y_g_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    [ax0.axvline(divider[i], linestyle='--', linewidth=1.0, alpha=1., color="k") for i in range(len(divider))]
    ax0.set_xlim(divider[0], divider[-1])

    ax0.plot(H[:], color="k")
    ax0.plot(H_pred[:], color=color_i[0], alpha=plot_alpha)

    ax2 = fig.add_subplot(3, 4, 12)
    ax2.text(s=r"\textbf{(d)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax2.transAxes)

    ax2.set_frame_on(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=0.)

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

    # fig.savefig(f"figures/forward_model_{module_key}_{model_id}_Performance.pdf", format="pdf")
    # fig.savefig(f"figures/forward_model_{module_key}_{model_id}_Performance.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")

