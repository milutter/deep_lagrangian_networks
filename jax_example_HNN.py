import argparse
import optax
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
from deep_lagrangian_networks.replay_memory import ReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env, activations, get_params

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'

if __name__ == "__main__":
    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[0, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the DeLaN model")
    parser.add_argument("-d", nargs=1, type=str, required=False, default=['uniform', ], help="Dataset")
    parser.add_argument("-t", nargs=1, type=str, required=False, default=['structured', ], help="Hamiltonian Type")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())
    rng_key = jax.random.PRNGKey(seed)

    dataset = str(parser.parse_args().d[0])
    model_id = str(parser.parse_args().t[0])

    # Construct Hyperparameters:
    if model_id == "structured":
        hamiltonian_type = hnn.structured_hamiltonian_fn

    elif model_id == "black_box":
        hamiltonian_type = hnn.blackbox_hamiltonian_fn

    else:
        raise ValueError

    # Construct Hyperparameters:
    hyper = {
        'dataset': dataset,
        'n_width': 64,
        'n_depth': 2,
        'n_minibatch': 512,
        'diagonal_epsilon': 0.1,
        'diagonal_shift': 0.0,
        'activation': 'tanh',
        'learning_rate': 1.e-04,
        'weight_decay': 1.e-5,
        'max_epoch': int(2.5 * 1e3) if dataset == "uniform" else int(5 * 1e3),
        'hamiltonian_type': hamiltonian_type,
        }

    model_id = "black_box"
    if hyper['hamiltonian_type'].__name__ == 'structured_hamiltonian_fn':
        model_id = "structured"

    if load_model:
        with open(f"data/hnn_models/hnn_{model_id}_{hyper['dataset']}_seed_{seed}.jax", 'rb') as f:
            data = pickle.load(f)

        hyper = data["hyper"]
        params = data["params"]

    else:
        params = None

    # Read the dataset:
    if hyper['dataset'] == "char":
        train_data, test_data, divider, dt = load_dataset(
            filename="data/character_data.pickle",
            test_label=["e", "q", "v"])

    elif hyper['dataset'] == "uniform":
        train_data, test_data, divider, dt = load_dataset(
            filename="data/uniform_data.pickle",
            test_label=["Test 0", "Test 1", "Test 2"])

    else:
        raise ValueError

    train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g = test_data
    n_dof = test_qp.shape[-1]

    # Generate Replay Memory:
    mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,), (n_dof,))
    mem = ReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim)
    mem.add_samples([train_qp, train_qv, train_p, train_pd, train_tau])

    print("\n\n################################################")
    print("Characters:")
    print("   Test Characters = {0}".format(test_labels))
    print("  Train Characters = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    print("\n################################################")
    print("Training Hamiltonian Neural Networks (HNN):\n")

    # Construct HNN:
    t0 = time.perf_counter()

    hamiltonian_fn = hk.transform(jax.partial(
        hyper['hamiltonian_type'],
        n_dof=n_dof,
        shape=(hyper['n_width'],) * hyper['n_depth'],
        activation=activations[hyper['activation']],
        epsilon=hyper['diagonal_epsilon'],
        shift=hyper['diagonal_shift'],
    ))

    q, qd, p, pd, tau = [jnp.array(x) for x in next(iter(mem))]
    rng_key, init_key = jax.random.split(rng_key)

    # Initialize Parameters:
    if params is None:
        params = hamiltonian_fn.init(init_key, q[0], qd[0])

    # Trace Model:
    hamiltonian = hamiltonian_fn.apply
    vmap_hamiltonian = jax.vmap(hamiltonian, [None, None, 0, 0])
    hnn_model = jax.jit(jax.partial(hnn.dynamics_model, hamiltonian=hamiltonian, n_dof=n_dof))
    _ = hnn_model(params, None, q[:5], p[:5], pd[:5], tau[:5])

    t_build = time.perf_counter() - t0
    print(f"HNN Build Time       = {t_build:.2f}s")

    # Generate & Initialize the Optimizer:
    t0 = time.perf_counter()

    optimizer = optax.adamw(
        learning_rate=hyper['learning_rate'],
        weight_decay=hyper['weight_decay']
    )

    opt_state = optimizer.init(params)
    loss_fn = jax.partial(
        hnn.forward_loss_fn,
        hamiltonian=hamiltonian,
        n_dof=n_dof,
        norm_tau=jnp.var(train_tau, axis=0),
        norm_qd=jnp.var(train_qv, axis=0),
        norm_pd=jnp.var(train_pd, axis=0),
    )

    def update_fn(params, opt_state, q, qd, p, pd, tau):
        (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, qd, p, pd, tau)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, logs

    update_fn = jax.jit(update_fn)
    _, _, logs = update_fn(params, opt_state, q[:1], qd[:1], p[:1], pd[:1], tau[:1])


    t_build = time.perf_counter() - t0
    print(f"Optimizer Build Time = {t_build:.2f}s")

    # Start Training Loop:
    t0_start = time.perf_counter()

    print("")
    epoch_i = 0
    while epoch_i < hyper['max_epoch'] and not load_model:

        n_batches = 0
        logs = jax.tree_map(lambda x: x * 0.0, logs)

        for data_batch in mem:
            t0_batch = time.perf_counter()

            q, qd, p, pd, tau = [jnp.array(x) for x in data_batch]
            params, opt_state, batch_logs = update_fn(params, opt_state, q, qd, p, pd, tau)

            # Update logs:
            n_batches += 1
            logs = jax.tree_map(lambda x, y: x + y, logs, batch_logs)
            t_batch = time.perf_counter() - t0_batch

        # Update Epoch Loss & Computation Time:
        epoch_i += 1
        logs = jax.tree_map(lambda x: x/n_batches, logs)

        if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print(f"Time = {time.perf_counter() - t0_start:05.1f}s", end=", ")
            print(f"Loss = {logs['loss']:.1e}", end=", ")
            print(f"Inv = {logs['inverse_mean']:.1e} \u00B1 {1.96 * np.sqrt(logs['inverse_var']):.1e}", end=", ")
            print(f"For = {logs['forward_mean']:.1e} \u00B1 {1.96 * np.sqrt(logs['forward_var']):.1e}", end=", ")
            print(f"Power = {logs['energy_mean']:.1e} \u00B1 {1.96 * np.sqrt(logs['energy_var']):.1e}")

    # Save the Model:
    if save_model:
        with open(f"data/hnn_models/hnn_{model_id}_{hyper['dataset']}_seed_{seed}.jax", "wb") as file:
            pickle.dump(
                {"epoch": epoch_i,
                 "hyper": hyper,
                 "params": params,
                 "seed": seed,
                 },
                file)

    print("\n################################################")
    print("Evaluating HNN:")

    # Convert NumPy samples to Jax:
    q, qd, qdd = jnp.array(test_qp), jnp.array(test_qv), jnp.array(test_qa)
    p, pd = jnp.array(test_p), jnp.array(test_pd)
    zeros = jnp.zeros_like(q)

    # Compute Mass Matrix by approximation from BLACK-BOX Hamiltonian:
    e10 = jnp.stack([jnp.ones(q.shape[0]), jnp.zeros(q.shape[0])], axis=-1)
    e01 = jnp.stack([jnp.zeros(q.shape[0]), jnp.ones(q.shape[0])], axis=-1)
    e11 = jnp.ones_like(q)

    V = vmap_hamiltonian(params, None, q, zeros)

    inv_mass_mat_00 = vmap_hamiltonian(params, None, q, e10) - V
    inv_mass_mat_11 = vmap_hamiltonian(params, None, q, e01) - V
    inv_mass_mat_01 = 1./2. * (vmap_hamiltonian(params, None, q, e11) - V - inv_mass_mat_00 - inv_mass_mat_11)

    inv_mass_mat = 2. * jnp.stack([
        jnp.stack([inv_mass_mat_00, inv_mass_mat_01], axis=-1),
        jnp.stack([inv_mass_mat_01, inv_mass_mat_11], axis=-1)],
        axis=-1)

    mass_mat = jax.vmap(jnp.linalg.inv, (0,))(inv_mass_mat)

    # Compute Mass Matrix from STRUCTURED Hamiltonian:
    if hyper['hamiltonian_type'].__name__ == 'structured_hamiltonian_fn':
        mass_matrix_fn = jax.partial(
            hnn.inv_mass_matrix_fn,
            n_dof=n_dof,
            shape=(hyper['n_width'],) * hyper['n_depth'],
            activation=activations[hyper['activation']],
            epsilon=hyper['diagonal_epsilon'],
            shift=hyper['diagonal_shift'],)

        params_mass_matrix, params_potential_energy = get_params(params, "mass_matrix")
        vmap_mass_matrix_fn = jax.vmap(hk.transform(mass_matrix_fn).apply, [None, None, 0])
        pred_inv_mass_mat = vmap_mass_matrix_fn(params_mass_matrix, None, q)
        pred_mass_mat = jax.vmap(jnp.linalg.inv, (0,))(pred_inv_mass_mat)
        # np.testing.assert_allclose(mass_mat, pred_mass_mat, atol=1.e-6)

    # Compute the torque decomposition:
    hnn_g = hnn_model(params, None, q, zeros, zeros, zeros)[2]
    hnn_m = jax.vmap(jnp.matmul, (0, 0))(mass_mat, qdd)
    hnn_minus_dHdq_quad = hnn_model(params, None, q, p, zeros, zeros)[2] - hnn_g
    hnn_dHdt_qd = pd - hnn_m
    hnn_c = hnn_dHdt_qd + hnn_minus_dHdq_quad

    t0_evaluation = time.perf_counter()
    hnn_tau = hnn_model(params, None, q, p, pd, 0.0 * q)[2]
    t_eval = (time.perf_counter() - t0_evaluation) / float(q.shape[0])

    # Compute Errors:
    test_dEdt = np.sum(test_tau * test_qv, axis=1).reshape((-1, 1))
    err_g = 1. / float(test_qp.shape[0]) * np.sum((hnn_g - test_g) ** 2)
    err_m = 1. / float(test_qp.shape[0]) * np.sum((hnn_m - test_m) ** 2)
    err_c = 1. / float(test_qp.shape[0]) * np.sum((hnn_c - test_c) ** 2)
    err_tau = 1. / float(test_qp.shape[0]) * np.sum((hnn_tau - test_tau) ** 2)

    print("\nPerformance:")
    print("                Torque MSE = {0:.3e}".format(err_tau))
    print("              Inertial MSE = {0:.3e}".format(err_m))
    print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
    print("         Gravitational MSE = {0:.3e}".format(err_g))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1./t_eval))

    print("\n################################################")
    print("Plotting Performance:")

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    y_t_low = np.clip(1.2 * np.min(np.vstack((test_tau, hnn_tau)), axis=0), -np.inf, -0.01)
    y_t_max = np.clip(1.5 * np.max(np.vstack((test_tau, hnn_tau)), axis=0), 0.01, np.inf)

    y_m_low = np.clip(1.2 * np.min(np.vstack((test_m, hnn_m)), axis=0), -np.inf, -0.01)
    y_m_max = np.clip(1.2 * np.max(np.vstack((test_m, hnn_m)), axis=0), 0.01, np.inf)

    y_c_low = np.clip(1.2 * np.min(np.vstack((test_c, hnn_c)), axis=0), -np.inf, -0.01)
    y_c_max = np.clip(1.2 * np.max(np.vstack((test_c, hnn_c)), axis=0), 0.01, np.inf)

    y_g_low = np.clip(1.2 * np.min(np.vstack((test_g, hnn_g)), axis=0), -np.inf, -0.01)
    y_g_max = np.clip(1.2 * np.max(np.vstack((test_g, hnn_g)), axis=0), 0.01, np.inf)

    plt.rc('text', usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = np.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    fig = plt.figure(figsize=(24.0/1.54, 8.0/1.54), dpi=100)
    fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
    fig.canvas.set_window_title('Seed = {0}'.format(seed))

    legend = [mp.patches.Patch(color=color_i[0], label="HNN"),
              mp.patches.Patch(color="k", label="Ground Truth")]

    # Plot Torque
    ax0 = fig.add_subplot(2, 4, 1)
    ax0.set_title(r"$\boldsymbol{\tau}$")
    ax0.text(s=r"\textbf{Joint 0}", x=-0.35, y=.5, fontsize=12, fontweight="bold", rotation=90, horizontalalignment="center", verticalalignment="center", transform=ax0.transAxes)
    ax0.set_ylabel("Torque [Nm]")
    ax0.get_yaxis().set_label_coords(-0.2, 0.5)
    ax0.set_ylim(y_t_low[0], y_t_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_t_low[0], y_t_max[0], linestyles='--', lw=0.5, alpha=1.)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 5)
    ax1.text(s=r"\textbf{Joint 1}", x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
             horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax1.text(s=r"\textbf{(a)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel("Torque [Nm]")
    ax1.get_yaxis().set_label_coords(-0.2, 0.5)
    ax1.set_ylim(y_t_low[1], y_t_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_t_low[1], y_t_max[1], linestyles='--', lw=0.5, alpha=1.)
    ax1.set_xlim(divider[0], divider[-1])

    # ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=1.)

    # Plot Ground Truth Torque:
    ax0.plot(test_tau[:, 0], color="k")
    ax1.plot(test_tau[:, 1], color="k")

    # Plot DeLaN Torque:
    ax0.plot(hnn_tau[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(hnn_tau[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Mass Torque
    ax0 = fig.add_subplot(2, 4, 2)
    ax0.set_title(r"$\displaystyle\mathbf{H}(\mathbf{q}) \ddot{\mathbf{q}}$")
    ax0.set_ylabel("Torque [Nm]")
    ax0.set_ylim(y_m_low[0], y_m_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_m_low[0], y_m_max[0], linestyles='--', lw=0.5, alpha=1.)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 6)
    ax1.text(s=r"\textbf{(b)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel("Torque [Nm]")
    ax1.set_ylim(y_m_low[1], y_m_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_m_low[1], y_m_max[1], linestyles='--', lw=0.5, alpha=1.)
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Inertial Torque:
    ax0.plot(test_m[:, 0], color="k")
    ax1.plot(test_m[:, 1], color="k")

    # Plot DeLaN Inertial Torque:
    ax0.plot(hnn_m[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(hnn_m[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Coriolis Torque
    ax0 = fig.add_subplot(2, 4, 3)
    ax0.set_title(r"$\displaystyle\mathbf{c}(\mathbf{q}, \dot{\mathbf{q}})$")
    ax0.set_ylabel("Torque [Nm]")
    ax0.set_ylim(y_c_low[0], y_c_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_c_low[0], y_c_max[0], linestyles='--', lw=0.5, alpha=1.)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 7)
    ax1.text(s=r"\textbf{(c)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel("Torque [Nm]")
    ax1.set_ylim(y_c_low[1], y_c_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_c_low[1], y_c_max[1], linestyles='--', lw=0.5, alpha=1.)
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Coriolis & Centrifugal Torque:
    ax0.plot(test_c[:, 0], color="k")
    ax1.plot(test_c[:, 1], color="k")

    # Plot DeLaN Coriolis & Centrifugal Torque:
    ax0.plot(hnn_c[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(hnn_c[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Gravity
    ax0 = fig.add_subplot(2, 4, 4)
    ax0.set_title(r"$\displaystyle\mathbf{g}(\mathbf{q})$")
    ax0.set_ylabel("Torque [Nm]")
    ax0.set_ylim(y_g_low[0], y_g_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_g_low[0], y_g_max[0], linestyles='--', lw=0.5, alpha=1.)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 8)
    ax1.text(s=r"\textbf{(d)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
             verticalalignment="center", transform=ax1.transAxes)

    ax1.set_ylabel("Torque [Nm]")
    ax1.set_ylim(y_g_low[1], y_g_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_g_low[1], y_g_max[1], linestyles='--', lw=0.5, alpha=1.)
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Gravity Torque:
    ax0.plot(test_g[:, 0], color="k")
    ax1.plot(test_g[:, 1], color="k")

    # Plot DeLaN Gravity Torque:
    ax0.plot(hnn_g[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(hnn_g[:, 1], color=color_i[0], alpha=plot_alpha)

    fig.savefig(f"figures/jax_HNN_{model_id}_{hyper['dataset']}_Performance.pdf", format="pdf")
    fig.savefig(f"figures/jax_HNN_{model_id}_{hyper['dataset']}_Performance.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")

