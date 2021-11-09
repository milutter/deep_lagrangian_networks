import argparse
import torch
import numpy as np
import time

import PyQt5

import matplotlib as mp


try:
    mp.use("Qt5Agg")
    mp.rc('text', usetex=True)
    #mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    mp.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

except ImportError:
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env


if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[42, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[0, ], help="Save the DeLaN model")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    # Read the dataset:
    n_dof = 2
    train_data, test_data, divider, _ = load_dataset()
    train_labels, train_qp, train_qv, train_qa, _, _, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, _, _, test_tau, test_m, test_c, test_g = test_data

    print("\n\n################################################")
    print("Characters:")
    print("   Test Characters = {0}".format(test_labels))
    print("  Train Characters = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    print("\n################################################")
    print("Training Deep Lagrangian Networks (DeLaN):")

    # Construct Hyperparameters:
    hyper = {'n_width': 64,
             'n_depth': 2,
             'diagonal_epsilon': 0.01,
             'activation': 'SoftPlus',
             'b_init': 1.e-4,
             'b_diag_init': 0.001,
             'w_init': 'xavier_normal',
             'gain_hidden': np.sqrt(2.),
             'gain_output': 0.1,
             'n_minibatch': 512,
             'learning_rate': 5.e-04,
             'weight_decay': 1.e-5,
             'max_epoch': 100}

    # Load existing model parameters:
    if load_model:
        load_file = "data/delan_model.torch"
        state = torch.load(load_file)

        delan_model = DeepLagrangianNetwork(n_dof, **state['hyper'])
        delan_model.load_state_dict(state['state_dict'])
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    else:
        # Construct DeLaN:
        delan_model = DeepLagrangianNetwork(n_dof, **hyper)
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

        # Generate & Initialize the Optimizer:
        optimizer = torch.optim.Adam(delan_model.parameters(),
                                 lr=hyper["learning_rate"],
                                 weight_decay=hyper["weight_decay"],
                                 amsgrad=True)

        delan_model.train_model(np.hstack(train_qp, train_qv, train_qa), train_tau, optimizer)

    # # Generate Replay Memory:
    # mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ))
    # mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
    # mem.add_samples([train_qp, train_qv, train_qa, train_tau])
    #
    # # Start Training Loop:
    # t0_start = time.perf_counter()
    #
    # epoch_i = 0
    # while epoch_i < hyper['max_epoch'] and not load_model:
    #     l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
    #     l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
    #     l_mem, n_batches = 0.0, 0.0
    #
    #     for q, qd, qdd, tau in mem:
    #         t0_batch = time.perf_counter()
    #
    #         # Reset gradients:
    #         optimizer.zero_grad()
    #
    #         # Compute the Rigid Body Dynamics Model:
    #         tau_hat, dEdt_hat = delan_model(q, qd, qdd)
    #
    #         # Compute the loss of the Euler-Lagrange Differential Equation:
    #         err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
    #         l_mean_inv_dyn = torch.mean(err_inv)
    #         l_var_inv_dyn = torch.var(err_inv)
    #
    #         # Compute the loss of the Power Conservation:
    #         dEdt = torch.matmul(qd.view(-1, n_dof, 1).transpose(dim0=1, dim1=2), tau.view(-1, n_dof, 1)).view(-1)
    #             # previous version
    #             # dEdt = torch.matmul(qd.view(-1, 2, 1).transpose(dim0=1, dim1=2), tau.view(-1, 2, 1)).view(-1)
    #         err_dEdt = (dEdt_hat - dEdt) ** 2
    #         l_mean_dEdt = torch.mean(err_dEdt)
    #         l_var_dEdt = torch.var(err_dEdt)
    #
    #         # Compute gradients & update the weights:
    #         loss = l_mean_inv_dyn + l_mem_mean_dEdt
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Update internal data:
    #         n_batches += 1
    #         l_mem += loss.item()
    #         l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
    #         l_mem_var_inv_dyn += l_var_inv_dyn.item()
    #         l_mem_mean_dEdt += l_mean_dEdt.item()
    #         l_mem_var_dEdt += l_var_dEdt.item()
    #
    #         t_batch = time.perf_counter() - t0_batch
    #
    #     # Update Epoch Loss & Computation Time:
    #     l_mem_mean_inv_dyn /= float(n_batches)
    #     l_mem_var_inv_dyn /= float(n_batches)
    #     l_mem_mean_dEdt /= float(n_batches)
    #     l_mem_var_dEdt /= float(n_batches)
    #     l_mem /= float(n_batches)
    #     epoch_i += 1
    #
    #     if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
    #         print("Epoch {0:05d}: ".format(epoch_i), end=" ")
    #         print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
    #         print("Loss = {0:.3e}".format(l_mem), end=", ")
    #         print("Inv Dyn = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)), end=", ")
    #         print("Power Con = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))
    #
    # # Save the Model:
    # if save_model:
    #     torch.save({"epoch": epoch_i,
    #                 "hyper": hyper,
    #                 "state_dict": delan_model.state_dict()},
    #                 "data/delan_model.torch")

    print("\n################################################")
    print("Evaluating DeLaN:")

    # Compute the inertial, centrifugal & gravitational torque using batched samples
    t0_batch = time.perf_counter()

    # Convert NumPy samples to torch:
    q = torch.from_numpy(test_qp).float().to(delan_model.device)
    qd = torch.from_numpy(test_qv).float().to(delan_model.device)
    qdd = torch.from_numpy(test_qa).float().to(delan_model.device)
    zeros = torch.zeros_like(q).float().to(delan_model.device)

    # Compute the torque decomposition:
    with torch.no_grad():
        delan_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
        delan_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_g
        delan_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_g

    t_batch = (time.perf_counter() - t0_batch) / (3. * float(test_qp.shape[0]))

    # Move model to the CPU:
    delan_model.cpu()

    # Compute the joint torque using single samples on the CPU. The evaluation is done using only single samples to
    # imitate the online control-loop. These online computation are performed on the CPU as this is faster for single
    # samples.

    delan_tau, delan_dEdt = np.zeros(test_qp.shape), np.zeros((test_qp.shape[0], 1))
    t0_evaluation = time.perf_counter()
    for i in range(test_qp.shape[0]):

        with torch.no_grad():

            # Convert NumPy samples to torch:
            q = torch.from_numpy(test_qp[i]).float().view(1, -1)
            qd = torch.from_numpy(test_qv[i]).float().view(1, -1)
            qdd = torch.from_numpy(test_qa[i]).float().view(1, -1)

            # Compute predicted torque:
            out = delan_model(q, qd, qdd)
            delan_tau[i] = out[0].cpu().numpy().squeeze()
            delan_dEdt[i] = out[1].cpu().numpy()

    t_eval = (time.perf_counter() - t0_evaluation) / float(test_qp.shape[0])

    # Compute Errors:
    test_dEdt = np.sum(test_tau * test_qv, axis=1).reshape((-1, 1))
    err_g = 1. / float(test_qp.shape[0]) * np.sum((delan_g - test_g) ** 2)
    err_m = 1. / float(test_qp.shape[0]) * np.sum((delan_m - test_m) ** 2)
    err_c = 1. / float(test_qp.shape[0]) * np.sum((delan_c - test_c) ** 2)
    err_tau = 1. / float(test_qp.shape[0]) * np.sum((delan_tau - test_tau) ** 2)
    err_dEdt = 1. / float(test_qp.shape[0]) * np.sum((delan_dEdt - test_dEdt) ** 2)

    print("\nPerformance:")
    print("                Torque MSE = {0:.3e}".format(err_tau))
    print("              Inertial MSE = {0:.3e}".format(err_m))
    print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
    print("         Gravitational MSE = {0:.3e}".format(err_g))
    print("    Power Conservation MSE = {0:.3e}".format(err_dEdt))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1./t_eval))

    print("\n################################################")
    print("Plotting Performance:")

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    y_t_low = np.clip(1.2 * np.min(np.vstack((test_tau, delan_tau)), axis=0), -np.inf, -0.01)
    y_t_max = np.clip(1.5 * np.max(np.vstack((test_tau, delan_tau)), axis=0), 0.01, np.inf)

    y_m_low = np.clip(1.2 * np.min(np.vstack((test_m, delan_m)), axis=0), -np.inf, -0.01)
    y_m_max = np.clip(1.2 * np.max(np.vstack((test_m, delan_m)), axis=0), 0.01, np.inf)

    y_c_low = np.clip(1.2 * np.min(np.vstack((test_c, delan_c)), axis=0), -np.inf, -0.01)
    y_c_max = np.clip(1.2 * np.max(np.vstack((test_c, delan_c)), axis=0), 0.01, np.inf)

    y_g_low = np.clip(1.2 * np.min(np.vstack((test_g, delan_g)), axis=0), -np.inf, -0.01)
    y_g_max = np.clip(1.2 * np.max(np.vstack((test_g, delan_g)), axis=0), 0.01, np.inf)

    plt.rc('text', usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = np.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    fig = plt.figure(figsize=(24.0/1.54, 8.0/1.54), dpi=100)
    fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
    #fig.canvas.set_window_title('Seed = {0}'.format(seed))
    fig.canvas.manager.set_window_title('Seed = {0}'.format(seed))

    legend = [mp.patches.Patch(color=color_i[0], label="DeLaN"),
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

    ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=1.)

    # Plot Ground Truth Torque:
    ax0.plot(test_tau[:, 0], color="k")
    ax1.plot(test_tau[:, 1], color="k")

    # Plot DeLaN Torque:
    ax0.plot(delan_tau[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_tau[:, 1], color=color_i[0], alpha=plot_alpha)

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
    ax0.plot(delan_m[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_m[:, 1], color=color_i[0], alpha=plot_alpha)

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
    ax0.plot(delan_c[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_c[:, 1], color=color_i[0], alpha=plot_alpha)

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
    ax0.plot(delan_g[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_g[:, 1], color=color_i[0], alpha=plot_alpha)

    fig.savefig("figures/DeLaN_Performance.pdf", format="pdf")
    fig.savefig("figures/DeLaN_Performance.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")

