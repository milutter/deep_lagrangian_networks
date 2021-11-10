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
import Utils
import pickle as pkl


if __name__ == "__main__":
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


    # Load existing model parameters:
    load_file = "data/delan_model.torch"
    state = torch.load(load_file)

    delan_model = DeepLagrangianNetwork(n_dof, **state['hyper'])
    delan_model.load_state_dict(state['state_dict'])
    delan_model = delan_model.cpu()

    print("\n################################################")
    print("Evaluating DeLaN:")

    # Compute the inertial, centrifugal & gravitational torque using batched samples
    t0_batch = time.perf_counter()

    t_batch = (time.perf_counter() - t0_batch) / (3. * float(test_qp.shape[0]))

    # Move model to the CPU:
    delan_model.cpu()

    # Compute the joint torque using single samples on the CPU. The evaluation is done using only single samples to
    # imitate the online control-loop. These online computation are performed on the CPU as this is faster for single
    # samples.

    delan_test_tau, delan_test_dEdt = np.zeros(test_qp.shape), np.zeros((test_qp.shape[0], 1))
    t0_evaluation = time.perf_counter()
    for i in range(test_qp.shape[0]):

        with torch.no_grad():

            # Convert NumPy samples to torch:
            q = torch.from_numpy(test_qp[i]).float().view(1, -1)
            qd = torch.from_numpy(test_qv[i]).float().view(1, -1)
            qdd = torch.from_numpy(test_qa[i]).float().view(1, -1)

            # Compute predicted torque:

            out = delan_model(q, qd, qdd)
            delan_test_tau[i] = out[0].cpu().numpy().squeeze()
            delan_test_dEdt[i] = out[1].cpu().numpy()

    # Convert NumPy samples to torch:
    q = torch.from_numpy(test_qp).float().to(delan_model.device)
    qd = torch.from_numpy(test_qv).float().to(delan_model.device)
    qdd = torch.from_numpy(test_qa).float().to(delan_model.device)
    zeros = torch.zeros_like(q).float().to(delan_model.device)
    #Computing torque decomposition
    with torch.no_grad():
        delan_test_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
        delan_test_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_test_g
        delan_test_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_test_g


    t_eval = (time.perf_counter() - t0_evaluation) / float(test_qp.shape[0])

    delan_tr_tau, delan_tr_dEdt = np.zeros(train_qp.shape), np.zeros((train_qp.shape[0], 1))
    delan_tr_m, delan_tr_c, delan_tr_g = np.zeros(delan_test_tau.shape), np.zeros(delan_test_tau.shape), \
                                               np.zeros(delan_test_tau.shape)
    for i in range(train_qp.shape[0]):

        with torch.no_grad():

            # Convert NumPy samples to torch:
            q = torch.from_numpy(train_qp[i]).float().view(1, -1)
            qd = torch.from_numpy(train_qv[i]).float().view(1, -1)
            qdd = torch.from_numpy(train_qa[i]).float().view(1, -1)

            # Compute predicted torque:
            out = delan_model(q, qd, qdd)
            delan_tr_tau[i] = out[0].cpu().numpy().squeeze()
            delan_tr_dEdt[i] = out[1].cpu().numpy()

    # Convert NumPy samples to torch:
    q = torch.from_numpy(train_qp).float().to(delan_model.device)
    qd = torch.from_numpy(train_qv).float().to(delan_model.device)
    qdd = torch.from_numpy(train_qa).float().to(delan_model.device)
    zeros = torch.zeros_like(q).float().to(delan_model.device)
    #Computing torque decomposition
    with torch.no_grad():
        delan_tr_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
        delan_tr_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_tr_g
        delan_tr_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_tr_g

    pd_test_estimates = Utils.convert_predictions_to_dataset(np.hstack([delan_test_tau, delan_test_m, delan_test_c, delan_test_g]),
                                                             ['tau_est', 'm_est', 'c_est', 'g_est'], range(n_dof))
    pkl.dump(pd_test_estimates, open('data/DeLaN_test_estimates.pkl', 'wb'))

    pd_tr_estimates = Utils.convert_predictions_to_dataset(np.hstack([delan_tr_tau, delan_tr_m, delan_tr_c, delan_tr_g]),
                                                            ['tau_est', 'm_est', 'c_est', 'g_est'], range(n_dof))
    pkl.dump(pd_tr_estimates, open('data/DeLaN_train_estimates.pkl', 'wb'))

    print("\n################################################\n\n\n")

