"""
RBF+LIN estimator of the Franka Emika Panda robot simulated in Pybullet

Author: Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
Edited by Niccolò Turcato (niccolo.turcato@studenti.unipd.it)

"""

# %%
# Preamble 

from torch._C import device

import Project_Utils
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import time
import argparse

import argparse
import torch
import numpy as np
import time

import PyQt5

import matplotlib as mp

from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory

try:
    mp.use("Qt5Agg")
    mp.rc('text', usetex=True)
    # mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    mp.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

except ImportError:
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork

data_path = ""
training_file = ""
test_file = ""
training_file_M = ""
test_file_M = ""
saving_path = ""
robot_name = ""
flg_norm = None
num_data_tr = None
flg_train = None
shuffle = 0
N_epoch_print = 0
flg_save = 0
# %%
# Set command line arguments
parser = argparse.ArgumentParser('FE panda GIP estimator')
parser.add_argument('-robot_name',
                    type=str,
                    default='FE_panda_pybul',
                    help='Name of the robot.')
parser.add_argument('-data_path',
                    type=str,
                    default='./Simulated_robots/Pybullet_sim/FE_panda/data/',
                    help='Path to the folder containing training and test dasets.')
parser.add_argument('-saving_path',
                    type=str,
                    default='./Results/GP_estimators/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-training_file',
                    type=str,
                    default='FE_panda_pybul_tr.pkl',
                    help='Name of the file containing the train dataset.')
parser.add_argument('-test_file',
                    type=str,
                    default='FE_panda_pybul_sum_of_sin_test.pkl',
                    help='Name of the file containing the test dataset.')
parser.add_argument('-flg_load',
                    type=bool,
                    default=False,
                    help='Flag load model. If True the model loaded from memory, otherwise they are computed.')
parser.add_argument('-flg_save',
                    type=bool,
                    default=True,
                    help='Flag save model. If true, the model parameters are saved in memory.')
parser.add_argument('-flg_train',
                    type=bool,
                    default=True,
                    help='Flag train. If True the model parameters are trained.')
parser.add_argument('-batch_size',
                    type=int,
                    default=500,
                    help='Batch size for the training procedure.')
parser.add_argument('-shuffle',
                    type=bool,
                    default=True,
                    help='Shuffle data before training.')
parser.add_argument('-flg_norm',
                    type=bool,
                    default=True,
                    help='Normalize signal.')
parser.add_argument('-N_epoch',
                    type=int,
                    default=4000,
                    help='Number of Epoch for the training procedure.')
parser.add_argument('-N_epoch_print',
                    type=int,
                    default=1,
                    help='Num epoch between two prints during training.')
parser.add_argument('-flg_cuda',
                    type=bool,
                    default=False,
                    help='Set the device type')
parser.add_argument('-num_data_tr',
                    type=int,
                    default=None,
                    help='Number of data to use in the training.')
parser.add_argument('-num_threads',
                    type=int,
                    default=1,
                    help='Number of computational threads.')
parser.add_argument('-downsampling',
                    type=int,
                    default=1,
                    help='Downsampling.')
locals().update(vars(parser.parse_known_args()[0]))

# %%
# Set flags -- for debug
flg_train = True

# flg_save = True

flg_load = False
# flg_load = True

flg_cuda = False
# flg_cuda = True

downsampling = 100
num_threads = 4
N_epoch = 500
batch_size = 512

# %%
# Set the paths 
print('Setting paths... ', end='')

# Datasets loading paths
tr_path = data_path + training_file
test_path = data_path + test_file

# %%
# Set robot params
print('Setting robot parameters... ', end='')

num_dof = 7
joint_index_list = range(0, num_dof)
robot_structure = [0] * num_dof  # 0 = revolute, 1 = prismatic
joint_names = [str(joint_index) for joint_index in range(1, num_dof + 1)]
# features_name_list = [str(i) for i in range(1, num_dof +1)]
output_feature = 'tau'

print('Done!')

# %%
# Load datasets
print('Loading datasets... ', end='')

q_names = ['q_' + joint_name for joint_name in joint_names]
dq_names = ['dq_' + joint_name for joint_name in joint_names]
ddq_names = ['ddq_' + joint_name for joint_name in joint_names]
input_features = q_names + dq_names + ddq_names
pos_indices = range(0, num_dof)
acc_indices = range(2 * num_dof, 3 * num_dof)
input_features_joint_list = [input_features] * num_dof

X_tr, Y_tr, active_dims_list, data_frame_tr = Project_Utils.get_data_from_features(tr_path,
                                                                                   input_features,
                                                                                   input_features_joint_list,
                                                                                   output_feature,
                                                                                   num_dof)
X_test, Y_test, active_dims_list, data_frame_test = Project_Utils.get_data_from_features(test_path,
                                                                                         input_features,
                                                                                         input_features_joint_list,
                                                                                         output_feature,
                                                                                         num_dof)

num_data_test = X_test.shape[0]
print('Done!')

# Read the dataset:
n_dof = num_dof

train_qp = X_tr[:, 0:7]  # joint positions
train_qv = X_tr[:, 7:14]  # joint velocities
train_qa = X_tr[:, 14:]  # joint accelerations

train_tau = Y_tr

test_qp = X_test[:, 0:7]  # joint positions
test_qv = X_test[:, 7:14]  # joint velocities
test_qa = X_test[:, 14:]  # joint accelerations

test_tau = Y_test

print("\n\n################################################")
print("Characters:")
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
         'max_epoch': 10}

cuda = False  # Watch this

delan_model = DeepLagrangianNetwork(n_dof, **hyper)
delan_model = delan_model.cuda() if cuda else delan_model.cpu()

# Generate & Initialize the Optimizer:
optimizer = torch.optim.Adam(delan_model.parameters(),
                             lr=hyper["learning_rate"],
                             weight_decay=hyper["weight_decay"],
                             amsgrad=True)

# Generate Replay Memory:
mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
mem.add_samples([train_qp, train_qv, train_qa, train_tau])

# Start Training Loop:
t0_start = time.perf_counter()

epoch_i = 0
while epoch_i < hyper['max_epoch']:
    l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
    l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
    l_mem, n_batches = 0.0, 0.0

    for q, qd, qdd, tau in mem:
        t0_batch = time.perf_counter()

        # Reset gradients:
        optimizer.zero_grad()

        # Compute the Rigid Body Dynamics Model:
        tau_hat, dEdt_hat = delan_model(q, qd, qdd)

        # Compute the loss of the Euler-Lagrange Differential Equation:
        err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
        l_mean_inv_dyn = torch.mean(err_inv)
        l_var_inv_dyn = torch.var(err_inv)

        # Compute the loss of the Power Conservation:
        dEdt = torch.matmul(qd.view(-1, n_dof, 1).transpose(1, 2), tau.view(-1, n_dof, 1)).view(-1)
        err_dEdt = (dEdt_hat - dEdt) ** 2
        l_mean_dEdt = torch.mean(err_dEdt)
        l_var_dEdt = torch.var(err_dEdt)

        # Compute gradients & update the weights:
        loss = l_mean_inv_dyn + l_mem_mean_dEdt
        loss.backward()
        optimizer.step()

        # Update internal data:
        n_batches += 1
        l_mem += loss.item()
        l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
        l_mem_var_inv_dyn += l_var_inv_dyn.item()
        l_mem_mean_dEdt += l_mean_dEdt.item()
        l_mem_var_dEdt += l_var_dEdt.item()

        t_batch = time.perf_counter() - t0_batch

    # Update Epoch Loss & Computation Time:
    l_mem_mean_inv_dyn /= float(n_batches)
    l_mem_var_inv_dyn /= float(n_batches)
    l_mem_mean_dEdt /= float(n_batches)
    l_mem_var_dEdt /= float(n_batches)
    l_mem /= float(n_batches)
    epoch_i += 1

    if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
        print("Epoch {0:05d}: ".format(epoch_i), end=" ")
        print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
        print("Loss = {0:.3e}".format(l_mem), end=", ")
        print("Inv Dyn = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)),
              end=", ")
        print("Power Con = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))

# Save the Model:

torch.save({"epoch": epoch_i,
            "hyper": hyper,
            "state_dict": delan_model.state_dict()},
           "data/delan_model.torch")

print("\n################################################")
print("Evaluating DeLaN:")

Y_tr_hat_list = []
Y_test_hat_list = []

# Training estimates
for i in range(train_qp.shape[0]):
    with torch.no_grad():
        # Convert NumPy samples to torch:
        q = torch.from_numpy(train_qp[i]).float().view(1, -1)
        qd = torch.from_numpy(train_qv[i]).float().view(1, -1)
        qdd = torch.from_numpy(train_qa[i]).float().view(1, -1)

        # Compute predicted torque:
        out = delan_model(q, qd, qdd)
        Y_tr_hat_list.append(out[0].cpu().numpy().squeeze())

# Test estimates
for i in range(test_qp.shape[0]):
    with torch.no_grad():
        # Convert NumPy samples to torch:
        q = torch.from_numpy(test_qp[i]).float().view(1, -1)
        qd = torch.from_numpy(test_qv[i]).float().view(1, -1)
        qdd = torch.from_numpy(test_qa[i]).float().view(1, -1)

        # Compute predicted torque:
        out = delan_model(q, qd, qdd)
        Y_test_hat_list.append(out[0].cpu().numpy().squeeze())

norm_coeff = 1

flg_norm=False

# Print estimates and stats
Y_tr_hat_pd, Y_test_hat_pd, Y_tr_pd, Y_test_pd, Y_tr_noiseless_pd, Y_test_noiseless_pd = Project_Utils.get_pandas_obj(
    output_tr=Y_tr,
    output_test=Y_test,
    noiseless_output_tr=Y_tr,
    noiseless_output_test=Y_test,
    Y_tr_hat_list=Y_tr_hat_list,
    Y_test_hat_list=Y_test_hat_list,
    flg_norm=flg_norm,
    norm_coeff=norm_coeff,
    joint_index_list=joint_index_list,
    output_feature=output_feature,
    noiseless_output_feature=output_feature,
    var_tr_list=[],
    var_test_list=[])

# get the erros stats
Project_Utils.get_stat_estimate(Y_tr_noiseless_pd, [Y_tr_hat_pd], joint_index_list, stat_name='nMSE',
                                output_feature=output_feature)
Project_Utils.get_stat_estimate(Y_test_noiseless_pd, [Y_test_hat_pd], joint_index_list, stat_name='nMSE',
                                output_feature=output_feature)
# Project_Utils.get_stat_estimate(Y_test2_noiseless_pd, [Y_test2_hat_pd], joint_index_list, stat_name='nMSE', output_feature=output_feature, output_feature_noiseless=noiseless_output_feature)
# print the estimates
Project_Utils.print_estimate(Y_tr_pd, [Y_tr_hat_pd], joint_index_list, flg_print_var=True,
                             output_feature=output_feature, data_noiseless=Y_tr_noiseless_pd,
                             noiseless_output_feature=output_feature)
# plt.show()
Project_Utils.print_estimate(Y_test_pd, [Y_test_hat_pd], joint_index_list, flg_print_var=True,
                             output_feature=output_feature)
plt.show()
# Project_Utils.print_estimate(Y_test2_pd, [Y_test2_hat_pd], joint_index_list, flg_print_var=True, output_feature=output_feature)
# plt.show()
