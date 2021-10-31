"""
Script for training/testing of Deep Lagrangian Network of simulated data of a Panda robot restricted to its first 3
joints and links.

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""
from matplotlib import pyplot as plt

import robust_fl_with_gps.Project_Utils as Project_FL_Utils

import argparse
import torch
import numpy as np
import time

import PyQt5

import matplotlib as mp

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory

try:
    mp.use("Qt5Agg")
    mp.rc('text', usetex=True)
    # mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    mp.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"
except ImportError:
    pass

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
                    default='FE_panda3DOF_sim',
                    help='Name of the robot.')
parser.add_argument('-data_path',
                    type=str,
                    default='./robust_fl_with_gps/Simulated_robots/SympyBotics_sim/FE_panda/',
                    help='Path to the folder containing training and test dasets.')
parser.add_argument('-saving_path',
                    type=str,
                    default='./data/Results/DeLan/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-model_saving_path',
                    type=str,
                    default='./data/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-training_file',
                    type=str,
                    default='FE_panda3DOF_sim_tr.pkl',
                    help='Name of the file containing the train dataset.')
parser.add_argument('-test_file',
                    type=str,
                    default='FE_panda3DOF_sim_test.pkl',
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
                    default=512,
                    help='Batch size for the training procedure.')
parser.add_argument('-shuffle',
                    type=bool,
                    default=True,
                    help='Shuffle data before training.')
parser.add_argument('-flg_norm',
                    type=bool,
                    default=False,
                    help='Normalize signal.')
parser.add_argument('-N_epoch',
                    type=int,
                    default=5000,
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

# flg_cuda = False
flg_cuda = True  # Watch this

downsampling = 100
num_threads = 4
N_epoch = 500
batch_size = 512
norm_coeff = 1

# Set the paths
print('Setting paths... ', end='')

# Datasets loading paths
tr_path = data_path + training_file
test_path = data_path + test_file

# Set robot params
print('Setting robot parameters... ', end='')

num_dof = 3
joint_index_list = range(0, num_dof)
robot_structure = [0] * num_dof  # 0 = revolute, 1 = prismatic
joint_names = [str(joint_index) for joint_index in range(1, num_dof + 1)]
# features_name_list = [str(i) for i in range(1, num_dof +1)]
output_feature = 'tau'

print('Done!')

# Load datasets
print('Loading datasets... ', end='')

q_names = ['q_' + joint_name for joint_name in joint_names]
dq_names = ['dq_' + joint_name for joint_name in joint_names]
ddq_names = ['ddq_' + joint_name for joint_name in joint_names]
input_features = q_names + dq_names + ddq_names
pos_indices = range(0, num_dof)
acc_indices = range(2 * num_dof, 3 * num_dof)
input_features_joint_list = [input_features] * num_dof

X_tr, Y_tr, active_dims_list, data_frame_tr = Project_FL_Utils.get_data_from_features(tr_path,
                                                                                      input_features,
                                                                                      input_features_joint_list,
                                                                                      output_feature,
                                                                                      num_dof)

X_test, Y_test, active_dims_list, data_frame_test = Project_FL_Utils.get_data_from_features(test_path,
                                                                                            input_features,
                                                                                            input_features_joint_list,
                                                                                            output_feature,
                                                                                            num_dof)

print("\n\n################################################")
print("# Training Samples = {0:05d}".format(int(X_tr.shape[0])))
print("# Test Samples = {0:05d}".format(int(X_test.shape[0])))
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
         'max_epoch': 10000,
         'save_file': model_saving_path + 'delan_panda3DOF_model.torch'}

if flg_train:
    delan_model = DeepLagrangianNetwork(num_dof, **hyper)
    delan_model = delan_model.cuda(torch.device('cuda:0')) if flg_cuda else delan_model.cpu()
    optimizer = torch.optim.Adam(delan_model.parameters(),
                                 lr=hyper["learning_rate"],
                                 weight_decay=hyper["weight_decay"],
                                 amsgrad=True)

    delan_model.train_model(X_tr, Y_tr, optimizer, save_model=flg_save)

elif flg_load:
    state = torch.load(hyper['save_file'])

    delan_model = DeepLagrangianNetwork(num_dof, **state['hyper'])
    delan_model.load_state_dict(state['state_dict'])
    delan_model = delan_model.cuda(torch.device('cuda:0')) if flg_cuda else delan_model.cpu()

else:
    raise RuntimeError('Aborting because no model training or loading was defined')

Y_tr_hat_list = delan_model.evaluate(X_tr) # [np.zeros((X_tr.shape[0],1)) for i in range(num_dof)]
Y_test_hat_list = delan_model.evaluate(X_test) # [np.zeros((X_tr.shape[0],1)) for i in range(num_dof)]

# Print estimates and stats
Y_tr_hat_pd, Y_test_hat_pd, Y_tr_pd, Y_test_pd, Y_tr_noiseless_pd, Y_test_noiseless_pd = Project_FL_Utils.get_pandas_obj(
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
    noiseless_output_feature=output_feature)

# get the erros stats
Project_FL_Utils.get_stat_estimate(Y_tr_noiseless_pd, [Y_tr_hat_pd], joint_index_list, stat_name='nMSE',
                                   output_feature=output_feature)
Project_FL_Utils.get_stat_estimate(Y_test_noiseless_pd, [Y_test_hat_pd], joint_index_list, stat_name='nMSE',
                                   output_feature=output_feature)
# Project_Utils.get_stat_estimate(Y_test2_noiseless_pd, [Y_test2_hat_pd], joint_index_list, stat_name='nMSE', output_feature=output_feature, output_feature_noiseless=noiseless_output_feature)
# print the estimates
Project_FL_Utils.print_estimate(Y_tr_pd, [Y_tr_hat_pd], joint_index_list, flg_print_var=True,
                                output_feature=output_feature, data_noiseless=Y_tr_noiseless_pd,
                                noiseless_output_feature=output_feature, label_prefix='tr_')
# plt.show()
Project_FL_Utils.print_estimate(Y_test_pd, [Y_test_hat_pd], joint_index_list, flg_print_var=True,
                                output_feature=output_feature, label_prefix='test_')
plt.show()
# Project_Utils.print_estimate(Y_test2_pd, [Y_test2_hat_pd], joint_index_list, flg_print_var=True, output_feature=output_feature)
# plt.show()
