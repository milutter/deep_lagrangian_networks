"""File with auxiliary functions
Author: Alberto Dalla Libera
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
import torch

def print_estimate(data, data_est_list, link_index_list, flg_print_var=False, output_feature='tau', data_noiseless=None, noiseless_output_feature=None):
    """Prints the estimates"""

    for i, joint in enumerate(link_index_list):
        plt.figure()
        plt.plot(data[output_feature+'_'+str(joint+1)], 'r', label=output_feature+'_'+str(joint+1))
        if data_noiseless is not None:
            plt.plot(data_noiseless[noiseless_output_feature+'_'+str(joint+1)], 'k', label=noiseless_output_feature+'_'+str(joint+1))
        for index, data_est in enumerate(data_est_list):
            plt.plot(data_est[output_feature+'_est_'+str(joint+1)], 'b', label=output_feature+'_'+str(joint+1)+'_est_'+str(index))
            if flg_print_var:
                X_indices = np.arange(0,data_est[output_feature+'_est_'+str(joint+1)].size)
                std = 3*np.sqrt(data_est['var_'+output_feature+'_'+str(joint+1)])
                plt.fill(np.concatenate([X_indices,np.flip(X_indices)]),
                                         np.concatenate([data_est[output_feature+'_est_'+str(joint+1)]+std,np.flip(data_est[output_feature+'_est_'+str(joint+1)]-std)]),
                                         'b', alpha=0.4)
        plt.legend()


def get_stat_estimate(data, data_est_list, link_index_list, stat_name='MSE', output_feature='tau', output_feature_noiseless='tau'):
    """Prints the estimate performances"""
    stat_list = []
    #select the stat
    if stat_name=='MSE':
        f_stat = lambda a,b: MSE(a,b)
    if stat_name=='nMSE':
        f_stat = lambda a,b: nMSE(a,b)

    print('\n')
    for i, joint in enumerate(link_index_list):
        stat_joint_list = []
        tau_label = output_feature_noiseless+'_'+str(joint+1)
        tau_est_label = output_feature+'_est_'+str(joint+1)
        print('Joint '+str(joint+1)+':')
        for index, data_est in enumerate(data_est_list):
            stat = f_stat(data[tau_label].values, data_est[tau_est_label].values)
            stat_joint_list.append(stat)
            print('-'+stat_name+' Estimator '+str(index)+': '+str(stat))
        stat_list.append(stat_joint_list)

    return stat_list


def nMSE(y, y_hat):
    num_sample = y.size
    return np.sum((y - y_hat)**2)/num_sample/np.var(y)

def MSE(y, y_hat):
    num_sample = y.size
    return np.sum((y - y_hat)**2)/num_sample


def get_phi_labels(joint_index_list, num_dof, num_par_dyn):
    """Returns a list with the phi labels, given the list of joint indices, num_dof and num_par_dyn"""
    return ['phi_'+str(joint_index)+'_'+str(j+1)
            for j in range(0, num_par_dyn*num_dof)
            for joint_index in joint_index_list]


def get_data_from_features(data_frame_pkl, input_features, input_features_joint_list, output_feature, num_dof, joint_name_list=None):
    """Function that returns:
       - a np matrix whose columns are the "input_features" elements of the pandas data pointed by data_frame_pkl
       - a np matrix containing the output vectors (the i-th conlumn corresponds to the output of the i-th link)
       - a list containing the active_dims list of each link gp
    """
    # get input output data
    if joint_name_list is None:
        joint_name_list = [str(i) for i in range(1,num_dof+1)]
    data_frame = pd.read_pickle(data_frame_pkl)
    input_vector = data_frame[input_features].values
    output_labels = [output_feature+'_'+joint_name for joint_name in joint_name_list]
    output_vector = data_frame[output_labels].values
    # get the active dims of each joint
    active_dims_list = []
    for joint_index in range(0,len(joint_name_list)):
        active_dims_list.append(np.array([input_features.index(name) for name in input_features_joint_list[joint_index]]))
    return input_vector, output_vector, active_dims_list, data_frame


def get_dataset_poly_from_structure(data_frame_pkl, num_dof, output_feature, robot_structure, features_name_list=None):
    """Returns dataset and kernel info from robot structure:
       Robot structure is a list of num_dof elements containing:
       - 0 when the joint is revolute
       - 1 when the joint is prismatic
       As regards the positions cos(q), sin(q) are considered when the joint ir revolute (q when prism) 
    """
    # load the pandas dataframe
    data_frame = pd.read_pickle(data_frame_pkl)
    # list with data
    data_list = []
    # lists with the active dims
    active_dims_acc_vel = []
    active_dims_friction = []
    active_dims_acc = []
    active_dims_q_dot = []
    active_dims_vel = []
    active_dims_mon_rev = []
    active_dims_mon_rev_cos = []
    active_dims_mon_rev_sin = []
    active_dims_mon_prism = []
    # init counters
    index_active_dim = 0
    num_rev = 0
    num_prism = 0
    # set names_list
    if features_name_list is None:
        features_name_list = [str(i) for i in range(1,num_dof+1)] 
    # get pos features
    for joint_index in range(0,num_dof):
        # get the q label
        q_label = 'q_'+ features_name_list[joint_index]
        # check the type of joint
        if robot_structure[joint_index]==0:
            # when the type is revolute add cos(q) and sin(q)
            data_list.append(np.cos(data_frame[q_label].values).reshape([-1,1]))
            active_dims_mon_rev_cos.append(np.array([index_active_dim]))
            index_active_dim +=1
            data_list.append(np.sin(data_frame[q_label].values).reshape([-1,1]))
            active_dims_mon_rev_sin.append(np.array([index_active_dim]))
            active_dims_mon_rev.append(np.array([index_active_dim-1,index_active_dim]))
            index_active_dim +=1
            num_rev +=2
        else:
            # when the type is prismatic add q
            data_list.append(data_frame[q_label].values.reshape([-1,1]))
            active_dims_mon_prism.append(np.array([index_active_dim]))
            index_active_dim +=1
            num_prism +=1
    # get acc/vel/frictions features
    for joint_index_1 in range(0,num_dof):
        # add acc
        data_list.append(data_frame['ddq_'+ features_name_list[joint_index_1]].values.reshape([-1,1]))
        active_dims_acc_vel.append(index_active_dim)
        active_dims_acc.append(index_active_dim)
        index_active_dim +=1
        # add q_dot/friction features
        vel_label_1 = 'dq_'+ features_name_list[joint_index_1]
        data_list.append((data_frame[vel_label_1].values).reshape([-1,1]))
        data_list.append(np.sign(data_frame[vel_label_1].values).reshape([-1,1]))
        active_dims_friction.append(np.array([index_active_dim, index_active_dim+1])) 
        active_dims_q_dot.append(index_active_dim)
        index_active_dim +=2
        # add vel features
        for joint_index_2 in range(joint_index_1,num_dof):
            vel_label_2 = 'dq_'+ features_name_list[joint_index_2]
            data_list.append((data_frame[vel_label_1].values*data_frame[vel_label_2].values).reshape([-1,1]))
            active_dims_acc_vel.append(index_active_dim)
            active_dims_vel.append(index_active_dim)
            index_active_dim +=1
    # get input output
    X = np.concatenate(data_list,1)
    Y = data_frame[[output_feature+'_'+features_name_list[joint_index]
                   for joint_index in range(0, num_dof)]].values
    # build the active dims diictionary
    active_dims_dict = dict()
    active_dims_dict['active_dims_mon_rev'] = active_dims_mon_rev
    active_dims_dict['active_dims_mon_rev_cos'] = active_dims_mon_rev_cos
    active_dims_dict['active_dims_mon_rev_sin'] = active_dims_mon_rev_sin
    active_dims_dict['active_dims_mon_prism'] = active_dims_mon_prism
    active_dims_dict['active_dims_acc_vel'] = np.array(active_dims_acc_vel)
    active_dims_dict['active_dims_acc'] = np.array(active_dims_acc)
    active_dims_dict['active_dims_q_dot'] = np.array(active_dims_q_dot)
    active_dims_dict['active_dims_vel'] = np.array(active_dims_vel)
    active_dims_dict['active_dims_friction'] =  active_dims_friction
    return X, Y, active_dims_dict, data_frame


def normalize_signals(signals, norm_coeff=None):
    """Normalize signals: constraint the module of the signal
    between zero and one"""
    if norm_coeff is None:
        norm_coeff = (np.abs(signals)).max(axis=0)
    return signals/norm_coeff, norm_coeff


def denormalize_signals(signals, norm_coeff):
    """Denormalize signals"""
    return signals*norm_coeff


def get_pandas_obj(output_tr, output_test,
                   noiseless_output_tr, noiseless_output_test,
                   Y_tr_hat_list, Y_test_hat_list,
                   flg_norm,norm_coeff,
                   joint_index_list, output_feature,noiseless_output_feature,
                   var_tr_list=None, var_test_list=None):
    """Denormalize signals and returns a pandas dataset"""
    # check variance signals
    if var_tr_list is None:
        var_tr_list = [np.zeros([output_tr.shape[0],1])
                       for joint_index in range(0, output_tr.shape[1])]
    if var_test_list is None:
        var_test_list = [np.zeros([output_test.shape[0],1])
                         for joint_index in range(0, output_tr.shape[1])]
    # denormalize signals
    if flg_norm:
        Y_tr = denormalize_signals(output_tr, norm_coeff)
        Y_tr_noiseless = denormalize_signals(noiseless_output_tr, norm_coeff)
        Y_test = denormalize_signals(output_test, norm_coeff)
        Y_test_noiseless = denormalize_signals(noiseless_output_test, norm_coeff)
        Y_tr_hat = denormalize_signals(np.concatenate(Y_tr_hat_list,1),
                                                     norm_coeff)
        Y_test_hat = denormalize_signals(np.concatenate(Y_test_hat_list,1),
                                                       norm_coeff)
        var_tr = denormalize_signals(np.concatenate(var_tr_list,1),
                                                   norm_coeff**2)
        var_test = denormalize_signals(np.concatenate(var_test_list,1),
                                                     norm_coeff**2)
    else:
        Y_tr = output_tr
        Y_test = output_test
        Y_tr_noiseless = noiseless_output_tr
        Y_test_noiseless = noiseless_output_test
        Y_tr_hat = np.concatenate(Y_tr_hat_list,1)
        Y_test_hat = np.concatenate(Y_test_hat_list,1)
        var_tr = np.concatenate(var_tr_list,1)
        var_test = np.concatenate(var_test_list,1)
    # Convert predictions in pandas dataset
    output_labels_hat = [output_feature+'_est_'+str(joint+1) for joint in joint_index_list]
    output_labels_hat += ['var_'+output_feature+'_'+str(joint+1) for joint in joint_index_list]
    output_labels = [output_feature+'_'+str(joint+1) for joint in joint_index_list]
    noiseless_output_labels = [noiseless_output_feature+'_'+str(joint+1) for joint in joint_index_list]
    Y_tr_hat_pd = pd.DataFrame(data=np.concatenate([Y_tr_hat, var_tr],1), columns=output_labels_hat)
    Y_test_hat_pd = pd.DataFrame(data=np.concatenate([Y_test_hat, var_test],1), columns=output_labels_hat)
    # Convert output in pandas datasets
    Y_tr_pd = pd.DataFrame(Y_tr, columns=output_labels)
    Y_test_pd = pd.DataFrame(Y_test, columns=output_labels)
    Y_tr_noiseless_pd = pd.DataFrame(Y_tr_noiseless, columns=noiseless_output_labels)
    Y_test_noiseless_pd = pd.DataFrame(Y_test_noiseless, columns=noiseless_output_labels)
    return Y_tr_hat_pd, Y_test_hat_pd, Y_tr_pd, Y_test_pd, Y_tr_noiseless_pd, Y_test_noiseless_pd


def get_acc_shared_par_indices(num_joints):
    """
    Generates the indices defining the Sigma_pos_par vector from the Sigma_pos_par_shared parameters,
    when considering constraints on the accelerations 
    """
    # get the number of parameters
    num_par = int((num_joints**2-num_joints)/2+num_joints)
    # generate the indices
    indices = np.arange(0,num_par)
    # get the indices for each output
    acc_indices = np.zeros((num_joints,num_joints))
    current_index = 0
    for row in range(0,num_joints):
        for column in range(row, num_joints):
            acc_indices[row,column] = indices[current_index]
            acc_indices[column,row] = indices[current_index]
            current_index +=1
    acc_indices_list = [acc_indices[row,:] for row in range(0,num_joints)]
    return num_par, acc_indices_list