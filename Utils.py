"""
Utilities Source File

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory


def unpack_dataset_joint_variables(dataset, n_dof):
    """
        Unpacks a dataset in the format with examples in rows with joint positions, velocities and accelerations in
        columns (in that order)

        Returns matrices q, qv, qa; containing rows of examples for joint positions, velocities and accelerations
    """
    q = dataset[:, 0:n_dof]  # joint positions
    qv = dataset[:, n_dof:n_dof * 2]  # joint velocities
    qa = dataset[:, n_dof * 2:]  # joint accelerations

    return q, qv, qa


def convert_predictions_to_dataset(prediction, features_name, joint_index_list):
    output_labels = []
    for feat_name in features_name:
        output_labels += [feat_name + '_' + str(joint + 1) for joint in joint_index_list]
    predictions_pd = pd.DataFrame(prediction, columns=output_labels)

    return predictions_pd


def nMSE(y, y_hat):
    num_sample = y.size
    return np.sum((y - y_hat) ** 2) / num_sample / np.var(y)


def MSE(y, y_hat):
    num_sample = y.size
    return np.sum((y - y_hat) ** 2) / num_sample


def k_fold_cross_val_model_selection(num_dof, optimizer_lambda, dataset, targets, hyperparameters_list, k_folds=5, flg_cuda=True):
    """
        Perfroms k-fold cross validation for model selection
    """

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=69)

    # Start print


    results_nmse_models=[]

    for model_idx, hyperparameters in enumerate(hyperparameters_list):
        print('\n\n--------------------------------')
        print(f'{model_idx}: training with hyperparamers: {hyperparameters}', flush=True)
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Print
            print(f'FOLD {fold}', flush=True)
            print('--------------------------------', flush=True)

            # Sample elements randomly from a given list of ids, no replacement.
            train_data = dataset[train_ids]
            train_targets = targets[train_ids]

            val_data = dataset[val_ids]
            val_targets = targets[val_ids]

            # Init the neural network
            delan_model = DeepLagrangianNetwork(num_dof, **hyperparameters)
            delan_model = delan_model.cuda(torch.device('cuda:0')) if flg_cuda else delan_model.cpu()
            optimizer = optimizer_lambda(delan_model.parameters(), hyperparameters["learning_rate"],
                                         hyperparameters["weight_decay"], True)
            delan_model.train_model(train_data, train_targets, optimizer, save_model=False)

            # Process is complete.
            print('Training process has finished.', flush=True)

            # Print about testing
            print('Starting testing', flush=True)

            # Evaluationfor this fold
            with torch.no_grad():
                est_val_targets = delan_model.evaluate(val_data)
            nMse = nMSE(val_targets, est_val_targets)
            results[fold] = nMse

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum_nmse = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value}')
            sum_nmse += value
        avg = sum_nmse / len(results.items())
        print(f'Average: {avg}')
        results_nmse_models.append(avg)

    return np.argmax(np.array(results_nmse_models))
