"""
Utilities Source File

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""
import numpy as np
import pandas as pd

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
    output_labels = [features_name + '_' + str(joint + 1) for joint in joint_index_list]
    predictions_pd = pd.DataFrame(prediction, columns=output_labels)

    return predictions_pd