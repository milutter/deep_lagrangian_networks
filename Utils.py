"""
Utilities Source File

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""


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
