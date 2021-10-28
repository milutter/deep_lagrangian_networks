import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np


def mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift):
    assert n_dof > 0
    n_output = int((n_dof ** 2 + n_dof) / 2)

    # Calculate the indices of the diagonal elements of L:
    idx_diag = np.arange(n_dof, dtype=int) + 1
    idx_diag = (idx_diag * (idx_diag + 1) / 2 - 1).astype(int)

    # Calculate the indices of the off-diagonal elements of L:
    idx_tril = np.setdiff1d(np.arange(n_output), idx_diag)

    # Indexing for concatenation of l_diagonal  and l_off_diagonal
    cat_idx = np.hstack((idx_diag, idx_tril))
    idx = np.arange(cat_idx.size)[np.argsort(cat_idx)]

    # Compute Matrix Indices
    mat_idx = np.tril_indices(n_dof)
    mat_idx = jax.ops.index[..., mat_idx[0], mat_idx[1]]

    # Compute Mass Matrix
    net = hk.nets.MLP(
        output_sizes= shape + (n_output,),
        activation=activation,
        name="mass_matrix"
    )

    # Apply feature transform:
    z = jnp.concatenate([jnp.cos(q), jnp.sin(q)], axis=-1)
    l_diagonal, l_off_diagonal = jnp.split(net(z), [n_dof,], axis=-1)

    # Ensure positive diagonal:
    # l_diagonal = jax.nn.softplus(l_diagonal) + epsilon
    l_diagonal = jax.nn.softplus(l_diagonal + shift) + epsilon

    vec_lower_triangular = jnp.concatenate((l_diagonal, l_off_diagonal), axis=-1)[..., idx]

    triangular_mat = jnp.zeros((n_dof, n_dof))
    triangular_mat = jax.ops.index_update(triangular_mat, mat_idx, vec_lower_triangular[:])

    mass_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    return mass_mat


def kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift):
    mass_mat = mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift)
    return 1. / 2. * jnp.dot(qd, jnp.dot(mass_mat, qd))


def potential_energy_fn(q, shape, activation):
    net = hk.nets.MLP(output_sizes= shape + (1,),
                      activation=activation,
                      name="potential_energy")

    # Apply feature transform
    z = jnp.concatenate([jnp.cos(q), jnp.sin(q)], axis=-1)
    return net(z)


def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
    e_kin = kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift)
    e_pot = potential_energy_fn(q, shape, activation).squeeze()
    return e_kin - e_pot


def blackbox_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
    del epsilon, n_dof
    net = hk.nets.MLP(output_sizes= shape + (1,),
                      activation=activation,
                      name="lagrangian")

    # Apply feature transform
    z = jnp.concatenate([jnp.cos(q), jnp.sin(q)], axis=-1)
    state = jnp.concatenate([z, qd], axis=-1)
    return net(state).squeeze()


def euler_lagrange_equation(params, key, q, qd, qdd, lagrangian):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)
    batch_matmul = jax.vmap(jnp.matmul, (0, 0))

    # Compute Lagrangian and Jacobians:
    lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
    L, (dLdq, dLdqd) = jax.vmap(lagrangian_value_and_grad, vmap_dim)(params, key, q, qd)
    print(f"L = {L.shape} - dLdq = {dLdq.shape} - dLdqd = {dLdqd.shape}")

    # Compute Hessian:
    lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
    (_, (d2L_dqddq, d2Ld2qd)) = jax.vmap(lagrangian_hessian, vmap_dim)(params, key, q, qd)

    # Compute the predicted generalized force:
    tau_pred = batch_matmul(d2Ld2qd, qdd) + batch_matmul(d2L_dqddq, qd) - dLdq
    return tau_pred


def dynamics_model(params, key, q, qd, qdd, tau, lagrangian, n_dof):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)
    batch_matmul = jax.vmap(jnp.matmul, (0, 0))
    batch_inverse = jax.vmap(jnp.linalg.inv, (0,))

    # Compute Lagrangian and Jacobians:
    lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
    L, (dLdq, dLdqd) = jax.vmap(lagrangian_value_and_grad, vmap_dim)(params, key, q, qd)

    # Compute Hessian:
    lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
    (_, (d2L_dqddq, d2Ld2qd)) = jax.vmap(lagrangian_hessian, vmap_dim)(params, key, q, qd)

    # Compute the inverse model:
    tau_pred = batch_matmul(d2Ld2qd, qdd) + batch_matmul(d2L_dqddq, qd) - dLdq

    # Compute the forward model:
    qdd_pred = batch_matmul(batch_inverse(d2Ld2qd + 1.e-4 * jnp.eye(n_dof)), (tau - batch_matmul(d2L_dqddq, qd) + dLdq))

    # Compute Hamiltonian & dH/dt:
    H = jax.vmap(jnp.dot, [0, 0])(dLdqd, qd) - L
    dHdt = jax.vmap(jnp.dot, [0, 0])(qd, tau_pred)
    return qdd_pred, tau_pred, H, dHdt


def forward_model(params, key, q, qd, tau, lagrangian, n_dof):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)
    batch_matmul = jax.vmap(jnp.matmul, (0, 0))
    batch_inverse = jax.vmap(jnp.linalg.inv, (0,))

    # Compute Lagrangian and Jacobians:
    lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
    L, (dLdq, dLdqd) = jax.vmap(lagrangian_value_and_grad, vmap_dim)(params, key, q, qd)

    # Compute Hessian:
    lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
    (_, (d2L_dqddq, d2Ld2qd)) = jax.vmap(lagrangian_hessian, vmap_dim)(params, key, q, qd)

    # Compute the forward model:
    qdd_pred = batch_matmul(batch_inverse(d2Ld2qd + 1.e-4 * jnp.eye(n_dof)), (tau - batch_matmul(d2L_dqddq, qd) + dLdq))
    return qd, qdd_pred


def inverse_model(params, key, q, qd, qdd, lagrangian):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)
    batch_matmul = jax.vmap(jnp.matmul, (0, 0))

    # Compute Lagrangian and Jacobians:
    lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
    L, (dLdq, dLdqd) = jax.vmap(lagrangian_value_and_grad, vmap_dim)(params, key, q, qd)

    # Compute Hessian:
    lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
    (_, (d2L_dqddq, d2Ld2qd)) = jax.vmap(lagrangian_hessian, vmap_dim)(params, key, q, qd)

    # Compute the inverse model:
    tau_pred = batch_matmul(d2Ld2qd, qdd) + batch_matmul(d2L_dqddq, qd) - dLdq
    return tau_pred


def inverse_loss_fn(params, q, qd, qdd, tau, lagrangian, n_dof, norm_tau, norm_qdd):
    qdd_pred, tau_pred, H_pred, dHdt_pred = dynamics_model(params, None, q, qd, qdd, tau, lagrangian, n_dof)

    # Forward Error
    qdd_error = jnp.sum((qdd - qdd_pred)**2 / norm_qdd, axis=-1)
    mean_forward_error = jnp.mean(qdd_error)
    var_forward_error = jnp.var(qdd_error)

    # Inverse Error:
    tau_error = jnp.sum((tau - tau_pred)**2 / norm_tau, axis=-1)
    mean_inverse_error = jnp.mean(tau_error)
    var_inverse_error = jnp.mean(tau_error)

    # Temporal Energy Conservation:
    dHdt = jax.vmap(jnp.dot, [0, 0])(qd, tau)
    dHdt_error = (dHdt_pred - dHdt) ** 2
    mean_energy_error = jnp.mean(dHdt_error)
    var_energy_error = jnp.var(dHdt_error)

    # Compute Loss
    loss = mean_inverse_error

    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
        'inverse_mean': mean_inverse_error,
        'inverse_var': var_inverse_error,
        'energy_mean': mean_energy_error,
        'energy_var': var_energy_error,
    }
    return loss, logs


def rollout(params, key, q0, qd0, p0, tau, lagrangian, forward_model, integrator, dt):
    L, (dLdqd,) = jax.value_and_grad(lagrangian, argnums=[3, ])(params, key, q0[0], qd0[0])
    H0 = (jnp.dot(dLdqd, qd0[0]) - L)[jnp.newaxis]

    def step(x, u):
        q, qd = x
        q_n, qd_n = integrator(params, key, q, qd, u, forward_model, dt)
        L, (dLdqd,) = jax.value_and_grad(lagrangian, argnums=[3,])(params, key, q_n[0], qd_n[0])

        p_n = dLdqd
        H = jnp.dot(dLdqd, qd_n[0]) - L
        return (q_n, qd_n), (q_n, qd_n, p_n[jnp.newaxis], H[jnp.newaxis])

    _, (q, qd, p, H) = jax.lax.scan(step, (q0, qd0), tau[:-1])

    # Append initial value to trajectory
    q, qd, p, H = jax.tree_map(
        lambda x0, x: jnp.concatenate([x0, x[:, 0]], axis=0),
        (q0, qd0, p0, H0), (q, qd, p, H))

    return q, qd, p, H
