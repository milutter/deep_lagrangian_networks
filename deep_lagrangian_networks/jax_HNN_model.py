import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from deep_lagrangian_networks.jax_DeLaN_model import mass_matrix_fn, potential_energy_fn

# For HNN the predicted mass matrix inverse is composed of the LL^{T} + eps I.
inv_mass_matrix_fn = mass_matrix_fn


def kinetic_energy_fn(q, p, n_dof, shape, activation, epsilon, shift):
    inv_mass_mat = inv_mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift)
    return 1. / 2. * jnp.dot(p, jnp.dot(inv_mass_mat, p))


def structured_hamiltonian_fn(q, p, n_dof, shape, activation, epsilon, shift):
    e_kin = kinetic_energy_fn(q, p, n_dof, shape, activation, epsilon, shift)
    e_pot = potential_energy_fn(q, shape, activation).squeeze()
    return e_kin + e_pot


def blackbox_hamiltonian_fn(q, p, n_dof, shape, activation, epsilon, shift):
    del epsilon, n_dof
    net = hk.nets.MLP(output_sizes= shape + (1,),
                      activation=activation,
                      name="hamiltonian")

    z = jnp.concatenate([jnp.cos(q), jnp.sin(q)], axis=-1)
    state = jnp.concatenate([z, p], axis=-1)
    return net(state).squeeze()


def hamiltons_equation(params, key, q, qd, p, tau, hamiltonian):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)
    batch_matmul = jax.vmap(jnp.matmul, (0, 0))

    # Compute Hamiltonian and Jacobians:
    hamiltonian_value_and_grad = jax.value_and_grad(hamiltonian, argnums=argnums)
    H, (dHdq, dHdp) = jax.vmap(hamiltonian_value_and_grad, vmap_dim)(params, key, q, p)
    print(f"H = {H.shape} - dHdq = {dHdq.shape} - dHdp = {dHdp.shape}")

    # Compute the predicted velocity and \dot{impulse}:
    pd_pred = tau - dHdq
    qd_pred = dHdp

    return qd_pred, pd_pred


def dynamics_model(params, key, q, p, pd, tau, hamiltonian, n_dof):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)
    batch_matmul = jax.vmap(jnp.matmul, (0, 0))

    # Compute Lagrangian and Jacobians:
    hamiltonian_value_and_grad = jax.value_and_grad(hamiltonian, argnums=argnums)
    H, (dHdq, dHdp) = jax.vmap(hamiltonian_value_and_grad, vmap_dim)(params, key, q, p)

    # Compute the forward model:
    pd_pred = tau - dHdq
    qd_pred = dHdp

    # Compute the inverse model:
    tau_pred = pd + dHdq

    # Compute \dot{\mathcal{H}}
    dHdt = jax.vmap(jnp.dot, [0, 0])(dHdp, tau_pred)

    return qd_pred, pd_pred, tau_pred, H, dHdt


def forward_model(params, key, q, p, tau, hamiltonian, n_dof):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)

    # Compute Lagrangian and Jacobians:
    hamiltonian_value_and_grad = jax.value_and_grad(hamiltonian, argnums=argnums)
    H, (dHdq, dHdp) = jax.vmap(hamiltonian_value_and_grad, vmap_dim)(params, key, q, p)

    # Compute the forward model:
    pd_pred = tau - dHdq
    qd_pred = dHdp

    return qd_pred, pd_pred


def inverse_model(params, key, q, p, pd, hamiltonian):
    argnums = [2, 3]
    vmap_dim = (None, None, 0, 0)

    # Compute Lagrangian and Jacobians:
    hamiltonian_value_and_grad = jax.value_and_grad(hamiltonian, argnums=argnums)
    H, (dHdq, dHdp) = jax.vmap(hamiltonian_value_and_grad, vmap_dim)(params, key, q, p)

    # Compute the inverse model:
    tau_pred = pd + dHdq
    return tau_pred


def forward_loss_fn(params, q, qd, p, pd, tau, hamiltonian, n_dof, norm_tau, norm_qd, norm_pd):
    qd_pred, pd_pred, tau_pred, H_pred, dHdt_pred = dynamics_model(params, None, q, p, pd, tau, hamiltonian, n_dof)

    # Forward Error:
    qd_error = jnp.sum((qd - qd_pred)**2 / norm_qd, axis=-1)
    pd_error = jnp.sum((pd - pd_pred) ** 2 / norm_pd, axis=-1)

    mean_forward_error = 1. / 2. * (jnp.mean(qd_error) + jnp.mean(pd_error))
    var_forward_error = 1./2. *  (jnp.var(qd_error) + jnp.var(pd_error))

    # Inverse Error:
    tau_error = jnp.sum((tau - tau_pred)**2 / norm_tau, axis=-1)
    mean_inverse_error = jnp.mean(tau_error)
    var_inverse_error = jnp.var(tau_error)

    # Temporal Power Conservation:
    dHdt = jax.vmap(jnp.dot, [0, 0])(qd, tau)
    dHdt_error = (dHdt_pred - dHdt) ** 2
    mean_energy_error = jnp.mean(dHdt_error)
    var_energy_error = jnp.var(dHdt_error)

    # Compute Loss
    loss = mean_forward_error

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


def rollout(params, key, q0, qd0, p0, tau, hamiltonian, forward_model, integrator, dt):
    H0 = hamiltonian(params, key, q0[0], p0[0])[jnp.newaxis]

    def step(x, u):
        q, p = x
        q_n, p_n = integrator(params, key, q, p, u, forward_model, dt)
        H_n, (qd_n,) = jax.value_and_grad(hamiltonian, argnums=[3,])(params, key, q_n[0], p_n[0])
        return (q_n, p_n), (q_n, qd_n[jnp.newaxis], p_n, H_n[jnp.newaxis])

    _, (q, qd, p, H) = jax.lax.scan(step, (q0, p0), tau[:-1])

    # Append initial value to trajectory
    q, qd, p, H = jax.tree_map(
        lambda x0, x: jnp.concatenate([x0, x[:, 0]], axis=0),
        (q0, qd0, p0, H0), (q, qd, p, H))

    return q, qd, p, H
