import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np


def dynamics_network(q, qd, qdd, tau, n_dof, shape, activation, epsilon, shift):
    forward_net = hk.nets.MLP(output_sizes= shape + (n_dof,),
                      activation=activation,
                      name="forward_model")

    inverse_net = hk.nets.MLP(output_sizes= shape + (n_dof,),
                      activation=activation,
                      name="inverse_model")

    # Apply feature transform
    z = jnp.concatenate([jnp.cos(q), jnp.sin(q)], axis=-1)
    qdd_pred = forward_net(jnp.concatenate([z, qd, tau], axis=-1))
    tau_pred = inverse_net(jnp.concatenate([z, qd, qdd], axis=-1))
    return qdd_pred, tau_pred


def dynamics_model(params, key, q, qd, qdd, tau, black_box_model, n_dof):
    n_samples = q.shape[0]

    # Compute the forward & inverse model:
    qdd_pred, tau_pred = black_box_model(params, key, q, qd, qdd, tau)

    # Compute Hamiltonian & dH/dt:
    H = jnp.zeros(n_samples)
    dHdt = jax.vmap(jnp.dot, [0, 0])(qd, tau_pred)
    return qdd_pred, tau_pred, H, dHdt


def forward_model(params, key, q, qd, tau, black_box_model, n_dof):
    qdd_pred, _ = black_box_model(params, key, q, qd, qd * 0.0, tau)
    return qd, qdd_pred


def inverse_model(params, key, q, qd, qdd, black_box_model):
    _, tau_pred = black_box_model(params, key, q, qd, qdd, qdd * 0.0)
    return tau_pred


def loss_fn(params, q, qd, qdd, tau, model, n_dof, norm_tau, norm_qdd):
    qdd_pred, tau_pred, H_pred, dHdt_pred = dynamics_model(params, None, q, qd, qdd, tau, model, n_dof)

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
    loss = mean_inverse_error + mean_forward_error

    logs = {
        'n_batch': 1,
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
        'inverse_mean': mean_inverse_error,
        'inverse_var': var_inverse_error,
        'energy_mean': mean_energy_error,
        'energy_var': var_energy_error,
    }
    return loss, logs


def rollout(params, key, q0, qd0, p0, tau, black_box_model, forward_model, integrator, dt):
    inv_model = jax.partial(inverse_model, black_box_model=black_box_model)
    H0 = jnp.zeros((1,))

    def step(x, u):
        q, qd = x
        q_n, qd_n = integrator(params, key, q, qd, u[jnp.newaxis], forward_model, dt)

        # Compute Mass Matrix by approximation from BLACK-BOX Forward Model:
        e10 = jnp.stack([jnp.ones(q.shape[0]), jnp.zeros(q.shape[0])], axis=-1)
        e01 = jnp.stack([jnp.zeros(q.shape[0]), jnp.ones(q.shape[0])], axis=-1)
        zeros = jnp.zeros_like(q)

        tau_g = inv_model(params, None, q_n, zeros, zeros)
        mass_mat_00 = inv_model(params, None, q_n, zeros, e10) - tau_g
        mass_mat_11 = inv_model(params, None, q_n, zeros, e01) - tau_g

        mass_mat = jnp.stack([mass_mat_00, mass_mat_11], axis=-1)
        p_n = jnp.matmul(mass_mat[0], qd_n[0])[jnp.newaxis]
        return (q_n, qd_n), (q_n, qd_n, p_n, jnp.zeros((1,)))

    _, (q, qd, p, H) = jax.lax.scan(step, (q0, qd0), tau[:-1])

    # Append initial value to trajectory
    q, qd, p, H = jax.tree_map(
        lambda x0, x: jnp.concatenate([x0, x[:, 0]], axis=0),
        (q0, qd0, p0, H0), (q, qd, p, H))

    return q, qd, p, H
