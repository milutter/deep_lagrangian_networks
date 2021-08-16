def explicit_euler(params, key, x0_t0, x1_t0, u_t0, forward_model, dt):
    dx0dt_t0, dx1dt_t0 = forward_model(params, key, x0_t0, x1_t0, u_t0)
    x0_t1 = x0_t0 + dt * dx0dt_t0
    x1_t1 = x1_t0 + dt * dx1dt_t0
    return x0_t1, x1_t1


def symplectic_euler(params, key, x0_t0, x1_t0, u_t0, forward_model, dt):
    _, dx1dt_t0 = forward_model(params, key, x0_t0, x1_t0, u_t0)
    x1_t1 = x1_t0 + dt * dx1dt_t0

    dx0_dt_t1, _ = forward_model(params, key, x0_t0, x1_t1, u_t0)
    x0_t1 = x0_t0 + dt * dx0_dt_t1
    return x0_t1, x1_t1


def runge_kutta_4(params, key, x0_t0, x1_t0, u_t0, forward_model, dt):
    x0_k1, x1_k1 = forward_model(params, key, x0_t0, x1_t0, u_t0)
    x0_k2, x1_k2 = forward_model(params, key, x0_t0 + dt/2 * x0_k1, x1_t0 + dt/2 * x1_k1, u_t0)
    x0_k3, x1_k3 = forward_model(params, key, x0_t0 + dt/2 * x0_k2, x1_t0 + dt/2 * x1_k2, u_t0)
    x0_k4, x1_k4 = forward_model(params, key, x0_t0 + dt   * x0_k3, x1_t0 + dt   * x1_k3, u_t0)

    x0_t1 = x0_t0 + dt/6. * (x0_k1 + 2. * x0_k2 + 2. * x0_k3 + x0_k4)
    x1_t1 = x1_t0 + dt/6. * (x1_k1 + 2. * x1_k2 + 2. * x1_k3 + x1_k4)
    return x0_t1, x1_t1
