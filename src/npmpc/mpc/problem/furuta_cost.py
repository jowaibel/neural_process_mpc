import casadi as ca


def furuta_cost(x: ca.MX, u: ca.MX, params: dict) -> ca.MX:
    """Total Furuta MPC cost: stage + interstage + LQR terminal cost.

    All three pieces share the 2π-periodic half-angle lift θ ↦ 2·sin(θ/2) on the
    pendulum angle so all upright configurations cost the same. They are split
    into the functions below so callers can take individual terms (e.g.
    closed-loop metrics that want the running cost without the terminal term).
    """
    return (stage_cost(x, u, params)
            + interstage_cost(x, u, params)
            + terminal_cost(x, params))


def stage_cost(x: ca.MX, u: ca.MX, params: dict) -> ca.MX:
    """State + input stage cost over the first N steps (N = u.shape[0]). The
    pendulum angle uses the squared half-angle lift (2·sin(θ/2))² = 2·(1 - cos θ)."""
    N = u.shape[0]

    state_stage = (params['x'][0] * 2.0 * (1.0 - ca.cos(x[:N, 0]))
                 + params['x'][1] * x[:N, 1]**2
                 + params['x'][2] * x[:N, 2]**2
                 + params['x'][3] * x[:N, 3]**2)
    state_cost = ca.sum1(state_stage)
    input_cost = ca.sum1(params['u'][0] * u[:N, 0]**2)
    return state_cost + input_cost


def interstage_cost(x: ca.MX, u: ca.MX, params: dict) -> ca.MX:
    """Differential cost on the N state transitions x_{k+1} - x_k (discourages
    jerky trajectories)."""
    N = u.shape[0]

    dx = x[1:N+1, :] - x[:N, :]
    diff_stage = (params['x_diff'][0] * dx[:, 0]**2
                + params['x_diff'][1] * dx[:, 1]**2
                + params['x_diff'][2] * dx[:, 2]**2
                + params['x_diff'][3] * dx[:, 3]**2)
    return ca.sum1(diff_stage)


def terminal_cost(x: ca.MX, params: dict) -> ca.MX:
    """LQR terminal cost on the lifted error e_N = [2·sin(θ_N/2), φ_N, θ̇_N, φ̇_N];
    P is the Riccati solution from FurutaNPMPC.compute_terminal_P."""
    P = ca.DM(params['terminal_P'])
    xN = x[-1, :]                                       # (1, 4)
    e_N = ca.horzcat(2.0 * ca.sin(xN[0] / 2.0),
                     xN[1], xN[2], xN[3])               # (1, 4)
    return e_N @ P @ e_N.T                              # (1, 1)
