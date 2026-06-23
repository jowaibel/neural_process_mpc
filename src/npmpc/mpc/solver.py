import casadi as ca


def configure_solver(problem: ca.Opti, solver_name: str):
    """Configure the CasADi Opti solver."""
    if solver_name == 'ipopt':
        problem.solver('ipopt', {
            'ipopt.max_iter': 50,
            'ipopt.tol': 1e-6,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
        })
    else:
        raise NotImplementedError(f"Solver '{solver_name}' not implemented")
