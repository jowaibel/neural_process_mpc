import numpy as np
import torch
import casadi as ca
from scipy.linalg import solve_discrete_are

from npmpc.dynamics.furuta import FurutaDynamics
from npmpc.dynamics.integrators import IntegrationBase, INTEGRATOR_REGISTRY
from npmpc.mpc.problem.MPCBase import MPCBase
from npmpc.mpc.problem.furuta_cost import furuta_cost

class FurutaMPC(MPCBase):
    integrator: IntegrationBase

    p: torch.Tensor
    p_opti: ca.DM

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

        self.integrator = INTEGRATOR_REGISTRY[self.params['integrator']](FurutaDynamics)
        self.p = torch.Tensor(self.params['p'])
        self.p_opti = ca.DM(self.params['p'])

        self.r = None

        # Pre-compute the LQR terminal cost matrix into params['cost']['terminal_P'].
        self.compute_terminal_P()


    def integrate(self, x0, u, dt):
        return self.integrator.integrate(x0, u, dt, p=self.p)


    def integration_constraints(self, x, u, dt):
        return self.integrator.casadi_implicit(x, u, dt, p=self.p_opti)


    def linearize(self):
        """Linearise the analytical Furuta one-step map at the upright
        equilibrium; returns (A_d, B_d) shaped (4, 4) and (4, 1).

        Uses the configured integrator's single-step rule (integrator.step) so
        the terminal LQR weight P matches the dynamics the optimiser enforces.
        step is functional, so it is safe to differentiate through with
        torch.autograd.functional.jacobian.
        """
        x_eq = torch.zeros(4, dtype=torch.float32)
        u_eq = torch.zeros(1, dtype=torch.float32)
        dt = self.params['dt']
        # Flatten params to a 1-D (un-batched) parameter set for FurutaDynamics.dynamics.
        p = self.p.flatten()

        A = torch.autograd.functional.jacobian(lambda x: self.integrator.step(x, u_eq, dt, p), x_eq)
        B = torch.autograd.functional.jacobian(lambda u: self.integrator.step(x_eq, u, dt, p), u_eq)
        return A.detach().numpy(), B.detach().numpy()


    def compute_terminal_P(self) -> np.ndarray:
        """Solve the discrete-time ARE for the LQR terminal cost from the
        linearised analytical dynamics; writes it to params['cost']['terminal_P']."""
        A_d, B_d = self.linearize()
        Q = np.diag(self.params['cost']['x_end']).astype(np.float64)
        R = np.diag(self.params['cost']['u']).astype(np.float64)
        P = solve_discrete_are(A_d.astype(np.float64), B_d.astype(np.float64), Q, R)
        self.params['cost']['terminal_P'] = P.tolist()
        return P


    def cost_function(self, x, u):
        return furuta_cost(x, u, self.params['cost'])
