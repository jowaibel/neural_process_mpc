import numpy as np
import torch
import casadi as ca
from scipy.linalg import solve_discrete_are

from npmpc.dynamics.integrators import FurutaNPIntegration
from npmpc.mpc.problem.MPCBase import MPCBase
from npmpc.mpc.problem.furuta_cost import furuta_cost


class FurutaNPMPC(MPCBase):
    integrator: FurutaNPIntegration
    z: torch.Tensor
    z_casadi: object  # ca.DM latent code for the CasADi decoder

    def __init__(self, model, params: dict):
        super().__init__()
        self.params = params

        self.integrator = FurutaNPIntegration(model)
        self.z = torch.tensor(self.params['z'], dtype=torch.float32) if self.params['z'] is not None else None
        self.z_casadi = model.decoder.casadi_z(self.params['z']) if self.params['z'] is not None else None

        self.p = None
        if self.z is not None:
            self.compute_terminal_P()


    def integrate(self, x0, u, dt):
        return self.integrator.integrate(x0, torch.as_tensor(u, dtype=torch.float32), dt, z=self.z.unsqueeze(-2))


    def integration_constraints(self, x, u, dt):
        return self.integrator.casadi_implicit(x, u, dt, z=self.z_casadi)


    def linearize(self):
        """Linearise the NP dynamics x_{k+1}=F(x_k,u_k;z) at the upright
        equilibrium; returns (A_d, B_d) shaped (4, 4) and (4, 1).

        Re-implements the single-step rule here because the integrator's
        in-place indexing breaks torch.autograd.functional.jacobian.
        """
        x_eq = torch.zeros(4, dtype=torch.float32)
        u_eq = torch.zeros(1, dtype=torch.float32)
        dt = self.params['dt']
        # The decoder broadcasts z (one fewer dim than x_nn) over the points
        # dim, so a flat `(z_dim,)` latent pairs with the `(1, 5)` x_nn below.
        z = self.z.reshape(-1)

        def step(x, u):
            x_nn = torch.cat([
                torch.sin(x[0:1]), torch.cos(x[0:1]),
                x[2:4], u,
            ], dim=-1).unsqueeze(-2)                            # (1, 5)
            y_nn, _ = self.integrator.np.decode(x_nn, z=z)
            y_nn = y_nn.reshape(-1)                             # → (n_y,)
            return x + torch.cat([dt * (x[2:4] + y_nn / 2), y_nn], dim=-1)

        A = torch.autograd.functional.jacobian(lambda x: step(x, u_eq), x_eq)
        B = torch.autograd.functional.jacobian(lambda u: step(x_eq, u), u_eq)
        return A.detach().numpy(), B.detach().numpy()


    def compute_terminal_P(self) -> np.ndarray:
        """Solve the discrete-time ARE for the LQR terminal cost from the
        linearised NP dynamics; writes it to params['cost']['terminal_P']."""
        A_d, B_d = self.linearize()
        Q = np.diag(self.params['cost']['x_end']).astype(np.float64)
        R = np.diag(self.params['cost']['u']).astype(np.float64)
        P = solve_discrete_are(A_d.astype(np.float64), B_d.astype(np.float64), Q, R)
        self.params['cost']['terminal_P'] = P.tolist()
        return P


    def cost_function(self, x, u):
        return furuta_cost(x, u, self.params['cost'])
