import torch
import casadi as ca

from npmpc.dynamics.furuta import FurutaDynamics
from npmpc.nps.NeuralProcess import NeuralProcess


class FurutaNPIntegration(torch.nn.Module):
    dynamics = FurutaDynamics
    np: NeuralProcess

    def __init__(self, np):
        super().__init__()
        self.np = np


    def integrate(self, x0: torch.Tensor, u: torch.Tensor, dt: float, z: torch.Tensor = None):
        """Integrate using the NP decoder mean output."""
        n_steps = u.shape[-2]
        z = z.reshape(-1)                                   # latent, (z_dim,)

        x = x0.unsqueeze(-2).repeat((*([1] * len(x0.shape[:-1])), n_steps+1, 1))
        for i in range(n_steps):
            x_nn = torch.cat([torch.sin(x[..., i, 0]).unsqueeze(-1), torch.cos(x[..., i, 0]).unsqueeze(-1),
                              x[..., i, 2:4], u[..., i, :]], dim=-1).unsqueeze(-2)
            y_nn, _ = self.np.decode(x_nn, z=z)
            y_nn = y_nn.squeeze(-2)
            x[...,i+1,:] = x[...,i,:] + torch.cat([dt*(x[...,i,2:4]+y_nn/2), y_nn], dim=-1)
        return x


    def montecarlo_integrate(self, x0: torch.Tensor, u: torch.Tensor, dt: float,
                             z: torch.Tensor = None, n: int = 100):
        """Sample n rollouts from the NP output (output noise) given a fixed latent z."""
        n_steps = u.shape[-2]
        z_mc = z.reshape(-1).unsqueeze(0).expand(n, -1)     # same z replicated n times, (n, z_dim)
        u_mc = u.unsqueeze(-3).expand(*u.shape[:-2], n, *u.shape[-2:])

        x = x0.unsqueeze(-2).unsqueeze(-2).repeat((*([1] * len(x0.shape[:-1])), n, n_steps + 1, 1))
        for i in range(n_steps):
            x_nn = torch.cat([torch.sin(x[..., i, 0]).unsqueeze(-1), torch.cos(x[..., i, 0]).unsqueeze(-1),
                              x[..., i, 2:4], u_mc[..., i, :]], dim=-1).unsqueeze(-2)
            y_mu, y_sigma = self.np.decode(x_nn, z=z_mc)
            y_mu = y_mu.squeeze(-2)
            y_sigma = y_sigma.squeeze(-2)
            y_sample = torch.distributions.Normal(y_mu, y_sigma).sample()
            x[...,i+1,:] = x[...,i,:] + torch.cat([dt*(x[...,i,2:4]+y_sample/2), y_sample], dim=-1)
            x[...,i+1,2:4] = torch.clamp(x[...,i+1,2:4], -50, 50)
        return x


    def casadi_implicit(self, x: ca.MX, u: ca.MX, dt: float, z=None):
        """CasADi implicit integration constraints: x[i+1] == f(x[i], u[i])."""
        constraints = []
        for i in range(u.size1()):
            x_nn = ca.horzcat(ca.sin(x[i, 0]), ca.cos(x[i, 0]),
                              x[i, 2:4], u[i, :])
            y_nn = self.np.casadi_decoder(x_nn, z=z)
            constraints.append(x[i+1, :] == x[i, :] + ca.horzcat(dt * (x[i, 2:4] + y_nn / 2), y_nn))
        return constraints


    def casadi_explicit(self, x0: ca.MX, u: ca.MX, dt: float, z=None):
        """CasADi explicit integration: returns the full predicted trajectory."""
        n_steps = u.size1()
        x = ca.MX.zeros(n_steps + 1, x0.size2())
        x[0, :] = x0
        for i in range(n_steps):
            x_nn = ca.horzcat(ca.sin(x[i, 0]), ca.cos(x[i, 0]),
                              x[i, 2:4], u[i, :])
            y_nn = self.np.casadi_decoder(x_nn, z=z)
            x[i + 1, :] = x[i, :] + ca.horzcat(dt * (x[i, 2:4] + y_nn / 2), y_nn)
        return x
