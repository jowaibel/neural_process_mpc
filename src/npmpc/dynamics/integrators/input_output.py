from npmpc.dynamics.integrators import IntegrationBase
import torch


class InputOutputIntegration(IntegrationBase):

    def integrate(self, x0: torch.Tensor, u: torch.Tensor, dt: float, p: torch.Tensor=None):
        x = x0.unsqueeze(-2).repeat((*([1]*len(x0.shape[:-1])), 2, 1))
        x[...,1,:] = self.dynamics.dynamics(x[...,0,:], None, p)
        return x