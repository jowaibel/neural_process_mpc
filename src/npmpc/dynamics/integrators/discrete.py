from npmpc.dynamics.integrators import IntegrationBase
import torch
import casadi as ca


class DiscreteIntegration(IntegrationBase):

    def integrate(self, x0: torch.Tensor, u: torch.Tensor, dt: float=None, p: torch.Tensor=None):
        x = x0.unsqueeze(-2).repeat((*([1]*len(x0.shape[:-1])), u.shape[-2]+1, 1))
        for i in range(u.shape[-2]):
            x[...,i+1,:] = self.dynamics.dynamics(x[...,i,:], u[...,i,:], p)
        return x
    
    
    def casadi_explicit(self, x0: ca.SX, u: ca.SX, dt: float, p: ca.SX=None):
        x = ca.SX.zeros(u.size1()+1, x0.size2())
        x[0] = x0[0]
        for i in range(u.size1()):
            x[i+1,:] = self.dynamics.casadi_dynamics(x[i,:], u[i,:], p)
        return x


    def casadi_implicit(self, x: ca.MX, u: ca.MX, dt: float, p: ca.MX=None):
        constraints = []
        for i in range(u.size1()):
            constraints.append(x[i+1,:] == self.dynamics.casadi_dynamics(x[i,:], u[i,:], p))
        return constraints