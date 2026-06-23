from npmpc.dynamics.integrators import IntegrationBase
import torch
import casadi as ca


class MidpointIntegration(IntegrationBase):

    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float, p: torch.Tensor=None):
        k1 = self.dynamics.dynamics(x, u, p)
        k2 = self.dynamics.dynamics(x + dt/2*k1, u, p)
        return x + dt*k2


    def integrate(self, x0: torch.Tensor, u: torch.Tensor, dt: float, p: torch.Tensor=None):
        x = x0.unsqueeze(-2).repeat((*([1]*len(x0.shape[:-1])), u.shape[-2]+1, 1))
        for i in range(u.shape[-2]):
            x[...,i+1,:] = self.step(x[...,i,:], u[...,i,:], dt, p)
        return x


    def casadi_explicit(self, x0: ca.SX, u: ca.SX, dt: float, p: ca.SX=None):
        x = ca.SX.zeros(u.size1()+1, x0.size2())
        x[0] = x0[0]
        for i in range(u.size1()):
            k1 = self.dynamics.casadi_dynamics(x[i,:], u[i,:], p)
            k2 = self.dynamics.casadi_dynamics(x[i,:] + dt/2*k1, u[i,:], p)
            x[i+1,:] = x[i,:] + dt*k2
        return x


    def casadi_implicit(self, x: ca.MX, u: ca.MX, dt: float, p: ca.MX=None):
        constraints = []
        for i in range(u.size1()):
            constraints.append(x[i+1,:] == x[i,:] + dt*self.dynamics.casadi_dynamics((x[i,:]+x[i+1,:])/2, u[i,:], p))
        return constraints
