from npmpc.dynamics.integrators import IntegrationBase
import torch
import casadi as ca


class TrapezoidalIntegration(IntegrationBase):
    """Trapezoidal integration of x' = f(x, u; p), holding u constant over each
    interval.

    * step / integrate / casadi_explicit: explicit trapezoid (Heun's method) for
      forward rollouts: x[i+1] = x[i] + dt/2·(f(x[i],u[i]) + f(x[i]+dt·f, u[i])).
    * casadi_implicit: implicit trapezoidal collocation used as the optimiser's
      dynamics equality: x[i+1] == x[i] + dt/2·(f(x[i],u[i]) + f(x[i+1],u[i])).
    """

    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float, p: torch.Tensor=None):
        f0 = self.dynamics.dynamics(x, u, p)
        f1 = self.dynamics.dynamics(x + dt*f0, u, p)
        return x + dt/2*(f0 + f1)


    def integrate(self, x0: torch.Tensor, u: torch.Tensor, dt: float, p: torch.Tensor=None):
        x = x0.unsqueeze(-2).repeat((*([1]*len(x0.shape[:-1])), u.shape[-2]+1, 1))
        for i in range(u.shape[-2]):
            x[...,i+1,:] = self.step(x[...,i,:], u[...,i,:], dt, p)
        return x


    def casadi_explicit(self, x0: ca.SX, u: ca.SX, dt: float, p: ca.SX=None):
        x = ca.SX.zeros(u.size1()+1, x0.size2())
        x[0] = x0[0]
        for i in range(u.size1()):
            f0 = self.dynamics.casadi_dynamics(x[i,:], u[i,:], p)
            f1 = self.dynamics.casadi_dynamics(x[i,:] + dt*f0, u[i,:], p)
            x[i+1,:] = x[i,:] + dt/2*(f0 + f1)
        return x


    def casadi_implicit(self, x: ca.MX, u: ca.MX, dt: float, p: ca.MX=None):
        constraints = []
        for i in range(u.size1()):
            f0 = self.dynamics.casadi_dynamics(x[i,:], u[i,:], p)
            f1 = self.dynamics.casadi_dynamics(x[i+1,:], u[i,:], p)
            constraints.append(x[i+1,:] == x[i,:] + dt/2*(f0 + f1))
        return constraints
