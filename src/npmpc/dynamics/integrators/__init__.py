from npmpc.dynamics import DynamicsBase
import torch
import casadi as ca


class IntegrationBase:
    
    dynamics: DynamicsBase
    
    def __init__(self, dynamics: DynamicsBase):
        self.dynamics = dynamics


    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float, p: torch.Tensor=None) -> torch.Tensor:
        """Single integration step x(t) -> x(t+dt) under input u. Functional
        (no in-place writes) so it is safe with torch.autograd.functional.jacobian."""
        raise NotImplementedError


    def integrate(self, x0: torch.Tensor, u: torch.Tensor, dt: float, p: torch.Tensor=None) -> torch.Tensor:
        """Integrates the dynamics from x0 to xN using the inputs u over the time dt.

        Args:
            x0 (Tensor): Initial input state (..., x_size).
            u (Tensor): Input sequence (..., N, u_size).
            dt (float): Time step.
            p (Tensor, optional): Parameters (..., p_size). Defaults to None.
            
        Returns:
            Tensor: Integrated state (..., N+1, x_size).
        """
        raise NotImplementedError
    
    
    def casadi_explicit(self, x0: ca.MX, u: ca.MX, dt: float, p: ca.MX=None) -> ca.MX:
        """Integrates the dynamics from x0 to xN using the inputs u over the time dt.

        Args:
            x0 (ca.MX): Initial input state (x_size).
            u (ca.MX): Input sequence (N, u_size).
            dt (float): Time step.
            p (ca.MX, optional): Parameters (p_size). Defaults to None.
            
        Returns:
            ca.MX: Integrated state (N+1, x_size).
        """
        raise NotImplementedError
    
    
    def casadi_implicit(self, x: ca.MX, u: ca.MX, dt: float, p: ca.MX=None) -> list:
        """Integrates the dynamics from x0 to xN using the inputs u over the time dt.

        Args:
            x (ca.MX): Initial input state (N+1, x_size).
            u (ca.MX): Input sequence (N, u_size).
            dt (float): Time step.
            p (ca.MX, optional): Parameters (p_size). Defaults to None.
            
        Returns:
            ca.MX: Integrated state (N+1, x_size).
        """
        raise NotImplementedError
    

from npmpc.dynamics.integrators.discrete import DiscreteIntegration
from npmpc.dynamics.integrators.midpoint import MidpointIntegration
from npmpc.dynamics.integrators.trapezoidal import TrapezoidalIntegration
from npmpc.dynamics.integrators.input_output import InputOutputIntegration
from npmpc.dynamics.integrators.furuta_np import FurutaNPIntegration

INTEGRATOR_REGISTRY = {
    'discrete': DiscreteIntegration,
    'midpoint': MidpointIntegration,
    'trapezoidal': TrapezoidalIntegration,
    'input_output': InputOutputIntegration,
}
