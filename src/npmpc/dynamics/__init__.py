import torch
import casadi as ca

class DynamicsBase:
    
    x_size: int
    u_size: int
    y_size: int
    p_size: int
    
    @classmethod
    def dynamics(cls, x: torch.Tensor, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        x: state of shape (..., points, x_size);
        u: command of shape (..., points, u_size);
        p: parameters of shape (..., p_size);\n
        Returns: of shape (..., points, y_size)\n
        Computes the state update given state and input.
        """
        raise NotImplementedError
    
    
    @classmethod
    def casadi_dynamics(cls, x: ca.MX, u: ca.MX, p: ca.DM) -> ca.MX:
        """
        x: state of shape (x_size);
        u: command of shape (u_size);
        p: parameters of shape (p_size);\n
        Returns: of shape (y_size)\n
        Computes the state update given state and input.
        """
        raise NotImplementedError



from npmpc.dynamics.furuta import FurutaDynamics

DYNAMICS_REGISTRY = {
    'furuta': FurutaDynamics,
}