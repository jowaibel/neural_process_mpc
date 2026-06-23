import torch
import casadi as ca

from npmpc.dynamics import DynamicsBase


class FurutaDynamics(DynamicsBase):
    """
    x: theta (pendulum angle), theta_dot (pendulum angular velocity), phi (arm angle), phi_dot (arm angular velocity)

    u: torque (arm torque)

    p: lp (pendulum length), mp (pendulum mass), lr (arm length), mr (arm mass)

    y: theta_dot (pendulum velocity), phi_dot (arm velocity), theta_ddot (pendulum angular acceleration), phi_ddot (arm angular acceleration)
    """
    g = 9.81

    x_size = 4
    u_size = 1
    y_size = 4
    p_size = 4
    
    @classmethod
    def dynamics(cls, x, u, p):
        theta, _phi, theta_dot, phi_dot = x[...,0], x[...,1], x[...,2], x[...,3]
        torque = u[...,0]
        lp, mp, lr, mr = p[...,0], p[...,1], p[...,2], p[...,3]

        # Inertia calculation
        j_r = mr*lr**2 / 3 + mp*lr**2
        j_p = mp*lp**2 / 3

        # B definition
        B = torch.zeros((*(x.shape[:-1]), 2, 2)).to(x.device)
        B[...,0,0] = j_p 
        B[...,0,1] = -mp*lr*lp/2*torch.cos(theta)
        B[...,1,0] = -mp*lr*lp/2*torch.cos(theta)
        B[...,1,1] = j_r + j_p*torch.sin(theta)**2

        # A definition
        A = torch.zeros((*(x.shape[:-1]), 2)).to(x.device)
        A[...,0] =  j_p*torch.sin(2*theta)/2*phi_dot**2 + mp*lp*cls.g/2*torch.sin(theta)
        A[...,1] = -j_p*torch.sin(2*theta)*phi_dot*theta_dot - mp*lr*lp/2*torch.sin(theta)*theta_dot**2 + torque        
        y = torch.cat([theta_dot.unsqueeze(-1), phi_dot.unsqueeze(-1), (B.inverse() @ A.unsqueeze(-1)).squeeze(-1)],dim=-1)
        
        return y
    
    
    @classmethod
    def casadi_dynamics(cls, x, u, p):
        theta, _phi, theta_dot, phi_dot = x[:,0], x[:,1], x[:,2], x[:,3]
        torque = u[:,0]
        lp, mp, lr, mr = p[:,0], p[:,1], p[:,2], p[:,3]

        # Inertia calculation
        j_r = mr*lr**2 / 3 + mp*lr**2
        j_p = mp*lp**2 / 3

        # B definition
        B = ca.MX.zeros((2, 2))
        B[0,0] = j_p
        B[0,1] = -mp*lr*lp/2*ca.cos(theta)
        B[1,0] = -mp*lr*lp/2*ca.cos(theta)
        B[1,1] = j_r + j_p*ca.sin(theta)**2
        

        # A definition
        A = ca.MX.zeros((2))
        A[0] =  j_p*ca.sin(2*theta)/2*phi_dot**2 + mp*lp*cls.g/2*ca.sin(theta)
        A[1] = -j_p*ca.sin(2*theta)*phi_dot*theta_dot - mp*lr*lp/2*ca.sin(theta)*theta_dot**2 + torque
        
        b_det = B[0,0]*B[1,1] - B[0,1]*B[1,0]; b_adj = ca.vertcat(ca.horzcat(B[1,1], -B[0,1]), ca.horzcat(-B[1,0], B[0,0]))
        y = ca.horzcat(theta_dot, phi_dot, ((b_adj / b_det) @ A).T)
        
        return y