import torch
from npmpc.data.generators import DataGeneratorBase


class RandomDataGenerator(DataGeneratorBase):
    
    def generate(self):
        torch.manual_seed(self.params['seed'])
        p = self.generate_random_p()
        x0 = self.generate_random_x0() 
        u = self.generate_random_u()
        
        x = self.oracle.integrate(x0, u, self.params['dt'], p)
        x[...,1,:] += self.generate_noise()
        return [{'p': p[i,0], 'x': x[i], 'u': u[i]} for i in range(self.params['p_size'])]
        
    
    def generate_random_p(self):
        p_range = torch.Tensor(self.params['range']['p'])
        p = torch.rand(self.params['p_size'], p_range.shape[0]) * (p_range[:,1]-p_range[:,0]) + p_range[:,0]
        return p.unsqueeze(-2).repeat((1,self.params['x0_size'],1))
    
    
    def generate_random_x0(self):
        x_range = torch.Tensor(self.params['range']['x'])
        x0 = torch.rand(self.params['p_size'],self.params['x0_size'], x_range.shape[0]) * (x_range[:,1]-x_range[:,0]) + x_range[:,0]
        return x0
    
    
    def generate_random_u(self):
        u_range = torch.Tensor(self.params['range']['u'])
        u = torch.rand(self.params['p_size'],self.params['x0_size'],self.params['u_size'], u_range.shape[0])* (u_range[:,1]-u_range[:,0]) + u_range[:,0]
        return u
    
    
    def generate_noise(self):
        x_range = torch.Tensor(self.params['range']['x'])
        noise = torch.Tensor(self.params['noise']) * torch.randn((self.params['p_size'],self.params['x0_size'], x_range.shape[0]))
        return noise