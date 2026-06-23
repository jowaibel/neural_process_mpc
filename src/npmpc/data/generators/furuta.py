import json
import logging
import os
import numpy as np
import torch
from npmpc.data.generators.random import RandomDataGenerator

logger = logging.getLogger(__name__)

BATCH_SIZE = 32

class FurutaDataGenerator(RandomDataGenerator):

    def generate(self):
        torch.manual_seed(self.params['seed'])
        p = self.generate_random_p()
        x0 = self.generate_random_x0()
        u = self.generate_random_u(p)

        x = self.oracle.integrate(x0, u, self.params['dt'], p)
        x[..., 1, :] += self.generate_noise()
        return [{'p': p[i, 0], 'x': x[i], 'u': u[i]} for i in range(self.params['p_size'])]


    def save_to_disk(self, output_dir: str, device: str = 'cpu'):
        """Generate trajectories and save pre-transformed tensors to disk as .pt files.
        Each file contains {'x': (n_steps, 5), 'y': (n_steps, 2)} ready for training.
        Uses GPU for simulation if device='cuda'.
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.manual_seed(self.params['seed'])

        self.device = device

        for start_id in range(0,self.params['p_size'],BATCH_SIZE):
            batch_size = min(BATCH_SIZE,self.params['p_size']-start_id)
            p = self.generate_random_p(batch_size)
            x0 = self.generate_random_x0(batch_size)
            u = self.generate_random_u(p)

            x = self.oracle.integrate(x0, u, self.params['dt'], p)
            x[..., 1, :] += self.generate_noise(batch_size)
            
            # Batch feature transform for all trajectories in this parameter set
            x_states = x[..., :-1, :]  # (x0_size, n_steps, 4)
            x_nn = torch.cat([
                torch.sin(x_states[..., 0:1]), torch.cos(x_states[..., 0:1]),
                x_states[..., 2:4], u], dim=-1).reshape((batch_size,-1,5))  # (x0_size, n_steps, 5)
            y_nn = (x[..., 1:, 2:4] - x[..., :-1, 2:4]).reshape((batch_size,-1,2))  # (x0_size, n_steps, 2)

            # Chan online mean/std update for x and y across all trajectories
            x_sum_new = x_nn.mean(dim=-2).sum(dim=0)
            x_M2_new = (x_nn-x_sum_new/batch_size).pow(2).mean(dim=-2).sum(dim=0)
            y_sum_new = y_nn.mean(dim=-2).sum(dim=0)
            y_M2_new = (y_nn-y_sum_new/batch_size).pow(2).mean(dim=-2).sum(dim=0)
            if start_id == 0:
                x_M2 = x_M2_new
                x_sum = x_sum_new
                y_M2 = y_M2_new
                y_sum = y_sum_new
            else:
                x_M2 += x_M2_new + (x_sum**2)/(batch_size+start_id)*batch_size/start_id \
                     + (x_sum_new**2)/(batch_size+start_id)*start_id/batch_size \
                     - 2*x_sum*x_sum_new/(batch_size+start_id)
                x_sum += x_sum_new
                y_M2 += y_M2_new + (y_sum**2)/(batch_size+start_id)*batch_size/start_id \
                     + (y_sum_new**2)/(batch_size+start_id)*start_id/batch_size \
                     - 2*y_sum*y_sum_new/(batch_size+start_id)
                y_sum += y_sum_new

            # Save batch to disk
            for p_id in range(batch_size):
                torch.save({'p': p[p_id].to('cpu'), 'x': x_nn[p_id].to('cpu'), 'y': y_nn[p_id].to('cpu')},
                           os.path.join(output_dir, f'{start_id+p_id:06d}.pt'))

        # Compute scaling
        x_mean = x_sum / self.params['p_size']
        x_var = (x_M2 / (self.params['p_size']-1)).sqrt()
        y_mean = y_sum / self.params['p_size']
        y_var = (y_M2 / (self.params['p_size']-1)).sqrt()

        metadata = {
            'params': self.params,
            'num_systems': self.params['p_size'],
            'scaling': {
                'x': {'mean': x_mean.tolist(), 'std': x_var.tolist()},
                'y': {'mean': y_mean.tolist(), 'std': y_var.tolist()},
            }
        }
        with open(os.path.join(output_dir, '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f'Saved trajectories to {output_dir}')

    # --- Batched random generators ---

    
    def generate_random_p(self, batch_size=None):
        batch_size = self.params['p_size'] if batch_size is None else batch_size
        p_range = torch.tensor(self.params['range']['p'], device=self.device)
        p = torch.rand(batch_size, p_range.shape[0], device=self.device) * (p_range[:,1]-p_range[:,0]) + p_range[:,0]
        return p.unsqueeze(-2).expand((batch_size,self.params['x0_size'],p_range.shape[0]))
    
    
    def generate_random_x0(self, batch_size=None):
        batch_size = self.params['p_size'] if batch_size is None else batch_size
        x_range = torch.tensor(self.params['range']['x'], device=self.device)
        x0 = torch.rand(batch_size, self.params['x0_size'], x_range.shape[0], device=self.device) * (x_range[:,1]-x_range[:,0]) + x_range[:,0]
        return x0
    
    
    def generate_random_u(self, p: torch.Tensor):
        u_range = torch.tensor(self.params['range']['u'], device=self.device) * p[...,(0,2,3)].prod(-1, keepdim=True).unsqueeze(-1)
        u = torch.rand(p.shape[0],self.params['x0_size'],self.params['u_size'], u_range.shape[-2], device=self.device) * (u_range[...,1:2]-u_range[...,0:1]) + u_range[...,0:1]
        return u
    
    
    def generate_noise(self, batch_size=None):
        batch_size = self.params['p_size'] if batch_size is None else batch_size
        x_range = torch.tensor(self.params['range']['x'], device=self.device)
        noise = torch.tensor(self.params['noise'], device=self.device) * torch.randn((batch_size,self.params['x0_size'], x_range.shape[0]), device=self.device)
        return noise