import logging
import time

import torch

from npmpc.hardware.QubeBase import QubeBase
from npmpc.utils.utils import resolve_path

logger = logging.getLogger(__name__)

class DataRecorder:
    params: dict
    qube: QubeBase
    
    
    def __init__(self, params: dict, qube: QubeBase=None):
        self.params = params

        # When loading a saved experiment, skip qube setup entirely.
        if params.get('load') is not None:
            self.qube = None
            return

        if qube is not None:
            self.qube = qube
        else:
            self.setup_qube()


    def terminate(self):
        if self.qube is not None:
            self.qube.terminate()


    @classmethod
    def get_trajectory(cls, params: dict) -> list:
        """Load a saved trajectory or run the experiment; return the trajectory list."""
        if params.get('load') is not None:
            logger.info(f"Loading experiment '{params['load']}' from disk")
            return cls.load_saved(params['load'])
        
        recorder = cls(params)
        try:
            return recorder.run_experiment()
        finally:
            recorder.terminate()


    def run_experiment(self):
        if "impulse" in self.params.keys():
            kind = 'impulse'; runner = self.impulse_experiment
        elif "sinusoid" in self.params.keys():
            kind = 'sinusoid'; runner = self.sinusoid_experiment
        elif "random" in self.params.keys():
            kind = 'random'; runner = self.random_experiment
        else:
            raise ValueError('Experiment not implemented')

        logger.info(
            f"Starting {kind} experiment on target={self.params.get('target', '?')}: "
            f"{self.params['experiment_length']} steps @ dt={self.params['dt']}s"
        )
        t_start = time.time()
        experiment = runner()
        logger.info(f"Experiment finished in {time.time() - t_start:.1f}s")

        if self.params.get('save_experiment', False):
            path = resolve_path(self.params['save_name'])
            torch.save(experiment, path)
            logger.info(f"Saved experiment to {path}")

        return experiment


    @classmethod
    def load_saved(cls, name: str) -> list:
        return torch.load(resolve_path(name), weights_only=False)
    
    
    def impulse_experiment(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.qube.get_state()
        experiemnt_length = self.params['experiment_length']
        x_experiment = torch.zeros((experiemnt_length+1, x0.shape[-1]))
        u_experiment = torch.zeros((experiemnt_length, 1))
        t_experiment = torch.zeros((experiemnt_length+1, 1))
        
        impulse_length = self.params['impulse']['length']
        impulse_magnitude = self.params['impulse']['magnitude']
        double_impulse = self.params['impulse']['double']
        dt = self.params['dt']
        t0 = time.time()
        t_experiment[0] = 0
        for k in range(experiemnt_length):
            # Compute commanded input
            if t_experiment[k] < impulse_length:
                u_experiment[k] = impulse_magnitude
            elif double_impulse and t_experiment[k] < 2*impulse_length:
                u_experiment[k] = -impulse_magnitude
            else:
                u_experiment[k] = 0
            # Get state and send command
            x_experiment[k] = self.qube.get_state()
            self.qube.set_torque_setpoint(u_experiment[k])
            # Wait until next time step
            current_time = time.time() - t0
            if t_experiment[k] > current_time - dt:
                time.sleep(t_experiment[k].item() + dt - current_time)
                t_experiment[k+1] = t_experiment[k] + dt
            else:
                t_experiment[k+1] = current_time
                
        x_experiment[-1] = self.qube.get_state()
        return [{'p': self.qube.p.squeeze(0), 'x': x_experiment.unsqueeze(0), 'u': u_experiment.unsqueeze(0), 't': t_experiment.unsqueeze(0)}]
    
    
    def sinusoid_experiment(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sin_period = self.params['sinusoid']['period']
        sin_magnitude = self.params['sinusoid']['magnitude']
        dt = self.params['dt']
        t0 = time.time()
        
        x0 = self.qube.get_state()
        experiemnt_length = self.params['experiment_length']
        x_experiment = torch.zeros((experiemnt_length+1, x0.shape[-1]))
        u_experiment = sin_magnitude * torch.sin(2*torch.pi/sin_period * torch.arange(0, experiemnt_length*dt, dt)).unsqueeze(-1)
        t_experiment = torch.zeros((experiemnt_length+1, 1))
        
        
        for k in range(experiemnt_length):
            # Get state and send command
            x_experiment[k] = self.qube.get_state()
            self.qube.set_torque_setpoint(u_experiment[k])
            # Wait until next time step
            current_time = time.time() - t0
            if t_experiment[k] > current_time - dt:
                time.sleep(t_experiment[k].item() + dt - current_time)
                t_experiment[k+1] = t_experiment[k] + dt
            else:
                t_experiment[k+1] = current_time
                
        x_experiment[-1] = self.qube.get_state()
        return [{'p': self.qube.p.squeeze(0), 'x': x_experiment.unsqueeze(0), 'u': u_experiment.unsqueeze(0), 't': t_experiment.unsqueeze(0)}]
    
    
    
    def random_experiment(self) -> list[dict]:
        """Record a trajectory with a uniformly-random torque each step.
        Range from `params['random']['range']` (Nm); seeded by `params['seed']`."""
        u_range = self.params['random']['range']
        dt = self.params['dt']
        experiment_length = self.params['experiment_length']
        generator = torch.Generator().manual_seed(self.params['seed'])
        u_experiment = (torch.rand(experiment_length, 1, generator=generator)
                        * (u_range[1] - u_range[0]) + u_range[0])

        x0 = self.qube.get_state()
        x_experiment = torch.zeros((experiment_length + 1, x0.shape[-1]))
        t_experiment = torch.zeros((experiment_length + 1, 1))
        t0 = time.time()
        for k in range(experiment_length):
            x_experiment[k] = self.qube.get_state()
            self.qube.set_torque_setpoint(u_experiment[k])
            current_time = time.time() - t0
            if t_experiment[k] > current_time - dt:
                time.sleep(t_experiment[k].item() + dt - current_time)
                t_experiment[k + 1] = t_experiment[k] + dt
            else:
                t_experiment[k + 1] = current_time
        x_experiment[-1] = self.qube.get_state()
        return [{'p': self.qube.p.squeeze(0) if self.qube.p is not None else None,
                 'x': x_experiment.unsqueeze(0),
                 'u': u_experiment.unsqueeze(0),
                 't': t_experiment.unsqueeze(0)}]


    def setup_qube(self):
        from npmpc.hardware import create_hardware
        self.qube = create_hardware(self.params)