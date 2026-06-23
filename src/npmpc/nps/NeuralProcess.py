import os
import torch

from npmpc.nps.nn_utils import Scaler
from npmpc.nps.encoder import CNPEncoder
from npmpc.nps.decoder import CNPDecoder
from npmpc.utils.utils import get_project_root, resolve_path


class NeuralProcess(torch.nn.Module):
    encoder: CNPEncoder
    decoder: CNPDecoder

    x_direct_scaler: Scaler
    y_direct_scaler: Scaler
    y_inverse_scaler: Scaler

    def __init__(self, params: dict, scaler_params: dict=None):
        super().__init__()
        self.params = params
        torch.manual_seed(params['seed'])

        self.encoder = CNPEncoder(params)
        self.decoder = CNPDecoder(params)

        if scaler_params is None:
            self.x_direct_scaler = Scaler(weight=torch.ones(params['sizes']['x'] * params['sizes']['history']),
                                          bias=torch.zeros(params['sizes']['x'] * params['sizes']['history']))
            self.y_direct_scaler = Scaler(weight=torch.ones(params['sizes']['y']),
                                          bias=torch.zeros(params['sizes']['y']))
            self.y_inverse_scaler = Scaler(weight=torch.ones(params['sizes']['y']),
                                           bias=torch.zeros(params['sizes']['y']))
        else:
            self.x_direct_scaler = Scaler(weight=1 / scaler_params['x']['scale'],
                                          bias=-scaler_params['x']['mean'] / scaler_params['x']['scale'])
            self.y_direct_scaler = Scaler(weight=1 / scaler_params['y']['scale'],
                                          bias=-scaler_params['y']['mean'] / scaler_params['y']['scale'])
            self.y_inverse_scaler = Scaler(weight=scaler_params['y']['scale'],
                                           bias=scaler_params['y']['mean'])

        if params.get('load', False):
            self._load_state_dicts(torch.load(
                resolve_path(params['load_path']),
                map_location='cpu', weights_only=False))


    def _load_state_dicts(self, checkpoint: dict):
        self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
        self.decoder.load_state_dict(checkpoint['decoder'], strict=False)
        self.x_direct_scaler.load_state_dict(self._convert_scaler_sd(checkpoint['x_direct_scaler']))
        self.y_direct_scaler.load_state_dict(self._convert_scaler_sd(checkpoint['y_direct_scaler']))
        self.y_inverse_scaler.load_state_dict(self._convert_scaler_sd(checkpoint['y_inverse_scaler']))

    @staticmethod
    def _convert_scaler_sd(sd: dict) -> dict:
        """Convert old nn.Linear scaler state_dict to new buffer format if needed."""
        if 'scaler.0.weight' in sd:
            return {'weight': sd['scaler.0.weight'].diag(), 'bias': sd['scaler.0.bias']}
        return sd


    def save_model(self, epoch):
        torch.save({
            'params': self.params,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'x_direct_scaler': self.x_direct_scaler.state_dict(),
            'y_direct_scaler': self.y_direct_scaler.state_dict(),
            'y_inverse_scaler': self.y_inverse_scaler.state_dict()
        }, resolve_path(self.params['save_path']) + f'_{epoch}.pth')


    @classmethod
    def from_checkpoint(cls, path: str):
        """Load a model from a checkpoint file without needing the JSON config."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        params = checkpoint['params']
        params['load'] = False
        model = cls(params)
        model._load_state_dicts(checkpoint)
        return model


    def casadi_decoder(self, x, z):
        return self.decoder.casadi_forward(
            x, z, self.x_direct_scaler.state_dict(), self.y_inverse_scaler.state_dict())


    def encode(self, x: torch.Tensor, y: torch.Tensor, context_mask: torch.Tensor = None):
        x_scaled = self.x_direct_scaler(x)
        y_scaled = self.y_direct_scaler(y)
        return self.encoder(x_scaled, y_scaled, context_mask)


    def decode(self, x: torch.Tensor, z: torch.Tensor):
        """Decode x given latent z. Returns (y_mu, y_sigma); y_sigma stays in scaled space."""
        x_scaled = self.x_direct_scaler(x)
        y_mu_scaled, y_sigma_scaled = self.decoder(x_scaled, z=z)
        y_mu = self.y_inverse_scaler(y_mu_scaled)
        return y_mu, y_sigma_scaled


    def check_nans(self):
        nans = any(p.grad is not None and p.grad.isnan().any() for p in self.parameters())
        if nans:
            print('NaNs in gradient, skipping step...')
        else:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        return nans


    # Backward-compat alias for model.to(device).
    def send_to_device(self, device):
        self.to(device)
