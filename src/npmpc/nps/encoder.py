import torch
from npmpc.nps.nn_utils import create_mlp, get_activation, initialize_mlp


class CNPEncoder(torch.nn.Module):
    """CNP encoder: maps (x, y, context_mask) -> z.

    Each context pair (x_i, y_i) is encoded by an MLP into r_i; the r_i are
    mean-pooled (respecting context_mask) into the latent z, returned directly.
    """

    def __init__(self, network):
        super().__init__()
        enc_config = network['cnp_encoder']
        inner_activation = get_activation(enc_config.get('inner_activation', 'GELU'))

        z_dim = network['sizes']['z']

        self.cnp_encoder = create_mlp(
            network['sizes']['x'] + network['sizes']['y'],
            z_dim,
            enc_config.get('layers', [64, 256, 128]),
            inner_activation=inner_activation)
        initialize_mlp(self.cnp_encoder)

    def forward(self, x, y, context_mask=None):
        xy = torch.cat([x, y], dim=-1)
        r = self.cnp_encoder(xy)                                # (..., n_points, z_dim)

        # Mean-pool over context points, respecting the mask.
        if context_mask is not None:
            mask = context_mask.unsqueeze(-1).to(r.dtype)       # (..., n_points, 1)
            z = (r * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1.0)
        else:
            z = r.mean(dim=-2)                                  # (..., z_dim)

        return z
