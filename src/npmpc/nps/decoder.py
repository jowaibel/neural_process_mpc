import torch
from npmpc.nps.nn_utils import create_mlp, initialize_mlp, get_activation


class CNPDecoder(torch.nn.Module):
    """CNP decoder: maps (x, z) -> (y_mu, y_sigma).

    An MLP on [x, z] predicts the per-output mean and a per-output
    (Softplus-positive) variance; the decoder returns its sqrt as the standard
    deviation sigma. Uncertainty is diagonal.
    """

    def __init__(self, network):
        super().__init__()
        self.decoder_params = network['cnp_decoder']
        self._n_y = network['sizes']['y']

        input_dim = network['sizes']['x'] + network['sizes']['z']

        # Names kept so `casadi_forward` can pass them to `create_casadi_mlp`.
        self._inner_act_name = self.decoder_params.get('inner_activation', 'GELU')
        self._mu_act_name = self.decoder_params.get('mu_activation', 'None')

        inner_activation = get_activation(self._inner_act_name)
        mu_activation = get_activation(self._mu_act_name)
        sigma_activation = get_activation(self.decoder_params.get('sigma_activation', 'Softplus'))

        self.mu_decoder = create_mlp(input_dim, self._n_y, self.decoder_params['layers'],
                                     inner_activation=inner_activation, final_activation=mu_activation)
        initialize_mlp(self.mu_decoder)

        self.sigma_decoder = create_mlp(input_dim, self._n_y, self.decoder_params['layers'],
                                        inner_activation=inner_activation, final_activation=sigma_activation)
        initialize_mlp(self.sigma_decoder)

    def forward(self, x, z):
        # Broadcast z over the points dim so each (x_i, z) pair is decoded jointly.
        z = z.unsqueeze(-2).expand(*x.shape[:-1], z.shape[-1])
        xz = torch.cat([x, z], dim=-1)

        mu_y = self.mu_decoder(xz)
        sigma_y = self.sigma_decoder(xz).sqrt()          # standard deviation σ
        return mu_y, sigma_y

    def loss(self, y_mu, y_sigma, y_target):
        """Per-sequence diagonal Gaussian negative log-likelihood.
        y_mu / y_sigma / y_target: (..., seq, n_y)."""
        var = y_sigma.pow(2)
        nll_per = torch.nn.functional.gaussian_nll_loss(
            y_mu, y_target, var, full=True, reduction='none')
        return nll_per.sum(dim=-1)

    def casadi_z(self, z_list):
        import casadi as ca
        return ca.DM([z_list])

    def casadi_forward(self, x, z, x_scaler_sd, y_inv_scaler_sd):
        import casadi as ca
        from npmpc.nps.casadi_utils import create_casadi_mlp, create_casadi_scaler

        x_scaled = create_casadi_scaler(x, x_scaler_sd)
        z_row = z.T if z.size1() > z.size2() else z
        decoder_input = ca.horzcat(x_scaled, z_row)
        mlp_network = {
            'inner_activation': self._inner_act_name,
            'mu_activation': self._mu_act_name,
        }
        y_scaled = create_casadi_mlp(decoder_input, 'mu_decoder', mlp_network, self.state_dict())
        y = create_casadi_scaler(y_scaled, y_inv_scaler_sd)
        return y
