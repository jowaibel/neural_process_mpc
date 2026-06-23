import logging

from matplotlib import pyplot as plt

import numpy as np
import torch
from npmpc.dynamics.integrators import FurutaNPIntegration
from npmpc.training import Training

logger = logging.getLogger(__name__)


class FurutaTraining(Training):

    def _build_context(self, traj: dict, n_context: int):
        """Pack the first n_context steps of a recorded trajectory into the
        (x_nn, y_nn) format the NP encoder expects: x = [sin θ, cos θ, θ̇, φ̇, u],
        y = Δ(θ̇, φ̇). `traj` is the single-trajectory dict from DataRecorder."""
        x_rec = traj['x'][0]                       # (n_steps+1, 4)
        u_rec = traj['u'][0]                       # (n_steps,   1)
        x_nn = torch.cat([
            torch.sin(x_rec[:n_context, 0:1]),
            torch.cos(x_rec[:n_context, 0:1]),
            x_rec[:n_context, 2:4],
            u_rec[:n_context],
        ], dim=-1)
        y_nn = x_rec[1:n_context + 1, 2:4] - x_rec[:n_context, 2:4]
        return x_nn, y_nn


    def estimate_z(self, traj: dict, n_context: int) -> torch.Tensor:
        """Encode the first `n_context` steps of a recorded trajectory and
        return the NP latent `z` shaped (1, z_dim)."""
        with torch.no_grad():
            self.model.send_to_device('cpu')
            x_nn, y_nn = self._build_context(traj, n_context)
            mask = torch.ones(1, n_context, dtype=torch.bool)
            z = self.model.encode(x_nn.unsqueeze(0), y_nn.unsqueeze(0), mask)
        return z


    def compare_with_np(self, traj: dict, params: dict):
        """Open-loop comparison of a recorded trajectory against the NP:

            * Encodes z from the first `params['n_context']` recorded steps.
            * Rolls the NP forward (mean + n_montecarlo MC samples) from the
              recorded `x0` using the recorded `u` sequence.
            * Computes one-step NLL / R² / z-score metrics on the full trajectory.
            * Plots θ, φ, θ̇, φ̇ + applied torque, recorded vs. NP mean vs MC band.
        """
        n_ctx = params['n_context']
        n_mc = params['n_montecarlo']
        dt = params['dt']

        x_real = traj['x'][0]                                   # (n_steps+1, 4)
        u = traj['u'][0]                                        # (n_steps,   1)
        x0 = x_real[0]
        n_steps = u.shape[0]

        with torch.no_grad():
            self.model.send_to_device('cpu')
            np_integrator = FurutaNPIntegration(self.model)

            # Encode context
            x_nn_ctx, y_nn_ctx = self._build_context(traj, n_ctx)
            mask = torch.ones(1, n_ctx, dtype=torch.bool)
            z = self.model.encode(x_nn_ctx.unsqueeze(0), y_nn_ctx.unsqueeze(0), mask)

            # Open-loop rollouts (mean + MC) from the recorded initial state
            x_mean = np_integrator.integrate(x0, u, dt, z=z)               # (n_steps+1, 4)
            x_mc = np_integrator.montecarlo_integrate(x0, u, dt,
                                                      z=z, n=n_mc)         # (n_mc, n_steps+1, 4)

            # ---- one-step metrics on the FULL recorded sequence ----
            x_nn_full = torch.cat([
                torch.sin(x_real[:n_steps, 0:1]),
                torch.cos(x_real[:n_steps, 0:1]),
                x_real[:n_steps, 2:4],
                u,
            ], dim=-1)
            y_target = x_real[1:n_steps+1,2:4] - x_real[:n_steps,2:4]   # (n_steps, 2)

            y_mu_step, y_sigma_step = self.model.decode(x_nn_full.unsqueeze(0), z=z)
            nll = self.model.decoder.loss(y_mu_step, y_sigma_step, y_target.unsqueeze(0)).mean().item()

            y_mu_squeezed = y_mu_step.squeeze(0)                             # (n_steps, 2)
            y_sigma = y_sigma_step.squeeze(0)                               # (n_steps, 2)
            ss_res = (y_mu_squeezed - y_target).pow(2).sum(-1).mean()
            ss_tot = (y_target - y_target.mean(0)).pow(2).sum(-1).mean().clamp_min(1e-12)
            r2 = 100.0 * (1 - ss_res / ss_tot).item()
            z_score = ((y_target - y_mu_squeezed) / y_sigma).norm(2, dim=-1).mean().item()

        logger.info(f"[experiment]  NLL = {nll:+.3f}   R² = {r2:5.2f}%   z-score = {z_score:.3f}")

        # ---- Plot ----
        t = dt * torch.arange(n_steps + 1)
        state_labels = [r'$\theta$ (rad)', r'$\varphi$ (rad)',
                        r'$\dot{\theta}$ (rad/s)', r'$\dot{\varphi}$ (rad/s)']
        colors = ['#4477AA', '#DDAA33', '#BB5566']
        lw = 1.5

        fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True, layout='constrained')
        for ix in range(4):
            axs[ix].plot(t, x_real[:, ix], 'k--', lw=lw, label='Recorded')
            for j in range(n_mc):
                axs[ix].plot(t, x_mc[j, :, ix],
                             c=colors[0], lw=0.3, alpha=max(0.05, 15 / n_mc))
            axs[ix].plot(t, x_mean[:, ix], '-', c=colors[0], lw=lw, label='NP mean')
            axs[ix].axvspan(0, n_ctx * dt, alpha=0.1, color='green')
            axs[ix].set_ylabel(state_labels[ix])
            axs[ix].grid(True, lw=0.3)
            if ix < 2:
                axs[ix].set_ylim(-2 * np.pi, 2 * np.pi)

        axs[4].step(t[:-1], 1e3 * u[:, 0], 'k-', lw=lw, where='post')
        axs[4].axvspan(0, n_ctx * dt, alpha=0.1, color='green', label='Context')
        axs[4].set_ylabel(r'$\tau$ (mNm)')
        axs[4].set_xlabel('Time (s)')
        axs[4].grid(True, lw=0.3)

        axs[0].legend(loc='upper right')
        axs[4].legend(loc='upper right')
        fig.suptitle(
            f"target={params.get('target', 'sim')}   context={n_ctx}   "
            f"NLL={nll:+.3f}   R²={r2:.2f}%   z-score={z_score:.3f}")

        plt.savefig('experiment.png', dpi=150)
        plt.show()
        return fig
