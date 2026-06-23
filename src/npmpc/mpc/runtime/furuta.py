import logging

import casadi as ca
import numpy as np
import torch

from npmpc.hardware import create_hardware
from npmpc.mpc.runtime import RuntimeBase
from npmpc.mpc.problem.furuta_cost import stage_cost, interstage_cost
from npmpc.dynamics.furuta import FurutaDynamics
from npmpc.dynamics.integrators.midpoint import MidpointIntegration
from npmpc.utils.utils import resolve_path

logger = logging.getLogger(__name__)


class FurutaRuntime(RuntimeBase):
    """Wires the solution window, hardware, logging, and metrics for the Furuta
    pendulum. Inherits the solution-window storage from RuntimeBase, so the
    buffer arrays (x0, u, k, ...) live directly on self."""

    def __init__(self, mpc, params: dict):
        x_size = mpc.integrator.dynamics.x_size
        u_size = mpc.integrator.dynamics.u_size
        super().__init__(params, x_size, u_size)
        self.mpc = mpc
        self.qube = create_hardware(params)

    def load_initial_solution(self):
        x0_measured = self.qube.get_state()

        def integrate_fn(x0, u):
            with torch.no_grad():
                return self.mpc.integrate(x0=torch.Tensor(x0), u=u, dt=self.params['dt'])

        return super().load_initial_solution(x0_measured, integrate_fn)

    def apply_solution(self, x, u, lam_g):
        # Safety check
        u_bounds = self.params['hard_bound']['u'][0]
        if u[1] < 1.2 * u_bounds[0] or u[1] > 1.2 * u_bounds[1]:
            logger.error(f'Control signal out of bounds: {u[1]}')
            self.terminate()
            exit()

        super().apply_solution(x, u, lam_g)
        self.qube.set_torque_setpoint(self.u[self.k, 0, 0])

        # Incremental dump each step so a mid-run crash leaves a usable
        # trajectory on disk (self.k records how far we got).
        if self.params.get('save', False):
            self._dump_data(verbose=False)

    def terminate(self):
        self.qube.terminate()
        metrics = self.compute_metrics()
        logger.info(
            f"Performance | closed-loop cost={metrics['cost']:.3f} | "
            f"convergence={metrics['convergence_str']} "
            f"(theta tol={metrics['theta_tol_deg']:.1f}deg)"
        )
        if self.params.get('save', False):
            # Final dump also includes the analytical-ODE open-loop predictions
            # (same control sequences) for comparison against the NP rollout.
            self._dump_data(include_oracle=True, metrics=metrics)

    def compute_metrics(self, theta_tol_deg: float = 10.0) -> dict:
        """Closed-loop metrics from the realized trajectory (measured states
        x0[k], applied controls u[k, 0]).

        * cost: realized running cost — sum over executed steps of
          stage_cost + interstage_cost on the 2-state window [x0[k], x0[k+1]]
          with the applied control (so N = 1). The terminal Riccati term is
          excluded; it belongs to the prediction horizon, not the realized run.
        * convergence: settling time of theta — first step after which theta
          stays within theta_tol_deg of the nearest upright (0 mod 2pi) for the
          rest of the run. None if it never settles.
        """
        n_valid = max(self.k + 1, 0)
        dt = self.params['dt']
        cw = self.params['cost']

        # Realized running cost: each step uses a 2-state window and u shaped
        # (1, 1) so the cost helpers see N = 1.
        cost_per_step = np.zeros(max(n_valid - 1, 0))
        for k in range(n_valid - 1):
            x_win = ca.DM(self.x0[k:k + 2])            # (2, 4): steps k and k+1
            u_win = ca.DM(self.u[k:k + 1, 0, :])       # (1, u_size): applied at k
            cost_per_step[k] = float(stage_cost(x_win, u_win, cw)
                                     + interstage_cost(x_win, u_win, cw))
        cost = float(cost_per_step.sum())

        # Convergence: settling time on |theta| wrapped to the nearest upright.
        theta = self.x0[:n_valid, 0]
        tol = np.deg2rad(theta_tol_deg)
        theta_err = np.abs((theta + np.pi) % (2.0 * np.pi) - np.pi)
        within = theta_err < tol
        if n_valid == 0 or not within.any() or not within[-1]:
            settle_step = None
        else:
            violations = np.where(~within)[0]
            settle_step = int(violations[-1] + 1) if violations.size else 0

        settle_time = settle_step * dt if settle_step is not None else None
        convergence_str = (f"{settle_time:.2f}s (step {settle_step})"
                           if settle_step is not None else "not converged")

        return {
            "cost": cost,
            "cost_per_step": cost_per_step,
            "convergence_step": settle_step,
            "convergence_time": settle_time,
            "convergence_str": convergence_str,
            "theta_tol_deg": theta_tol_deg,
        }

    def _compute_oracle_open_loop(self):
        """Per MPC step, roll the analytical Furuta ODE forward from x0[k] with
        the chosen controls. Returns an array shaped like self.x."""
        horizon = self.params['horizon_steps']
        dt = self.params['dt']
        n_valid = max(self.k + 1, 0)

        integrator = MidpointIntegration(FurutaDynamics)
        p = torch.tensor(self.params['p'][0], dtype=torch.float32)

        oracle = np.zeros_like(self.x)
        with torch.no_grad():
            for k in range(n_valid):
                x0_k = torch.tensor(self.x[k, 0], dtype=torch.float32)
                u_seq = torch.tensor(self.u[k, :, :], dtype=torch.float32)
                # MidpointIntegration returns (horizon+1, x_size), starting with x0_k.
                x_traj = integrator.integrate(x0_k, u_seq, dt, p).numpy()
                oracle[k, :, :] = x_traj
        return oracle

    def _compute_np_open_loop(self):
        """True NP open-loop rollout via self.mpc.integrate from x0[k] with the
        chosen controls. Differs from the optimizer's iterate x when the SQP
        didn't drive the dynamics residuals to zero."""
        horizon = self.params['horizon_steps']
        dt = self.params['dt']
        n_valid = max(self.k + 1, 0)

        x_np = np.zeros_like(self.x)
        with torch.no_grad():
            for k in range(n_valid):
                x0_k = torch.tensor(self.x[k, 0], dtype=torch.float32)
                u_seq = torch.tensor(self.u[k, :, :], dtype=torch.float32)
                # mpc.integrate already binds the latent code z internally.
                x_traj = self.mpc.integrate(x0=x0_k, u=u_seq, dt=dt).numpy()
                x_np[k, :, :] = x_traj
        return x_np

    def _compute_np_montecarlo(self):
        """Monte Carlo NP rollouts per MPC step: sample the decoder's diagonal
        output distribution at the fixed latent z. Returns shape
        (sim_steps+1, n_montecarlo, horizon+1, 4)."""
        dt = self.params['dt']
        n_valid = max(self.k + 1, 0)
        n_mc = self.params.get('n_montecarlo', 100)

        z = self.mpc.z.unsqueeze(-2)
        x_mc = np.zeros((self.x.shape[0], n_mc, *self.x.shape[1:]))
        with torch.no_grad():
            for k in range(n_valid):
                x0_k = torch.tensor(self.x[k, 0], dtype=torch.float32)
                u_seq = torch.tensor(self.u[k, :, :], dtype=torch.float32)
                x_mc[k] = self.mpc.integrator.montecarlo_integrate(x0_k, u_seq, dt, z=z, n=n_mc).numpy()
        return x_mc

    def _dump_data(self, verbose: bool = True, include_oracle: bool = False, metrics: dict = None):
        dump = {
            "t": self.t, "mpc_t": self.mpc_t,
            "x0": self.x0, "u0": self.u0,
            "x": self.x, "u": self.u,
            "k": self.k,
            "method": self.params['method'],
            "horizon_steps": self.params['horizon_steps'],
            "sim_steps": self.params['experiment_options']['sim_steps'],
            "dt": self.params['dt'],
        }
        if metrics is not None:
            dump["metrics"] = metrics
        if include_oracle:
            dump["x_oracle"] = self._compute_oracle_open_loop()
            # NP rollouts (only meaningful for method='neural').
            if self.params.get('method') == 'neural' and self.mpc.z is not None:
                dump["x_np"] = self._compute_np_open_loop()
                dump["x_mc"] = self._compute_np_montecarlo()
        np.savez(resolve_path(self.params['save_path']), **dump)
        if verbose:
            logger.info(f"Saved to {resolve_path(self.params['save_path'])}")
