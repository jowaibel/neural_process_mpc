import time
import numpy as np


class RuntimeBase:
    """Base MPC runtime: manages the solution window (stores trajectories, shifts
    warm-start data between steps) and declares the metrics interface. System
    runtimes subclass this to add hardware, logging, and metrics."""

    def __init__(self, params: dict, x_size: int, u_size: int):
        sim_steps = params['experiment_options']['sim_steps']
        horizon = params['horizon_steps']

        self.x0 = np.zeros((sim_steps+1, x_size))
        self.u0 = np.zeros((sim_steps+1, u_size))
        self.x = np.zeros((sim_steps+1, horizon+1, x_size))
        self.u = np.zeros((sim_steps+1, horizon, u_size))
        self.t = np.zeros(sim_steps + 1)
        self.mpc_t = np.zeros(sim_steps)
        self.lam_g = None

        self.params = params
        self.k = -1

    def init_solution(self, x: np.ndarray, u: np.ndarray, lam_g: np.ndarray):
        self.k = -1
        self.x[0, :, :] = x
        self.u[0, :, 0] = u
        self.lam_g = lam_g
        self.t[0] = time.time()

    def load_initial_solution(self, x0_measured: np.ndarray, integrate_fn=None):
        """Prepare warm-start data for the next MPC step.

        Args:
            x0_measured: current measured state from hardware
            integrate_fn: callable(x0, u) -> x_predicted, for one-step integration
        """
        self.k += 1
        self.x0[self.k] = x0_measured
        self.u0[self.k] = self.u[self.k-1,0]

        if self.k > 0:
            self.u[self.k, :-1] = self.u[self.k-1, 1:]
            self.u[self.k,-1] = self.u[self.k-1,-1]
            x0_delayed = integrate_fn(x0_measured, self.u0[self.k].reshape(1,1))[1]
            if integrate_fn is not None:
                self.x[self.k] = integrate_fn(x0_delayed, self.u[self.k])

        x_init = np.array(self.x[self.k])
        u_init = np.array(self.u[self.k])
        lam_g_init = np.array(self.lam_g)

        return x_init, u_init, lam_g_init, self.x[self.k, 0]

    def apply_solution(self, x: np.ndarray, u: np.ndarray, lam_g: np.ndarray):
        self.x[self.k, :, :] = x
        self.u[self.k, :, 0] = u
        self.lam_g = lam_g

        self.mpc_t[self.k] = time.time() - self.t[self.k]
        if self.mpc_t[self.k] < self.params['dt']:
            time.sleep(self.params['dt'] - self.mpc_t[self.k])
        self.t[self.k + 1] = time.time()

    def compute_metrics(self) -> dict:
        """Closed-loop performance metrics from the realized trajectory.
        System-specific; implemented by subclasses."""
        raise NotImplementedError
