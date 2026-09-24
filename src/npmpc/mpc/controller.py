import logging
import time

import casadi as ca
import torch

from npmpc.mpc.problem import PROBLEM_REGISTRY
from npmpc.mpc.runtime.furuta import FurutaRuntime
from npmpc.mpc.solver import configure_solver

logger = logging.getLogger(__name__)

RUNTIME_REGISTRY = {
    'furuta': FurutaRuntime,
}


class MPCController:

    def __init__(self, model, params: dict):
        self.params = params

        # Create problem and runtime via registries
        problem_cls = PROBLEM_REGISTRY[(params['system'], params['method'])]
        runtime_cls = RUNTIME_REGISTRY[params['system']]

        if params['method'] == 'neural':
            self.mpc = problem_cls(model, params)
        else:
            self.mpc = problem_cls(params)

        self.runtime = runtime_cls(self.mpc, params)

        # Build CasADi optimization problem
        self.problem = ca.Opti()
        self._build_optimization()
        configure_solver(self.problem, params['solver'])

    def _build_optimization(self):
        N = self.params['horizon_steps']
        x_size = self.mpc.integrator.dynamics.x_size
        u_size = self.mpc.integrator.dynamics.u_size

        self.x = self.problem.variable(N + 1, x_size)
        self.u = self.problem.variable(N, u_size)
        self.slack = self.problem.variable(1, x_size)
        self.x0 = self.problem.parameter(1, x_size)

        # Initial state
        self.problem.subject_to(self.x[0, :] <= self.x0 + 1e-3)
        self.problem.subject_to(self.x[0, :] >= self.x0 - 1e-3)

        # Constraints
        for c in self.mpc.integration_constraints(self.x, self.u, self.params['dt']):
            self.problem.subject_to(c)
        for c in self.mpc.state_constraints(self.x, self.slack, self.params):
            self.problem.subject_to(c)
        for c in self.mpc.input_constraints(self.u, self.params):
            self.problem.subject_to(c)
        for c in self.mpc.slack_constraints(self.slack, self.params['slack_bound']):
            self.problem.subject_to(c)

        # Cost
        cost = self.mpc.cost_function(self.x, self.u) \
             + self.mpc.slack_cost(self.slack, self.params['slack_bound'])
        self.problem.minimize(cost)

    def warm_start(self, x0: torch.Tensor):
        N = self.params['horizon_steps']
        u_init = torch.zeros(N, self.mpc.integrator.dynamics.u_size)
        x_init = x0 + torch.linspace(0, 1, steps=N + 1).unsqueeze(1) * (torch.Tensor([2 * torch.pi, 0, 0, 0]) - x0)

        self.problem.set_value(self.x0, x0.numpy())
        self.problem.set_initial(self.x, x_init.numpy())
        self.problem.set_initial(self.u, u_init.numpy())

        for attempt in range(100):
            try:
                sol = self.problem.solve()
                break
            except Exception as exc:
                sol = self.problem.debug
                logger.warning(
                    f"Warm-start attempt {attempt} failed: {exc} | "
                    f"stats={self.problem.stats()}"
                )
                self.problem.set_initial(self.x, sol.value(self.x))
                self.problem.set_initial(self.u, sol.value(self.u))
                self.problem.set_initial(self.slack, sol.value(self.slack))
                self.problem.set_initial(self.problem.lam_g, sol.value(self.problem.lam_g))
        else:
            logger.warning("Warm start failed after 100 attempts")

        self.runtime.init_solution(sol.value(self.x), sol.value(self.u), sol.value(self.problem.lam_g))

    def run_controller(self):
        with torch.no_grad():
            x0 = torch.Tensor(self.params['experiment_options']['x0'])
            sim_steps = self.params['experiment_options']['sim_steps']
            dt = self.params['dt']

            # Fixed run length in (real-time) simulation time, the same for every
            # agent (C++ clients too): sim_steps * dt. Steps are paced to at least
            # dt (RuntimeBase.apply_solution), so sim_steps also bounds the number
            # of steps that fit in it -- and sizes the runtime's buffers.
            sim_time = sim_steps * dt

            logger.info(
                f"Starting MPC ({self.params['method']}/{self.params['system']}): "
                f"{sim_time:g}s (at most {sim_steps} steps), horizon={self.params['horizon_steps']}, dt={dt}s"
            )
            t_start = time.time()
            self.warm_start(x0)

            solve_times = []
            failures = 0
            t_loop = time.time()
            for step in range(sim_steps):
                if time.time() - t_loop >= sim_time:
                    break
                x_init, u_init, lam_g_init, x0 = self.runtime.load_initial_solution()
                self.problem.set_initial(self.x, x_init)
                self.problem.set_initial(self.u, u_init)
                self.problem.set_initial(self.problem.lam_g, lam_g_init)
                self.problem.set_value(self.x0, x0)

                t_solve = time.perf_counter()
                try:
                    sol = self.problem.solve()
                except Exception as exc:
                    sol = self.problem.debug
                    failures += 1
                    logger.warning(
                        f"Optimization failed at step {step}: {exc} | "
                        f"stats={self.problem.stats()} | x0={x0}"
                    )
                solve_times.append(time.perf_counter() - t_solve)

                self.runtime.apply_solution(sol.value(self.x), sol.value(self.u), sol.value(self.problem.lam_g))

            self.runtime.terminate()

            if solve_times:
                solve = torch.tensor(solve_times)
                excess = (solve - dt).clamp_min(0.0)
                n_viol = int((solve > dt).sum().item())
                pct_viol = 100.0 * n_viol / solve.numel()
                mean_excess = excess[excess > 0].mean().item() if n_viol > 0 else 0.0
                max_excess = excess.max().item()
                logger.info(
                    f"MPC finished in {time.time() - t_start:.1f}s | "
                    f"avg solve={solve.mean().item() * 1e3:.1f}ms (budget {dt * 1e3:.1f}ms) | "
                    f"violations={pct_viol:.1f}% ({n_viol}/{solve.numel()}) | "
                    f"mean excess={mean_excess * 1e3:.2f}ms  max excess={max_excess * 1e3:.2f}ms | "
                    f"failures={failures}"
                )

    # Backward-compatibility alias for self.runtime.
    @property
    def starter(self):
        return self.runtime
