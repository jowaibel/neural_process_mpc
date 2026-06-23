import logging

import torch
import casadi as ca
import matplotlib.pyplot as plt

from npmpc.dynamics.furuta import FurutaDynamics
from npmpc.dynamics.integrators import IntegrationBase
from npmpc.dynamics.integrators import INTEGRATOR_REGISTRY
from npmpc.dynamics.integrators import FurutaNPIntegration
from npmpc.nps.NeuralProcess import NeuralProcess

logger = logging.getLogger(__name__)


def post_experiment_analysis(traj: list, experiment_params: dict, trainer=None):
    """Run opt-in post-experiment analyses: `compare_with_np` (needs a trainer) and
    `estimate_parameters` (nonlinear physical-parameter identification)."""
    if experiment_params.get('compare_with_np', False):
        if trainer is None:
            raise ValueError("compare_with_np=True requires a trainer")
        trainer.compare_with_np(traj[0], experiment_params)

    if experiment_params.get('estimate_parameters', False):
        p_est = ParameterEstimation(experiment_params).estimate(traj)
        logger.info(f"Estimated parameters p: {p_est.tolist()}")
        return p_est
    return None


class ParameterEstimation:

    def __init__(self, params: dict):
        self.params = params
        # Integrator is built lazily in `estimate()`; compare_with_np callers need no `estimation` block.
        self._integrator: IntegrationBase = None

    @property
    def integrator(self):
        if self._integrator is None:
            if self.params['system'] == 'furuta':
                dynamics = FurutaDynamics
            else:
                raise NotImplementedError
            self._integrator = INTEGRATOR_REGISTRY[self.params['integrator']](dynamics)
        return self._integrator


    def estimate(self, data: dict):
        experiment_length = self.params['experiment_length']
        
        problem = ca.Opti()
        problem.solver(
            'ipopt', {
                'ipopt.max_iter': 500,
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
                'print_time': 0,
            })
        
        # optimization variables
        p_opti = problem.variable(1, self.integrator.dynamics.p_size)
        problem.set_initial(p_opti, ca.DM(self.params['p']))
        x_opti = problem.variable(experiment_length+1, self.integrator.dynamics.x_size)
        problem.set_initial(x_opti, ca.DM(data[0]['x'][0].numpy()))

        # measured trajectory
        x_truth = problem.parameter(experiment_length+1, self.integrator.dynamics.x_size)
        problem.set_value(x_truth, ca.DM(data[0]['x'][0].numpy()))
        u_truth = problem.parameter(experiment_length, self.integrator.dynamics.u_size)
        problem.set_value(u_truth, ca.DM(data[0]['u'][0].numpy()))

        # integration constraints
        for chunk in range(int(experiment_length / self.params['estimation']['chunk_size'])):
            i_min = chunk * self.params['estimation']['chunk_size']; i_max = min((chunk+1) * self.params['estimation']['chunk_size'], experiment_length)
            problem.subject_to(x_opti[i_min,:] == x_truth[i_min,:])
            constraints = self.integrator.casadi_implicit(x_opti[i_min:i_max,:], u_truth[i_min:i_max-1,:], self.params['dt'], p_opti)
            problem.subject_to(constraints)

        # parameter bounds
        problem.subject_to([p_opti >= ca.DM(self.params['estimation']['range']['p'])[:,0].T,
                            p_opti <= ca.DM(self.params['estimation']['range']['p'])[:,1].T])

        # minimize the difference between measured and estimated states
        cost = ca.sumsqr(x_opti[:,(0,2,3)] - x_truth[:,(0,2,3)])
        problem.minimize(cost)
        
        try:
            solution = problem.solve()
        except Exception as exc:
            solution = problem.debug
            logger.warning(f"Parameter estimation failed: {exc} | stats={problem.stats()}")
        
        fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(12,9))
        for i, ax in enumerate(axis.flatten()):
            ax.plot(data[0]['t'][0,:,0], data[0]['x'][0,:,i].numpy(), linestyle='--', color='black')
            ax.plot(data[0]['t'][0,:,0], solution.value(x_opti)[:,i], color='red')
        plt.show()
        
        return solution.value(p_opti).reshape((1,-1))
        
        
    def np_estimate(self, dataset: torch.utils.data.Dataset, model: NeuralProcess):
        r = None; z = None
        with torch.no_grad():
            model.send_to_device('cpu')
            # Encode the first dataset sample as context
            xc = dataset[0]['x']
            yc = dataset[0]['y']
            if model.params['z_encoder']['active']:
                z = model.latent_representation(xc, yc)[0].unsqueeze(0).numpy()
            else:
                r = model.conditional_representation(xc, yc)[0].unsqueeze(0).numpy()
        return r, z