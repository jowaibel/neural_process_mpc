"""Python-side counterpart to src_cpp/tests/eval_furuta_mpc.cpp: builds the
same MPC problem (FurutaNPMPC or FurutaMPC + MPCController) from the same
config files (model/furuta_np.json, model/furuta_mpc.json) and solves one
warm-start NLP for the configured x0.

--method neural (default): NP dynamics with the given z (placeholder: zeros).
Used to cross-check the C++ port's constraints/cost/warm-start logic against
the original Python implementation: the printed x/u trajectories should
match src_cpp/tests/eval_furuta_mpc.cpp's output to solver tolerance.

--method equation: the analytical Furuta ODE (FurutaMPC). With --dump, the
solution (x, u, objective) is written to YAML (e.g. model/eq_python_solution.yaml),
against which src_cpp/tests/eval_furuta_eq_laopt.cpp and eval_furuta_eq_casadi.cpp
check their solutions (they solve from its x0; --x0 overrides the config's
initial state).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from npmpc.nps.NeuralProcess import NeuralProcess
from npmpc.mpc.controller import MPCController
from npmpc.utils import utils


def _flow_list(values) -> str:
    return '[' + ', '.join(repr(float(v)) for v in values) + ']'


def eval_furuta_mpc(np_config_path: str, mpc_config_path: str, z: list,
                    method: str = 'neural', dump_path: str = None, x0: list = None) -> None:
    mpc_params = json.loads(Path(mpc_config_path).read_text())
    mpc_params['method'] = method
    mpc_params['system'] = 'furuta'
    if x0 is not None:
        mpc_params['experiment_options']['x0'] = x0

    if method == 'neural':
        utils.set_folder(str(Path(np_config_path).parent))
        np_params = json.loads(Path(np_config_path).read_text())
        np_model = NeuralProcess(np_params)
        mpc_params['z'] = z
    else:
        np_model = None

    controller = MPCController(np_model, mpc_params)
    try:
        x0 = torch.tensor(mpc_params['experiment_options']['x0'], dtype=torch.float32)
        controller.warm_start(x0)

        x = controller.problem.value(controller.x)
        u = controller.problem.value(controller.u).reshape(-1, 1)
        objective = float(controller.problem.value(controller.problem.f))

        print('x0:', x0.numpy())
        print('x trajectory (rows = time steps):')
        print(x)
        print('u trajectory:')
        print(u)
        print('objective:', objective)

        if dump_path is not None:
            x_rows = '\n'.join('  - ' + _flow_list(row) for row in x)
            u_rows = '\n'.join('  - ' + _flow_list(row) for row in u)
            Path(dump_path).write_text(
                f"method: {method}\n"
                f"x0: {_flow_list(x0.numpy())}\n"
                f"objective: {objective!r}\n"
                f"x:\n{x_rows}\n"
                f"u:\n{u_rows}\n")
            print(f'Wrote {dump_path}')
    finally:
        # MPCController's runtime starts the simulator process; stop it (not
        # runtime.terminate(), which would also write the experiment dump).
        controller.runtime.qube.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--np-config', default=str(PROJECT_ROOT / 'model/furuta_np.json'))
    parser.add_argument('--mpc-config', default=str(PROJECT_ROOT / 'model/furuta_mpc.json'))
    parser.add_argument('--method', choices=['neural', 'equation'], default='neural')
    parser.add_argument('--z', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument('--dump', default=None, help='Write the solution (x, u, objective) to this YAML file.')
    parser.add_argument('--x0', type=float, nargs=4, default=None,
                        help='Initial state [theta, phi, theta_dot, phi_dot] (default: experiment_options.x0 of the config).')
    args = parser.parse_args()
    eval_furuta_mpc(args.np_config, args.mpc_config, args.z, args.method, args.dump, args.x0)
