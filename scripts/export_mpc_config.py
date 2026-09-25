"""Export a FurutaNPMPC or FurutaMPC config to YAML for the C++ port (src_cpp/npmpc/mpc).

Computes terminal_P (the LQR terminal-cost matrix) the same way the Python
problem does -- torch autograd to linearise the dynamics at the upright
equilibrium, then scipy's discrete ARE -- since the C++ port has no
autodiff/Riccati solver and treats P as a precomputed constant (loaded, not
recomputed).

--method neural (default): FurutaNPMPC for a fixed latent z (--z; the default
all-zero latent is a placeholder, a real deployment infers z from context
data via the NP encoder). Writes `z`.

--method equation: FurutaMPC, the analytical Furuta ODE with the plant
parameters `p` from the MPC config. Writes `p` instead of `z`.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from npmpc.nps.NeuralProcess import NeuralProcess
from npmpc.mpc.problem.FurutaMPC import FurutaMPC
from npmpc.mpc.problem.FurutaNPMPC import FurutaNPMPC
from npmpc.utils import utils


def _flow_list(values) -> str:
    return '[' + ', '.join(repr(float(v)) for v in values) + ']'


def _bound_pair(pair) -> str:
    def fmt(v):
        if v == float('inf'):
            return '.inf'
        if v == -float('inf'):
            return '-.inf'
        return repr(float(v))
    return f'[{fmt(pair[0])}, {fmt(pair[1])}]'


def export_mpc_config(np_config_path: str, mpc_config_path: str, output_path: str, z: list,
                      method: str = 'neural') -> None:
    mpc_params = json.loads(Path(mpc_config_path).read_text())
    if method == 'neural':
        utils.set_folder(str(Path(np_config_path).parent))
        np_params = json.loads(Path(np_config_path).read_text())
        np_model = NeuralProcess(np_params)
        mpc_params['z'] = z
        FurutaNPMPC(np_model, mpc_params)  # computes params['cost']['terminal_P'] since z is not None
        model_line = f"z: {_flow_list(z)}"
    else:
        FurutaMPC(mpc_params)  # computes params['cost']['terminal_P'] from the analytical dynamics
        model_line = f"p: {_flow_list(mpc_params['p'][0])}  # [lp, mp, lr, mr]"

    cost = mpc_params['cost']
    hard_bound = mpc_params['hard_bound']
    slack_bound = mpc_params['slack_bound']

    p_rows = '\n'.join('  - ' + _flow_list(row) for row in cost['terminal_P'])
    hard_bound_x = '\n'.join('    - ' + _bound_pair(pair) for pair in hard_bound['x'])
    hard_bound_u = '\n'.join('    - ' + _bound_pair(pair) for pair in hard_bound['u'])
    slack_bound_x = '\n'.join(
        '    - ' + ('.inf' if b[0] == float('inf') else repr(float(b[0])))
        for b in slack_bound['x']
    )

    yaml_text = f"""\
dt: {mpc_params['dt']}
horizon_steps: {mpc_params['horizon_steps']}
x_size: 4
u_size: 1
solver: {mpc_params['solver']}
{model_line}
hard_bound:
  x:
{hard_bound_x}
  u:
{hard_bound_u}
slack_bound:
  x:
{slack_bound_x}
cost:
  x: {_flow_list(cost['x'])}
  x_diff: {_flow_list(cost['x_diff'])}
  x_end: {_flow_list(cost['x_end'])}
  u: {_flow_list(cost['u'])}
  terminal_p:
{p_rows}
experiment_options:
  x0: {_flow_list(mpc_params['experiment_options']['x0'])}
  sim_steps: {mpc_params['experiment_options']['sim_steps']}
"""

    Path(output_path).write_text(yaml_text)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--np-config', default='model/furuta_np.json')
    parser.add_argument('--mpc-config', default='model/furuta_mpc.json')
    parser.add_argument('--method', choices=['neural', 'equation'], default='neural')
    parser.add_argument('--output', default=None,
                        help='Default: model/mpc_config.yaml (neural), model/mpc_config_equation.yaml (equation)')
    parser.add_argument('--z', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0])
    args = parser.parse_args()
    output = args.output or ('model/mpc_config.yaml' if args.method == 'neural' else 'model/mpc_config_equation.yaml')
    export_mpc_config(args.np_config, args.mpc_config, output, args.z, args.method)
