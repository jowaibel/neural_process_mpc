"""Python-side counterpart to src_cpp/tests/eval_furuta_mpc.cpp: builds the
same NP-dynamics MPC problem (FurutaNPMPC + MPCController) from the same
config files (model/furuta_np.json, model/furuta_mpc.json) and the same
placeholder z, and solves one warm-start NLP for the configured x0.

Used to cross-check the C++ port's constraints/cost/warm-start logic against
the original Python implementation: the printed x/u trajectories should
match src_cpp/tests/eval_furuta_mpc.cpp's output to solver tolerance.
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


def eval_furuta_mpc(np_config_path: str, mpc_config_path: str, z: list) -> None:
    utils.set_folder(str(Path(np_config_path).parent))
    np_params = json.loads(Path(np_config_path).read_text())
    np_model = NeuralProcess(np_params)

    mpc_params = json.loads(Path(mpc_config_path).read_text())
    mpc_params['z'] = z
    mpc_params['method'] = 'neural'
    mpc_params['system'] = 'furuta'

    controller = MPCController(np_model, mpc_params)
    x0 = torch.tensor(mpc_params['experiment_options']['x0'], dtype=torch.float32)
    controller.warm_start(x0)

    print('x0:', x0.numpy())
    print('x trajectory (rows = time steps):')
    print(controller.problem.value(controller.x))
    print('u trajectory:')
    print(controller.problem.value(controller.u))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--np-config', default=str(PROJECT_ROOT / 'model/furuta_np.json'))
    parser.add_argument('--mpc-config', default=str(PROJECT_ROOT / 'model/furuta_mpc.json'))
    parser.add_argument('--z', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0])
    args = parser.parse_args()
    eval_furuta_mpc(args.np_config, args.mpc_config, args.z)
