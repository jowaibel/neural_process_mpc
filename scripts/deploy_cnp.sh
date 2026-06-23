#!/bin/bash
# Run closed-loop MPC on the simulated Furuta pendulum using the trained CNP model.
# The first thing this does is encode an open-loop trajectory (defined in
# model/furuta_experiment.json) to estimate z, then it
# spins up the simulator and the SQP receding-horizon loop.
#
# Set "save": true in model/furuta_mpc.json to dump the run to an .npz, then view it:
#   PYTHONPATH=src python -m npmpc.mpc.plotting.plot_mpc_trajectory <run>.npz
#
# Run from the project root: `./scripts/deploy_cnp.sh` (chmod +x first if needed).

set -euo pipefail


PYTHONPATH=src conda run -n np-mpc --no-capture-output python -m npmpc.main \
    --system furuta \
    --folder model/ \
    --deploy furuta_mpc.json \
    --model  furuta_np.json