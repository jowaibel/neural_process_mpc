#!/bin/bash
# Train the furuta CNP model (CNP encoder + CNP decoder), saving checkpoints to model/.
# The train/test split below (`:16384` / `16384:`) assumes the dataset has at least
# 16384 trajectories — raise `p_size` in datasets/furuta_system.json or adjust the split.
# Run from the project root: `./scripts/train_cnp.sh` (chmod +x first if needed).

set -euo pipefail


PYTHONPATH=src conda run -n np-mpc --no-capture-output python -m npmpc.main \
    --system furuta \
    --folder model/ \
    --train furuta_training.json \
    --model furuta_np.json \
    --train_data "datasets/furuta/:16384" \
    --test_data  "datasets/furuta/16384:"
