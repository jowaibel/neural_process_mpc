#!/bin/bash
# Generate the furuta training dataset from the system-parameter ranges in
# datasets/furuta_system.json, writing per-trajectory .pt files to datasets/furuta/.
# Run from the project root: `./scripts/create_dataset.sh` (chmod +x first if needed).

set -euo pipefail


PYTHONPATH=src conda run -n np-mpc --no-capture-output python -m npmpc.main \
    --system furuta \
    --folder model/ \
    --create_dataset datasets/furuta_system.json \
    --output datasets/furuta/ \
    --device cpu
