# Conditional Neural Process MPC for the Furuta Pendulum

Code accompanying the paper:

> **Neural Process Model Predictive Control**
> Johannes Waibel*, Pietro Mello Rella*, Colin N. Jones.
> Automatic Control Lab, EPFL — preprint submitted to Elsevier (2026).

A **Conditional Neural Process (CNP)** learns the residual dynamics of a Furuta
(rotary inverted) pendulum across a family of physical parameterisations. At
deployment the CNP encodes a short context trajectory of the current pendulum
into a latent code `z`, and a CasADi/IPOPT model-predictive controller uses the
`z`-conditioned learned model to swing the pendulum up and stabilise it
upright — adapting to an unseen pendulum from a few samples, without retraining.

> **Notation:** the latent the code calls `z` is the aggregated representation
> denoted `r` in the paper — they are the same quantity.

The paper compares several Neural Process variants for this rapid-adaptation
task; this repository is a clean, single-variant release of the deployed CNP and
its MPC.

> **Simulation only:** this release runs against the bundled Furuta simulator
> (`npmpc/hardware/QubeSimulator`); the driver for the physical Qube is not
> included, so the experiment config and deploy use `target: sim`. For
> implementing it on real hardware, contact pietro.mellorella@epfl.ch.

## Method in one paragraph

- **Encoder** (`npmpc/nps/encoder.py`): an MLP maps each context pair
  `(x_i, y_i)` to `r_i`; the `r_i` are mean-pooled into a single latent `z`.
- **Decoder** (`npmpc/nps/decoder.py`): an MLP maps `[x, z]` to a Gaussian over
  the one-step state change — a mean $\mu$ and a per-output standard deviation
  $\sigma$. Trained by Gaussian NLL.
- **MPC** (`npmpc/mpc/`): the model is exported to CasADi; an IPOPT
  receding-horizon controller (`FurutaNPMPC`) plans against the `z`-conditioned
  dynamics.

## Installation

Create the `np-mpc` conda environment (Python 3.12 + pinned dependencies):

```bash
conda env create -f environment.yml
conda activate np-mpc
```

Or with an existing environment:

```bash
pip install -r requirements.txt
```

The code is **not installed as a package** — it is run straight from `src/` by
putting it on `PYTHONPATH` (the helper scripts do this for you). Tested with
Python 3.12 (PyTorch 2.5.1, CasADi 3.7.0 with bundled IPOPT) on macOS (Apple
Silicon, M4) and on Ubuntu 22.04.5 LTS (AMD Ryzen 9 7940HS, the machine used in
the paper). The figure scripts use
LaTeX (`text.usetex`); a TeX install (e.g. TeX Live) is needed to reproduce them
exactly, otherwise set `text.usetex` to `False`.

The shell helpers below call `conda run -n np-mpc`; adjust the env name or run the
underlying `PYTHONPATH=src python -m npmpc.main …` commands directly.

## Usage

Run from the project root. Each helper script documents what it does and the
exact `python -m npmpc.main` command it runs.

```bash
./scripts/create_dataset.sh   # generate the training dataset -> datasets/furuta/
./scripts/train_cnp.sh        # train the CNP -> checkpoints in model/
./scripts/deploy_cnp.sh       # closed-loop MPC swing-up on the simulated pendulum
```

### Reproduce the paper figures

```bash
python scripts/plot_furuta_np_adaptation.py             # context-adaptation diagnostics
python scripts/plot_furuta_np_mpc.py                    # nominal vs NP-MPC comparison
python scripts/plot_furuta_np_adaptation_closedloop.py  # closed-loop cost vs context size
```

## Citation

```bibtex
@article{waibel2026neuralprocessmpc,
  title   = {Neural Process Model Predictive Control},
  author  = {Waibel, Johannes and Mello Rella, Pietro and Jones, Colin N.},
  year    = {2026},
  note    = {Preprint submitted to Elsevier. Johannes Waibel and Pietro Mello Rella contributed equally.},
}
```

## Acknowledgements

This work was supported by the Swiss National Science Foundation under the NCCR
Automation (grant agreement 51NF40_180545).
