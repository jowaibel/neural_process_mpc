#!/usr/bin/env python3
"""Plot a closed-loop log dumped by scripts/run_qube_server.py --dump.

Each dump is an .npz with:
  t   (n,)   wall-clock timestamps of each state sample
  x   (n,4)  measured state [theta, phi, theta_dot, phi_dot]
  u   (n,1)  torque applied at (i.e. most recently set before) each sample
  dt  ()     nominal control period (MPC model step)
and, in newer dumps:
  solve_t  (m,)  wall-clock arrival time of each input that reported a solve time
  solve_ms (m,)  MPC computation time of that solve (ms), plotted lower right
  state_period, sim_period ()  state-stream and simulator integration periods (s)
  pred_t (k,), pred_x (k, N+1, 4), pred_u (k, N, 1)  the logged open-loop MPC
      predictions (pred_t: time of the state each was solved from); overlaid on
      the state and torque plots, with a marker on each closed-loop state solved from
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def display_path(path: str) -> str:
    """`path` relative to the project root (e.g. model/experiment_x.npz), or as given if outside it."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='Path to the .npz file written by run_qube_server.py --dump')
    parser.add_argument('--no-pred', action='store_true', help='Do not plot the logged open-loop predictions.')
    args = parser.parse_args()

    d = np.load(args.path, allow_pickle=True)
    t = d['t'] - d['t'][0]
    x = d['x']
    u = d['u']

    labels = [r'pendulum angle $\theta$ (rad)', r'arm angle $\varphi$ (rad)',
              r'pendulum ang. vel. $\dot{\theta}$ (rad/s)', r'arm ang. vel. $\dot{\varphi}$ (rad/s)']

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    state_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for i, ax in enumerate(state_axes):
        ax.plot(t, x[:, i], c='C0', lw=1.2)
        ax.set_ylabel(labels[i])
        ax.grid(True, lw=0.3)

    ax_u = axes[2, 0]
    ax_u.step(t, u.reshape(-1), c='C1', lw=1.2, where='post')
    ax_u.set_ylabel(r'arm torque $\tau$ (Nm)')
    ax_u.set_xlabel('Time (s)')
    ax_u.grid(True, lw=0.3)

    # Logged open-loop MPC predictions (run_qube_server.py --pred-count/--pred-period),
    # prediction node k at t_state + k * dt, and a marker on the closed-loop
    # state each prediction was solved from.
    if not args.no_pred and 'pred_x' in d.files and d['pred_x'].size > 0:
        pred_t = d['pred_t'] - d['t'][0]
        pred_x = d['pred_x']              # (k, N+1, 4)
        pred_u = d['pred_u']              # (k, N, 1)
        dt = float(d['dt'])
        n_nodes = pred_x.shape[1]
        style = dict(c='C3', lw=0.8, ms=3, zorder=2)
        # Closed-loop sample the prediction started from (the server logs each
        # state at the same time t it sends it, so pred_t matches a sample).
        start_idx = np.clip(np.searchsorted(t, pred_t - 1e-9), 0, len(t) - 1)
        for j in range(len(pred_t)):
            t_nodes = pred_t[j] + dt * np.arange(n_nodes)
            for i, ax in enumerate(state_axes):
                ax.plot(t_nodes, pred_x[j, :, i], '.-', **style)
            ax_u.step(t_nodes[:-1], pred_u[j, :, 0], where='post', c='C3', lw=0.8, zorder=2)
            ax_u.plot(t_nodes[:-1], pred_u[j, :, 0], '.', c='C3', ms=3, zorder=2)
        for i, ax in enumerate(state_axes):
            ax.plot(t[start_idx], x[start_idx, i], 'o', c='k', ms=4, zorder=3)
        state_axes[0].plot([], [], c='C0', lw=1.2, label='closed loop')
        state_axes[0].plot([], [], '.-', label=f'open-loop predictions ({len(pred_t)})', **style)
        state_axes[0].plot([], [], 'o', c='k', ms=4, label='closed-loop state solved from')
        state_axes[0].legend(loc='best', fontsize=7)

    ax_c = axes[2, 1]
    if 'solve_ms' in d.files and d['solve_ms'].size > 0:
        t_solve = d['solve_t'] - d['t'][0]
        solve_ms = d['solve_ms']
        ax_c.plot(t_solve, solve_ms, c='C2', lw=0.8, marker='.', ms=3,
                  label=f'mean {solve_ms.mean():.2f} ms, max {solve_ms.max():.2f} ms')
        if 'state_period' in d.files:
            ax_c.axhline(1e3 * float(d['state_period']), c='k', ls='--', lw=0.8,
                         label=f'state period {1e3 * float(d["state_period"]):g} ms')
        ax_c.set_ylim(bottom=0)
        ax_c.set_ylabel('MPC computation time (ms)')
        ax_c.set_xlabel('Time (s)')
        ax_c.grid(True, lw=0.3)
        ax_c.legend(loc='upper right', fontsize=8)
    else:
        ax_c.axis('off')  # older dumps without solve times

    fig.suptitle(f'{display_path(args.path)} -- C++ NP-MPC closed loop')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
