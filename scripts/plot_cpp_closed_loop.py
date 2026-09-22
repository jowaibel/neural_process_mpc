#!/usr/bin/env python3
"""Plot a closed-loop log dumped by scripts/run_qube_server.py --dump.

Each dump is an .npz with:
  t   (n,)   wall-clock timestamps of each state sample
  x   (n,4)  measured state [theta, phi, theta_dot, phi_dot]
  u   (n,1)  torque applied at (i.e. most recently set before) each sample
  dt  ()     nominal control period
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='Path to the .npz file written by run_qube_server.py --dump')
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

    axes[2, 1].axis('off')

    fig.suptitle(f'{args.path} -- C++ NP-MPC closed loop')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
