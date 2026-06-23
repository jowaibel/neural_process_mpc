"""Closed-loop Furuta MPC comparison — nominal (equation) vs Neural-Process MPC.

3×2 trajectory figure from saved closed-loop experiments: for each controller
the realized state trajectory, the applied torque and the per-step solver time,
plus the NP controller's open-loop planned horizons at a few instants.

Each `.npz` stores, per closed-loop step:
  x0  (realized state)        u0  (applied input)
  x   (planned state horizon) u   (planned input horizon)
  x_oracle (true-dynamics rollout of the plan)
  mpc_t (solver time)         dt, horizon_steps, sim_steps
The Monte-Carlo band (x_mc) is drawn only if stored in the file.

All parameters are frozen in CONFIG below; the script takes no command-line
arguments.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from npmpc.utils.utils import get_project_root
from npmpc.utils import matplotlib_config


COLOR_NP = '#DDAA33'      # NP closed-loop
COLOR_OPEN = '#004488'    # NP open-loop plan
COLOR_REF = '#BB5566'     # reference
COLOR_BASE = 'lightgrey'  # baseline (equation) closed-loop
COLOR_NP_T = 'dimgrey'  # baseline solver time

STATE_LABELS = [r'$\theta$ (rad)', r'$\varphi$ (rad)',
                r'$\dot{\theta}$ (rad/s)', r'$\dot{\varphi}$ (rad/s)',
                r'$\tau$ (mNm)', 'comp. time (ms)']
LW = 1.5
MSZ = 2.5
ALPH = 0.4

# Frozen configuration (no command-line arguments).
CONFIG = SimpleNamespace(
    data_dir='datasets/furuta_mpc',
    baseline='experiment_eq_m3.npz',   # nominal/equation-MPC experiment file
    neural='experiment_np_m3.npz',     # neural-process-MPC experiment file
    n_pred=4,                          # number of NP open-loop horizons to overlay
    theta_ref=2 * np.pi,               # reference value for the theta panel (upright)
    rt_ms=20.0,                        # real-time requirement line (ms)
    show_oracle=False,                 # overlay true-dynamics rollout of each shown plan
    t_max=0.86,                         # trim the time axis to [0, t_max] seconds
    save='plots/furuta_np_mpc_v3.pdf',
)


def load_experiment(path):
    """Load a closed-loop run, trimmed to the valid steps and with t zeroed."""
    d = np.load(path, allow_pickle=True)
    n = int(d['sim_steps'])  # the trailing pre-allocated row is unfilled (all zeros)
    return dict(
        t=d['t'][:n] - d['t'][0],
        x0=d['x0'][:n],
        u0=d['u0'][:n],
        x=d['x'][:n],
        u=d['u'][:n],
        x_oracle=d['x_oracle'][:n],
        x_mc=d['x_mc'][:n] if 'x_mc' in d.files else None,  # (n, n_mc, horizon+1, 4)
        mpc_t=d['mpc_t'][:n],
        dt=float(d['dt']),
        horizon=int(d['horizon_steps']),
    )


def plot_closed_loop(axes, exp, fmt, color):
    """Realized state trajectory, applied torque (mNm) — one line per panel."""
    for ix in range(4):
        axes[ix].plot(exp['t'], exp['x0'][:, ix], fmt, lw=LW, ms=MSZ, c=color)
    axes[4].step(exp['t'], 1e3 * exp['u0'][:, 0], fmt, where='post', lw=LW, ms=MSZ, c=color)


def plot_open_loop(axes, exp, n_pred, color, show_oracle):
    """NP planned horizons at a few instants: Monte-Carlo band, optional oracle
    rollout, and the mean plan on top."""
    n = len(exp['t'])
    step = max(1, 2 + n // n_pred)
    t_pred = exp['dt'] * np.arange(exp['horizon'] + 1)
    mc = exp['x_mc']
    mc_alpha = max(0.03, 6.0 / mc.shape[1]) if mc is not None else 0
    for i in range(0, n - (exp['horizon'] + 1), step):
        # The stored plan at step i starts at the next closed-loop step (x[i,0]≈x0[i+1]).
        t = exp['t'][i + 1] + t_pred
        for ix in range(4):
            if mc is not None:
                axes[ix].plot(t, mc[i, :, :, ix].T, '-', lw=0.5 * LW, c=color, alpha=mc_alpha)
            if show_oracle:
                axes[ix].plot(t, exp['x_oracle'][i, :, ix], '-', lw=0.6 * LW, c=color, alpha=ALPH)
            axes[ix].plot(t, exp['x'][i, :, ix], '.-', lw=LW, ms=MSZ, c=color)
        axes[4].step(t, 1e3 * np.r_[exp['u'][i, :, 0], exp['u'][i, -1, 0]], where='post', lw=LW, c=color)
        axes[4].plot(t[:-1], 1e3 * exp['u'][i, :, 0], '.', ms=MSZ, c=color)


def main():
    args = CONFIG
    root = get_project_root()
    base = load_experiment(os.path.join(root, args.data_dir, args.baseline))
    neural = load_experiment(os.path.join(root, args.data_dir, args.neural))

    # Offset theta by 2pi for plotting, to converge to 0
    args.theta_ref -= 2 * np.pi
    base["x0"][:, 0] -= 2 * np.pi
    base["x"][:, :, 0] -= 2 * np.pi
    neural["x0"][:, 0] -= 2 * np.pi
    neural["x"][:, :, 0] -= 2 * np.pi
    neural["x_oracle"][:, :, 0] -= 2 * np.pi
    neural["x_mc"][:, :, :, 0] -= 2 * np.pi

    matplotlib_config.matplotlib_set_publication()

    fig = plt.figure(figsize=(10, 4.5), layout='constrained')
    axs = fig.add_gridspec(3, 2, wspace=0, hspace=0.1).subplots(sharex=True)
    axes = list(axs.flatten())  # [theta, phi, theta_dot, phi_dot, tau, comp_time]
    t_span = [neural['t'][0], neural['t'][-1]]

    # Reference lines
    axes[0].plot(t_span, [args.theta_ref, args.theta_ref], '--', lw=1.5 * LW, c=COLOR_REF)
    axes[5].plot(t_span, [args.rt_ms, args.rt_ms], '--', lw=1.5 * LW, c='black')

    # Baseline (equation) MPC: closed-loop + solver time
    plot_closed_loop(axes, base, 'o--', COLOR_BASE)
    axes[5].plot(base['t'], 1e3 * base['mpc_t'], 'o--', lw=LW, ms=MSZ, c=COLOR_BASE)

    # Neural-process MPC: closed-loop + open-loop plans + solver time
    plot_closed_loop(axes, neural, 'o-', COLOR_NP)
    plot_open_loop(axes, neural, args.n_pred, COLOR_OPEN, args.show_oracle)
    axes[5].plot(neural['t'], 1e3 * neural['mpc_t'], 'o-', lw=LW, ms=MSZ, c=COLOR_NP_T)

    for ax, label in zip(axes, STATE_LABELS):
        ax.set_ylabel(label, labelpad=0)
        ax.grid(True, lw=0.3)
    axes[4].set_xlabel('Time (s)', labelpad=0)
    axes[5].set_xlabel('Time (s)', labelpad=0)
    axes[0].set_xlim(0, args.t_max)  # shared x-axis

    # Legends
    axes[0].plot([], [], '--', lw=1.5 * LW, c=COLOR_REF, label='reference')
    axes[0].plot([], [], 'o--', lw=LW, ms=MSZ, c=COLOR_BASE, label='baseline')
    axes[0].plot([], [], lw=0, label=' ')
    axes[0].plot([], [], 'o-', lw=LW, ms=MSZ, c=COLOR_NP, label='closed-L')
    axes[0].plot([], [], '.-', lw=LW, ms=MSZ, c=COLOR_OPEN, label='open-L')
    if neural['x_mc'] is not None:
        axes[0].plot([], [], '-', lw=0.5 * LW, c=COLOR_OPEN, alpha=ALPH, label='MC')
    if args.show_oracle:
        axes[0].plot([], [], '-', lw=0.6 * LW, c=COLOR_OPEN, alpha=ALPH, label='oracle')
    # axes[0].legend(loc='best', ncol=2)
    axes[0].legend(ncol=2, loc="lower right", bbox_to_anchor=(0.93, 0.02))

    axes[5].plot([], [], '--', lw=1.5 * LW, c='black', label='real-time req.')
    axes[5].plot([], [], 'o--', lw=LW, ms=MSZ, c=COLOR_BASE, label='baseline')
    axes[5].plot([], [], 'o-', lw=LW, ms=MSZ, c=COLOR_NP_T, label='NP-MPC')
    # axes[5].legend(loc='best')
    axes[5].legend(ncol=2, loc="lower left", bbox_to_anchor=(0.152, 0.08))

    # Axis limits and ticks
    axes[0].set_ylim(-np.pi-0.5, 1.0)
    axes[1].set_ylim(-1.9, 0.7)
    axes[2].set_ylim(-9, 25)
    axes[3].set_ylim(-24, 24)

    # axes[0].set_xlim(0, neural['t'][-1])
    axes[0].set_xlim(0, min(args.t_max, neural['t'][-1]))
    axes[0].xaxis.set_major_locator(MultipleLocator(0.2))  # tick every 5
    # axes[0].xaxis.set_minor_locator(MultipleLocator(0.1))  # minor every 1

    axes[-1].set_ylim(-1.2, 24)

    fig.align_ylabels()

    plt.show()

    if False:
        save_path = os.path.join(root, args.save)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved figure to {save_path}")
    else:
        print(f"Figure not saved.")


if __name__ == '__main__':
    main()
