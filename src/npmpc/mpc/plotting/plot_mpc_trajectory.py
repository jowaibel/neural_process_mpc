#!/usr/bin/env python3
"""Interactive visualizer for a saved MPC trajectory (FurutaRuntime._dump_data).

Four state panels (θ, φ, θ̇, φ̇), the applied control torque, and the per-step
solve time. A slider selects an MPC step k; the overlay then shows the
open-loop trajectory (states and control) predicted at that step.
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?',
                        default='mpc_run.npz',
                        help='Path to the .npz file written by FurutaRuntime._dump_data')
    args = parser.parse_args()

    data = np.load(args.path, allow_pickle=True)
    x0 = data['x0']                  # (sim_steps, 4)             measured closed-loop
    u0 = data['u0']                  # (sim_steps, 1)             measured closed-loop
    # Optimizer's accepted iterate of x. (sim_steps+1, horizon+1, 4)
    x = data['x']
    # True NP open-loop rollout via torch (saved only at the end of a
    # method='neural' run); may differ from x if the SQP didn't fully
    # converge the dynamics constraints.
    x_np = data['x_np'] if 'x_np' in data.files else x
    x_oracle = data['x_oracle'] if 'x_oracle' in data.files else None  # analytical ODE rollout
    u = data['u']              # (sim_steps+1, horizon+1, 1) open-loop control predictions
    mpc_t = data['mpc_t']            # (sim_steps,)                SQP solve wall-times
    horizon = int(data['horizon_steps'])
    sim_steps = int(data['sim_steps'])
    dt = float(data['dt'])
    k_last = int(data['k']) if 'k' in data.files else sim_steps - 1
    n_valid = k_last + 1

    # Wall-clock timestamps from the runtime, offset so the plot starts at t = 0.
    t_full = data['t']
    t_axis = t_full[:n_valid] - t_full[0]
    labels = [r'$\theta$ (rad)', r'$\varphi$ (rad)',
              r'$\dot{\theta}$ (rad/s)', r'$\dot{\varphi}$ (rad/s)']

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    plt.subplots_adjust(bottom=0.14, hspace=0.25)

    state_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    ax_u = axes[2, 0]
    ax_t = axes[2, 1]

    # -- Closed-loop background (always visible) --
    for i, ax in enumerate(state_axes):
        ax.plot(t_axis[:n_valid], x0[:n_valid, i],
                c='C0', lw=1.2, alpha=0.85, label='closed-loop')
        ax.set_ylabel(labels[i])
        ax.grid(True, lw=0.3)
    ax_u.step(t_axis[:n_valid], u0[:n_valid, 0],
              c='C1', lw=1.2, alpha=0.85, where='post', label='applied $u$')
    ax_u.set_ylabel(r'$\tau$ (Nm)')
    ax_u.set_xlabel('Time (s)')
    ax_u.grid(True, lw=0.3)

    ax_t.plot(t_axis[:n_valid], 1e3 * mpc_t[:n_valid], c='C2', lw=1.2)
    ax_t.axhline(1e3 * dt, c='black', lw=0.6, ls='--', label=f'dt = {1e3*dt:.0f} ms')
    ax_t.set_ylabel('MPC time (ms)')
    ax_t.set_xlabel('Time (s)')
    ax_t.grid(True, lw=0.3)
    ax_t.legend(loc='best', fontsize=8)

    # -- Open-loop highlighted overlay (driven by the slider) --
    ol_state_lines, ol_oracle_lines = [], []
    for ax in state_axes:
        line, = ax.plot([], [], c='C3', lw=2.0, marker='o', ms=3,
                        label='NP open-loop @ k')
        ol_state_lines.append(line)
        if x_oracle is not None:
            oracle_line, = ax.plot([], [], c='C2', lw=1.5, ls='--', marker='s', ms=3,
                                    label='oracle open-loop @ k')
            ol_oracle_lines.append(oracle_line)
    ol_u_line, = ax_u.step([], [], c='C3', lw=2.0, where='post',
                            label='open-loop @ k')
    # Vertical markers for the current k on the time axes
    k_marks_x = [ax.axvline(t_axis[0], c='C3', lw=0.8, alpha=0.6)
                 for ax in state_axes + [ax_u]]
    k_mark_t = ax_t.axvline(t_axis[0], c='C3', lw=0.8, alpha=0.6)

    state_axes[0].legend(loc='best', fontsize=8)
    ax_u.legend(loc='best', fontsize=8)

    # Lock axis limits: set_data doesn't autoscale, so take the union of the
    # closed-loop trajectory and every open-loop prediction once at startup.
    margin = 0.05

    def padded(lo, hi, default_pad=0.1):
        span = hi - lo
        pad = span * margin if span > 0 else default_pad
        return lo - pad, hi + pad

    # x-axis: closed-loop ends at t_axis[k_last]; the longest open-loop overlay
    # extends horizon extra steps past that.
    t_min = t_axis[0]
    t_max = t_axis[k_last] + horizon * dt
    state_axes[0].set_xlim(*padded(t_min, t_max))   # sharex propagates to all

    # y-axis per state: union of closed-loop, NP open-loop, and oracle.
    for i, ax in enumerate(state_axes):
        cl = x0[:n_valid, i]
        ol = x_np[:n_valid, 1:, i]
        lo = min(cl.min(), ol.min())
        hi = max(cl.max(), ol.max())
        if x_oracle is not None:
            orc = x_oracle[:n_valid, 1:, i]
            lo, hi = min(lo, orc.min()), max(hi, orc.max())
        ax.set_ylim(*padded(lo, hi))

    # y-axis for control: applied u + every predicted open-loop u
    applied = u[:n_valid, 0, 0]
    ol_u = u[:n_valid, 1:horizon, 0]
    ax_u.set_ylim(*padded(min(applied.min(), ol_u.min()),
                          max(applied.max(), ol_u.max()),
                          default_pad=0.01))

    # -- Slider + step buttons --
    btn_prev_ax = plt.axes([0.10, 0.04, 0.04, 0.04])
    slider_ax   = plt.axes([0.22, 0.045, 0.62, 0.03])
    btn_next_ax = plt.axes([0.86, 0.04, 0.04, 0.04])
    slider = Slider(slider_ax, '$k$', 0, k_last,
                    valinit=0, valstep=1, valfmt='%d')
    btn_prev = Button(btn_prev_ax, '◀')
    btn_next = Button(btn_next_ax, '▶')

    def step(delta):
        slider.set_val(int(np.clip(int(slider.val) + delta, 0, k_last)))

    btn_prev.on_clicked(lambda _: step(-1))
    btn_next.on_clicked(lambda _: step(+1))

    # Left/right arrow keys give the same one-step nudges.
    def on_key(event):
        if event.key == 'left':
            step(-1)
        elif event.key == 'right':
            step(+1)
    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(val):
        k = int(val)
        # Open-loop state prediction at step k: measured x0[k] then the
        # predicted x_np[k], at times t_axis[k] + 0, dt, …
        t_ol_x = t_axis[k] + dt * np.arange(0,horizon+2)
        for i, line in enumerate(ol_state_lines):
            line.set_data(t_ol_x, np.concat([x0[k:k+1, i], x_np[k, :, i]]))
        if x_oracle is not None:
            for i, line in enumerate(ol_oracle_lines):
                line.set_data(t_ol_x, np.concat([x0[k:k+1, i], x_oracle[k, :, i]]))

        # Open-loop control prediction at step k.
        t_ol_u = t_axis[k] + dt * np.arange(horizon+1)
        ol_u_line.set_data(t_ol_u, np.concat([u0[k:k+1, 0], u[k, :, 0]]))

        # Vertical markers
        for vline in k_marks_x:
            vline.set_xdata([t_axis[k], t_axis[k]])
        k_mark_t.set_xdata([t_axis[k], t_axis[k]])

        fig.suptitle(
            f"{args.path}  —  step k={k}/{k_last} (t={t_axis[k]:.2f}s), "
            f"horizon={horizon}, dt={dt}",
            fontsize=10)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()


if __name__ == '__main__':
    main()
