"""NP adaptation diagnostics on per-system Furuta trajectories (one file per
physical system: sys1/sys2/sys3/...).

Left column:
  • Adaptation progress: 2D PCA projection of the encoded latent r(|C|) as the
    context grows; one polyline per system, star marker at the infinite-context
    reference r̂_∞.
  • Prediction MSE: mean squared error of the decoder's mean prediction over the
    full trajectory vs |C| (encoder fed only the first |C| samples).
  • Decoder performance: mean NLL of the decoder over the full trajectory,
    evaluated with the encoder fed only the first |C| samples.

Right column: three prediction panels at the selected context sizes. Every panel
rolls out open-loop from the same step — max(|C|) — so they are directly
comparable; only the amount of context fed to the encoder differs. The rollout
predicts the window, resetting to ground truth every reset interval, with
Monte-Carlo samples drawn from the decoder's output distribution.

All parameters are frozen in CONFIG below; the script takes no command-line
arguments.
"""
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import matplotlib.pyplot as plt
import numpy as np
import torch

from npmpc.nps.NeuralProcess import NeuralProcess
from npmpc.dynamics.integrators import FurutaNPIntegration
from npmpc.utils.utils import get_project_root, set_folder
from npmpc.utils import matplotlib_config

# Per-system colors/markers (sys1/sys2/sys3).
SYSTEM_STYLE = {
    'sys1': dict(color='#004488', marker='o', msz=1.1, mfc='full'),
    'sys2': dict(color='#DDAA33', marker='^', msz=1.2, mfc='full'),
    'sys3': dict(color='#BB5566', marker='s', msz=0.9, mfc='full'),
}
SYSTEM_ORDER = ['sys1', 'sys2', 'sys3']

STATE_IDX = {'theta_dot': 2, 'phi_dot': 3}
STATE_LABEL = {'theta_dot': r'$\dot{\theta}$ (rad/s)', 'phi_dot': r'$\dot{\varphi}$ (rad/s)'}

LW = 1.5
MSZ = 2.5
ALPH = 0.7

# Frozen configuration (no command-line arguments).
CONFIG = SimpleNamespace(
    model='model/furuta_np.json',
    data_dir='datasets/furuta_sinusoids',
    dt=0.02,
    ctx_step=1,            # stride for the |C| sweep (distance + loss curves)
    ctx_max=150,           # cap on |C|
    ctx_plot_max=None,     # max |C| shown on the left curves (display only)
    pred_ctx=[1, 20, 50],  # three context sizes for the prediction panels
    pred_systems=['sys1', 'sys3'],  # two systems overlaid in the prediction panels
    reset_ms=250.0,        # open-loop reset interval (ms)
    pred_duration=1.5,     # time window (s) shown in the prediction panels
    pred_state='phi_dot',
    n_mc=100,
    hide_context=True,     # show only the prediction window
    save='plots/furuta_np_adaptation.pdf',
)


# --------------------------------------------------------------------------- #
# Styling helpers                                                             #
# --------------------------------------------------------------------------- #
def style_of(label, i=0):
    st = SYSTEM_STYLE.get(label, dict(color=plt.cm.tab10(i), marker='o', mfc='full'))
    return dict(st, label=label)


def disp_label(label):
    """Render 'sys1' as the math label '$\\mathrm{sys}_1$' (and 'sys1+2' as '$\\mathrm{sys}_{1+2}$')."""
    # s = re.match(r'sys(\d+(?:\+\d+)?)$', label)
    # return rf'$\mathrm{{sys}}_{{{s.group(1)}}}$' if s else label
    return label #.capitalize()


def nearest_idx(grid, value):
    return min(range(len(grid)), key=lambda k: abs(grid[k] - value))


# --------------------------------------------------------------------------- #
# Model / data                                                                #
# --------------------------------------------------------------------------- #
def build_xy(traj, n):
    """NN inputs (sinθ, cosθ, θ̇, φ̇, u) and targets (Δθ̇, Δφ̇) over the first n steps."""
    x, u = traj['x'][0], traj['u'][0]
    x_nn = torch.cat([torch.sin(x[:n, 0:1]), torch.cos(x[:n, 0:1]), x[:n, 2:4], u[:n]], dim=-1)
    y_nn = x[1:n + 1, 2:4] - x[:n, 2:4]
    return x_nn, y_nn


def encode(model, traj, n):
    x_nn, y_nn = build_xy(traj, n)
    mask = torch.ones(1, n, dtype=torch.bool)
    return model.encode(x_nn.unsqueeze(0), y_nn.unsqueeze(0), mask)


def decoder_loss(model, traj, z):
    """Mean NLL of the decoder over the full trajectory, given a context latent."""
    x_nn, y_target = build_xy(traj, traj['u'].shape[1])
    y_mu, y_sigma = model.decode(x_nn.unsqueeze(0), z=z)
    return model.decoder.loss(y_mu, y_sigma, y_target.unsqueeze(0)).mean().item()


def decoder_mse(model, traj, z):
    """MSE of the decoder's mean prediction against the recorded targets over the
    full trajectory."""
    x_nn, y_target = build_xy(traj, traj['u'].shape[1])
    y_mu, _ = model.decode(x_nn.unsqueeze(0), z=z)
    y_pred = y_mu.squeeze(0)
    return ((y_target - y_pred) ** 2).mean().item()


def open_loop_resets(integrator, traj, z, dt, reset_steps, n_mc, start, n_pred):
    """Open-loop rollout over [start, start+n_pred), reset to ground truth every reset_steps."""
    x, u = traj['x'][0], traj['u'][0]
    end = min(u.shape[0], start + n_pred)
    means, mcs, t_segs = [], [], []
    for s in range(start, end, reset_steps):
        e = min(s + reset_steps, end)
        means.append(integrator.integrate(x[s], u[s:e], dt, z=z))
        mcs.append(integrator.montecarlo_integrate(x[s], u[s:e], dt, z=z, n=n_mc))
        t_segs.append(dt * (s + np.arange(e - s + 1)))
    return t_segs, means, mcs


def label_of(path):
    """System id from the filename: the trailing `sysN` token (e.g.
    `furuta_sinusoid_experiment_sys1` -> `sys1`), or the whole stem otherwise."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'(sys\d+(?:\+\d+)?)$', stem)
    return m.group(1) if m else stem


def load_model(model_path):
    with open(model_path) as f:
        params = json.load(f)
    model = NeuralProcess(params)
    model.send_to_device('cpu')
    return model, FurutaNPIntegration(model)


def load_systems(root, data_dir):
    """Return per-system trajectories, ordered sys1/sys2/sys3 first then any extras."""
    files = sorted(f for f in glob.glob(os.path.join(root, data_dir, 'furuta_*'))
                   if not f.endswith('.json'))
    if not files:
        raise FileNotFoundError(f"No furuta_* trajectory files in {data_dir}")
    by_label = {label_of(f): torch.load(f, weights_only=False)[0] for f in files}
    labels = ([s for s in SYSTEM_ORDER if s in by_label]
              + [s for s in by_label if s not in SYSTEM_ORDER])
    return labels, by_label


# --------------------------------------------------------------------------- #
# Computation                                                                 #
# --------------------------------------------------------------------------- #
def sweep_context(model, trajs, ctx_grid, n_steps):
    """For each system, encode r(|C|) over the |C| grid and score the decoder (MSE, NLL)."""
    z_curve, z_inf = [], []
    mse = np.zeros((len(trajs), len(ctx_grid)))
    loss = np.zeros_like(mse)
    with torch.no_grad():
        for i, tr in enumerate(trajs):
            z_inf.append(encode(model, tr, n_steps).squeeze())
            curve = []
            for j, c in enumerate(ctx_grid):
                r = encode(model, tr, c)
                curve.append(r.squeeze())
                mse[i, j] = decoder_mse(model, tr, r)
                loss[i, j] = decoder_loss(model, tr, r)
            z_curve.append(torch.stack(curve))
    return z_curve, torch.stack(z_inf), mse, loss


def pca_2d(z_curve, z_inf):
    """Project every latent (all curves + the r̂_∞ references) onto a shared 2D PCA basis."""
    Z_all = torch.cat(z_curve + [z_inf], dim=0)
    mean = Z_all.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Z_all - mean, full_matrices=False)
    # Vt[2,:] *= 2
    proj = lambda Z: ((Z - mean) @ Vt[:2].T).numpy()
    return [proj(zc) for zc in z_curve], proj(z_inf)


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def make_figure():
    # set_publication_style()
    matplotlib_config.matplotlib_set_publication()

    fig = plt.figure(figsize=(10, 5), layout='constrained')
    grid = fig.add_gridspec(2, 2, wspace=0, hspace=0,
                            width_ratios=[1.2, 1.0], height_ratios=[1.2, 1.0])
    ax_rep = fig.add_subplot(grid[0, 0])
    ax_dist, ax_loss = grid[1, 0].subgridspec(1, 2, wspace=0, hspace=0).subplots(
        sharex='all', sharey='none')
    ax_pred = list(grid[:, 1].subgridspec(3, 1, wspace=0, hspace=0).subplots(
        sharex='all', sharey='all', squeeze=False)[:, 0])
    return fig, ax_rep, ax_dist, ax_loss, ax_pred


def plot_adaptation(ax, z2d_curves, z2d_inf, labels, marker_idx, initial_marker=True):
    ax.set_title('Adaptation progress')
    rinf_color = 0.8 * np.ones(3)

    # Draw initial r (before adaptation)
    for i, label in enumerate(labels):
        st = style_of(label, i)

        ax.plot(z2d_curves[i][:, 0], z2d_curves[i][:, 1], '-', lw=1.0, c=st['color'])
        ax.plot(z2d_inf[i, 0], z2d_inf[i, 1], linestyle='None', marker='*', ms=13,
                mec='black', mfc=st['color'], mew=1.0, zorder=900)
        for k in marker_idx:
            ax.plot(z2d_curves[i][k, 0], z2d_curves[i][k, 1], linestyle='None', marker=st['marker'],
                    ms=6*st['msz'], mec='black', mfc=st['color'], mew=0.5, zorder=901)

    # Initial marker (at marker_idx 0)
    if initial_marker:
        ax.plot(z2d_curves[0][0, 0], z2d_curves[0][0, 1], linestyle='None', marker='h',
                ms=10, mec='black', mfc='white', mew=1.0)
        ax.plot([], [], linestyle='None', marker='h',
                ms=8, mec='black', mfc='white', mew=1.0, label=r'$r_0$')

    ax.plot([], [], '-', lw=1.0, c='black', label=r'$r$ as $C\uparrow$')
    ax.plot([], [], '*', ms=11, mec='black', mfc=rinf_color, mew=1.0, label=r'$\hat{r}_\infty$')
    ax.set_ylabel(r'$r$ (proj. on 2D)')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Wide panel (~1.35:1) with equal data scaling.
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_box_aspect(0.74)
    ax.grid(True, lw=0.3)
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', handlelength=0.8, handletextpad=0.8)


def plot_ctx_curve(ax, title, ctx_grid, values, labels, marker_idx, legend, xlim_right, initial_marker=True):
    """A '<metric> vs |C|' curve per system, marked at the prediction context sizes."""
    ax.set_title(title)
    ctx = np.asarray(ctx_grid)
    for i, label in enumerate(labels):
        st = style_of(label, i)

        ax.plot(ctx, values[i], '-', lw=LW, c=st['color'])
        ax.plot(ctx[marker_idx], np.asarray(values[i])[marker_idx], linestyle='None',
                marker=st['marker'], ms=2*MSZ*st['msz'], mec='black', mfc=st['color'], mew=0.5,
                zorder=900+len(labels)-i)
        ax.plot([], [], linestyle='-', marker=st['marker'], lw=1.4*LW, ms=2*MSZ*st['msz'],
                mec='black', c=st['color'], mew=0.5, label=disp_label(label))

        if initial_marker:
            ax.plot(ctx[0], values[i][0], linestyle='None', marker='h',
                    ms=8, mec='black', mfc='white', mew=1.0)


    ax.set_xlabel(r'Context size $|C|$', labelpad=0)
    ax.grid(True, lw=0.3)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97),
                  handlelength=1.4, handletextpad=0.4)
    if xlim_right:
        ax.set_xlim(right=xlim_right)


def plot_predictions(axes, model, integrator, by_label, args, ctx_max, n_steps):
    reset_steps = max(1, int(round(args.reset_ms / 1000.0 / args.dt)))
    pred_steps = min(int(round(args.pred_duration / args.dt)), n_steps)
    pred_labels = [l for l in args.pred_systems if l in by_label]
    state_idx = STATE_IDX[args.pred_state]

    # Every panel rolls out from the same step (the largest context), over the same window.
    pred_start = min(max(args.pred_ctx), ctx_max)
    end = min(pred_start + pred_steps, n_steps)
    t_full = args.dt * np.arange(end + 1)
    t_split = pred_start * args.dt
    # When the context is hidden, re-zero the time axis at the rollout start so
    # the prediction window reads 0..pred_duration instead of t_split..end.
    t0 = t_split if args.hide_context else 0.0
    mc_alpha = max(0.03, 15 / args.n_mc * ALPH)

    with torch.no_grad():
        for ax, c_req in zip(axes, args.pred_ctx):
            c_val = min(c_req, ctx_max)
            ax.text(0.05, 0.935, rf'$|C|={c_req}$', transform=ax.transAxes,
                    ha='left', va='top', bbox=dict(linewidth=0.8, edgecolor='black', fill=False))
            ax.grid(True, lw=0.3)
            ax.set_ylabel(STATE_LABEL[args.pred_state], labelpad=0)
            ax.set_xlim(0, end * args.dt - t0)
            if not args.hide_context:
                ax.axvspan(0, t_split, color='0.92', lw=0)
                ax.axvline(t_split, c='gray', ls=':', lw=0.8)

            for label in pred_labels:
                tr, st = by_label[label], style_of(label)
                z = encode(model, tr, c_val)
                t_segs, _, mcs = open_loop_resets(
                    integrator, tr, z, args.dt, reset_steps, args.n_mc, pred_start, pred_steps)
                if not args.hide_context:
                    ax.plot(t_full[:c_val + 1] - t0, true[:c_val + 1], '-', c=st['color'], lw=1.8)
                for t_seg, x_mc in zip(t_segs, mcs):
                    for m in range(args.n_mc):
                        ax.plot(t_seg - t0, x_mc[m, :, state_idx].numpy(), c=st['color'], lw=0.5 * LW, alpha=mc_alpha)

            for label in pred_labels:
                tr, st = by_label[label], style_of(label)
                z = encode(model, tr, c_val)
                t_segs, means, _ = open_loop_resets(
                    integrator, tr, z, args.dt, reset_steps, args.n_mc, pred_start, pred_steps)
                true = tr['x'][0, :end + 1, state_idx].numpy()
                ax.plot(t_full - t0, true, '--', c='black', lw=0.7*LW)
                for t_seg, x_mean in zip(t_segs, means):
                    ax.plot(t_seg - t0, x_mean[:, state_idx].numpy(), c='white', lw=1.4*LW)
                    ax.plot(t_seg - t0, x_mean[:, state_idx].numpy(), c=st['color'], lw=1.2*LW)

    axes[0].set_title('Prediction')
    axes[-1].set_xlabel('Time (s)', labelpad=0)

    axes[1].plot([], [], '--', c='black', lw=LW, label='true')
    axes[1].plot([], [], '-', c='black', lw=LW, label='mean')
    axes[1].plot([], [], '-', c='grey', lw=0.5 * LW, alpha=ALPH, label='MC')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.685, 1.0), handlelength=1.0, handletextpad=0.4)

    # for label in pred_labels:
    #     st = style_of(label)
    #     axes[0].plot([], [], c=st['color'], lw=1.2*LW, label=disp_label(label))
    # axes[0].legend(loc='upper center', bbox_to_anchor=(0.337, 0.93), handlelength=1.0, handletextpad=0.4)

def main():
    args = CONFIG
    root = get_project_root()
    set_folder(os.path.dirname(os.path.abspath(args.model)))

    model, integrator = load_model(args.model)
    labels, by_label = load_systems(root, args.data_dir)
    trajs = [by_label[s] for s in labels]

    n_steps = trajs[0]['u'].shape[1]
    ctx_max = min(args.ctx_max or n_steps, n_steps)
    ctx_grid = list(range(1, ctx_max + 1, args.ctx_step))
    # Markers on the left plots sit at the same |C| as the right prediction panels.
    marker_idx = [nearest_idx(ctx_grid, c) for c in args.pred_ctx]

    z_curve, z_inf, mse, loss = sweep_context(model, trajs, ctx_grid, n_steps)
    z2d_curves, z2d_inf = pca_2d(z_curve, z_inf)

    fig, ax_rep, ax_mse, ax_loss, ax_pred = make_figure()
    plot_adaptation(ax_rep, z2d_curves, z2d_inf, labels, marker_idx)
    plot_ctx_curve(ax_mse, 'MSE of mean prediction', ctx_grid, mse, labels,
                   marker_idx, legend=False, xlim_right=args.ctx_plot_max)
    plot_ctx_curve(ax_loss, 'Decoder performance (loss)', ctx_grid, loss, labels,
                   marker_idx, legend=True, xlim_right=args.ctx_plot_max)
    plot_predictions(ax_pred, model, integrator, by_label, args, ctx_max, n_steps)

    ax_mse.set_ylim(0, 0.36)
    ax_pred[0].set_ylim(-18, 18)

    fig.align_ylabels(ax_pred)

    plt.show()

    if True:
        save_path = os.path.join(root, args.save)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved figure to {save_path}")
    else:
        print(f"Figure not saved.")


if __name__ == '__main__':
    main()
