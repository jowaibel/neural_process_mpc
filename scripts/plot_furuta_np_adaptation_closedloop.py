"""Closed-loop adaptation comparison — CNP across context sizes vs the
equation-based baseline.

Bar figures summarising closed-loop performance per controller group:
  • per-system figure: mean closed-loop cost (top) and mean convergence
    (settling) time (bottom), one figure for every system present;
  • combined figure: grouped bars over all systems, cost (top) and
    convergence time (bottom).

The per-run metrics in `cnp_metrics.csv` are aggregated into mean ± std per
group. Groups that never converged (e.g. CNP ctx=1) are drawn with a
placeholder bar at the full run duration (sim_steps * dt) and annotated.

All parameters are frozen in CONFIG below; the script takes no command-line
arguments.
"""
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from npmpc.utils.utils import get_project_root
from npmpc.utils import matplotlib_config


SYSTEM_COLORS = {'sys1': '#004488', 'sys2': '#DDAA33', 'sys3': '#BB5566'}
# Per-system colors/markers (matching plot_furuta_np_adaptation.py).
SYSTEM_STYLE = {
    'sys1': dict(color='#004488', marker='o', msz=1.1, mfc='full'),
    'sys2': dict(color='#DDAA33', marker='^', msz=1.2, mfc='full'),
    'sys3': dict(color='#BB5566', marker='s', msz=0.9, mfc='full'),
}
COLOR_BASE = 'lightgrey'

BASELINE = 'baseline'  # substring identifying the baseline controller group
LW = 1.5
MSZ = 2.5

def is_baseline(group):
    """True for the equation-based baseline group (used as the normaliser)."""
    return BASELINE in group.lower()

# Frozen configuration (no command-line arguments).
CONFIG = SimpleNamespace(
    metrics='datasets/furuta_mpc/all experiments/cnp_metrics.csv',  # per-run metrics, relative to project root
    dt=0.02,                                    # control period (s), matches the experiments
)

def group_sort_key(group):
    """CNP groups ordered by increasing context size; baseline always last."""
    if is_baseline(group):
        return (1, 0)
    m = re.search(r'ctx=(\d+)', group)
    return (0, int(m.group(1)) if m else 0)
def disp_group(group):
    """Render a 'CNP ctx=N' group as the math label '$|C|=N$' (others unchanged)."""
    m = re.search(r'ctx=(\d+)', group)
    return rf'$|C|={m.group(1)}$' if m else group
def aggregate(df):
    """Mean ± std per group, plus convergence count and run length."""
    g = df.groupby('group', observed=True)
    agg = g.agg(
        cost_mean=('cost', 'mean'), cost_std=('cost', 'std'),
        conv_mean=('conv_time_s', 'mean'), conv_std=('conv_time_s', 'std'),
        conv_count=('conv_time_s', 'count'),
        sim_steps=('sim_steps', 'max'),
    )
    order = sorted(agg.index, key=group_sort_key)
    return agg.reindex(order)

def draw_bars(ax, groups, means, stds, colors, ylabel, title, value_fmt,
              missing_mask=None, missing_height=None, missing_label=None):
    """One bar per group with std error bars; non-converged groups get a
    placeholder bar of height `missing_height` and a rotated annotation."""
    x = np.arange(len(groups))
    means = np.asarray(means, dtype=float)
    stds = np.nan_to_num(np.asarray(stds, dtype=float), nan=0.0)
    missing_mask = (np.zeros(len(groups), dtype=bool) if missing_mask is None
                    else np.asarray(missing_mask))

    heights = np.where(missing_mask, missing_height, np.nan_to_num(means, nan=0.0))
    err = np.where(missing_mask, 0.0, stds)

    bars = ax.bar(x, heights, yerr=err, capsize=5, color=colors, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([disp_group(g) for g in groups], rotation=20, ha='right')
    ax.set_ylabel(ylabel, labelpad=0)
    ax.set_title(title)
    ax.grid(axis='y', lw=0.3)
    ax.set_axisbelow(True)

    # Value labels above each bar (cleared past the error bar via padding).
    ax.bar_label(bars, labels=[value_fmt.format(h) for h in heights],
                 padding=3, fontsize=9)
    # "did not converge" annotation inside the placeholder bars.
    if missing_label is not None and missing_mask.any():
        ax.bar_label(bars, labels=[missing_label if mm else '' for mm in missing_mask],
                     label_type='center', rotation=90, fontsize=9, color='white')

def plot_combined(df, dt):
    """Combined figure: grouped bars, one x-group per context size (+ baseline),
    one bar per system. Top panel = cost, bottom = convergence time."""
    systems = sorted(df['system'].unique())
    groups = sorted(df['group'].unique(), key=group_sort_key)

    g = df.groupby(['system', 'group'], observed=True)

    def mat(series, how):
        return getattr(g[series], how)().unstack('system').reindex(index=groups, columns=systems)

    cost_m, cost_s = mat('cost', 'mean'), mat('cost', 'std')
    # conv_m, conv_s = mat('conv_time_s', 'mean'), mat('conv_time_s', 'std')
    # conv_cnt = mat('conv_time_s', 'count')
    # steps = mat('sim_steps', 'max')

    # Normalise each system's columns by its own baseline cost, then drop the
    # baseline row (baseline == 1).
    base_group = [g for g in groups if is_baseline(g)][0]
    base_cost = cost_m.loc[base_group]
    cost_m, cost_s = cost_m.div(base_cost, axis=1), cost_s.div(base_cost, axis=1)
    groups = [g for g in groups if not is_baseline(g)]
    cost_m, cost_s = cost_m.reindex(groups), cost_s.reindex(groups)

    x = np.array([int(re.search(r'ctx=(\d+)', g).group(1)) for g in groups])  # true context sizes
    nsys = len(systems)

    fig, (axc) = plt.subplots(1, 1, figsize=(4.75, 3.75), sharex=True, layout='constrained')

    axc.axhline(1.0, linestyle='-', color='lightgrey', lw=1.5*LW, label='baseline')
    axc.plot([], [], lw=0, label=' ')
    axc.plot([], [], lw=0, label=' ')

    for i, s in enumerate(systems):
        st = SYSTEM_STYLE[s]
        color, marker, ms = st['color'], st['marker'], 2.5 * MSZ * st['msz']

        # --- cost panel ---
        cm = cost_m[s].to_numpy(dtype=float)
        cs = np.nan_to_num(cost_s[s].to_numpy(dtype=float), nan=0.0)
        axc.plot(x, cm, '--', lw=1.0, color=color)  # connect the points
        axc.errorbar(x, cm, yerr=cs, fmt=marker, ms=ms, capsize=6, capthick=0.8, elinewidth=0.8,
                     color=color, ecolor='black', mec='black', mew=0.8)
        axc.plot([], [], marker, ls='--', ms=ms, color=color, mec='black', mew=0.8, label=s)  # legend proxy (no error bar)

    for ax, ylabel, title in [
        (axc, 'Cost relative to baseline', 'Closed-loop cost'),
        # (axv, 'Convergence time (s)', 'Convergence time: CNP vs baseline across systems'),
    ]:
        ax.set_xlabel(r'Context size $|C|$', labelpad=0)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_title(title)
        ax.grid(axis='y', lw=0.3)
        ax.set_axisbelow(True)
    x_pad = 0.1  # padding as a fraction of the x-range on each end
    axc.set_xticks(x)  # ticks at the true context sizes
    axc.set_xlim(x[0] - x_pad * (x[-1] - x[0]), x[-1] + x_pad * (x[-1] - x[0]))
    axc.set_ylim(bottom=0.6)
    axc.set_ylim(top=3.5)
    axc.yaxis.set_major_locator(MultipleLocator(1))
    axc.yaxis.set_major_formatter(lambda v, _: '0' if v == 0 else rf'${v:g}\mathsf{{x}}$')
    axc.legend(ncol=2, loc='upper right', bbox_to_anchor=(0.98, 0.76), handletextpad=0.4)

    return fig

def main():
    args = CONFIG
    root = get_project_root()
    df = pd.read_csv(os.path.join(root, args.metrics))

    matplotlib_config.matplotlib_set_publication()

    fig = plot_combined(df, args.dt)

    plt.show()

    if True:
        for f, name in [(fig, 'np_adaptation_closedloop.pdf'),
                        # (fig_box, 'np_adaptation_closedloop_box.pdf')
                        ]:
            save_path = os.path.join(root, 'plots', name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            f.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
            print(f"Saved figure to {save_path}")
    else:
        print(f"Figure not saved.")


if __name__ == '__main__':
    main()
