"""Plot saved evaluations in the compact style of the supplied reference figure."""
import argparse
from pathlib import Path

from compare_runs import get_all_runs_data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--num_runs', type=int, default=20)
    parser.add_argument('--output_prefix', default='comparison_reference_style')
    args = parser.parse_args()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    bounds = []
    reference_steps = None
    for algo, color, zorder in [('ddpg', '#009E73', 3), ('ppo', '#FF4B2B', 2)]:
        steps, mean, ci, _ = get_all_runs_data(algo, args.num_runs, args.root)
        if reference_steps is not None and not np.array_equal(steps, reference_steps):
            raise ValueError('Algorithms have different evaluation checkpoints')
        reference_steps = steps
        x = steps / 1e6
        ax.plot(x, mean, label=algo.upper(), color=color, linestyle='-',
                linewidth=1.0, zorder=zorder)
        # Keep the experiment's 95% t intervals, not the reference's min/max bands.
        if args.num_runs > 1:
            ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=0.12,
                            zorder=zorder-1)
            bounds.extend([float(np.min(mean-ci)), float(np.max(mean+ci))])
        else:
            bounds.extend([float(np.min(mean)), float(np.max(mean))])
    ax.set_xlabel('Training steps (millions)')
    ax.set_ylabel('Evaluation return')
    ax.set_xlim(x[0], x[-1])
    margin = max(1.0, (max(bounds)-min(bounds))*0.04)
    ax.set_ylim(min(bounds)-margin, max(bounds)+margin)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.grid(True)
    ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False,
              ncol=1, framealpha=0.8, handlelength=1.2)
    plt.tight_layout(pad=0.2)
    for extension in ('png', 'pdf', 'svg'):
        path = Path(args.root) / f'{args.output_prefix}.{extension}'
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
