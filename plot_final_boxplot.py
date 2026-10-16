"""Final evaluation returns across seeds, in the supplied compact figure style."""
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
    parser.add_argument('--output_prefix', default='final_performance_boxplot')
    args = parser.parse_args()
    results = [get_all_runs_data(algo, args.num_runs, args.root) for algo in ('ddpg', 'ppo')]
    if not np.array_equal(results[0][0], results[1][0]):
        raise ValueError('Algorithms have different evaluation checkpoints')
    scores = [result[3][:, -1] for result in results]
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['cmr10'],
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'axes.unicode_minus': False,
        'ps.fonttype': 42,
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
        'axes.linewidth': 0.8, 'grid.alpha': 0.2, 'grid.linestyle': '--',
        'figure.dpi': 300, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.set_axisbelow(True)
    ax.grid(True)
    colors = ['#0000FF', '#FF0000']
    for position, (values, color) in enumerate(zip(scores, colors), start=1):
        ax.boxplot(
            values, positions=[position], widths=0.48, patch_artist=True,
            whis=1.5, showfliers=True, manage_ticks=False,
            boxprops={'facecolor': '#CCCCFF' if position == 1 else '#FFCCCC',
                      'edgecolor': color, 'linewidth': 1.0},
            whiskerprops={'color': color, 'linewidth': 1.0},
            capprops={'color': color, 'linewidth': 1.0},
            medianprops={'color': color, 'linewidth': 1.3},
            flierprops={'marker': 'o', 'markersize': 3.5,
                        'markerfacecolor': 'none', 'markeredgecolor': color,
                        'markeredgewidth': 0.8},
            zorder=3,
        )
    ax.set_xticks([1, 2], ['DDPG', 'PPO'])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0, max(map(np.max, scores)) * 1.07)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Final evaluation return')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    plt.tight_layout(pad=0.2)
    for extension in ('png', 'pdf', 'svg'):
        path = Path(args.root) / f'{args.output_prefix}.{extension}'
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
