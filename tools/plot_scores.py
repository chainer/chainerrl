"""Plot scores.txt

Examples:
    python plot_scores.py --label 1 --file 1/scores.txt --label 2 --file 2/scores.txt --savefile out.png
    python plot_scores.py --label 1 --file 1/scores.txt --label 2 --file 2/scores.txt --x episodes --y-range stdev
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import argparse
import os

import matplotlib
# Don't `import matplotlib.pyplot` here
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefile', type=str, default=None)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--file', action='append', dest='files',
                        default=[], type=str,
                        help='specify paths of scores.txt')
    parser.add_argument('--label', action='append', dest='labels',
                        default=[], type=str,
                        help='specify labels for scores.txt files')
    parser.add_argument('--x', type=str, default='steps')
    parser.add_argument('--y-range', type=str, default=None)
    parser.add_argument('--fill-alpha', type=float, default=0.3)
    args = parser.parse_args()

    if args.savefile:
        matplotlib.use('Agg')  # Needed to run without X-server
    import matplotlib.pyplot as plt_
    global plt
    plt = plt_

    assert len(args.files) > 0
    assert len(args.labels) == len(args.files)

    if args.y_range:
        yerr = args.y_range.split(',')
        assert 1 <= len(yerr) <= 2
    else:
        yerr = []

    for fpath, label in zip(args.files, args.labels):
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, 'scores.txt')
        assert os.path.exists(fpath)
        scores = pd.read_csv(fpath, delimiter='\t')
        xs = scores[args.x]
        ys = scores['mean']
        p, = plt.plot(xs, ys, label=label)
        if yerr:
            color = p.get_color()
            if len(yerr) == 1:
                es = scores[yerr[0]]
                ys_high = ys + es
                ys_low = ys - es
            else:
                ys_high = scores[yerr[0]]
                ys_low = scores[yerr[1]]
            plt.fill_between(xs, ys_high, ys_low,
                             label=None, color=color, linewidth=0.0, alpha=args.fill_alpha)

    plt.xlabel(args.x)
    plt.ylabel('score')
    plt.legend(loc='best')
    if args.title:
        plt.title(args.title)

    if args.savefile:
        fig_fname = args.savefile

        _, ext = os.path.splitext(fig_fname)
        if ext != '.png':
            fig_fname += '.png'

        plt.savefig(fig_fname)
        print('Saved a figure as {}'.format(fig_fname))
    else:
        plt.show()

if __name__ == '__main__':
    main()
