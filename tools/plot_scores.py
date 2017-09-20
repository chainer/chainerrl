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
    args = parser.parse_args()

    if args.savefile:
        matplotlib.use('Agg')  # Needed to run without X-server
    import matplotlib.pyplot as plt_
    global plt
    plt = plt_

    assert len(args.files) > 0
    assert len(args.labels) == len(args.files)

    for fpath, label in zip(args.files, args.labels):
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, 'scores.txt')
        assert os.path.exists(fpath)
        scores = pd.read_csv(fpath, delimiter='\t')
        plt.plot(scores['steps'], scores['mean'], label=label)

    plt.xlabel('steps')
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
