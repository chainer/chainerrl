import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--out', type=str, default='plot.png')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('score_files', type=str, nargs='*')
    args = parser.parse_args()

    assert len(args.score_files) > 0

    for score_file_path in args.score_files:
        scores = pd.read_csv(score_file_path, delimiter='\t')
        for col in ['mean', 'median']:
            plt.plot(scores['steps'], scores[col],
                     label=score_file_path + ' ' + col)

    if args.title is not None:
        plt.title(args.title)
    if args.steps is not None:
        plt.xlim([0, args.steps])
    plt.xlabel('steps')
    plt.ylabel('score')
    plt.legend(loc='best')
    plt.savefig(args.out)
    print('Saved a figure as {}'.format(args.out))

if __name__ == '__main__':
    main()
