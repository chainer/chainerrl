import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', type=str, help='specify path of scores.txt')
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, delimiter='\t')
    for col in ['mean', 'median']:
        plt.plot(scores['steps'], scores[col], label=col)
    plt.title('A3C Breakout')
    plt.xlabel('steps')
    plt.ylabel('score')
    plt.legend(loc='best')
    # plt.show()
    fig_fname = args.scores + '.png'
    plt.savefig(fig_fname)
    print('Saved a figure as {}'.format(fig_fname))

if __name__ == '__main__':

    main()
