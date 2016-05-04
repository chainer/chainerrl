import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', type=str, help='specify path of scores.txt')
    args = parser.parse_args()
    scores = pd.read_csv(args.scores, names=('t', 'score'), delimiter=' ')
    scores.plot(x='t', y='score')
    plt.show()

if __name__ == '__main__':

    main()
