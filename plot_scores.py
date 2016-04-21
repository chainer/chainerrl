import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    scores_path = os.path.join(args.dir, 'scores.txt')
    scores = pd.read_csv(scores_path, names=('t', 'score'), delimiter=' ')
    scores.plot(x='t', y='score')
    plt.show()

if __name__ == '__main__':

    main()
