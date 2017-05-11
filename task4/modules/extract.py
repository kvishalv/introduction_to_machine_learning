import numpy as np
import pandas as pd

SAMPLES = 2000


def main():
    train_labeled = pd.read_hdf("data/train_labeled.h5", "train")
    train_unlabeled = pd.read_hdf("data/train_unlabeled.h5", "train")
    test = pd.read_hdf("data/test.h5", "test")

    test.to_csv('data/test.csv')
    train_labeled.to_csv('data/train_labeled.csv')
    train_unlabeled.to_csv('data/train_unlabeled.csv')


if __name__ == '__main__':
    main()
