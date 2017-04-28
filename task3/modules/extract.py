import random

import h5py
import numpy as np
import pandas as pd

SAMPLES = 2000


def main():
    random.seed(1742)

    data_file = h5py.File('data/train.h5', 'r')
    columns = list(data_file['train/axis0'])
    a0 = data_file['train/axis1']

    pop = len(data_file['train/block0_values'])
    indexes = random.sample(range(pop), SAMPLES))
    indexes.sort()

    datax = data_file['train/block0_values'][indexes]
    datay = data_file['train/block1_values'][indexes].astype(int)

    data = pd.DataFrame(
        np.column_stack((datay, datax)),
        columns=columns)
    data.to_csv('data/trainsample.csv', index_label='Id')


if __name__ == '__main__':
    main()
