#! /usr/bin/env python3

from modules import datasets


def main():
    lx, ly, ux, tx = read_data()

    print('Labeled samples:\t', len(lx))
    print('Unlabeled samples:\t', len(ux))
    print('Test samples:\t\t', len(tx))

    features = tx.shape[1]
    classes = sorted(set(ly))
    print('\nFeatures:\t\t', features)
    print('Classes:\t\t', classes)

    print()
    for c in classes:
        print('Labeled samples of class', c, ':', len(ly[ly == c]))


def read_data():
    ldata = datasets.H5DataSet.from_labeled_data('./data/train_labeled.h5')
    lx, ly = ldata.features, ldata.outputs

    udata = datasets.H5DataSet.from_unlabeled_data('./data/train_unlabeled.h5')
    ux = udata.features

    tdata = datasets.H5DataSet.from_test_data('./data/test.h5')
    tx = tdata.features

    return lx, ly, ux, tx


if __name__ == '__main__':
    main()
