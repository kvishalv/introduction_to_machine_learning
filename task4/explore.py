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

    columnstats = []
    for f in range(features):
        cl = lx[:, f]
        cu = ux[:, f]
        ct = tx[:, f]

        sl = len(set(cl))
        su = len(set(cu))
        st = len(set(ct))

        zl = 100 * len(cl[cl != 0.0]) / len(cl)
        zu = 100 * len(cu[cu != 0.0]) / len(cu)
        zt = 100 * len(ct[ct != 0.0]) / len(ct)

        columnstats.append((sl, su, st, f, zl, zu, zt))

    print()
    print('\n\t\tUnique values\t    | % of elements non zero')
    print('\t\t=============================================')
    print('\t\tLabeled\tUnlab.\tTest| Labeled\tUnlab.\tTest')
    for sl, su, st, f, zl, zu, zt in sorted(columnstats):
        print('Feature: x%3d: \t%3d \t%3d \t%3d | \t%.3f \t%.3f \t%.3f' % (
            f, sl, su, st, zl, zu, zt,
        ))

   # 'Features [0 3 5 14 26 35 40 65 72 95 99 104 124] are constant.'


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
