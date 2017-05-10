import numpy as np
from keras import utils as ku


class H5DataSet(object):

    def __init__(self, ids, features, outputs):
        self.ids = ids
        self.features = features
        self.outputs = outputs

    @classmethod
    def from_labeled_data(cls, filename, dataname='train', shuffle=True):
        mi, mx, my = cls._read_hdf5(filename, dataname)
        if shuffle:
            mi, my, mx = cls._shuffle(mi, my, mx)
        return cls(mi, mx, my)

    @classmethod
    def from_unlabeled_data(cls, filename, dataname='train'):
        mi, mx, my = cls._read_hdf5(filename, dataname)
        return cls(mi, mx, my)

    @classmethod
    def from_test_data(cls, filename):
        return cls.from_unlabeled_data(filename, 'test')

    @staticmethod
    def _read_hdf5(filename, dataname):
        train_i = ku.HDF5Matrix(filename, '%s/axis1' % dataname)
        train_x = ku.HDF5Matrix(filename, '%s/block0_values' % dataname)
        try:
            train_y = ku.HDF5Matrix(filename, '%s/block1_values' % dataname)
        except KeyError:
            train_y = None

        train_i = np.array(train_i.data)
        train_x = np.array(train_x.data)
        if train_y is not None:
            train_y = np.array(train_y.data)

        return train_i, train_x, train_y

    @staticmethod
    def _shuffle(mi, my, mx):
        np.random.seed(1742)
        data = np.column_stack((mi, my, mx))
        np.random.shuffle(data)
        mi = data[:, 0]
        my = data[:, 1]
        mx = data[:, 2:]
        return mi, my, mx

    def write_labelled_output(self, filename):
        np.savetxt(
            filename, np.column_stack((self.ids, self.outputs)),
            header="Id,y", comments="",
            delimiter=",", fmt="%i,%r"
        )
