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

        train_i = np.array(train_i.data).astype('int')
        train_x = np.array(train_x.data)
        if train_y is not None:
            train_y = np.array(train_y.data).astype('int')

        return train_i, train_x, train_y

    @staticmethod
    def _shuffle(mi, my, mx):
        np.random.seed(1742)
        data = np.column_stack((mi, my, mx))
        np.random.shuffle(data)
        mi = data[:, 0].astype('int')
        my = data[:, 1].astype('int')
        mx = data[:, 2:]
        return mi, my, mx

    def write_labelled_output(self, filename):
        np.savetxt(
            filename, np.column_stack((self.ids, self.outputs)),
            header="Id,y", comments="",
            delimiter=",", fmt="%i,%r"
        )


class CSVDataSet(object):

    def __init__(self, ids, features, outputs):
        self.ids = ids
        self.features = features
        self.outputs = outputs

    @classmethod
    def from_labeled_data(cls, filename, shuffle=True):
        data = cls._csv_to_array(filename)
        if shuffle:
            np.random.seed(1742)
            np.random.shuffle(data)
        mi = data[:, 0].astype('int')
        my = data[:, 1].astype('int')
        mx = data[:, 2:]
        return cls(mi, mx, my)

    @classmethod
    def from_unlabeled_data(cls, filename):
        data = cls._csv_to_array(filename)
        mi = data[:, 0].astype('int')
        mx = data[:, 1:]
        return cls(mi, mx, None)

    @classmethod
    def from_test_data(cls, filename):
        return cls.from_unlabeled_data(filename)

    @staticmethod
    def _csv_to_array(filename):
        return np.genfromtxt(filename, delimiter=',', skip_header=True)

    def write_labelled_output(self, filename):
        np.savetxt(
            filename, np.column_stack((self.ids, self.outputs)),
            header="Id,y", comments="",
            delimiter=",", fmt="%i,%r"
        )
