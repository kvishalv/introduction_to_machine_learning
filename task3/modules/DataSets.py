import numpy as np
from keras import utils as ku

class DataSets(object):

    def __init__(self, ids, features, outputs):
        self.ids = ids
        self.features = features
        self.outputs = outputs

    @classmethod
    def from_train_data(cls, filename, shuffle=True):
        train_i, train_x, train_y = cls._read_hdf5(filename, 'train')
        di = np.array(train_i.data)
        dx = np.array(train_x.data)
        dy = np.array(train_y.data)
        if shuffle:
            np.random.seed(1742)
            data = np.column_stack((di, dy, dx))
            np.random.shuffle(data)
            di = data[:, 0]
            dy = data[:, 1:6]
            dx = data[:, 6:]
        return cls(di, dx, dy)

    @classmethod
    def from_test_data(cls, filename):
        test_i, test_x, _ = cls._read_hdf5(filename, 'test')
        di = np.array(test_i.data)
        dx = np.array(test_x.data)
        return cls(di, dx, None)

    @staticmethod
    def _read_hdf5(filename, dataname):
        train_i = ku.HDF5Matrix(filename, '%s/axis1' % dataname)
        train_x = ku.HDF5Matrix(filename, '%s/block0_values' % dataname)
        try:
            train_y = ku.HDF5Matrix(filename, '%s/block1_values' % dataname)
        except KeyError as e:
            train_y = None
        else:
            train_y.data = ku.to_categorical(train_y.data, num_classes=5)
        return train_i, train_x, train_y

    def write_labelled_output(self, filename):
        np.savetxt(
            filename, np.column_stack((self.ids, self.outputs)),
            header="Id,y", comments="",
            delimiter=",", fmt="%i,%r"
        )
