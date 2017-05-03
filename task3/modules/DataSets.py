import numpy as np
import pandas as pd
from keras.utils import to_categorical

SHUFFLE = True

class DataSets(object):

    def __init__(self, ids, features, outputs):
        self.ids = ids
        self.features = features
        self.outputs = outputs

    @classmethod
    def from_train_data(cls, filename, shuffle=SHUFFLE):
        data = cls._hdf_to_array(filename, "train")
        train_id = data.index.values
        train_y = data['y'].as_matrix()
        train_y_cat = to_categorical(train_y, num_classes=5)        # Converts a class vector to binary class matrix
        train_features = data.ix[:, 1:].astype(float).as_matrix()
        return cls(train_id, train_features, train_y_cat)

    @classmethod
    def from_test_data(cls, filename):
        data = cls._hdf_to_array(filename, "test")
        test_id = data.index.values
        test_features = data.astype(float).as_matrix()
        return cls(test_id, test_features, None)

    @staticmethod
    def _hdf_to_array(filename, key):
        return pd.read_hdf(filename, key)

    def write_labelled_output(self, filename):
        y_est  = [np.argmax(x) for x in self.outputs]

        np.savetxt(
            filename, np.column_stack((self.ids, y_est)),
            header="Id,y", comments="",
            delimiter=",", fmt="%i,%r"
        )


