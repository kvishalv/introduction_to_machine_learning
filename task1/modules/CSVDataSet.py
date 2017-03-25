import numpy as np
from sklearn import cross_validation


class CSVDataSet(object):

    def __init__(self, ids, features, outputs):
        self.ids = ids
        self.features = features
        self.outputs = outputs

    @classmethod
    def from_train_data(cls, filename, dtype=np.longdouble):
        data = cls._csv_to_array(filename, dtype)
        train_id = data[:, 0].astype('int')
        train_y = data[:, 1]
        train_features = data[:, 2:]
        return cls(train_id, train_features, train_y)

    @classmethod
    def from_test_data(cls, filename, dtype=np.longdouble):
        data = cls._csv_to_array(filename, dtype)
        test_id = data[:, 0].astype('int')
        test_features = data[:, 1:]
        return cls(test_id, test_features, None)

    @staticmethod
    def _csv_to_array(filename, dtype):
        """
        Returns contents of `filename` CSV file as a numpy array.

        dtype: NumPy type

        Note: Assumes and ignores exactly one header line.
        """
        return np.genfromtxt(
            filename, delimiter=',', dtype=dtype, skip_header=True
        )

    def write_labelled_output(self, filename):
        np.savetxt(
            filename, np.column_stack((self.ids, self.outputs)),
            header="Id,y", comments="",
            delimiter=",", fmt="%i,%r"
        )

    def split(self, train_size=0.90):
        if self.outputs is None:
            x1, x2, id1, id2 = cross_validation.train_test_split(
                self.features,
                self.ids,
                train_size=train_size,
                random_state=42
            )
            y1 = y2 = None
        else:
            id1, id2, x1, x2, y1, y2 = cross_validation.train_test_split(
                self.ids,
                self.features,
                self.outputs,
                train_size=train_size,
                random_state=42
            )
        return CSVDataSet(id1, x1, y1), CSVDataSet(id2, x2, y2)
