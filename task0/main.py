#!/usr/bin/env python3

import abc

import numpy as np


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


class AbstractLearner(object):

    __metaclass__ = abc.ABCMeta

    _model = NotImplemented

    def __init__(self):
        self._train_set = None
        self._test_set = None

    def learn_from(self, train_set):
        self._train_set = train_set
        self._train()

    def predict_from(self, test_set):
        self._test_set = test_set
        self._test_set.outputs = self._predict(self._test_set.features)
        return self._test_set

    @property
    def train_error(self):
        predictions = self._predict(self._train_set.features)
        return self.rms_error(predictions, self._train_set.outputs)

    @staticmethod
    def rms_error(predictions, true_values):
        mse = ((true_values - predictions) ** 2).mean()
        return mse ** 0.5

    @abc.abstractmethod
    def _train(self):
        raise NotImplementedError

    def _predict(self, features):
        return np.apply_along_axis(self._model, 1, features)


class MeanLearner(AbstractLearner):

    def _train(self):
        self._model = np.mean


class AdvMeanLearner(AbstractLearner):

    def _train(self):
        self._model = self._mymean

    @staticmethod
    def _mymean(v):
        v.sort()
        return np.mean(v)


class AutoLeastSquaresLearner(AbstractLearner):

    def _train(self):
        A = self._train_set.features
        b = self._train_set.outputs
        x, _residuals, _rank, _s = np.linalg.lstsq(A, b)
        self._model = lambda v: v.dot(x)


def main():
    train_set = CSVDataSet.from_train_data('data/train.csv', dtype=np.double)
    test_set = CSVDataSet.from_test_data('data/test.csv', dtype=np.double)

    learner = AutoLeastSquaresLearner()
    learner.learn_from(train_set)
    out_set = learner.predict_from(test_set)
    out_set.write_labelled_output('test_result.csv')

    #print(calc_error(predict(model, features), y))


if __name__ == '__main__':
    main()
