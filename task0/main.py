#!/usr/bin/env python3

import abc

import numpy as np


def write_result(filename, result):
    np.savetxt(
        filename, result,
        header="Id,y", comments="",
        delimiter=",", fmt="%i,%r"
    )


class AbstractLearner(object):

    __metaclass__ = abc.ABCMeta

    _model = NotImplemented

    def __init__(self):
        self._train_id = None
        self._train_features = None
        self._train_y = None

        self._test_id = None
        self._test_features = None
        self._test_y = None

    def learn_from(self, filename):
        self._get_train_data(filename)
        self._train()

    def predict_from(self, filename):
        self._get_test_data(filename)
        return self._predict()

    @staticmethod
    def rms_error(predictions, true_values):
        errors = true_values - predictions
        return (np.sum(errors ** 2) / errors.size) ** 0.5

    def _get_train_data(self, filename):
        data = self._csv_to_array(filename)
        self._train_id = data[:, 0].astype('int')
        self._train_y = data[:, 1]
        self._train_features = data[:, 2:]

    def _get_test_data(self, filename):
        data = self._csv_to_array(filename)
        self._test_id = data[:, 0].astype('int')
        self._test_features = data[:, 1:]

    @staticmethod
    def _csv_to_array(filename, dtype=np.longdouble):
        """
        Returns contents of `filename` CSV file as a numpy array.

        dtype: NumPy type

        Note: Assumes and ignores exactly one header line.
        """
        return np.genfromtxt(
            filename, delimiter=',', dtype=dtype, skip_header=True
        )

    @abc.abstractmethod
    def _train(self):
        raise NotImplementedError

    def _predict(self):
        self._test_y = np.apply_along_axis(self._model, 1, self._test_features)
        return np.column_stack((self._test_id, self._test_y))

class MeanLearner(AbstractLearner):

    def _train(self):
        self._model = np.mean


class AdvMeanLearner(AbstractLearner):

    def _train(self):
        self._model = _mymean

    @staticmethod
    def _mymean(v):
        v.sort()
        return np.mean(v)


def main():
    learner = MeanLearner()

    learner.learn_from('data/train.csv')
    result = learner.predict_from('data/test.csv')
    write_result('test_result.csv', result)

    #print(calc_error(predict(model, features), y))


if __name__ == '__main__':
    main()
