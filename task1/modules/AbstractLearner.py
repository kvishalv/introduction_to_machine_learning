import abc

import numpy as np


class _AbstractLearner(object):

    __metaclass__ = abc.ABCMeta

    _model = NotImplemented

    # _train_set and _test_set are CSVDataSet instances
    def __init__(self):
        self._train_set = None
        self._test_set  = None

    # Calls self._train
    def learn_from(self, train_set):
        self._train_set = train_set
        self._train()

    # Calls self._predict
    def predict_from(self, test_set):
        self._test_set = test_set
        self._test_set.outputs = self._predict(self._test_set.features)
        return self._test_set

    @property
    # Calls self._predict
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

    @abc.abstractmethod
    def _predict(self, features):
        raise NotImplementedError


class AbstractLearner(_AbstractLearner):

    def _predict(self, features):
        return np.apply_along_axis(self._model, 1, features)


class SciKitLearner(_AbstractLearner):

    def _predict(self, features):
        return self._model(features)
