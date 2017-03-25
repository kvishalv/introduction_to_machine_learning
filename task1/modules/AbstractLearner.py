import abc

import numpy as np
from sklearn import metrics


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

    # Calls self._predict
    def validate_against(self, validation_set):
        predictions = self._predict(validation_set.features)
        return self.rms_error(predictions, validation_set.outputs)

    @property
    # Calls self._predict
    def train_error(self):
        predictions = self._predict(self._train_set.features)
        return self.rms_error(predictions, self._train_set.outputs)

    @staticmethod
    def rms_error(predictions, true_values):
        return metrics.mean_squared_error(predictions, true_values) ** 0.5

    @abc.abstractmethod
    def _train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, features):
        raise NotImplementedError


class NumPyLearner(_AbstractLearner):

    def _predict(self, features):
        return np.apply_along_axis(self._model, 1, features)


class SciKitLearner(_AbstractLearner):

    def _predict(self, features):
        return self._model(features)


class TransformingSciKitLearner(SciKitLearner):

    _transform = None

    def _predict(self, features):
        return super()._predict(
            self._transform.fit_transform(features)
        )
