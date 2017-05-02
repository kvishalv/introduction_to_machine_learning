#!/usr/bin/env python3
import abc
from sklearn import metrics
import numpy as np

class AbstractNN(object):

    __metaclass__ = abc.ABCMeta
    _model = NotImplemented

    # _train_features and _train_outputs are NumPy arrays
    def __init__(self):
        self._train_features = None
        self._train_outputs = None

    # Calls self._train
    def learn_from(self, features, outputs):
        self._train_features = features
        self._train_outputs = outputs
        self._train()

    def predict_from(self, features):
        return self._model(features)


    # Calls self.predict_from
    def validate_against(self, features, outputs):
        predictions = self.predict_from(features)
        y_est  = [np.argmax(x) for x in predictions]
        y_true = [np.argmax(x) for x in outputs]
        return self.accuracy(y_est, y_true)

    @property
    # Calls self.predict_from
    def train_error(self):
        predictions = self.predict_from(self._train_features)
        y_est = [np.argmax(x) for x in predictions]           # To revert the categorial labels to multiclass labels
        y_true = [np.argmax(x) for x in self._train_outputs]

        return self.accuracy(y_est, y_true)

    @staticmethod
    def accuracy(predictions, true_values):
        return metrics.accuracy_score(predictions, true_values)

    @abc.abstractmethod
    def _train(self):
        raise NotImplementedError
