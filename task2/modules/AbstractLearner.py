import abc

from sklearn import metrics


class _AbstractLearner(object):

    __metaclass__ = abc.ABCMeta

    _model = NotImplemented

    # _train_set and _test_set are CSVDataSet instances
    def __init__(self):
        self._train_set = None

    # Calls self._train
    def learn_from(self, train_set):
        self._train_set = train_set
        self._train()

    # Calls self._predict
    def predict_from(self, test_set):
        return self._predict(test_set.features)

    # Calls self._predict
    def validate_against(self, validation_set):
        predictions = self._predict(validation_set.features)
        return self.accuracy(predictions, validation_set.outputs)

    @property
    # Calls self._predict
    def train_error(self):
        predictions = self._predict(self._train_set.features)
        return self.accuracy(predictions, self._train_set.outputs)

    @staticmethod
    def accuracy(predictions, true_values):
        return metrics.accuracy_score(predictions, true_values)

    @staticmethod
    def rms_error(predictions, true_values):
        return metrics.mean_squared_error(predictions, true_values) ** 0.5

    @abc.abstractmethod
    def _train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, features):
        raise NotImplementedError


class SciKitLearner(_AbstractLearner):

    _transform = None

    def _predict(self, features):
        if hasattr(self._transform, 'transform'):
            features = self._transform.transform(features)
        return self._model(features)
