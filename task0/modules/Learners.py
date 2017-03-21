import numpy as np

from modules.AbstractLearner import AbstractLearner


class MeanLearner(AbstractLearner):

    def _train(self):
        self._model = np.mean


class AdvMeanLearner(AbstractLearner):
    """ Sorts the training data, then takes the mean """

    def _train(self):
        self._model = self._mymean

    @staticmethod
    def _mymean(v):
        v.sort()
        return np.mean(v)


class LeastSquaresLearner(AbstractLearner):

    def _train(self):
        a = self._train_set.features
        b = self._train_set.outputs
        x, _residuals, _rank, _s = np.linalg.lstsq(a, b)
        self._model = lambda v: v.dot(x)


class MoorePenroseLearner(AbstractLearner):

    def _train(self):
        a  = self._train_set.features
        b  = self._train_set.outputs
        at = np.linalg.pinv(a)
        x  = at.dot(b)
        self._model = lambda v: v.dot(x)


class QRFactorizationLearner(AbstractLearner):

    def _train(self):
        a    = self._train_set.features
        b    = self._train_set.outputs
        q, r = np.linalg.qr(a)
        # QRx = b; where Q is orthogonal and R is upper triangular
        # Rx = Q^T b = (b^T Q)^T = b_proj
        b_proj = b.transpose().dot(q).transpose()
        x    = np.linalg.solve(r, b_proj)
        self._model = lambda v: v.dot(x)
