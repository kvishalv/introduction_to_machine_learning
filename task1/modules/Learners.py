import numpy as np
from scipy import linalg as splin
from sklearn import linear_model

from modules.AbstractLearner import (
    NumPyLearner,
    SciKitLearner
)


class MoorePenroseLearner(NumPyLearner):

    def _train(self):
        a  = self._train_set.features
        b  = self._train_set.outputs
        at = np.linalg.pinv(a)
        x  = at.dot(b)
        self._model = lambda v: v.dot(x)


class QRFactorizationLearner(NumPyLearner):

    def _train(self):
        a    = self._train_set.features
        b    = self._train_set.outputs
        q, r = np.linalg.qr(a)
        # QRx = b; where Q is orthogonal and R is upper triangular
        # Rx = Q^T b = (b^T Q)^T = b_proj
        b_proj = b.transpose().dot(q).transpose()
        x    = splin.solve_triangular(r, b_proj, overwrite_b=True)
        self._model = lambda v: v.dot(x)


class LinearRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs
        clf = linear_model.LinearRegression(fit_intercept=True)
        clf.fit(x, y)
        self._model = clf.predict


class RidgeRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs
        clf = linear_model.Ridge(
            alpha=0.1,
            fit_intercept=True,
            random_state=42
        )
        clf.fit(x, y)
        self._model = clf.predict
