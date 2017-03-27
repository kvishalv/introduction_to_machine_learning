import numpy as np
from scipy import linalg as splin
from sklearn import (
    feature_selection,
    linear_model,
    pipeline,
    preprocessing,
)

from modules.AbstractLearner import (
    NumPyLearner,
    SciKitLearner,
)

import matplotlib.pyplot as plt


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


class PolyRidgeRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(2)

        clf = linear_model.Ridge(
            alpha=1.0,
            fit_intercept=True,
        )
        clf.fit(self._transform.fit_transform(x), y)
        self._model = clf.predict


class LassoRegressionLearner(SciKitLearner):
    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        clf = linear_model.Lasso(1, max_iter=1e8)
        clf.fit(self._transform.fit_transform(x), y)
        self._model = clf.predict

class ElasticNetLearner(SciKitLearner):
    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        clf = linear_model.ElasticNet(1)
        clf.fit(x, y)
        self._model = clf.predict


class PolyTheilSenRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(
            2,
            interaction_only=False,
        )

        clf = linear_model.TheilSenRegressor(
            fit_intercept=True,
            verbose=True,
            n_jobs=4,
            max_subpopulation=10000
        )
        clf.fit(self._transform.fit_transform(x), y)
        self._model = clf.predict


class Model0(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = pipeline.Pipeline([
            #('scale', preprocessing.StandardScaler()),
            ('proliferate', preprocessing.PolynomialFeatures(3)),
            ('pselect', feature_selection.SelectPercentile(feature_selection.f_regression, percentile=98)),
            #('kselect', feature_selection.SelectKBest(feature_selection.f_regression, k=750)),
        ])

        clf = linear_model.Ridge(alpha=800, fit_intercept=True,)
        clf.fit(self._transform.fit_transform(x, y), y)
        self._model = clf.predict

    def findOptimalAlpha(self, validation_set):
        x    = self._train_set.features
        y    = self._train_set.outputs

        transformer = pipeline.Pipeline([
            ('proliferate', preprocessing.PolynomialFeatures(3)),
            ('pselect', feature_selection.SelectPercentile(feature_selection.f_regression, percentile=98)),
        ])

        nrIterations = 10
        start        = 1
        increment    = 100

        rms = np.zeros(nrIterations)
        alp = np.zeros(nrIterations)
        for i in range(0, nrIterations-1):
            alp[i] = (i*increment+start)
            clf = linear_model.Ridge(alpha=alp[i], fit_intercept=True,)
            clf.fit(transformer.fit_transform(x, y), y)
            predictions = clf.predict(transformer.transform(validation_set.features))
            rms[i] = self.rms_error(predictions, validation_set.outputs)
        plt.plot(alp, rms)
        plt.title('alpha values vs error')
        plt.show()
        print(rms, "\n", alp)


class SelectKBestLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=10)

        clf = linear_model.Ridge(alpha=1.0, fit_intercept=True,)
        clf.fit(self._transform.fit_transform(x, y), y)
        self._model = clf.predict
