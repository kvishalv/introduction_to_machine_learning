import numpy as np
from scipy import linalg as splin
from sklearn import (
    feature_selection,
    ensemble,
    linear_model,
    pipeline,
    preprocessing,
)

from modules.AbstractLearner import SciKitLearner

import matplotlib.pyplot as plt


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
            alpha=10,
            fit_intercept=True,
            random_state=42
        )
        clf.fit(x, y)
        self._model = clf.predict


class PolyRidgeRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        clf = linear_model.Ridge(
            alpha=100.0,
            fit_intercept=True,
        )
        clf.fit(self._transform.fit_transform(x), y)
        self._model = clf.predict


class LassoRegressionLearner(SciKitLearner):
    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        clf = linear_model.Lasso(alpha = 100, max_iter=1e8)
        clf.fit(x, y)
        self._model = clf.predict


class PolyLassoRegressionLearner(TransformingSciKitLearner):
    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        clf = linear_model.Lasso(alpha=0.4, max_iter=1e8)
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


def filter_outliers(x, y, **kwargs):
    xy = np.column_stack((x,y))
    filter_estimator = ensemble.IsolationForest(random_state=42, **kwargs)
    filter_estimator.fit(xy)
    is_inlier = filter_estimator.predict(xy)
    return x[is_inlier == 1], y[is_inlier == 1]


class Model0(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        x, y = filter_outliers(x, y, n_estimators=200, contamination=0.005)

        self._transform = pipeline.Pipeline([
            #('scale', preprocessing.StandardScaler()),
            ('proliferate', preprocessing.PolynomialFeatures(3)),
            ('pselect', feature_selection.SelectPercentile(feature_selection.f_regression, percentile=98)),
            #('kselect', feature_selection.SelectKBest(feature_selection.f_regression, k=750)),
        ])

        clf = linear_model.Ridge(alpha=500, fit_intercept=True,)
        clf.fit(self._transform.fit_transform(x, y), y)
        self._model = clf.predict

    def findOptimalAlpha(self, validation_set):
        x    = self._train_set.features
        y    = self._train_set.outputs

        transformer = pipeline.Pipeline([
            ('proliferate', preprocessing.PolynomialFeatures(3)),
            ('pselect', feature_selection.SelectPercentile(feature_selection.f_regression, percentile=98)),
        ])

        nrIterations = 200
        start        = 0.1
        increment    = 10

        tr_err  = np.zeros(nrIterations)
        val_err = np.zeros(nrIterations)
        alp = np.zeros(nrIterations)
        for i in range(0, nrIterations):
            alp[i] = (i*increment+start)
            clf = linear_model.Ridge(alpha=alp[i], fit_intercept=True,)
            #clf = linear_model.Ridge(alpha=500, fit_intercept=True, )
            clf.fit(transformer.fit_transform(x, y), y)
            predictions = clf.predict(transformer.transform(validation_set.features))
            val_err[i] = self.rms_error(predictions, validation_set.outputs)
            predictions = clf.predict(transformer.transform(x))
            tr_err[i]  = self.rms_error(predictions, y)
        plt.plot(alp, val_err, 'r', alp, tr_err, 'b', alp, val_err-tr_err, 'g--')
        plt.title('alpha values vs errors')
        plt.show()
        print(val_err, "\n", tr_err, "\n", alp)


class SelectKBestLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=10)

        clf = linear_model.Ridge(alpha=500, fit_intercept=True,)
        clf.fit(self._transform.fit_transform(x, y), y)
        self._model = clf.predict
