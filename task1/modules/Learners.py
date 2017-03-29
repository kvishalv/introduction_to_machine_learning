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
    TransformingSciKitLearner
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
            alpha=10,
            fit_intercept=True,
            random_state=42
        )
        clf.fit(x, y)
        self._model = clf.predict


class PolyRidgeRegressionLearner(TransformingSciKitLearner):

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

        clf = linear_model.Lasso(alpha=0.4, max_iter=1e8, tol=1e-10)
        clf.fit(self._transform.fit_transform(x), y)
        self._model = clf.predict


class ElasticNetLearner(SciKitLearner):
    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        clf = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.1)
        clf.fit(x, y)
        self._model = clf.predict


class PolyTheilSenRegressionLearner(TransformingSciKitLearner):

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


class Model0(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

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


class SelectKBestLearner(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=10)

        clf = linear_model.Ridge(alpha=500, fit_intercept=True,)
        clf.fit(self._transform.fit_transform(x, y), y)
        self._model = clf.predict


class BayesianRidgeRegression(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        #alpha_1 alpha_2 lambda_1 lambda_2

        clf = linear_model.BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, compute_score=True, fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class LarsLearner(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(1)

        clf = linear_model.Lars(n_nonzero_coefs=400 ,fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class LassoLarsLearner(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        clf = linear_model.LassoLars(alpha=1e-3,fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class OrthogonalMatchingPursuit(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        clf = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=235, fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class OrthogonalMatchingPursuit(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        clf = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=235, fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class ARDRegressionLearner(TransformingSciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        #alpha_1 alpha_2 lambda_1 lambda_2, threshold_lambda
        clf = linear_model.ARDRegression(n_iter=70, fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict