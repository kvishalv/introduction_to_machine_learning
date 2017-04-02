import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as splin
from sklearn import (
    feature_selection,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    pipeline,
    preprocessing,
)

from modules.AbstractLearner import (
    SciKitLearner,
)


NAMES = [
'y',
'x1', 'x2', 'x3',
'x4', 'x5', 'x6',
'x7', 'x8', 'x9',
'x10', 'x11', 'x12',
'x13', 'x14', 'x15'
]


class LinearRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs
        clf = linear_model.LinearRegression(fit_intercept=True)
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


class PolyLassoRegressionLearner(SciKitLearner):
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


class PolyTheilSenRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(2)

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


class BayesianRidgeRegression(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        #alpha_1 alpha_2 lambda_1 lambda_2

        clf = linear_model.BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, compute_score=True, fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class LarsLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(1)

        clf = linear_model.Lars(n_nonzero_coefs=400 ,fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class LassoLarsLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        clf = linear_model.LassoLars(alpha=1e-3,fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict



class LassoLarsCVLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(1)

        clf = linear_model.LassoLarsCV(fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class LassoLarsICLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(1)

        clf = linear_model.LassoLarsCV(fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class OrthogonalMatchingPursuit(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        clf = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=235, fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


def filter_outliers(x, y, **kwargs):
    xy = np.column_stack((x,y))
    filter_estimator = ensemble.IsolationForest(random_state=42, **kwargs)
    filter_estimator.fit(xy)
    is_inlier = filter_estimator.predict(xy)
    return x[is_inlier == 1], y[is_inlier == 1]


class GridLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        x, y = filter_outliers(x, y, n_estimators=200, contamination=0.003)

        pipe = pipeline.Pipeline([
            #('kselect', feature_selection.SelectKBest(feature_selection.f_regression, k=15)),
            ('expand', preprocessing.PolynomialFeatures()),
            ('estim', linear_model.LassoLars())
        ])

        param_grid = [{
            'expand__include_bias': [False],
            'expand__degree': [3],

            'estim__normalize': [False],
            'estim__fit_intercept': [True],
            'estim__alpha': [0.313]
            #'estim__alpha': list(0.01 + 1 * i for i in range(0, 9))
            #'estim__l1_ratio': list(0.80 + 0.01 * i for i in range(0, 6))
        }]

        grid = model_selection.GridSearchCV(
            pipe, cv=3, n_jobs=1, param_grid=param_grid, verbose=1,
            scoring = metrics.make_scorer(
                metrics.mean_squared_error,
                greater_is_better=False
            )
        )
        grid.fit(x, y)

        print(grid.best_estimator_)
        print(grid.cv_results_)

        estim = grid.best_estimator_.named_steps['estim']
        coeffs = estim.coef_

        polyexp = grid.best_estimator_.named_steps['expand']
        f_names = polyexp.get_feature_names(NAMES[1:])

        """
        for elem in sorted(zip(coeffs, f_names), reverse=True):
            print(elem)
        """

        self._model = grid.predict


class ARDRegressionLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        self._transform = preprocessing.PolynomialFeatures(3)

        #alpha_1 alpha_2 lambda_1 lambda_2, threshold_lambda
        clf = linear_model.ARDRegression(n_iter=3, fit_intercept=True)
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict
