import numpy as np
from sklearn import *
from modules.AbstractLearner import (
    SciKitLearner,
)





class GridLearner(SciKitLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('scale', preprocessing.StandardScaler()),
            ('expand', preprocessing.PolynomialFeatures()),
            ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis())
        ])

        param_grid = [{
            'scale__with_mean': [True, False],
            'scale__with_std': [True, False],

            'expand__include_bias': [False, True],
            'expand__interaction_only': [False, True],
            'expand__degree': [1, 2]

            #'estim__reg_param': [0.5]
            #'estim__alpha': list(0.001 + 1 * i for i in range(0, 5))
        }]

        grid = model_selection.GridSearchCV(
            pipe, cv=10, n_jobs=1, param_grid=param_grid, verbose=1,
            scoring = metrics.make_scorer(metrics.accuracy_score),
        )
        grid.fit(x, y)

        print('Optimal Hyperparametres:')
        print('=======================')
        for step in grid.best_estimator_.steps:
            print(step)
        print("CV Score:", grid.best_score_)


        self._model = grid.predict



class NaiveBayesLearner(SciKitLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        self._transform = preprocessing.PolynomialFeatures(2)

        clf = naive_bayes.GaussianNB()
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict



class LinearDiscriminantAnalysis(SciKitLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        self._transform = preprocessing.PolynomialFeatures(2)

        clf = discriminant_analysis.LinearDiscriminantAnalysis()
        clf.fit(self._transform.fit_transform(x, y), y)

        self._model = clf.predict


class QuadraticDiscriminantLearner(SciKitLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        """
        StandardScaler(copy=True, with_mean=True, with_std=True)
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariances=False, tol=0.0001)
        """

        self._scale = preprocessing.StandardScaler(with_mean=True, with_std=True)
        self._transform = preprocessing.PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

        clf = discriminant_analysis.QuadraticDiscriminantAnalysis()
        clf.fit(self._transform.fit_transform(self._scale.fit_transform(x), y), y)

        self._model = clf.predict

#class SelectKBest(SciKitLearner):
#    sklearn.feature_selection.SelectKBest(, k = 14)
