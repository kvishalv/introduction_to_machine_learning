import numpy as np
from sklearn import (
# Please don't use import *, it causes a wall of Deprecation Warnings. Thanks!
    discriminant_analysis,
    feature_selection,
    metrics,
    model_selection,
    naive_bayes,
    pipeline,
    preprocessing
)


from modules.AbstractLearner import AbstractLearner
from modules import transformers


class GridLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            #('drop', transformers.ColumnDropper(columns=(7, 13))),
            #('select', feature_selection.SelectKBest()),
            ('scale', preprocessing.StandardScaler()),
            ('expand', preprocessing.PolynomialFeatures()),
            ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis()),
        ])

        param_grid = [{
            #'select__k': [i for i in range(15, 21)],
            #'select__score_func': [feature_selection.f_classif],

            'scale__with_mean': [True, False],
            'scale__with_std': [True],

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



class NaiveBayesLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(7, 13)
            )),
            ('scale', preprocessing.StandardScaler()),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                score_func=feature_selection.f_classif,
                k=18
            )),
            ('estim', naive_bayes.GaussianNB()),
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class LinearDiscriminantLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
            )),
            ('estim', discriminant_analysis.LinearDiscriminantAnalysis())
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class QuadraticDiscriminantLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('scale', preprocessing.StandardScaler(with_mean=True, with_std=True)),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False
            )),
            #QuadraticDiscriminantAnalysis(
            #    priors=None, reg_param=0.0, store_covariances=False, tol=0.0001
            #)
            ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis())
        ])

        pipe.fit(x, y)
        self._model = pipe.predict

#class SelectKBest(AbstractLearner):
#    sklearn.feature_selection.SelectKBest(, k = 14)
