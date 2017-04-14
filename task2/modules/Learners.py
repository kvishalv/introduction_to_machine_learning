import itertools

import numpy as np
from sklearn import (
# Please don't use import *, it causes a wall of Deprecation Warnings. Thanks!
    decomposition,
    discriminant_analysis,
    ensemble,
    feature_selection,
    metrics,
    model_selection,
    naive_bayes,
    neighbors,
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


class VotingLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(columns=(7, 8, 13))),
            ('estim', ensemble.VotingClassifier(
                estimators=[
                    ('knn', pipeline.Pipeline([
                        ('scale', preprocessing.StandardScaler(with_mean=True, with_std=True)),
                        ('expand', preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
                        ('select', feature_selection.SelectPercentile(score_func=feature_selection.f_classif)),
                        ('estim', neighbors.KNeighborsClassifier())
                    ])),
                    ('qda', pipeline.Pipeline([
                        ('scale', preprocessing.StandardScaler(with_mean=True, with_std=False)),
                        ('expand', preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
                        ('select', feature_selection.SelectPercentile(score_func=feature_selection.f_classif)),
                        ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis())
                    ]))
            ]))
        ])

        param_grid = [{
            'estim__knn__select__percentile': [i for i in range(5, 8)],
            'estim__knn__estim__n_neighbors': [i for i in range(2, 7)],
            #'estim__knn__estim__weights': ['distance', 'uniform'],
            'estim__knn__estim__weights': ['distance'],
            'estim__knn__estim__metric': ['euclidean', 'manhattan', 'chebyshev'],
            #'estim__knn__estim__metric': ['chebyshev'],

            'estim__qda__select__percentile': [i for i in range(41, 51)],
            'estim__qda__estim__reg_param': [0.02 + 0.001 for i in range(-5, 6)],

            'estim__voting': ['soft'],
            'estim__weights': list(itertools.product(range(1, 2), range(1, 3)))
        }]

        grid = model_selection.GridSearchCV(
            pipe, cv=10, n_jobs=4, param_grid=param_grid, verbose=1,
            scoring = metrics.make_scorer(metrics.accuracy_score),
        )
        grid.fit(x, y)

        print('Optimal Hyperparametres:')
        print('=======================')
        for name, step in grid.best_estimator_.steps:
            if name == 'estim':
                for (name2, _), estim2 in zip(step.estimators, step.estimators_):
                    print('  ', name2)
                    for name3, step3 in estim2.steps:
                        print('    ', step3)
                print('Weights:', step.voting, step.weights)
            else:
                print(step)
        print("CV Score:", grid.best_score_)

        self._model = grid.predict


class NaiveBayesLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(7, 8, 11, 12, 13, 14)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=1,
                interaction_only=False,
                include_bias=False
            )),
            ('reduce', decomposition.FastICA(
                fun='cube',
                random_state=1742,
            )),
            ('select', feature_selection.SelectKBest(
                k=7,
                score_func=feature_selection.mutual_info_classif,
            )),
            ('estim', naive_bayes.GaussianNB()),
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class KNNLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(7, 8, 11, 12, 13, 14)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=1,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                k=8,
                score_func=feature_selection.f_classif
            )),
            ('estim', neighbors.KNeighborsClassifier(
                n_neighbors=16,
                weights='distance',
                metric='chebyshev'
            ))
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class NearestCentroidLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            # x14 == x10
            # x8 == x3
            # x9 == x6^2 - C
            ('drop', transformers.ColumnDropper(
                columns=(7, 8, 13)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                k=26,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', neighbors.NearestCentroid(
                metric='euclidean',
                shrink_threshold=None
            )),
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
            ('drop', transformers.ColumnDropper(columns=(7, 8, 13))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=False # this is not a typo!
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=50,
                score_func=feature_selection.f_classif
            )),
            ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis(
                reg_param=0.022
            ))
        ])

        pipe.fit(x, y)
        self._model = pipe.predict
