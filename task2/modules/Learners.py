import itertools

import numpy as np

from sklearn import (
    decomposition,
    discriminant_analysis,
    ensemble,
    feature_selection,
    metrics,
    model_selection,
    naive_bayes,
    neighbors,
    pipeline,
    preprocessing,
    linear_model,
    svm,
    ensemble,
    tree,
    svm
)


from modules.AbstractLearner import AbstractLearner
from modules import transformers


class GridLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(columns=(6, 7, 8, 11, 12, 13, 14))),
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


class NuSVCLearner(AbstractLearner):

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
            ('estim', svm.NuSVC(
                nu=0.19,
                kernel='rbf',
                gamma='auto',
                shrinking=True,
                class_weight='balanced',
                random_state=1742
            )),
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


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
                columns=(7, 8, 11, 12, 13, 14)
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
            ('drop', transformers.ColumnDropper(
                columns=(6, 7, 8, 11, 12, 13, 14))
            ),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=False # this is not a typo!
            )),
            #('scale', preprocessing.RobustScaler(
            #    with_centering=True, with_scaling=False, quantile_range=(1.0, 99.0)
            #)),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=98,
                score_func=feature_selection.f_classif
            )),
            ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis(
                reg_param=0.0043
            ))
        ])

        pipe.fit(x, y)
        self._model = pipe.predict



class StochasticGradientLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(columns=(6, 7, 8, 11, 12, 13, 14))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True # this is not a typo!
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                k=25,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', linear_model.SGDClassifier(alpha=0.001
            ))
        ])

        pipe.fit(x, y)
        self._model = pipe.predict



class SVMLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(columns=(6, 7, 8, 11, 12, 13, 14))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True # this is not a typo!
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                k=25,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', svm.LinearSVC(C=10
            ))
            #svm.SVC, svm.NuSVC, svm.LinearSVC
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class GradientBoostingLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(columns=(6, 7, 8, 11, 12, 13, 14))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True # this is not a typo!
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                k=25,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', ensemble.GradientBoostingClassifier(learning_rate=0.1, max_depth=2
            ))
            #svm.SVC, svm.NuSVC, svm.LinearSVC
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class DecisionTreeLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(columns=(6, 7, 8, 11, 12, 13, 14))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True # this is not a typo!
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                k=25,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', tree.DecisionTreeClassifier(
            ))
            #svm.SVC, svm.NuSVC, svm.LinearSVC
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class HierarchicalLearner(AbstractLearner):

    def _train(self):

        #Train to distinguish 2 first!
        x = self._train_features
        y = self._train_outputs

        print(x.shape)
        print(y.T.shape)

        #xy = np.concatenate((x, y.T), axis=0)
        #print(y[0:10,])


        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(columns=(6, 7, 8, 11, 12, 13, 14))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True # this is not a typo!
            )),
            ('expand', preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('select', feature_selection.SelectKBest(
                k=25,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', tree.DecisionTreeClassifier(
            ))
            #svm.SVC, svm.NuSVC, svm.LinearSVC
        ])

        pipe.fit(x, y)
        self._model = pipe.predict
