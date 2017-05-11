import itertools

import numpy as np

from sklearn import (
    decomposition,
    discriminant_analysis,
    dummy,
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
            ('drop', transformers.ColumnDropper(
                columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124)
            )),
            ('scale', preprocessing.StandardScaler()),
            ('select', feature_selection.SelectPercentile()),
            ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis()),
        ])

        param_grid = [{
            'scale__with_mean': [True],
            'scale__with_std': [True],

            #'select__percentile': [i for i in range(40, 81, 3)],
            'select__percentile': [i for i in range(40, 51)],
            'select__score_func': [
                feature_selection.f_classif,
                feature_selection.mutual_info_classif
            ],

            'estim__reg_param': [0.1 + 0.025 * i for i in range(-1, 2)]
        }]

        grid = model_selection.GridSearchCV(
            pipe, cv=9, n_jobs=16, param_grid=param_grid, verbose=1,
            scoring = metrics.make_scorer(metrics.accuracy_score),
        )
        grid.fit(x, y)

        print('Optimal Hyperparametres:')
        print('=======================')
        for step in grid.best_estimator_.steps:
            print(step)
        print("CV Score:", grid.best_score_)

        self._model = grid.predict


class NuSVCLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=57,
                score_func=feature_selection.f_classif
            )),
            ('estim', svm.NuSVC(
                nu=0.05,
                kernel='rbf',
                gamma='auto',
                shrinking=True,
                class_weight='balanced',
                random_state=1742
            )),
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class QuadraticDiscriminantLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=46,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', discriminant_analysis.QuadraticDiscriminantAnalysis(
                reg_param=0.1
            ))
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class LinearDiscriminantLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=80,
                score_func=feature_selection.f_classif
            )),
            ('estim', discriminant_analysis.LinearDiscriminantAnalysis()),
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class NaiveBayesLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=False
            )),
            ('reduce', decomposition.FastICA(
                n_components=40,
                fun='exp',
                random_state=1742,
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=57,
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
                columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=False
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=73,
                score_func=feature_selection.f_classif
            )),
            ('estim', neighbors.KNeighborsClassifier(
                n_neighbors=16,
                weights='distance',
                metric='euclidean'
            ))
        ])

        pipe.fit(x, y)
        self._model = pipe.predict


class NearestCentroidLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('drop', transformers.ColumnDropper(
                columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124)
            )),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('select', feature_selection.SelectKBest(
                k=101,
                score_func=feature_selection.f_classif
            )),
            ('estim', neighbors.NearestCentroid(
                metric='euclidean',
                shrink_threshold=None
            )),
        ])

        pipe.fit(x, y)
        self._model = pipe.predict
