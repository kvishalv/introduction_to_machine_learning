import numpy as np
from sklearn import (
    metrics,
    model_selection,
    naive_bayes,
    pipeline,
    preprocessing,
)

from modules.AbstractLearner import (
    SciKitLearner,
)


class NaiveBayesLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        pipe = pipeline.Pipeline([
            ('scale', preprocessing.StandardScaler()),
            ('expand', preprocessing.PolynomialFeatures()),
            ('estim', naive_bayes.GaussianNB())
        ])

        param_grid = [{
            'scale__with_mean': [True, False],
            'scale__with_std': [True, False],

            'expand__include_bias': [False, True],
            'expand__degree': [1, 2, 3],
        }]

        grid = model_selection.GridSearchCV(
            pipe, cv=3, n_jobs=1, param_grid=param_grid, verbose=1,
            scoring = metrics.make_scorer(metrics.accuracy_score),
        )
        grid.fit(x, y)

        print('Optimal Hyperparametres:')
        print('=======================')
        for step in grid.best_estimator_.steps:
            print(step)

        self._model = grid.predict
