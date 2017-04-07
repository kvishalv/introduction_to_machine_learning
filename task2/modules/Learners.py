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
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            ('scale', preprocessing.StandardScaler()),
            ('expand', preprocessing.PolynomialFeatures()),
            ('estim', naive_bayes.GaussianNB())
        ])

        param_grid = [{
            'scale__with_mean': [True, False],
            'scale__with_std': [True, False],

            'expand__include_bias': [False, True],
            'expand__interaction_only': [False, True],
            'expand__degree': [1, 2, 3, 4],
        }]

        grid = model_selection.GridSearchCV(
            pipe, cv=5, n_jobs=4, param_grid=param_grid, verbose=1,
            scoring = metrics.make_scorer(metrics.accuracy_score),
        )
        grid.fit(x, y)

        print('Optimal Hyperparametres:')
        print('=======================')
        for step in grid.best_estimator_.steps:
            print(step)

        self._model = grid.predict
