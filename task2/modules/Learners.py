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


NAMES = [
'y',
'x1', 'x2', 'x3',
'x4', 'x5', 'x6',
'x7', 'x8', 'x9',
'x10', 'x11', 'x12',
'x13', 'x14', 'x15'
]


def filter_outliers(x, y, **kwargs):
    xy = np.column_stack((x,y))
    filter_estimator = ensemble.IsolationForest(random_state=42, **kwargs)
    filter_estimator.fit(xy)
    is_inlier = filter_estimator.predict(xy)
    return x[is_inlier == 1], y[is_inlier == 1]


class NaiveBayesLearner(SciKitLearner):

    def _train(self):
        x    = self._train_set.features
        y    = self._train_set.outputs

        #x, y = filter_outliers(x, y, n_estimators=200, contamination=0.003)

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
