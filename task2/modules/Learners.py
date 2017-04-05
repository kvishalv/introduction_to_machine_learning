import numpy as np
import matplotlib.pyplot as plt
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
            ('expand', preprocessing.PolynomialFeatures()),
            ('estim', linear_model.LassoLars())
        ])

        param_grid = [{
            'expand__include_bias': [False],
            'expand__degree': [3],

            'estim__normalize': [False],
            'estim__fit_intercept': [True],
            'estim__alpha': [0.313]
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

        """
        estim = grid.best_estimator_.named_steps['estim']
        coeffs = estim.coef_

        polyexp = grid.best_estimator_.named_steps['expand']
        f_names = polyexp.get_feature_names(NAMES[1:])

        for elem in sorted(zip(coeffs, f_names), reverse=True):
            print(elem)
        """

        self._model = grid.predict
