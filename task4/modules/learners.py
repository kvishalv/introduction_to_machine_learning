from sklearn import (
    decomposition,
    discriminant_analysis,
    feature_selection,
    metrics,
    model_selection,
    naive_bayes,
    neighbors,
    pipeline,
    preprocessing,
    semi_supervised,
    svm,
    tree,
    gaussian_process,
    neural_network,
    manifold
)


from modules.AbstractLearner import AbstractLearner
from modules import transformers


from modules.AbstractNN import AbstractNN
import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils, to_categorical
from keras.callbacks import History, Callback
# from keras.layers.convolutional import *
from keras.layers import *
from keras import regularizers
from keras import metrics
from keras import utils as ku

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
            scoring=metrics.make_scorer(metrics.accuracy_score),
        )
        grid.fit(x, y)

        print('Optimal Hyperparametres:')
        print('=======================')
        for step in grid.best_estimator_.steps:
            print(step)
        print("CV Score:", grid.best_score_)

        estimator = pipe.named_steps['estim']
        if hasattr(estimator, 'transduction_'):
            self._transduction = estimator.transduction_
        self._model = grid.predict


class LabelPropagationLearner(AbstractLearner):

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
                percentile=54,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', semi_supervised.LabelPropagation(
                kernel='rbf',
                alpha=0.65,
                n_neighbors=4,
                n_jobs=-1
            )),
        ])

        pipe.fit(x, y)
        self._transduction = pipe.named_steps['estim'].transduction_
        self._model = pipe.predict


class LabelSpreadingLearner(AbstractLearner):

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
                percentile=71,
                score_func=feature_selection.f_classif
            )),
            ('estim', semi_supervised.LabelSpreading(
                kernel='knn',
                alpha=0.17,
                n_neighbors=7,
                n_jobs=-1
            )),
        ])

        pipe.fit(x, y)
        self._transduction = pipe.named_steps['estim'].transduction_
        self._model = pipe.predict


class NuSVCLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        pipe = pipeline.Pipeline([
            #('kselect', feature_selection.SelectKBest(feature_selection.f_regression, k=115)),
            ('drop', transformers.ColumnDropper(columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=85,#59,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', svm.NuSVC(
                nu=0.0525,
                kernel='rbf',
                gamma='auto',
                shrinking=True,
                class_weight=None,
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
                metric='euclidean',
                n_jobs=-1
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



class ManifoldLLELearner(AbstractLearner):

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
                percentile=59,#59,
                score_func=feature_selection.mutual_info_classif
            )),
            ('select', feature_selection.SelectKBest(
                k=101,
                score_func=feature_selection.f_classif
            )),
            ('estim', manifold.locally_linear_embedding(
                x,
                n_neighbors=6,
                n_components=101,
                eigen_solver='auto',
                method='standard'

            )),
        ])

        pipe.fit_transform(x)
        self._model = pipe.predict




class BaselineModel3(AbstractNN):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        x, x_val, y, y_val = model_selection.train_test_split(
            x, y,
            train_size=0.90,
            stratify=y,
            random_state=2345
        )

        y = ku.to_categorical(y, num_classes=10)
        y_val = ku.to_categorical(y_val, num_classes=10)

        model = Sequential()
        model.add(Dense(700, input_dim=128, init="he_uniform", activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(700, init="he_uniform", activation="relu")) #, activity_regularizer=regularizers.l1(0.1), kernel_regularizer=regularizers.l2(0.1)
        model.add(Dense(10))
        model.add(Activation("softmax"))

        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = History()

        # Batchsize is number of samples you use for gradient descent update
        model.fit(x, y, epochs=30, batch_size=128, callbacks=[history], verbose=2, validation_data=(x_val, y_val), shuffle=2)

        self._model = model.predict_classes




class BaselineModel2(AbstractNN):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        x, x_val, y, y_val = model_selection.train_test_split(
            x, y,
            train_size=0.90,
            stratify=y,
            random_state=2345
        )

        y = ku.to_categorical(y, num_classes=10)
        y_val = ku.to_categorical(y_val, num_classes=10)

        model = Sequential()
        model.add(Dense(700, input_dim=128, init="he_uniform", activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(350, init="he_uniform", activation="relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = History()

        pipe = pipeline.Pipeline([
            #('kselect', feature_selection.SelectKBest(feature_selection.f_regression, k=115)),
            ('drop', transformers.ColumnDropper(columns=(0, 3, 5, 14, 26, 35, 40, 65, 72, 95, 99, 104, 124))),
            ('scale', preprocessing.StandardScaler(
                with_mean=True,
                with_std=True
            )),
            ('select', feature_selection.SelectPercentile(
                percentile=59,#59,
                score_func=feature_selection.mutual_info_classif
            )),
            ('estim', model),
        ])

        pipe.fit(x, y, epochs=50, batch_size=128, callbacks=[history], verbose=2, validation_data=(x_val, y_val), shuffle=2)
        self._model = pipe.predict_classes
