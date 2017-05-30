#!/usr/bin/env python3

from keras.layers import *
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

from keras import regularizers
from keras import metrics
from keras.utils import to_categorical

from sklearn import model_selection

from modules.AbstractNN import AbstractNN


class GridLearner(AbstractNN):

    @staticmethod
    def make_model(
        layers=(384, 384),
        dropout_low=0.15,
        dropout_high=0.25,
        lr=0.1,
        momentum=0.9,
        decay=0.0,
        nesterov=True,
        monitor='val_loss',
        optimizer='sgd'
    ):
        model = Sequential()

        model.add(Dropout(dropout_low, input_shape=(128,)))

        for count in layers[:-1]:
            model.add(Dense(count, kernel_initializer='he_uniform')),
            model.add(Activation(PReLU(shared_axes=[1]))),
            model.add(Dropout(dropout_high))

        model.add(Dense(layers[-1], kernel_initializer='he_uniform')),
        model.add(Activation(PReLU(shared_axes=[1]))),

        model.add(Dropout(dropout_low))

        model.add(Dense(10, kernel_initializer='he_uniform'))
        model.add(Activation('softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer, #SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov),
            metrics=['accuracy'],
        )
        return model

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        classifier = KerasClassifier(self.make_model)

        param_grid = [{
            'batch_size': [64, 128],
            'layers': [(1024,)],
            #'layers': [(768,), (384, 384)],
            'dropout_low': [0.05 * i for i in range(5)],
            #'dropout_high': [0.25],
            'shuffle': [True],
            'optimizer': ['adam', 'adadelta', 'rmsprop', 'nadam'],
            'epochs': [400],
            'verbose': [0],
            'callbacks': [[
                EarlyStopping(
                    monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto'
                )
            ]]
        }]

        grid = model_selection.GridSearchCV(
            classifier, cv=10, n_jobs=8, param_grid=param_grid, verbose=1,
            scoring = 'neg_log_loss'
        )
        grid_result = grid.fit(x, to_categorical(y, num_classes=10))

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        best_model = grid.best_estimator_.model
        self._model = best_model.predict_classes


class BaselineModel(AbstractNN):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        model = Sequential([
            Dropout(0.15, input_shape=(128,)),
            Dense(1024, kernel_initializer='he_uniform'),
            Activation(PReLU(shared_axes=[1])),
            Dropout(0.15),
            Dense(10),
            Activation('softmax'),
        ])

        sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy']
        )

        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            x, y,
            train_size=0.95,
            stratify=y,
            random_state=1742
        )

        es = EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=5,
            verbose=0,
            mode='auto'
        )

        model.fit(
            # x, to_categorical(y, num_classes=10),
            x_train, to_categorical(y_train, num_classes=10),
            epochs=200,
            batch_size=128,
            validation_data=(x_val, to_categorical(y_val, num_classes=10)),
            shuffle=True,
            callbacks=[],
            #callbacks=[es],
            verbose=1
        )

        self._model = model.predict_classes
