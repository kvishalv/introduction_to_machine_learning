#!/usr/bin/env python3

from keras.layers import *
from keras.layers.advanced_activations import ELU, PReLU
from keras.models import Sequential
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier

from keras import regularizers
from keras import metrics

from sklearn import model_selection

from modules.AbstractNN import AbstractNN


class GridLearner(AbstractNN):

    @staticmethod
    def make_model(
        layers=(768, 384),
        initialization='he_uniform',
        activation='prelu',
        dropout=0.1,
        lr=0.1,
        momentum=0.9,
        decay=0.0,
        nesterov=True,
        bn=False
    ):
        if activation == 'prelu':
            activation = PReLU(shared_axes=[1])

        model = Sequential()

        model.add(Dropout(dropout, input_shape=(100,)))

        for count in layers:
            model.add(Dense(count, kernel_initializer=initialization)),
            model.add(Activation(activation))
            if bn:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(5, kernel_initializer=initialization))
        model.add(Activation('softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov),
            metrics=['accuracy']
        )
        return model

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        classifier = KerasClassifier(self.make_model)

        okk = PReLU(shared_axes=[1])
        param_grid = [{
            'batch_size': [64],
            'layers': [(768, 128)],
            #'dropout': [0.1],
            'shuffle': [True],
            'epochs': [5, 10, 20, 40],
            #'lr': [0.1],
            'momentum': [0.9],
            'decay': [0.001],
            #'momentum': [0.9, 0.99],
            #'decay': [0.001, 0.0001],
            'bn': [False],
            'verbose': [0]
        }]

        grid = model_selection.GridSearchCV(
            classifier, cv=2, n_jobs=4, param_grid=param_grid, verbose=1,
            scoring = 'neg_log_loss'
        )
        grid_result = grid.fit(x, y)

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
            Dropout(0.1, input_shape=(100,)),
            Dense(1536, kernel_initializer='he_uniform'),
            Activation(PReLU(shared_axes=[1])),
            Dropout(0.1),
            Dense(1536, kernel_initializer='he_uniform'),
            Activation(PReLU(shared_axes=[1])),
            Dropout(0.1),
            Dense(5),
            Activation('softmax'),
        ])

        sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy']
        )

        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            x, y,
            train_size=0.85,
            stratify=y,
            random_state=1743
        )

        model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=64,
            validation_data=(x_val, y_val),
            shuffle=True,
            verbose=1
        )

        self._model = model.predict_classes


class ConvolutionalModel(AbstractNN):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        model = Sequential([
            Reshape((10, 10, -1), input_shape=(100,)),

            Conv2D(32, (3, 3), padding='same'),
            Activation(PReLU(shared_axes=[1, 2])),
            Conv2D(64, (2, 2), padding='same'),
            Activation(PReLU(shared_axes=[1, 2])),
            Conv2D(64, (2, 2), padding='same'),
            Activation(PReLU(shared_axes=[1, 2])),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Flatten(),

            Dense(256, kernel_initializer='he_uniform'),
            Activation(PReLU(shared_axes=[1])),
            Dropout(0.5),

            Dense(256, kernel_initializer='he_uniform'),
            Activation(PReLU(shared_axes=[1])),
            Dropout(0.2),

            Dense(5, kernel_initializer='he_uniform'),
            Activation('softmax'),
        ])

        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy']
        )

        #history = History()
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            x, y,
            train_size=0.85,
            stratify=y,
            random_state=1742
        )

        model.fit(
            x_train, y_train,
            epochs=50,
            batch_size=64,
            validation_data=(x_val, y_val),
            shuffle=True,
            verbose=1
        )

        self._model = model.predict_classes
