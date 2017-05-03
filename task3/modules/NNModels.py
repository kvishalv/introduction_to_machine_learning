#!/usr/bin/env python3

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

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from modules.evals import *

from modules.AbstractNN import AbstractNN


class BaselineModel(AbstractNN):
    def _train(self):
        x = self._train_features
        y = self._train_outputs
        if self._fwdTransform is not NotImplemented:
            x = self._fwdTransform(x)
            y = self._fwdTransform(y)

        model = Sequential()
        # model.add(Dense(64, activation='relu', input_dim=100))
        # model.add(Dropout(0.2))
        # model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        # model.add(Dropout(0.2))
        # model.add(Dense(5, activation='softmax'))

        model.add(Dense(64, activation='relu', input_dim=100))
        model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dense(5, activation='softmax'))

        # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
        # sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        history = History()

        # Batchsize is number of samples you use for gradient descent update
        model.fit(x, y, epochs=20, batch_size=100, callbacks=[history], verbose=0)

        # Plotting
        #plot_lossvsepoch(history.epoch, history.history["loss"], "baseline_losshistory.png")

        # cores = model.evaluate(x, y)

        self._model    = model.predict


    def _addDim(self, data):
        return data[None, :, :]

    def _rmvDim(self, data):
        return np.squeeze(data)


class ConvolutionalModel(AbstractNN):
    def _train(self):
        self._fwdTransform = self._addDim
        self._bckTransform = self._rmvDim

        x = self._train_features
        y = self._train_outputs
        if self._fwdTransform is not NotImplemented:
            x = self._fwdTransform(x)
            y = self._fwdTransform(y)

        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(None, x.shape[2])))
        model.add(Conv1D(filters=5, kernel_size=3, strides=1, input_shape=(None, x.shape[2]), padding='same'))
        model.add(Dense(5, activation='relu', input_shape=(None, x.shape[2])))
        model.add(Dense(5, activation='softmax', input_shape=(None, x.shape[2])))

        # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
        # sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        history = History()

        # Batchsize is number of samples you use for gradient descent update
        model.fit(x, y, epochs=20, batch_size=100, callbacks=[history], verbose=0)

        # Plotting
        #plot_lossvsepoch(history.epoch, history.history["loss"], "baseline_losshistory.png")

        # cores = model.evaluate(x, y)

        self._model    = model.predict


    def _addDim(self, data):
        return data[None, :, :]

    def _rmvDim(self, data):
        return np.squeeze(data)
