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

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from modules.evals import *

from modules.AbstractNN import AbstractNN


class BaselineModel(AbstractNN):
    def _train(self):
        x = self._train_features
        y = self._train_outputs

        model = Sequential()


        model.add(Dense(64, activation='relu', input_dim=100))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(5, activation='softmax'))
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        history = History()

        # Batchsize is number of samples you use for gradient descent update
        model.fit(x, y, epochs=20, batch_size=100, callbacks=[history], verbose=2)

        # Plotting
        #plot_lossvsepoch(history.epoch, history.history["loss"], "baseline_losshistory.png")

        # cores = model.evaluate(x, y)
        self._model = model.predict

