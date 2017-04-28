#!/usr/bin/env python3

import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical
from keras.callbacks import History, Callback

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from modules.AbstractNN import AbstractNN


class BaselineModel(AbstractNN):
    def _train(self):
        x = self._train_features
        y = self._train_outputs

        model = Sequential()
        model.add(Dense(32, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        history = History()

        model.fit(x, y, epochs=15, batch_size=100, callbacks=[history])

        scores = model.evaluate(x, y)
        predictions = model.predict(X_test)
        rounded = [np.argmax(x) for x in predictions]
        self._model =

