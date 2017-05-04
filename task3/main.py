#!/usr/bin/env python3
from sklearn import model_selection
from sklearn.cross_validation import StratifiedKFold
from keras.utils import to_categorical

from modules.AbstractNN import *
from modules.NNModels import *
from modules.DataSets import *
import matplotlib.pyplot as plt

import numpy as np

OUTPUT =    True
MANUALVALIDATE =  False
Kfolds = 10

learner = BaselineModel()
# learner = ConvolutionalModel()

"""
To do:
data preprocessing
architecture optimization, drop, layers nodes etc
custom loss function

"""

def main():

    train_set = DataSets.from_train_data('data/train.h5')
    x_train, y_train = train_set.features, train_set.outputs
    if MANUALVALIDATE:
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            x_train, y_train,
            train_size=0.80,
            stratify=y_train,
            random_state=42
        )

    # y_train_cat = np.asarray([np.argmax(x) for x in y_train])
    # print(np.where(y_train_cat == 0)[0].__len__())
    # print(np.where(y_train_cat == 1)[0].__len__())
    # print(np.where(y_train_cat == 2)[0].__len__())
    # print(np.where(y_train_cat == 3)[0].__len__())
    # print(np.where(y_train_cat == 4)[0].__len__())

    skf = StratifiedKFold(y_train[:, 1], n_folds=Kfolds, shuffle=True)
    cvscores = []
    for i, (train, test) in enumerate(skf):
        print('K-fold:',i+1, 'of', Kfolds)
        learner.learn_from(x_train[train], y_train[train])
        print("Training error for", learner.__class__.__name__, "is:", learner.train_error)
        kfolderror = learner.validate_against(x_train[test], y_train[test])
        print("K-Fold Validation error for", learner.__class__.__name__, "is:", kfolderror)
        cvscores.append(kfolderror)

        if MANUALVALIDATE:
            verror = learner.validate_against(x_val, y_val)
            print("Validation error for", learner.__class__.__name__, "is:", verror)
            print("Training error - Validation error:", learner.__class__.__name__, "is:", learner.train_error - verror)

    print("%.6f%% (+/- %.6f%%)" % (np.mean(cvscores), np.std(cvscores)))

    if OUTPUT:
        test_set = DataSets.from_test_data('data/test.h5')
        test_set.outputs = learner.predict_from(test_set.features)
        test_set.write_labelled_output('test_result.csv')

if __name__ == '__main__':
    main()