#!/usr/bin/env python3

import warnings

import numpy as np
from sklearn import model_selection

from modules.datasets import H5DataSet
from modules.learners import *

warnings.simplefilter('ignore')

OUTPUT   = True
VALIDATE = True

learner = QuadraticDiscriminantLearner()
#learner = GridLearner()

def main():
    train_set = H5DataSet.from_labeled_data('./data/train_labeled.h5')
    x_train, y_train = train_set.features, train_set.outputs

    if VALIDATE:
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            x_train, y_train,
            train_size=0.90,
            stratify=y_train,
            random_state=1742
        )

    learner.learn_from(x_train, y_train)

    train_acc = learner.train_error
    print('Scoring:')
    print('=======')
    print("Classifier:", learner.__class__.__name__)
    print('Training accuracy  :', train_acc)

    if VALIDATE:
        val_acc = learner.validate_against(x_val, y_val)
        print('Validation accuracy:', val_acc)
        print('Difference         :', train_acc - val_acc)

    if OUTPUT:
        test_set = H5DataSet.from_test_data('./data/test.h5')
        test_set.outputs = learner.predict_from(test_set.features)
        test_set.write_labelled_output('test_result.csv')


if __name__ == '__main__':
    main()
