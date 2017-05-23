#!/usr/bin/env python3

import warnings

import numpy as np
from sklearn import model_selection

from modules.datasets import H5DataSet, CSVDataSet
from modules.learners import *

warnings.simplefilter('ignore')

VALIDATE      = True
USE_UNLABELED = True
USE_TRANSDUCE = False
OUT_TRANSDUCE = False
OUTPUT        = True

learner = QuadraticDiscriminantLearner()
#learner = GridLearner()
#learner = ManifoldLLELearner()

def main():
    labeled_set = H5DataSet.from_labeled_data('./data/train_labeled.h5')
    x_label, y_label = labeled_set.features, labeled_set.outputs

    if VALIDATE:
        x_label, x_val, y_label, y_val = model_selection.train_test_split(
            x_label, y_label,
            train_size=0.80,
            stratify=y_label,
            random_state=1742
        )

    if USE_UNLABELED:
        unlabeled_set = H5DataSet.from_unlabeled_data('./data/train_unlabeled.h5')
        x_unlabel = unlabeled_set.features

        if USE_TRANSDUCE:
            trsd_set = CSVDataSet.from_unlabeled_data('./data/transduced.csv')
            y_unlabel = trsd_set.features[:, 0]
        else:
            y_unlabel = -np.ones(len(x_unlabel))

        x_train = np.concatenate((x_label, x_unlabel))
        y_train = np.concatenate((y_label, y_unlabel))
    else:
        x_train = x_label
        y_train = y_label

    learner.learn_from(x_train, y_train)
    train_acc = learner.train_error

    if USE_UNLABELED and OUT_TRANSDUCE:
        unlabeled_set.outputs = learner.get_transduction()[len(x_label):]
        unlabeled_set.write_labelled_output('./data/transduced.csv')

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
