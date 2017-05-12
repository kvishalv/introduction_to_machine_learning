#!/usr/bin/env python3

import warnings

import numpy as np
from sklearn import model_selection

from modules.datasets import H5DataSet
from modules.learners import *

warnings.simplefilter('ignore')

VALIDATE      = True
USE_UNLABELED = True
OUTPUT        = True

learner = QuadraticDiscriminantLearner()
#learner = GridLearner()

def main():
    labeled_set = H5DataSet.from_labeled_data('./data/train_labeled.h5')
    x_label, y_label = labeled_set.features, labeled_set.outputs

    if VALIDATE:
        x_label, x_val, y_label, y_val = model_selection.train_test_split(
            x_label, y_label,
            train_size=0.80,
            stratify=y_train,
            random_state=1742
        )

    if USE_UNLABELED:
        unlabeled_set = H5DataSet.from_unlabeled_data('./data/train_unlabeled.h5')
        x_unlabel = unlabeled_set.features
        y_unlabel = -np.ones(len(x_unlabel))

        x_label = np.concatenate((x_label, x_unlabel))
        y_label = np.concatenate((y_label, y_unlabel))

    learner.learn_from(x_label, y_label)
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
