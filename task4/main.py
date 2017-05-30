#!/usr/bin/env python3

import warnings

import numpy as np
from sklearn import model_selection

from modules.datasets import H5DataSet, CSVDataSet
from modules.learners import *
from random import randint
from modules.NNModels import *
from sklearn import semi_supervised

warnings.simplefilter('ignore')

VALIDATE      = False
USE_UNLABELED = True
USE_TRANSDUCE = True
OUT_TRANSDUCE = False
OUTPUT        = True


# learner = test()
# learner = GridLearner()
# learner = BaselineModel3()
learner = BaselineModel_C()
# learner = ManifoldLLELearner()

def main():
    labeled_set = H5DataSet.from_labeled_data('./data/train_labeled.h5')
    x_label, y_label = labeled_set.features, labeled_set.outputs

    if VALIDATE:
        x_label, x_val, y_label, y_val = model_selection.train_test_split(
            x_label, y_label,
            train_size=0.90,
            stratify=y_label,
            random_state=randint(0, 2000)
        )

    if USE_UNLABELED:
        unlabeled_set = H5DataSet.from_unlabeled_data('./data/train_unlabeled.h5')
        x_unlabel = unlabeled_set.features
        y_unlabel = -np.ones(len(x_unlabel))

        if USE_TRANSDUCE:

            x_train = np.concatenate((x_label, x_unlabel))
            y_train = np.concatenate((y_label, y_unlabel))

            model = semi_supervised.LabelSpreading(
                kernel='rbf',
                gamma=2.0
            )
            model.fit(x_train, y_train)
            y_train = model.transduction_



    else:
        x_train = x_label
        y_train = y_label

    # learner.learn_from(x_train, y_train)
    # train_acc = learner.train_error

    if USE_UNLABELED and OUT_TRANSDUCE:
        unlabeled_set.outputs = learner.get_transduction()[len(x_label):]
        unlabeled_set.write_labelled_output('./data/transduced.csv')

    print('Scoring:')
    print('=======')
    print("Classifier:", learner.__class__.__name__)
    #print('Training accuracy  :', train_acc)

    if VALIDATE:
        val_acc = learner.validate_against(x_val, y_val)
        print('Validation accuracy:', val_acc)
        print('Difference         :', train_acc - val_acc)

    if OUTPUT:
        test_set = H5DataSet.from_unlabeled_data('./data/train_unlabeled.h5')
        test_set.outputs = model.get_transduction() #.predict_from(test_set.features)
        #test_set.write_labelled_output('test_result.csv')
        test_set.write_labelled_output('transduction2.csv')


if __name__ == '__main__':
    main()
