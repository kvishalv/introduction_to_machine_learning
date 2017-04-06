#!/usr/bin/env python3

import warnings

import numpy as np

from modules.CSVDataSet import *
from modules.Learners import *

warnings.simplefilter("ignore")
OUTPUT    = True
VALIDATE  = True
learner = NaiveBayesLearner()


def main():
    train_set = CSVDataSet.from_train_data('data/train.csv')
    if VALIDATE:
        train_set, validation_set = train_set.split(train_size=0.70)
    learner.learn_from(train_set.features, train_set.outputs)

    tacc = learner.train_error
    print('Scoring:')
    print('=======')
    print("Classifier:", learner.__class__.__name__)
    print("    Training accuracy  : ", tacc)

    if VALIDATE:
        x_val, y_val = validation_set.features, validation_set.outputs
        vacc = learner.validate_against(x_val, y_val)
        print("    Validation accuracy: ", vacc)
        print("    Difference         : ", vacc - tacc)

    if OUTPUT:
        test_set = CSVDataSet.from_test_data('data/test.csv')
        test_set.outputs = learner.predict_from(test_set.features)
        test_set.write_labelled_output('test_result.csv')


if __name__ == '__main__':
    main()
