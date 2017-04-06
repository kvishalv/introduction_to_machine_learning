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
    learner.learn_from(train_set)

    tacc = learner.train_error
    print('Scoring:')
    print('=======')
    print("Classifier:", learner.__class__.__name__)
    print("    Training accuracy  : ", tacc)

    if VALIDATE:
        vacc = learner.validate_against(validation_set)
        print("    Validation accuracy: ", vacc)
        print("    Difference         : ", vacc - tacc)

    if OUTPUT:
        test_set = CSVDataSet.from_test_data('data/test.csv')
        test_set.outputs = learner.predict_from(test_set)
        test_set.write_labelled_output('test_result.csv')


if __name__ == '__main__':
    main()
