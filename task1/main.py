#!/usr/bin/env python3

import warnings

import numpy as np

from modules.CSVDataSet import *
from modules.Learners import *

warnings.simplefilter("ignore")
OUTPUT    = True
VALIDATE  = True
learner = LassoLarsLearner()
#learner = PolyLassoRegressionLearner()
#learner = GridLearner()

"""
    train_size = 0.9
    generalize optimum finder and its reporting
    add distributions for all x values as well and their correlations
"""

def main():
    train_set = CSVDataSet.from_train_data('data/train.csv', dtype=np.double)
    if VALIDATE:
        train_set, validation_set = train_set.split(train_size=0.95)
    #train_set.printOverview()

    learner.learn_from(train_set)
    print("Training error for", learner.__class__.__name__, "is:", learner.train_error)
    #learner.findOptimalAlpha(validation_set)

    if VALIDATE:
        verror = learner.validate_against(validation_set)
        print("Validation error for", learner.__class__.__name__, "is:", verror)
        print("Validation error - training error:", learner.__class__.__name__, "is:", verror-learner.train_error)

    if OUTPUT:
        test_set = CSVDataSet.from_test_data('data/test.csv', dtype=np.double)
        out_set = learner.predict_from(test_set)
        out_set.write_labelled_output('test_result.csv')


if __name__ == '__main__':
    main()
