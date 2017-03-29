#!/usr/bin/env python3

import numpy as np

from modules.CSVDataSet import *
from modules.Learners import *
import warnings

warnings.simplefilter("ignore")
OUTPUT    = True
VALIDATE  = True
learner = LassoLarsLearner()
#learner = PolyLassoRegressionLearner()
#learner = GridLearner()

"""
    train_size = 0.9
    PolyRidgeRegressionLearner: 1.40300900669e-07, alpha 100, poly 2, score: 25.3610838203
    PolyRidgeRegressionLearner: 1.40301129493e-08, alpha 10, poly 2, score: 25.0035653032
    PolyRidgeRegressionLearner: 1.91697708794e-07, alpha 100, poly 3, score: 41.0132351669
    Model0: 3.82868235438e-06, alpha 2000, poly 3, %90, score: 19.4278482076
    Model0: 1.61280520677e-06, alpha 800, poly 3, %98, score: 17.4999891433

    generalize optimum finder and its reporting
    add distributions for all x values as well and their correlations
"""

def main():
    """ Get the training data """
    train_set = CSVDataSet.from_train_data('data/train.csv', dtype=np.double)
    if VALIDATE:
        train_set, validation_set = train_set.split(train_size=0.95)
    #train_set.printOverview()

    """ Train """
    learner.learn_from(train_set)
    print("Training error for", learner.__class__.__name__, "is:", learner.train_error)
    #learner.findOptimalAlpha(validation_set)

    """ Validate """
    if VALIDATE:
        verror = learner.validate_against(validation_set)
        print("Validation error for", learner.__class__.__name__, "is:", verror)
        print("Validation error - training error:", learner.__class__.__name__, "is:", verror-learner.train_error)

    """ Get the test data, process & write the output """
    if OUTPUT:
        test_set = CSVDataSet.from_test_data('data/test.csv', dtype=np.double)
        out_set = learner.predict_from(test_set)
        out_set.write_labelled_output('test_result.csv')


if __name__ == '__main__':
    main()
