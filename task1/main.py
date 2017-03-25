#!/usr/bin/env python3

import numpy as np

from modules.CSVDataSet import *
from modules.Learners import *


"""
To do:
    validation framework
"""

OUTPUT = True
learner = PolyTheilSenRegressionLearner()


def main():
    """ Get the training data """
    train_set = CSVDataSet.from_train_data('data/train.csv', dtype=np.double)

    """ Train """
    learner.learn_from(train_set)
    print("Training error for", learner.__class__.__name__, "is:", learner.train_error)

    """ Get the test data, process & write the output """
    if OUTPUT:
        test_set = CSVDataSet.from_test_data('data/test.csv', dtype=np.double)
        out_set = learner.predict_from(test_set)
        out_set.write_labelled_output('test_result.csv')


if __name__ == '__main__':
    main()
