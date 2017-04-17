#!/usr/bin/env python3

import warnings

import numpy as np
from sklearn import model_selection

from modules.CSVDataSet import *
from modules.Learners import *

warnings.simplefilter("ignore")
OUTPUT     = True
VALIDATE   = True

MULTIPLEVALIDATE = False
PRINTSTEPS    = False #if MULTIPLEVALIDATE
NRVALIDATIONS = 20    #if MULTIPLEVALIDATE

learner = QuadraticDiscriminantLearner()
#learner = GridLearner()



""" To Do:

Try Removing outliers
Try robustScaler

"""


def main():

    if MULTIPLEVALIDATE:
        nrval = NRVALIDATIONS
    else:
        nrval = 1

    tacc = np.zeros(nrval)
    vacc = np.zeros(nrval)

    for i in range (0,nrval):
        print(i+1, "of", nrval)
        train_set = CSVDataSet.from_train_data('data/train.csv')
        x_train, y_train = train_set.features, train_set.outputs
        if VALIDATE:
            x_train, x_val, y_train, y_val = model_selection.train_test_split(
                x_train, y_train,
                train_size=0.90,
                stratify=y_train,
                random_state=1742
            )

        learner.learn_from(x_train, y_train)

        tacc[i] = learner.train_error
        if PRINTSTEPS:
            print('Scoring:')
            print('=======')
            print("Classifier:", learner.__class__.__name__)
            print("    Training accuracy  : ", tacc[i])

        if VALIDATE:
            vacc[i] = learner.validate_against(x_val, y_val)
            if PRINTSTEPS:
                print("    Validation accuracy: ", vacc[i])
                print("    Difference         : ", tacc[i] - vacc[i] )

    print('Scoring:')
    print('=======')
    print("Classifier:", learner.__class__.__name__)
    print("    Mean Training accuracy  : ", np.mean(tacc))
    print("    Std  Training accuracy  : ", np.std(tacc))
    print("    Mean Validation accuracy: ", np.mean(vacc))
    print("    Std  Validation accuracy: ", np.std(vacc))

    if MULTIPLEVALIDATE:
        plt.hist(tacc, 50)
        plt.title('training accuracies')
        plt.show()

        plt.hist(vacc, 50)
        plt.title('validation accuracies')
        plt.show()

    if OUTPUT:
        test_set = CSVDataSet.from_test_data('data/test.csv')
        test_set.outputs = learner.predict_from(test_set.features)
        test_set.write_labelled_output('test_result.csv')


if __name__ == '__main__':
    main()
