#!/usr/bin/env python3

from modules.NNModels import *
from modules.DataSets import *

OUTPUT = True

#learner = GridLearner()
learner = BaselineModel()
#learner = ConvolutionalModel()


def main():
    train_set = DataSets.from_train_data('data/train.h5', shuffle=True)
    learner.learn_from(train_set.features, train_set.outputs)

    if OUTPUT:
        test_set = DataSets.from_test_data('data/test.h5')
        test_set.outputs = learner.predict_from(test_set.features)
        test_set.write_labelled_output('test_result.csv')

if __name__ == '__main__':
    main()
