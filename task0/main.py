#!/usr/bin/env python3

import numpy as np


def main():
    """ Data import """
    train_set = get_train()
    model = train(train_set)

    validation_set = get_validation()
    score = validate(model, validation_set)

    test_set = get_test()
    prediction = evaluate(model, test_set)

    write_result(train_set, prediction)


def csv_to_np(filename):
    return np.genfromtxt(filename, delimiter=',', skip_header=True)


def get_train():
    #FIXME
    return csv_to_np("data/test.csv")


def get_validation():
    return csv_to_np("data/test.csv")


def get_test():
    return csv_to_np("data/test.csv")


def train(train_set):
    return np.mean(train_set[:, 1:], axis=1)


def validate(prediction, validation_set):
    pass


def evaluate(model, test_set):
    #FIXME
    return model


def write_result(test_set, prediction):
    result_test = np.column_stack((test_set[:, 0], prediction))
    np.savetxt("test_result.csv", result_test, delimiter=",", fmt="%i,%1.20f", header="Id,y", comments="")


if __name__ == '__main__':
    main()
