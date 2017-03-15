#!/usr/bin/env python3

import numpy as np


def main():
    """ Data import """
    train_set = get_train()
    model = train(train_set)

    validation_set = get_validation()
    score = validate(model, validation_set)

    test_set = get_test()
    prediction = predict(model, test_set)

    write_result(train_set, prediction)


def csv_to_np(filename):
    return np.genfromtxt(filename, delimiter=',', skip_header=True)


def get_train():
    #FIXME
    return csv_to_np("data/test.csv")


def get_validation():
    pass


def get_test():
    return csv_to_np("data/test.csv")


def train(train_set):
    return lambda v: np.mean(v)


def validate(prediction, validation_set):
    pass


def predict(model, test_set):
    return np.apply_along_axis(model, 1, test_set[:, 1:])


def write_result(test_set, prediction):
    result_test = np.column_stack((test_set[:, 0], prediction))
    np.savetxt("test_result.csv", result_test, delimiter=",", fmt="%i,%1.20f", header="Id,y", comments="")


if __name__ == '__main__':
    main()
