#!/usr/bin/env python3

import numpy as np


def main():
    """ Data import """
    train_set = get_train_data('data/train.csv')
    model = train(train_set)



    test_set = get_test()
    prediction = predict(model, test_set)

    write_result(train_set, prediction)


def csv_to_np(filename):
    return np.genfromtxt(filename, delimiter=',', skip_header=True)


def get_train_data(filename):
    return csv_to_np(filename)




def get_test():
    return csv_to_np("data/test.csv")


def train(train_set):
    return lambda v: np.mean(v)




def predict(model, test_set):
    return np.apply_along_axis(model, 1, test_set[:, 1:])


def write_result(col_id, predictions):
    result_test = np.column_stack((col_id[:, 0], predictions))
    np.savetxt(
        "test_result.csv", result_test,
        header="Id,y", comments="",
        delimiter=",", fmt="%i,%1.20f"
    )


if __name__ == '__main__':
    main()
