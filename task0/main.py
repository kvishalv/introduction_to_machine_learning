#!/usr/bin/env python3

import numpy as np

""" Data import """
train = np.genfromtxt("data/train.csv", delimiter=',', skip_header=True)
test = np.genfromtxt("data/test.csv", delimiter=',', skip_header=True)
y_pred = np.mean(test[:, 1:], axis=1)

result_test = np.column_stack((test[:, 0], y_pred))
np.savetxt("test_result.csv", result_test, delimiter=",", fmt="%i, %1.4f", header="Id, y")

# y = np.mean(train[:, 2:], axis=1)
# result_train = np.column_stack((train[:, 0], y))
# np.savetxt("train_result.csv", result_train, delimiter=",", fmt="%i, %1.1f", header="Id, y")
