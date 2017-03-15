
""" Import packages """
import numpy as np

""" Data import """
train = np.genfromtxt("data/train.csv", delimiter=',', skip_header=True)
test = np.genfromtxt("data/test.csv", delimiter=',', skip_header=True)
y_pred = np.mean(test[:, 1:], axis=1)

result = np.column_stack((test[:, 0], y_pred))
np.savetxt("result.csv", result, delimiter=",", fmt="%i, %1.4f", header="Id, y")


