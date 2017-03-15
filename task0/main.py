
""" Import packages """
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

""" Data import """
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
