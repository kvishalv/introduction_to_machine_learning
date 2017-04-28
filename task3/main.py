#!/usr/bin/env python3
from sklearn import model_selection
from keras.utils import to_categorical

from modules.AbstractNN import *
from modules.NNModels import *
from modules.DataSets import *

OUTPUT =    False
VALIDATE =  True

learner = BaselineModel()

def main():

    train_set = DataSets.from_train_data('data/train.h5')
    x_train, y_train = train_set.features, train_set.outputs
    if VALIDATE:
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            x_train, y_train,
            train_size=0.90,
            stratify=y_train,
            random_state=42
        )

    learner.learn_from(x_train, y_train)

    print("Training error for", learner.__class__.__name__, "is:", learner.train_error)

    if VALIDATE:
        verror = learner.validate_against(x_val, y_val)
        print("Validation error for", learner.__class__.__name__, "is:", verror)
        print("Validation error - training error:", learner.__class__.__name__, "is:", verror - learner.train_error)

    if OUTPUT:
        test_set = DataSets.from_test_data('data/test.h5')
        test_set.outputs = learner.predict_from(test_set.features)
        test_set.write_labelled_output('test_result.csv')

if __name__ == '__main__':
    main()