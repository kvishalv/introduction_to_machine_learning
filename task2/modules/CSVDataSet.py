import numpy as np
import matplotlib.pyplot as plt


class CSVDataSet(object):

    def __init__(self, ids, features, outputs):
        self.ids = ids
        self.features = features
        self.outputs = outputs

    @classmethod
    def from_train_data(cls, filename):
        data = cls._csv_to_array(filename)
        train_id = data[:, 0].astype('int')
        train_y = data[:, 1]
        train_features = data[:, 2:]
        return cls(train_id, train_features, train_y)

    @classmethod
    def from_test_data(cls, filename):
        data = cls._csv_to_array(filename)
        test_id = data[:, 0].astype('int')
        test_features = data[:, 1:]
        return cls(test_id, test_features, None)

    @staticmethod
    def _csv_to_array(filename):
        """
        Returns contents of `filename` CSV file as a numpy array.

        dtype: NumPy type

        Note: Assumes and ignores exactly one header line.
        """
        return np.genfromtxt(
            filename, delimiter=',', skip_header=True
        )

    def write_labelled_output(self, filename):
        np.savetxt(
            filename, np.column_stack((self.ids, self.outputs)),
            header="Id,y", comments="",
            delimiter=",", fmt="%i,%r"
        )

    def printOverview(self):

        np.set_printoptions(threshold=np.nan)
        print("\n\n------------------------------ overview ------------------------------")
        print("Means of x values:\n", np.apply_along_axis(np.mean, 0, self.features))
        print("Std of x values:\n", np.apply_along_axis(np.std, 0, self.features))
        print("Correlation coefficients of x values:\n", np.corrcoef(self.features,rowvar=0))
        print("\n")
        print("Means of y values:", np.mean(self.outputs))
        print("Std of y values:", np.std(self.outputs))
        print("----------------------------------------------------------------------\n\n")

        plt.hist(self.outputs, 50)
        plt.title('y histogram')
        plt.show()
