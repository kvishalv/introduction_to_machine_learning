import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas.tools import plotting
from scipy import stats
from sklearn import preprocessing


NAMES = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
              'x10', 'x11', 'x12', 'x13', 'x14', 'x15' ]
DROPC = ['x7', 'x8', 'x9', 'x12', 'x13', 'x14', 'x15']
NNAMES = [n for n in NAMES if n not in DROPC]

# 0. Read data
data = pd.read_csv('./data/train.csv', names=NAMES, header=1)
PALETTE = {0: "red", 1: "green", 2: "blue"}

# 1. Drop unuseful columns.
ddata = data.drop(DROPC, axis=1)
ddatax = ddata.drop(['y'], axis=1)
ddatay = ddata.y


def main():
    # 2. Plot scatter mattrix of all features
    scatterplot(data.drop(['y'], axis=1), title='All features')

    # 3. Show duplicate data
    scatterplot(data[['x3', 'x8', 'x10', 'x14']], title='Duplicate columns')

    # 3. Show redundant data
    data69 = pd.DataFrame(
        data=np.column_stack((
            data[['x6']]**2,
            data[['x9']]
        )),
        columns=('x6^2', 'x9')
    )
    scatterplot(data69, title='Redundant columns')

    # 4. Show noise data
    yscatterplot(
        data[['y', 'x12', 'x13', 'x15']],
        title='Uncorrelated columns (gaussian noise?)')

    # 5. Plot x7 against loggamma noise
    for yclass in range(3):
        data7c = data[(data.y == yclass)].x7
        c, loc, scale = stats.loggamma.fit(data7c)
        rdata = stats.loggamma.rvs(c, loc, scale, size=len(data7c))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data7c.values, bins=25, histtype='stepfilled', alpha=0.5, label='Class %d' % yclass)
        ax.hist(rdata, bins=25, histtype='stepfilled', alpha=0.5, label='random')
        ax.legend(loc='best', frameon=False)
        plt.title('Loggamma noise in column x7')
        plt.show()

    # 6. Show box-n-whisker plot
    boxplot(ddatax, 'Unscaled features')

    # 7. Plot feature summarizers
    plotting.radviz(ddata, 'y')
    plt.title('RadViz')
    plt.show()

    plotting.parallel_coordinates(ddata, 'y')
    plt.title('Parallel coordinates')
    plt.show()

    # 8. Plot scatter matrix for all classes
    yscatterplot(ddata, title='Features per class')

    # 9. Plot scatter matrix for each class
    for yclass in range(3):
        ddatac = ddata[(ddata.y == yclass)]
        scatterplot(
            ddatac.drop(['y'], axis=1),
            title='Class %d' % yclass,
        )

    # 10. Plot scatter matrix for all classes except one
    for yclass in range(3):
        ddatac = ddata[(ddata.y != yclass)]
        yscatterplot(ddatac, title='Not class %d' % yclass)
    """

    # 11. Plot 3D scatters for all classes, 3 columns per time.
    for x, y, z in itertools.combinations(NNAMES[1:], r=3):
        h = plt.figure().gca(projection='3d')
        h.scatter(ddata[x], ddata[y], ddata[z], color=[PALETTE[c] for c in ddata.y])
        h.set_xlabel(x); h.set_ylabel(y); h.set_zlabel(z)
        plt.show()


def boxplot(data, title=None):
    plt.figure()
    data.boxplot()
    if title is not None:
        plt.title(title)
    plt.show()


def yscatterplot(data, title=None, palette=None):
    if palette is None:
        palette = PALETTE
    scatterplot(data, title=title, color=[palette[int(c)] for c in np.nditer(data.y)])


def scatterplot(data, title=None, color=None):
    pd.scatter_matrix(data, alpha=0.3, diagonal='kde', color=color)
    if title is not None:
        plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    main()
