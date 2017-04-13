import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools import plotting
from sklearn import preprocessing

NAMES = [
'y',
'x1', 'x2', 'x3',
'x4', 'x5', 'x6',
'x7', 'x8', 'x9',
'x10', 'x11', 'x12',
'x13', 'x14', 'x15'
]
DROPC = ['x8', 'x9', 'x12', 'x13', 'x14', 'x15']
NNAMES = [n for n in NAMES if n not in DROPC]


def boxplot(data, title=None):
    plt.figure()
    data.boxplot()
    if title is not None:
        plt.title(title)
    plt.show()


def scatterplot(data, color=None):
    pd.scatter_matrix(data, alpha=0.3, diagonal='kde', color=color)
    plt.show()


# 0. Read data
data = pd.read_csv('./data/train.csv', names=NAMES, header=1)
palette = {0: "red", 1: "green", 2: "blue"}

# 1. Plot scatter mattrix of all features
scatterplot(data)

# 2. Show duplicate data
scatterplot(data[['x3', 'x8', 'x10', 'x14']])

# 2. Show redundant data
data69 = pd.DataFrame(
    data=np.column_stack((
        data[['x6']]**2,
        data[['x9']]
    )),
    columns=('x6^2', 'x9')
)
scatterplot(data69)

# 3. Show noise data
scatterplot(
    data[['y', 'x12', 'x13', 'x15']],
    color=tuple(palette[c] for c in data.y)
)

# 4. Filter useful columns.
ddata = data.drop(DROPC, axis=1)
ddatax = ddata.drop(['y'], axis=1)
ddatay = ddata.y

# 5. Show box-n-whisker plot
boxplot(ddatax, 'Unscaled features')

# 6. Plot feature summarizers
plotting.radviz(ddata, 'y')
plt.title('RadViz')
plt.show()

plotting.parallel_coordinates(ddata, 'y')
plt.title('Parallel coordinates')
plt.show()
