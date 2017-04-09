import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import decomposition

NAMES = [
'y',
'x1', 'x2', 'x3',
'x4', 'x5', 'x6',
'x7', 'x8', 'x9',
'x10', 'x11', 'x12',
'x13', 'x14', 'x15'
]

data = pd.read_csv('./data/train.csv', names=NAMES, header=1)
datax = data.drop('y', axis=1)
datay = data.drop(NAMES[1:], axis=1)

# Drop correlated data.
datax = datax.drop(['x8', 'x9', 'x14'], axis=1)

datax.boxplot()
plt.figure()
pd.scatter_matrix(datax)
plt.show()

for n in NAMES[1:]:
    plt.figure()
    x = data[n]
    y = datay
    plt.scatter(data[n], datay)
    m,b = np.polyfit(x, y, 1) 
    plt.plot(x, y, 'yo', x, m*x+b, '--k') 
    plt.show()

pca = decomposition.PCA(n_components=15)
pca.fit(datax)
print(pca.components_)
ppp = pd.DataFrame(pca.components_)
ppp.boxplot()
plt.show()
