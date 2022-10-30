import sklearn as sk
import numpy as np
from sklearn.ensemble import RandomForestRegressor



from common import readData


X, y = readData('plot.csv')
X, y = readData('data/testsplit0.csv')
X = X.reshape(-1, 1)

# print(X)
# print(X.shape, y.shape)
test_X, test_y = readData('data/testsplit8.csv')

clf = RandomForestRegressor()

clf.fit(X, y)

# print(clf.score(X, y))
print(clf.score(test_X, test_y))

