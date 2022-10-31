import sklearn as sk
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor



from common import readData


X, y = readData('data/testsplit0.csv')
print('Train data 0 read in')

clf = RandomForestRegressor(max_samples=.2, verbose=1)
clf.fit(X, y)
print('Regressor trained :)')

joblib.dump(clf, 'output/model1.pkl')
print('Regressor saved!')

test_X, test_y = readData('data/testsplit8.csv')
print('Test Data 8 read in')
print(clf.score(test_X, test_y))

