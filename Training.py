import sklearn as sk
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common import readData


X, y = readData('data/testsplit0.csv')
test_X, test_y = readData('data/testsplit8.csv')

clf = RandomForestClassifier()

clf.fit(X, y)

print(clf.score(test_X, test_y))

