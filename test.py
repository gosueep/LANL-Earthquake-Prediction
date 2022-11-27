import joblib
from Score import readTestData, scoreModel
from Training import createModel
from WindowMaker import getWindows
from sklearn.metrics import mean_absolute_error

# i = int(input())

# clf = createModel(f'./windows/testsplit{i}.pkl', 0)
clf = createModel('train.csv', 0)
# X, y = getWindows('windows/testsplit9.pkl', )
test_X, test_y = joblib.load('test.csv')

print(clf.score(test_X, test_y))
print(mean_absolute_error(clf.predict(test_X), test_y))