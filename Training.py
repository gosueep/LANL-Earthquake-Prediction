import os
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor


def createModel(windowPath, modelNum):
    X, y = joblib.load(windowPath)
    print('Train data read in')

    clf = RandomForestRegressor(max_samples=.2, verbose=3, n_jobs=-1)
    clf.fit(X, y)
    print('Regressor trained :)')

    joblib.dump(clf, f'output/model{modelNum}.pkl')
    print('Regressor saved!')

    return clf


def createAll(retrain=False):
    for modelNum in range(0, 8):
        file = f'windows/testsplit{modelNum}.pkl'
        outfile = f'output/model{modelNum}.pkl'
        print(file)

        if retrain or not os.path.isfile(outfile):
            createModel(file, modelNum)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == 'retrain':
            createAll(True)
        else:
            createAll(bool(sys.argv[1]))
    elif len(sys.argv) == 1:
        createAll()
