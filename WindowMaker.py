import os
from multiprocessing import Process

import numpy as np
from scipy import stats
from common import readData
import joblib


def calcStats(window):
    winMin = np.min(window)
    winMax = np.max(window)
    calculatedStats = np.array([winMax, winMin,
                                np.mean(window),
                                np.std(window),
                                winMax - winMin
                                # mode(window),
                                # np.median(window)
                                ])
    return calculatedStats


def getWindows(X, y, WINDOW_SIZE=100, STEP_SIZE=1, filename='test.out'):
    windows = []
    y_values = []

    for start in range(0, len(X) - WINDOW_SIZE, STEP_SIZE):
        window = X[start:start+WINDOW_SIZE]
        features = calcStats(window)
        windows.append(features)

        window_y = np.mean(y[start:start+WINDOW_SIZE])
        y_values.append(window_y)

        if start % 100000 == 0:
            print(f'Iteration: {start}')

    windows = np.array(windows)
    joblib.dump((windows, y_values), filename)
    return windows, y_values


def makeWindows(WINDOW_SIZE, STEP_SIZE, recreate=True):
    procs = []
    for i in range(0, 10):
        if recreate or not os.path.isfile(f'windows/testsplit{i}.pkl'):
            print(f'Making Window {i}')
            X, y = readData(f'data/testsplit{i}.csv')
            # windows, y_values = getWindows(X, y, WINDOW_SIZE, STEP_SIZE)
            # joblib.dump((windows, y_values), f'windows/testsplit{i}.pkl')

            proc = Process(target=getWindows, args=(X, y, WINDOW_SIZE, STEP_SIZE, f'windows/testsplit{i}.pkl'))
            procs.append(proc)
            proc.start()
            proc.join()


if __name__ == '__main__':
    WINDOW_SIZE = 150000
    STEP_SIZE   = 1000

    X, y = readData(f'data/train.csv')
    stopPoint = int(len(X) * 0.8)
    getWindows(X[:stopPoint], y[:stopPoint], WINDOW_SIZE, STEP_SIZE, 'train.csv')
    getWindows(X[stopPoint:], y[stopPoint:], WINDOW_SIZE, STEP_SIZE, 'test.csv')

    # makeWindows(WINDOW_SIZE, STEP_SIZE, recreate=False)

    # X, y = readData('./data/testsplit0.csv')
    # print('READ IN DATA')
    # windows, y_values = getWindows(X, y, WINDOW_SIZE, STEP_SIZE)
    #
    # print(f'Total Windows: {len(windows)}')
    # joblib.dump((windows, y_values), 'windows/testsplit0.pkl')


# ~ 3-4 mins for 500 step size

# W: 100000, SS: 10000 - .33
# W: 100000
# W: 1000000, 100000 - .498
# W: 1000000, 1000000


# all as one - 1000000, 10000