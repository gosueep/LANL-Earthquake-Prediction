import os

import numpy as np
from common import readData
import joblib


def calcStats(window):
    stats = np.array([np.max(window), np.min(window),
                      np.mean(window),
                      np.std(window)
                      ])
    return stats


def getWindows(X, y, WINDOW_SIZE=100, STEP_SIZE=1):
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
    return windows, y_values


def makeWindows(WINDOW_SIZE, STEP_SIZE, recreate=False):
    for i in range(0, 10):
        if recreate or not os.path.isfile(f'windows/testsplit{i}.pkl'):
            X, y = readData(f'data/testsplit{i}.csv')
            windows, y_values = getWindows(X, y, WINDOW_SIZE, STEP_SIZE)
            joblib.dump((windows, y_values), f'windows/testsplit{i}.pkl')


if __name__ == '__main__':
    WINDOW_SIZE = 10000
    STEP_SIZE = 1000

    # makeWindows(WINDOW_SIZE, STEP_SIZE)

    X, y = readData('./data/testsplit0.csv')
    print('READ IN DATA')
    windows, y_values = getWindows(X, y, WINDOW_SIZE, STEP_SIZE)

    print(f'Total Windows: {len(windows)}')
    joblib.dump((windows, y_values), 'windows/testsplit0.pkl')




