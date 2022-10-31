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


# def makeWindows(filename):
#     X, y = readData('plot.csv')
#     windows = getWindows(X, 100)
#     print(windows)
#
#     joblib.dump(windows, 'windows/plot')


if __name__ == '__main__':
    X, y = readData('./data/testsplit0.csv')
    print('READ IN DATA')
    # X, y = readData('plot.csv')
    WINDOW_SIZE = 10000
    STEP_SIZE = 5000
    windows, y_values = getWindows(X, y, WINDOW_SIZE, STEP_SIZE)

    # joblib.dump([windows, y[:-WINDOW_SIZE]], 'windows/plot.pkl')
    print(len(windows))
    print(len(y[:-WINDOW_SIZE]))
    joblib.dump((windows, y_values), 'windows/testsplit0.pkl')




