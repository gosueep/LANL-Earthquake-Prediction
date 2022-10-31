import numpy as np
from common import readData
import joblib


def calcStats(window):
    stats = np.array([np.max(window), np.min(window),
                      np.mean(window),
                      np.std(window)
                      ])
    return stats


def getWindows(X, WINDOW_SIZE=100):
    windows = []

    for start in range(0, len(X) - WINDOW_SIZE):
        window = X[start:start+WINDOW_SIZE]
        features = calcStats(window)
        windows.append(features)

    windows = np.array(windows)
    return windows


# def makeWindows(filename):
#     X, y = readData('plot.csv')
#     windows = getWindows(X, 100)
#     print(windows)
#
#     joblib.dump(windows, 'windows/plot')


if __name__ == '__main__':
    X, y = readData('plot.csv')
    windows = getWindows(X, 100)
    print(windows)

    joblib.dump(windows, 'windows/plot.pkl')




