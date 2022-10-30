import numpy as np


# Reads data from a specified filepath
def readData(filename):
    X = []
    y = []
    with open(filename, 'r') as trainData:
        for line in trainData:
            acousticData, timeToImpact = line.strip().split(',')
            X.append(acousticData)
            y.append(timeToImpact)

        return np.array(X), np.array(y)
