import numpy as np


# Reads data from a specified filepath
def readData(filename):
    X = []
    y = []
    with open(filename, 'r') as trainData:
        for line in trainData:
            line = line.strip().split(',')
            if len(line) != 2:
                continue
            acousticData, timeToImpact = line
            if acousticData == 'acoustic_data':     # if col header
                continue
            X.append(float(acousticData))
            y.append(float(timeToImpact))

        X = np.array(X)
        y = np.array(y)

        return X.reshape(-1, 1), y
