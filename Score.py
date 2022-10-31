import joblib


def readTestData(filename):
    test_X, test_y = joblib.load(filename)
    print('Test Data 8 read in')
    return test_X, test_y
    # print(clf.score(test_X, test_y))


def scoreModel(test_X, test_y, modelNum):
    clf = joblib.load(f'output/model{modelNum}.pkl')
    return clf.score(test_X, test_y)


def scoreAll(test_X, test_y):
    scores = []
    for modelNum in range(0, 8):
        score = scoreModel(test_X, test_y, modelNum)
        scores.append(score)
        print(score)

    return scores


if __name__ == '__main__':

    test_X, test_y = readTestData('windows/testsplit0.pkl')
    print(scoreModel(test_X, test_y, 0))

    # if len(sys.argv) == 2:
    #     if sys.argv[1].lower() == 'retrain':
    #         createAll(True)
    #     else:
    #         createAll(bool(sys.argv[1]))
    # elif len(sys.argv) == 1:
    #     createAll()