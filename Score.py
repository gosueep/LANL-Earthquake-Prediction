import sys
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
    if len(sys.argv) == 3:
        testSet, modelNum = sys.argv[1:]
        test_X, test_y = readTestData(f'windows/testsplit{testSet}.pkl')
        print(scoreModel(test_X, test_y, modelNum))
    elif len(sys.argv) == 2:
        modelNum = sys.argv[1]
        test_X, test_y = readTestData(f'windows/testsplit8.pkl')
        print(scoreModel(test_X, test_y, modelNum))
    else:
        test_X, test_y = readTestData(f'windows/testsplit.pkl')
        scores = []
        for modelNum in range(8):
            score = scoreModel(test_X, test_y, modelNum)
            print(score)
            scores.append(score)
        print('Average Score')
        print(sum(scores)/len(scores))