import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    np.random.seed(100)
    n1 = 80
    mean1 = [1.5, 3]
    cov1 = [[0.5, 0], [0, 0.5]]
    n2 = 10
    mean2 = [3, 2]
    cov2 = [[0.1, 0], [0, 1]]
    n3 = 10
    mean3 = [0, 1]
    cov3 = [[1, 0], [0, 1]]
    features = [np.random.multivariate_normal(mean1, cov1, size=n1),
            np.random.multivariate_normal(mean2, cov2, size=n2),
            np.random.multivariate_normal(mean3, cov3, size=n3),
           ]
    features = np.concatenate(features, axis=0)
    labels = np.array([0]*n1 + [1]*n2 + [2]*n3)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=100, test_size=0.5)
    model = joblib.load('model.pkl')
    print(sum(model.predict(X_test)==y_test))

if __name__ == "__main__":
    main()