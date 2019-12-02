from collections import Counter
from datetime import datetime

from scipy.spatial import distance
from scipy.stats import mode
import numpy as np

from util import get_mnist_data


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        dist = distance.cdist(self.X, X, 'euclidean')
        return mode(self.y[np.argsort(dist, axis=0)[:self.k]])[0][0]

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)


if __name__ == "__main__":
    X, Y = get_mnist_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    for k in (1, 2, 3, 4, 5):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print(f"Training time: {datetime.now() - t0}")

        t0 = datetime.now()
        print(f"Train accuracy: {knn.score(Xtrain, Ytrain)}")
