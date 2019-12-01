from collections import Counter
from datetime import datetime

from sortedcontainers import SortedList
import numpy as np

from util import get_data


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        pred = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):
                diff = x - xt
                dist = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add(self.y[j])
                else:
                    if dist < sl[-1][0]:
                        del sl[-1]
                        sl.add(self.y[j])

            counts = Counter(sl)
            pred[i] = counts.most_common()[0][0]
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)


if __name__ == "__main__":
    X, Y = get_data(2000)
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