from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import numpy as np

from util import get_mnist_data


class NaiveBayes(object):
    def fit(self, X, y, smoothing=1e-3):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()

        # calculate the gaussian and prior for each class based on data
        labels = set(y)
        for c in labels:
            X_c = X[y == c]
            self.gaussians[c] = {
                "mean": X_c.mean(axis=0),
                "cov": np.cov(X_c.T) + np.eye(D) * smoothing,
            }
            self.priors[c] = np.mean(y == c)

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)  # K is number of classes
        P = np.zeros((N, K))

        for c, g in self.gaussians.items():
            mean, cov = g["mean"], g["cov"]
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(
                self.priors[c]
            )  # calculate log-likelihood
        return np.argmax(P, axis=1)


if __name__ == "__main__":
    X, Y = get_mnist_data(10000)
    Ntrain = int(len(Y) / 2)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = NaiveBayes()

    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time: ", datetime.now() - t0)

    t0 = datetime.now()
    print("Train accuracy: ", model.score(Xtrain, Ytrain))
    print(
        "Time to compute train accuracy: ",
        datetime.now() - t0,
        "Train size: ",
        len(Ytrain),
    )

    t0 = datetime.now()
    print("Test accuracy: ", model.score(Xtest, Ytest))
    print(
        "Time to compute test accuracy: ",
        datetime.now() - t0,
        "Test size: ",
        len(Ytest),
    )
