from datetime import datetime
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import get_mnist_data


def entropy(y):
    """Calculate entropy for a binary (0-1) array of classes.

    Parameters
    ----------
    y: Array of labels/classes

    Returns
    -------
    entropy of y
    """
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


class TreeNode:
    """Set docstring here.
    For each feature (X_i), sort the feature and calculate all the possible split points of that feature,
    then calculate the entropy of y assuming we split on that feature.
    Split on the feature that would give the highest information gain.


    """

    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, Y):
        """
        for c in columns:
            condition = find_split(X, Y, c)
            Y_left = Y[X[c] meets condition]
            Y_right = Y[X[c] doesn't meet condition]
            information_gain = H(Y) - P(left)H(Y_left) - p(right)H(Y_right)
            select c corresponding to the highest information_gain
        after the best feature is determined, need to split the data
        X_left, Y_left, X_right, Y_right = split_by_best_attribute
        self.left_node = TreeNode()
        self.left_node.fit(X_left, Y_left)

        self.right_node = TreeNode()
        self.right_node.fit(X_right, Y_right)

        base case for recursion:
        1. if max_information_gain = 0: nothing to gain from splitting -> make
        that a leaf node. To predict for a leaf node, take the most likely class.
        2. to avoid overfitting, set a max_depth, when we hit max_depth, stop
        recursing, make a leaf node => every TreeNode must know its own depth,
        and max_depth.
        3. if there's only 1 sample in the dataset, predict that sample's label.
        4. if there's > 1 sample, but they all have the same label, predit that label.

        """
        if len(Y) == 1 or len(set(Y)) == 1:  # base cases 3 and 4
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]

        else:
            D = X.shape[1]
            cols = range(D)

            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, Y, col)
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split
            if max_ig == 0:  # base case 1
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())  # since Y is a binary 0-1 array

            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:  # base case 2:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:, best_col] < self.split].mean()),
                        np.round(Y[X[:, best_col] >= self.split].mean()),
                    ]
                else:
                    left_idx = X[:, best_col] < self.split
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = X[:, best_col] >= self.split
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)

    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        boundaries = np.nonzero(y_values[1:] != y_values[:-1])[0]

        best_split = None
        max_ig = 0
        for i in boundaries:
            split = (x_values[i] + x_values[i + 1]) / 2.0
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split):
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return entropy(y) - p0 * entropy(y0) - p1 * entropy(y1)

    def predict_one(self, x):
        if self.col is not None and self.split is not None:
            if x[self.col] < self.split:
                if self.left is not None:
                    p = self.left.predict_one(x)
                else:  # leaf node & prediction already computed
                    p = self.prediction[0]
            else:
                if self.right is not None:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == "__main__":
    X, Y = get_mnist_data()

    # since we're doing binary classification
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    Ntrain = int(len(Y) / 2)

    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = DecisionTree()
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
