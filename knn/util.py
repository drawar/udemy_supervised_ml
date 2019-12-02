import numpy as np
import pandas as pd


def get_mnist_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv("data/mnist.csv")
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def get_social_network_data():
    print("Reading in and transforming data...")
    df = pd.read_csv("data/social_network.csv")
    return df


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5
    X[50:100] = np.random.random((50, 2)) / 2
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])
    Y = np.array([0] * 100 + [1] * 100)
    return X, Y
