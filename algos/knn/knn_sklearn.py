import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from util import get_social_network_data

if __name__ == "__main__":
    dataset = get_social_network_data()
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
