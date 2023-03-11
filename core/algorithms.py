import numpy as np


def euclidean(x1, x2):
    dist = np.sqrt(np.sum(x1 - x2) ** 2)
    return dist


class KNN:
    def __init__(self, k=3):
        self.y_label = None
        self.X_train = None
        self.k = k

    def fit(self, x, y):
        self.X_train = x
        self.y_label = y

    def predict(self, x_new):
        y_prediction = [self.calculation(x) for x in x_new]
        return y_prediction

    def calculation(self, x):
        eq_dist = [euclidean(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(eq_dist)[:self.k]
        knn_result = [self.y_label[i] for i in k_idx]
        return max(knn_result, key=knn_result.count)
