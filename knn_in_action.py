from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

# custom functions
from core.algorithms import KNN


def clf(x_train, x_test, y_train, y_test):
    knn = KNN(k=3)
    knn.fit(x_train, y_train)
    y_predicted = knn.predict(x_test)
    accuracy = np.sum(y_test == y_predicted) / len(y_test)
    return y_predicted, accuracy


if __name__ == "__main__":
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=145)
    y_predict, accuracy_score = clf(X_train, X_test, Y_train, Y_test)
    print(accuracy_score)
