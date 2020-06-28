from helpers_05_08 import visualize_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt


def plot_iris_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3]):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes()
    ax.contourf(x1, x2, y_pred, alpha=0.3, cmap='rainbow') ## color the areas based on prediction
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
    ax.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18, rotation=0)
    ax.legend()


def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3]):

    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes()
    ax.contourf(x1, x2, y_pred, alpha=0.3, cmap='rainbow')
    unique_y = np.unique(y)
    style_list = ['yo', 'ro', 'bs', 'g^', 'k*', 'r>']
    for i in range(len(unique_y)):
        ax.plot(X[:, 0][y==unique_y[i]], X[:, 1][y==unique_y[i]], style_list[i])
   
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18, rotation=0)
    ax.axis(axes)



