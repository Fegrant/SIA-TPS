import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron

def plot_decision_boundary_2d(X, Y, perceptron: Perceptron, title):
    # Define the range of x-axis and y-axis
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a grid of points to evaluate the decision boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title + ' Decision Boundary')
    plt.show()


def plot_decision_boundary_3d(X, Y, perceptron: Perceptron):
    w0 = perceptron.weights[0]
    w1 = perceptron.weights[1]
    w2 = perceptron.weights[2]
    w3 = perceptron.weights[3]
    
    # Plot the data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)

    # Plot the decision boundary
    x1_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 10)
    x2_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    x3 = (-w1 * x1 - w2 * x2 - w0) / (w3)

    ax.plot_surface(x1, x2, x3)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title("Hiperplane decision boundary")
    plt.show()