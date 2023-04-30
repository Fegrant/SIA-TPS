import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron, SimpleLinealPerceptron, SimpleNonLinealPerceptron


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
    # Define the range of x-axis, y-axis and z-axis
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    # Create a grid of points to evaluate the decision boundary
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), np.arange(z_min, z_max, 0.1))

    z_plot = (perceptron.weights[0] - perceptron.weights[1] * xx - perceptron.weights[2] * yy) / perceptron.weights[3]

    print(z_plot)

    # Plot the decision boundary and the data points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.Set1, edgecolor='k', s=100)
    
    # ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    ax.plot_wireframe(xx, yy, z_plot, linewidth=1, alpha=0.8)
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.title('Decision Boundary')
    plt.show()