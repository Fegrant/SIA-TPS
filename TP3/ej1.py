from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

# Creation, training and values of and perceptron
and_perceptron = Perceptron(2)

and_X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
and_y = np.array([-1, -1, -1, 1])

and_perceptron.train(and_X, and_y)

print("AND Weights: ", and_perceptron.weights[1:])
print("AND Bias: ", and_perceptron.weights[0])
print("AND Predictions: ", and_perceptron.activation(and_X))

print()

# Creation, training and values of xor perceptron
xor_perceptron = Perceptron(2)

xor_X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
xor_y = np.array([1, 1, -1, -1])

xor_perceptron.train(xor_X, xor_y)

print("XOR Weights: ", xor_perceptron.weights[1:])
print("XOR Bias: ", xor_perceptron.weights[0])
print("XOR Predictions: ", xor_perceptron.activation(xor_X))


# Define the range of x-axis and y-axis
x_min, x_max = and_X[:, 0].min() - 1, and_X[:, 0].max() + 1
y_min, y_max = and_X[:, 1].min() - 1, and_X[:, 1].max() + 1

# Create a grid of points to evaluate the decision boundary
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = and_perceptron.activation(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# Plot the decision boundary and the data points
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(and_X[:, 0], and_X[:, 1], c=and_y, cmap=plt.cm.RdBu)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('AND Perceptron')
plt.show()


# Define the range of x-axis and y-axis
x_min, x_max = xor_X[:, 0].min() - 1, xor_X[:, 0].max() + 1
y_min, y_max = xor_X[:, 1].min() - 1, xor_X[:, 1].max() + 1

# Create a grid of points to evaluate the decision boundary
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = xor_perceptron.activation(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(xor_X[:, 0], xor_X[:, 1], c=xor_y, cmap=plt.cm.RdBu)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('XOR Perceptron')
plt.show()