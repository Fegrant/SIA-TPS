from perceptron import Perceptron
from utils.linearBoundary import plot_decision_boundary_2d
from config import load_config

import matplotlib.pyplot as plt
import numpy as np

# Load config
perceptron_config = load_config()

# Pull out simple perceptron config
simple_perceptron_config = perceptron_config["simple"]

num_inputs = int(simple_perceptron_config["number_of_inputs"])
epochs = int(simple_perceptron_config["epochs"])
learning_rate = float(simple_perceptron_config["learning_rate"])
accepted_error = float(simple_perceptron_config["accepted_error"])


# Creation, training and values of and perceptron
and_perceptron = Perceptron(num_inputs, epochs, learning_rate, accepted_error)

and_X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
and_y = np.array([-1, -1, -1, 1])

and_perceptron.train(and_X, and_y)

print("AND Weights: ", and_perceptron.weights[1:])
print("AND Bias: ", and_perceptron.weights[0])
print("AND Predictions: ", and_perceptron.predict(and_X))

print()

# Creation, training and values of xor perceptron
xor_perceptron = Perceptron(num_inputs, epochs, learning_rate, accepted_error)

xor_X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
xor_y = np.array([1, 1, -1, -1])

xor_perceptron.train(xor_X, xor_y)

print("XOR Weights: ", xor_perceptron.weights[1:])
print("XOR Bias: ", xor_perceptron.weights[0])
print("XOR Predictions: ", xor_perceptron.predict(xor_X))


# Plotting
plot_decision_boundary_2d(and_X, and_y, and_perceptron, "AND")