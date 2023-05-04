import numpy as np
from config import load_config
from multilayer_perceptron import MultilayerPerceptron, MLP

# Read from config.json file
config = load_config()
multi_layer_perceptron_config = config['multilayer']

# Read from config.json file
num_inputs = int(multi_layer_perceptron_config["number_of_inputs"])
num_outputs = int(multi_layer_perceptron_config["number_of_outputs"])
epochs = int(multi_layer_perceptron_config["epochs"])
learning_rate = float(multi_layer_perceptron_config["learning_rate"])
beta = float(multi_layer_perceptron_config["beta"])
hidden_layers = multi_layer_perceptron_config["hidden_layers"]
momentum = float(multi_layer_perceptron_config["momentum"])

# XOR inputs and outputs
X_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
y_train = np.array([[1], [1], [0], [0]])


# compute the XOR function using a MLP with 2 inputs, 2 hidden units and 1 output unit
mlp = MultilayerPerceptron(num_inputs, hidden_layers, num_outputs, momentum)

# fit the MLP
mlp.fit(X_train, y_train, epochs, learning_rate, beta)

# print the prediction for the four possible inputs
print("XOR prediction")
print("0, 1: ", mlp.predict(np.array([0, 1])))
print("1, 0: ", mlp.predict(np.array([1, 0])))
print("0, 0: ", mlp.predict(np.array([0, 0])))
print("1, 1: ", mlp.predict(np.array([1, 1])))
