import numpy as np
import matplotlib.pyplot as plt
from config import load_config
from multilayer_perceptron import MultilayerPerceptron

# Read from config.json file
config = load_config()
multi_layer_perceptron_config = config['multilayer']

# Read from config.json file
num_inputs = int(multi_layer_perceptron_config["number_of_inputs"])
num_outputs = int(multi_layer_perceptron_config["number_of_outputs"])
learning_rate = float(multi_layer_perceptron_config["learning_rate"])
beta = float(multi_layer_perceptron_config["beta"])
hidden_layers = multi_layer_perceptron_config["hidden_layers"]
momentum = float(multi_layer_perceptron_config["momentum"])

# XOR inputs and outputs
X_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
y_train = np.array([[1], [1], [0], [0]])


# compute the XOR function using a MLP with 2 inputs, 2 hidden units and 1 output unit
mlp = MultilayerPerceptron(num_inputs, hidden_layers, num_outputs, momentum)

# MSE vs Epochs
errors = []
for i in range(1, 10):
    epochs = i * 100
    mlp.fit(X_train, y_train, epochs, learning_rate, beta)
    errors.append(mlp.mse(y_train, mlp.predict(X_train)))

plt.plot(errors)
plt.legend()
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.title('MSE vs Epochs')
plt.show()
