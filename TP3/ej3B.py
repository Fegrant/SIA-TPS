import numpy as np
from multilayer_perceptron import MultilayerPerceptron
from config import load_config
from utils.parser import parse_txt_file
from sklearn.model_selection import train_test_split



# create the perceptron
config = load_config()
multi_layer_perceptron_config = config['multilayer']

num_inputs = int(multi_layer_perceptron_config["number_of_inputs"])
num_outputs = int(multi_layer_perceptron_config["number_of_outputs"])
epochs = int(multi_layer_perceptron_config["epochs"])
learning_rate = float(multi_layer_perceptron_config["learning_rate"])
beta = float(multi_layer_perceptron_config["beta"])
hidden_layers = multi_layer_perceptron_config["hidden_layers"]

mlp = MultilayerPerceptron([35] + [10] + [1])

input = parse_txt_file('input_files/TP3-ej3-digitos.txt')

ytrain = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

X = input.reshape(10, 35)

X_train, X_test, y_train, y_test = train_test_split(X, ytrain, test_size=0.2)

mlp.train(X_train, y_train, epochs, learning_rate)

print("Multilayer Weights: ", mlp.weights)
print("Multilayer Bias: ", mlp.bias)

print("Test: ", X_test)
print("Predictions: ", mlp.predict(X_test))
print("Expected: ", y_test)