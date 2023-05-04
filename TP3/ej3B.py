import numpy as np
from multilayer_perceptron import MultilayerPerceptron
from config import load_config
from utils.parser import parse_txt_file_3b

input = parse_txt_file_3b('input_files/TP3-ej3-digitos.txt')


ytrain = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

# create the perceptron
config = load_config()
multi_layer_perceptron_config = config['multilayer']

num_inputs = int(multi_layer_perceptron_config["number_of_inputs"])
num_outputs = int(multi_layer_perceptron_config["number_of_outputs"])
epochs = int(multi_layer_perceptron_config["epochs"])
learning_rate = float(multi_layer_perceptron_config["learning_rate"])
beta = float(multi_layer_perceptron_config["beta"])
hidden_layers = multi_layer_perceptron_config["hidden_layers"]
momentum = float(multi_layer_perceptron_config["momentum"])

mlp = MultilayerPerceptron([35] + [10] + [1], momentum)


matrix = np.reshape(input, (10, 35))

mlp.train(matrix, ytrain, epochs, learning_rate)

print("0:",mlp.predict(matrix[0]))
