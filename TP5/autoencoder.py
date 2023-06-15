from multilayer_perceptron import MultilayerPerceptron

from utils.parser import *
from config import load_config_multilayer
import numpy as np
import matplotlib.pyplot as plt

letters = load_data('inputs/font.json')
monocromatic_cmap = plt.get_cmap('binary')
plt.imshow(to_bin_array(letters[0]), cmap=monocromatic_cmap)
plt.show()

# input = letters
# output = letters

# multi_layer_perceptron_config = load_config_multilayer()
# num_inputs = int(multi_layer_perceptron_config["number_of_inputs"])
# num_outputs = int(multi_layer_perceptron_config["number_of_outputs"])
# epochs = int(multi_layer_perceptron_config["epochs"])
# learning_rate = float(multi_layer_perceptron_config["learning_rate"])
# beta = float(multi_layer_perceptron_config["beta"])
# hidden_layers = multi_layer_perceptron_config["hidden_layers"]
# momentum = float(multi_layer_perceptron_config["momentum"])

# mlp = MultilayerPerceptron([num_inputs] + hidden_layers + [num_outputs], momentum)

# mlp.train(input, output, epochs, learning_rate)

# print("Errors: ", mlp.errors)


