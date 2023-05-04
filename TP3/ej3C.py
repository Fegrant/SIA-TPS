import numpy as np

from sklearn.model_selection import train_test_split
from config import load_config
from multilayer_perceptron import MultilayerPerceptron
from utils.parser import parse_txt_file_3c
# determinate the number from matrix using mlp


input = parse_txt_file_3c('input_files/TP3-ej3-digitos.txt')

ytrain = np.array([
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 1, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0], 
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0 ,1 ,0 ,0 ,0 ,0 ,0 ,0], 
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0 ,0 ,0 ,1 ,0 ,0 ,0 ,0], 
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0 ,0 ,0 ,0 ,0 ,1 ,0 ,0], 
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,1]
])

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

# compute the XOR function using a MLP with 2 inputs, 2 hidden units and 1 output unit
mlp = MultilayerPerceptron([35] + hidden_layers + [10], momentum)

input_train, input_test = train_test_split(input, test_size=0.2)

X_train = input_train[:][0]
y_train = input_train[:][1]

X_test = input_test[0]
y_test = input_test[1]

# X_train = input_train[:,:-1]
# y_train = input_train[:,-1]

# X_test = input_test[:,:-1]
# y_test = input_test[:,-1]

print(X_test)
print(y_test)

set_size = len(input)
train_set_size = len(X_train)

# print(np.reshape(input, (10, 35)))
# print()

matrix_train = np.reshape(X_train, (train_set_size, train_set_size * 35))
matrix_test = np.reshape(X_test, (set_size-train_set_size, (set_size-train_set_size) * 35))

mlp.train(matrix_train, y_train, epochs, learning_rate)

# print the prediction for the four possible inputs
print("Numbers prediction")
for i in np.arange(10):
    predicted_num = 0
    predicted_values = mlp.predict(matrix[i])
    # print(predicted_values)
    for j in np.arange(10):
        if i == 0:
            predicted_num += 10 * predicted_values[0][j]
        else:
            predicted_num += j * predicted_values[0][j]
    print('{}: {}'.format(i, predicted_num))