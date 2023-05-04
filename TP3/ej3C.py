import numpy as np

from sklearn.model_selection import train_test_split
from config import load_config, ex3c_noise
from multilayer_perceptron import MultilayerPerceptron
from utils.parser import parse_txt_file
# determinate the number from matrix using mlp

def add_noise(matrix, noise):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if np.random.rand() < noise:
                matrix[i][j] = 1 if matrix[i][j] == 0 else 0
    return matrix


def add_noise_2(matrix, pixels_to_change):
    for i in range(pixels_to_change):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 35)
        matrix[x][y] = 1 if matrix[x][y] == 0 else 0
    return matrix

Xraw = parse_txt_file('input_files/TP3-ej3-digitos.txt')

y = np.array([
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
noise = ex3c_noise()

# compute the XOR function using a MLP with 2 inputs, 2 hidden units and 1 output unit
mlp = MultilayerPerceptron([35] + hidden_layers + [10], momentum)

X = np.reshape(Xraw, (10, 35))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X_train = input_train[:,:-1]
# y_train = input_train[:,-1]

# X_test = input_test[:,:-1]
# y_test = input_test[:,-1]

print(X_test)
print(y_test)

set_size = len(X)
train_set_size = len(X_train)
test_set_size = set_size - train_set_size

# print(np.reshape(input, (10, 35)))
# print()

Xt = add_noise(X, noise)

mlp.train(X, y, epochs, learning_rate)

# print the prediction for the four possible inputs
print("Numbers prediction")


# Con ~<0.6 aunque el error es alto, podemos diferenciar el número
for i in np.arange(set_size):
    predicted_values = mlp.predict(X[i])[0]
    # Averiguo cual es el valor que quiero
    highest_error = 0
    for j in np.arange(10):
        if predicted_values[j] > highest_error and y[i][j] != 1:
            highest_error = predicted_values[j]
        if y[i][j] == 1:
            num_expected = j
    error = 1 - (predicted_values[num_expected] - highest_error) if predicted_values[num_expected] > highest_error else 1
    print('Error: {}'.format(error))