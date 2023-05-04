import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from config import load_config, ex3c_noise
from multilayer_perceptron import MultilayerPerceptron
from utils.parser import parse_txt_file
import csv
import copy
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

X = np.reshape(Xraw, (10, 35))

############################### Learning_rate #################################

runs = 10
# learning_rates = np.arange(0.05, 0.8, 0.05)
# errors = []
# errors_std = []
# for rate in learning_rates:
#     rate_errors = []
#     for i in np.arange(runs):
#         mlp = MultilayerPerceptron([35] + hidden_layers + [10], momentum)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#         set_size = len(X)
#         train_set_size = len(X_train)
#         test_set_size = set_size - train_set_size

#         # Xt = add_noise(X, noise)
#         mlp.train(X, y, epochs, rate)

#         mlp_errors = []

#         for i in np.arange(set_size):
#             predicted_values = mlp.predict(X[i])[0]
#             # Averiguo cual es el valor que quiero
#             highest_error = 0
#             for j in np.arange(10):
#                 if predicted_values[j] > highest_error and y[i][j] != 1:
#                     highest_error = predicted_values[j]
#                 if y[i][j] == 1:
#                     num_expected = j
#             error = 1 - (predicted_values[num_expected] - highest_error) if predicted_values[num_expected] > highest_error else 1
#             mlp_errors.append(error)
#         rate_errors.append(np.mean(mlp_errors))
#     rate_errors_np = np.array(rate_errors)
#     errors.append(np.mean(rate_errors_np))
#     errors_std.append(np.std(rate_errors_np))

# print(errors)
# print(errors_std)

# with open('learning_rates_data.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['learning_rate, error, error_std'])
#     rate_count = 0
#     rate_idx = 0
#     for i in np.arange(len(errors)):
#         writer.writerow([learning_rates[rate_idx], errors[i], errors_std[i]])
#         rate_count += 1
#         if rate_count == runs:
#             rate_idx += 1
#             rate_count = 0

# Con ~<0.6 aunque el error es alto, podemos diferenciar el número
# for i in np.arange(set_size):
#     predicted_values = mlp.predict(X[i])[0]
#     # Averiguo cual es el valor que quiero
#     highest_error = 0
#     for j in np.arange(10):
#         if predicted_values[j] > highest_error and y[i][j] != 1:
#             highest_error = predicted_values[j]
#         if y[i][j] == 1:
#             num_expected = j
#     error = 1 - (predicted_values[num_expected] - highest_error) if predicted_values[num_expected] > highest_error else 1
#     print('Error: {}'.format(error))

#################################### Noise #####################################

# noise_rates = np.arange(0, 0.5, 0.05)
# errors = []
# errors_std = []
# for rate in noise_rates:
#     rate_errors = []
#     for i in np.arange(runs):
#         mlp = MultilayerPerceptron([35] + hidden_layers + [10], momentum)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#         set_size = len(X)
#         train_set_size = len(X_train)
#         test_set_size = set_size - train_set_size

#         Xt = add_noise(copy.deepcopy(X), rate)
#         mlp.train(Xt, y, epochs, learning_rate)

#         mlp_errors = []

#         for i in np.arange(set_size):
#             predicted_values = mlp.predict(X[i])[0]
#             # Averiguo cual es el valor que quiero
#             highest_error = 0
#             for j in np.arange(10):
#                 if predicted_values[j] > highest_error and y[i][j] != 1:
#                     highest_error = predicted_values[j]
#                 if y[i][j] == 1:
#                     num_expected = j
#             error = 1 - (predicted_values[num_expected] - highest_error) if predicted_values[num_expected] > highest_error else 1
#             mlp_errors.append(error)
#         rate_errors.append(np.mean(mlp_errors))
#     rate_errors_np = np.array(rate_errors)
#     errors.append(np.mean(rate_errors_np))
#     errors_std.append(np.std(rate_errors_np))

# print(errors)
# print(errors_std)

# with open('noise_data.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['learning_rate, error, error_std'])
#     for i in np.arange(len(errors)):
#         writer.writerow([noise_rates[i], errors[i], errors_std[i]])

inner_perceptrons = np.arange(10, 50, 5)
errors = []
errors_std = []
for perceptrons in inner_perceptrons:
    rate_errors = []
    for i in np.arange(runs):
        mlp = MultilayerPerceptron([35] + [perceptrons] + [perceptrons] + [10], momentum)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        set_size = len(X)
        train_set_size = len(X_train)
        test_set_size = set_size - train_set_size

        # Xt = add_noise(X, noise)
        mlp.train(X, y, epochs, learning_rate)

        mlp_errors = []

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
            mlp_errors.append(error)
        rate_errors.append(np.mean(mlp_errors))
    rate_errors_np = np.array(rate_errors)
    errors.append(np.mean(rate_errors_np))
    errors_std.append(np.std(rate_errors_np))

print(errors)
print(errors_std)

with open('inner_layers.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['perceptron_amount', 'error', 'error_std'])
    rate_count = 0
    rate_idx = 0
    for i in np.arange(len(errors)):
        writer.writerow([inner_perceptrons[i], errors[i], errors_std[i]])