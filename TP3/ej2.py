from utils.parser import parse_csv_file
from perceptron import SimpleLinealPerceptron, SimpleNonLinealPerceptron, UpdateMode
from config import load_config, ex2_test_size
from normalize import feature_scaling, inverse_feature_scaling

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Load config
perceptron_config = load_config()
test_size = float(ex2_test_size())
print(test_size)

# Pull out lineal perceptron config
lineal_perceptron_config = perceptron_config["lineal"]

num_inputs = int(lineal_perceptron_config["number_of_inputs"])
epochs = int(lineal_perceptron_config["epochs"])
learning_rate = float(lineal_perceptron_config["learning_rate"])
accepted_error = float(lineal_perceptron_config["accepted_error"])

input = parse_csv_file('input_files/TP3-ej2-conjunto.csv')

# Extract the input and output data
x = input[:, 0:3]
y = input[:, 3]

X = np.array(x)
Y = np.array(y)

# input_train, input_test = train_test_split(input, test_size=test_size)

# X_train = input_train[:,:-1]
# y_train = input_train[:,-1]

# X_test = input_test[:,:-1]
# y_test = input_test[:,-1]

# X_train, X_test = train_test_split(X, test_size=0.2)

# Creation, training and values of lineal perceptron
lineal_perceptron = SimpleLinealPerceptron(num_inputs, learning_rate, epochs, accepted_error, UpdateMode.BATCH)

mse, errors = lineal_perceptron.train(X, Y)

print("Lineal Weights: ", lineal_perceptron.weights[1:])
print("Lineal Bias: ", lineal_perceptron.weights[0])
print("Lineal Predictions: ", lineal_perceptron.predict(X))
print("Lineal MSE after training: ", mse)

#PLOT ERROR

x = list(range(len(errors)))
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("MSE - Lineal")
plt.plot(x, errors, label='Error')
plt.ylim((0, None))
plt.legend()
plt.show()


# TODO: COMO LOS DATOS NO SON DE CLASIFICACION (NO SON 0 Y 1) NO SE PUEDE HACER UNA LINEA DE DECISION, POR LO QUE NO TIENE MUCHO SENTIDO ESTE GRAFICO DE HIPERPLANO
# plot_decision_boundary_3d(X, y, lineal_perceptron)

# Pull out non lineal perceptron config
no_lineal_perceptron_config = perceptron_config["no-lineal"]

num_inputs = int(no_lineal_perceptron_config["number_of_inputs"])
epochs = int(no_lineal_perceptron_config["epochs"])
learning_rate = float(no_lineal_perceptron_config["learning_rate"])
accepted_error = float(no_lineal_perceptron_config["accepted_error"])
beta = float(no_lineal_perceptron_config["beta"])
activation_function = no_lineal_perceptron_config["activation_function"]

# Creation, training and values of non lineal perceptron
non_lineal_perceptron = SimpleNonLinealPerceptron(num_inputs, learning_rate, epochs, accepted_error, UpdateMode.BATCH, beta, activation_function)


# # activation_function = 1 -> tanh
# if activation_function == 1:
#     ynorm = feature_scaling(Y, -1, 1)
# # activation_function = 2 -> logistic
# elif activation_function == 2:
#     ynorm = feature_scaling(Y, 0, 1)
    

mse, errors = non_lineal_perceptron.train(X, Y)

print("Non lineal Weights: ", non_lineal_perceptron.weights[1:])
print("Non lineal Bias: ", non_lineal_perceptron.weights[0])

prediction = non_lineal_perceptron.predict(X)
prediction_denormalized = inverse_feature_scaling(prediction, 0, 1, np.min(Y), np.max(Y))


print("Non lineal Predictions: ", prediction_denormalized)
print("Non lineal Orinal Y: ", Y)
print("Non lineal MSE after training: ", mse)

#PLOT ERROR

x = list(range(len(errors)))
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("MSE", fontsize=12)
if activation_function == 1:
    plt.title("MSE - Tangente Hiperbolica")
elif activation_function == 2:
    plt.title("MSE - Logistica")
plt.plot(x, errors, label='Error')
plt.legend()
plt.show()

