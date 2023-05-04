import numpy as np
from multilayer_perceptron import MultilayerPerceptron
from config import load_config
from utils.parser import parse_txt_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def number(matrix, matrices):
    for i in range(len(matrices)):
        if np.array_equal(matrix, matrices[i]):
            return i

# create the perceptron
config = load_config()
multi_layer_perceptron_config = config['multilayer']

num_inputs = int(multi_layer_perceptron_config["number_of_inputs"])
num_outputs = int(multi_layer_perceptron_config["number_of_outputs"])
epochs = int(multi_layer_perceptron_config["epochs"])
learning_rate = float(multi_layer_perceptron_config["learning_rate"])
beta = float(multi_layer_perceptron_config["beta"])
convergence_threshold = float(multi_layer_perceptron_config["convergence_threshold"])

mlp = MultilayerPerceptron([35] + [10] + [1])

input = parse_txt_file('input_files/TP3-ej3-digitos.txt')

y= np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

X = input.reshape(10, 35)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.001)

# errors = []
# for i in range(len(X_test)):
#     print("Numero:", number(X_test[i], X))
#     prediction = mlp.predict(X_test[i])
#     print("Predictions: ", prediction)
#     print("Expected: ", y_test[i])
#     error = np.abs(prediction - y_test[i])
#     print("Error: ", np.abs(mlp.predict(X_test[i]) - y_test[i]))
#     errors.append(error)

# print("Mean error: ", np.mean(errors))


# plot mean error change learning rate
learning_rates = np.arange(0.01, 0.2, 0.01)
mean_errors = []
for learning_rate in learning_rates:
    mlp = MultilayerPerceptron([35] + [10] + [1])
    mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.0001)
    errors = []
    for i in range(len(X_test)):
        prediction = mlp.predict(X_test[i])
        error = np.abs(prediction - y_test[i])
        errors.append(error)
    mean_errors.append(np.mean(errors))

plt.plot(learning_rates, mean_errors)
plt.xlabel('Learning rate')
plt.ylabel('Mean error')
plt.show()


