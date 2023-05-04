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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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
learning_rates = np.arange(0.05, 0.8, 0.05)
mean_errors_3b = []
for learning_rate in learning_rates:
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        mlp = MultilayerPerceptron([35] + [10] + [1])
        mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.001)
        errors = []
        for i in range(len(X_test)):
            prediction = mlp.predict(X_test[i])
            error = np.abs(prediction - y_test[i])
            errors.append(error)
    mean_errors_3b.append(np.mean(errors))
       

plt.plot(learning_rates, mean_errors_3b)
plt.xlabel('Learning rate')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b1.png')


# plot mean error changing inner layer size
inner_layer_sizes = np.arange(5, 20, 1)
mean_errors_3b2 = []
for inner_layer_size in inner_layer_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    mlp = MultilayerPerceptron([35] + inner_layer_size + [1])
    mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.001)
    errors = []
    for i in range(len(X_test)):
        prediction = mlp.predict(X_test[i])
        error = np.abs(prediction - y_test[i])
        errors.append(error)
    mean_errors_3b2.append(np.mean(errors))

plt.clf()
plt.plot(inner_layer_sizes, mean_errors_3b2)
plt.xlabel('Inner layer size')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b2.png')

# plot mean error changing epochs
epochs = np.arange(100, 10000, 100)
mean_errors_3b3 = []
for epoch in epochs:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    mlp = MultilayerPerceptron([35] + [10] + [1])
    mlp.train(X_train, y_train, epoch, learning_rate, convergence_threshold=0.001)
    errors = []
    for i in range(len(X_test)):
        prediction = mlp.predict(X_test[i])
        error = np.abs(prediction - y_test[i])
        errors.append(error)
    mean_errors_3b3.append(np.mean(errors))

plt.clf()
plt.plot(epochs, mean_errors_3b3)
plt.xlabel('Epochs')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b3.png')

# plot mean error with 2 hidden layers and change neurons
inner_layer_sizes = np.arange(5, 20, 1)
mean_errors_3b4 = []
for learning_rate in learning_rates:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for i in range(10):
        mlp = MultilayerPerceptron([35] + [inner_layer_sizes, inner_layer_sizes] + [1])
        mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.001)
        errors = []
        for i in range(len(X_test)):
            prediction = mlp.predict(X_test[i])
            error = np.abs(prediction - y_test[i])
            errors.append(error)
    mean_errors_3b4.append(np.mean(errors))

plt.clf()
plt.plot(inner_layer_sizes, mean_errors_3b4)
plt.xlabel('Inner layers size')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b4.png')