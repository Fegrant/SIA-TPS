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
learning_rates = np.arange(0.001, 0.1, 0.01)
mean_errors_3b = []
std_errors_3b = []
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
    std_errors_3b.append(np.std(errors))
       
plt.clf()
plt.errorbar(learning_rates, mean_errors_3b, yerr=std_errors_3b, fmt='o')
plt.xlabel('Learning rate')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b1.png')


# plot mean error changing inner layer size
inner_layer_sizes = np.arange(5, 20, 1)
mean_errors_3b2 = []
std_errors_3b2 = []
for inner_layer_size in inner_layer_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    mlp = MultilayerPerceptron([35] + [inner_layer_size] + [1])
    mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.001)
    errors = []
    for i in range(len(X_test)):
        prediction = mlp.predict(X_test[i])
        error = np.abs(prediction - y_test[i])
        errors.append(error)
    mean_errors_3b2.append(np.mean(errors))
    std_errors_3b2.append(np.std(errors))

plt.clf()
plt.errorbar(inner_layer_sizes, mean_errors_3b2, yerr=std_errors_3b2, fmt='o')
plt.xlabel('Inner layer size')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b2.png')

# plot mean error changing epochs
epochs = np.arange(100, 10000, 100)
mean_errors_3b3 = []
std_errors_3b3 = []
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
    std_errors_3b3.append(np.std(errors))

plt.clf()
plt.errorbar(epochs, mean_errors_3b3, yerr=std_errors_3b3, fmt='o')
plt.xlabel('Epochs')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b3.png')

# plot mean error with 2 hidden layers and change neurons
inner_layer_sizes = np.arange(5, 20, 1)
mean_errors_3b4 = []
std_errors_3b4 = []
epochs = 1000
for inner_layer_size in inner_layer_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for i in range(10):
        mlp = MultilayerPerceptron([35] + [inner_layer_size, inner_layer_size] + [1])
        mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.001)
        errors = []
        for i in range(len(X_test)):
            prediction = mlp.predict(X_test[i])
            error = np.abs(prediction - y_test[i])
            errors.append(error)
    mean_errors_3b4.append(np.mean(errors))
    std_errors_3b4.append(np.std(errors))

plt.clf()
plt.errorbar(mean_errors_3b4, mean_errors_3b4, yerr=std_errors_3b4, fmt='o')
plt.xlabel('Inner layers size')
plt.ylabel('Mean error')
plt.savefig('plots/TP3-ej3b4.png')


# Define the learning rates and epochs
learning_rates = [0.001, 0.01, 0.05, 0.1]
epochs_array = np.arange(0, 5000, 100)

# Initialize an empty list to store the mean errors
mean_errors = []

# Loop over the learning rates and epochs
for learning_rate in learning_rates:
    error_list = []
    for epochs in epochs_array:
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        # Train the MLP with the given learning rate and epochs
        mlp = MultilayerPerceptron([35] + [10] + [1])
        mlp.train(X_train, y_train, epochs, learning_rate, convergence_threshold=0.0001)
        
        # Calculate the errors on the test set
        errors = []
        for i in range(len(X_test)):
            prediction = mlp.predict(X_test[i])
            error = np.abs(prediction - y_test[i])
            errors.append(error)
        
        # Calculate the mean error and add it to the list
        mean_error = np.mean(errors)
        error_list.append(mean_error)
    
    # Add the list of errors for this learning rate to the main list
    mean_errors.append(error_list)

# Plot the lines for the different learning rates
for i in range(len(learning_rates)):
    plt.plot(epochs_array, mean_errors[i], label='lr={}'.format(learning_rates[i]))

# Set the axis labels and legend
plt.xlabel('Epochs')
plt.ylabel('Mean Error')
plt.legend()
plt.savefig('plots/TP3-ej3b5.png')
