from utils.linearBoundary import plot_decision_boundary_3d
from utils.parser import parse_csv_file, parse_txt_file
from perceptron import SimpleLinealPerceptron, SimpleNonLinealPerceptron
from config import load_config
from normalize import feature_scaling

# Load config
perceptron_config = load_config()

# Pull out lineal perceptron config
lineal_perceptron_config = perceptron_config["lineal"]

num_inputs = int(lineal_perceptron_config["number_of_inputs"])
epochs = int(lineal_perceptron_config["epochs"])
learning_rate = float(lineal_perceptron_config["learning_rate"])
accepted_error = float(lineal_perceptron_config["accepted_error"])


input = parse_csv_file('input_files/TP3-ej2-conjunto.csv')

X = input[:,:-1]
y = input[:,-1]

# Creation, training and values of lineal perceptron
lineal_perceptron = SimpleLinealPerceptron(num_inputs, learning_rate, epochs, accepted_error)

mse = lineal_perceptron.train(X, y)

print("Lineal Weights: ", lineal_perceptron.weights[1:])
print("Lineal Bias: ", lineal_perceptron.weights[0])
print("Lineal Predictions: ", lineal_perceptron.predict(X))
print("Lineal MSE after training: ", mse)


# TODO: COMO LOS DATOS NO SON DE CLASIFICACION (NO SON 0 Y 1) NO SE PUEDE HACER UNA LINEA DE DECISION, POR LO QUE NO TIENE MUCHO SENTIDO ESTE GRAFICO DE HIPERPLANO
plot_decision_boundary_3d(X, y, lineal_perceptron)

# Pull out non lineal perceptron config
no_lineal_perceptron_config = perceptron_config["no-lineal"]

num_inputs = int(no_lineal_perceptron_config["number_of_inputs"])
epochs = int(no_lineal_perceptron_config["epochs"])
learning_rate = float(no_lineal_perceptron_config["learning_rate"])
accepted_error = float(no_lineal_perceptron_config["accepted_error"])
beta = float(no_lineal_perceptron_config["beta"])
activation_function = no_lineal_perceptron_config["activation_function"]

# Creation, training and values of non lineal perceptron
non_lineal_perceptron = SimpleNonLinealPerceptron(num_inputs, learning_rate, epochs, accepted_error, beta, activation_function)


# activation_function = 1 -> tanh
if activation_function == 1:
    ynorm = feature_scaling(y, -1, 1)
# activation_function = 2 -> logistic
elif activation_function == 2:
    ynorm = feature_scaling(y, 0, 1)
    

mse = non_lineal_perceptron.train(X, ynorm)

print("Non lineal Weights: ", non_lineal_perceptron.weights[1:])
print("Non lineal Bias: ", non_lineal_perceptron.weights[0])
print("Non lineal Predictions: ", non_lineal_perceptron.predict(X))
print("Non lineal nomralized y: ", ynorm)
print("Non lineal MSE after training: ", mse)
