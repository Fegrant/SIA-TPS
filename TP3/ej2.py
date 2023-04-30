import numpy as np

from utils.linearBoundary import plot_decision_boundary_3d
from utils.parser import parse_csv_file, parse_txt_file
from perceptron import SimpleLinealPerceptron, SimpleNonLinealPerceptron

learning_rate = 0.1
accepted_error = 0.5

input = parse_csv_file('input_files/TP3-ej2-conjunto.csv')

X = input[:,:-1]
y = input[:,-1]

lineal_perceptron = SimpleLinealPerceptron(np.shape(input[0])[0] - 1, learning_rate=learning_rate, max_epochs=20, accepted_error=accepted_error)

lineal_perceptron.train(X, y)

print("Lineal Weights: ", lineal_perceptron.weights[1:])
print("Lineal Bias: ", lineal_perceptron.weights[0])
print("Lineal Predictions: ", lineal_perceptron.predict(X))
# print("Lineal Error: ", lineal_perceptron.error(X))

plot_decision_boundary_3d(X, y, lineal_perceptron)