import numpy as np
import copy

class MultilayerPerceptron:
    def __init__(self, layer_sizes, momentum=None):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.momentum = momentum
        self.errors = []

        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        if self.momentum is not None:
            # self.prev_weights = np.zeros()
            # self.prev_biases = np.zeros()
            self.prev_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
            self.prev_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, X):
        self.activations = [X]
        self.outputs = []
        for i in range(self.num_layers - 1):
            self.outputs.append(np.dot(self.activations[i], self.weights[i]) + self.biases[i])
            self.activations.append(self.sigmoid(self.outputs[i]))
        return self.activations[-1]
    
    def backpropagation(self, y, learning_rate):
        error = y - self.activations[-1]
        deltas = [error * self.sigmoid_derivative(self.activations[-1])]
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases
        for i in range(self.num_layers - 1):
            if self.momentum is not None:
                delta_w = learning_rate * np.dot(self.activations[i].T, deltas[i]) + self.momentum * self.prev_weights[i]
                delta_b = learning_rate * np.sum(deltas[i], axis=0, keepdims=True) + self.momentum * self.prev_biases[i]
                self.weights[i] += delta_w
                self.biases[i] += delta_b
                self.prev_weights[i] = copy.deepcopy(delta_w)
                self.prev_biases[i] = copy.deepcopy(delta_b)
            else:
                self.weights[i] += learning_rate * np.dot(self.activations[i].T, deltas[i])
                self.biases[i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
        
    def train(self, X, y, epochs, learning_rate, convergence_threshold=0.01):
        for _ in range(epochs):
            self.feedforward(X)
            self.backpropagation(y, learning_rate)

            self.errors.append(self.mse(y, self.activations[-1]))
            if self.mse(y, self.predict(X)) < convergence_threshold:
                print('Convergence reached at epoch ' + str(_ + 1) + '.')
                break
        
    def predict(self, X):
        return self.feedforward(X)

    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred)) 