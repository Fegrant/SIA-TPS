import numpy as np

class MultilayerPerceptron:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.init_weights()
        self.init_biases()

    def init_weights(self):
        self.weights = np.zeros(self.n_inputs) 
    
    def init_biases(self):
        self.biases = np.ones(self.n_inputs, 1)
    
    def train(self):
        pass

    def predict(self, X):
        X = np.array(X).reshape(self.n_inputs, 1)
        return np.round(self.feedforward(X))

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return self.activation(x) * (1 - self.activation(x))
        
    def feedforward(self, X):
        hidden_output = self.activation(np.dot(self.weights, X) + self.biases)
        return hidden_output

    def backpropagation(self):
        pass