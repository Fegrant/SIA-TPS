import numpy as np

class MultilayerPerceptron:
    def __init__(self, n_inputs, hidden_layers, n_outputs):
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs
        self.init_weights()

    # initialize weights with random values
    def init_weights(self):
        self.weights = np.random.randn(self.n_inputs, self.hidden_layers[0])
        for i in range(1, len(self.hidden_layers)):
            self.weights = np.concatenate((self.weights, np.random.randn(self.hidden_layers[i-1], self.hidden_layers[i])), axis=1)
        self.weights = np.concatenate((self.weights, np.random.randn(self.hidden_layers[-1], self.n_outputs)), axis=1)
        
    def fit(self, X, y, epochs=1000, learning_rate=0.1, beta=0.1):
        for epoch in range(epochs):
            for input, solution in zip(X, y):
                prediction = self.feedforward(input)
                self.backpropagation(input, solution, prediction, learning_rate, beta)
                

    def feedforward(self, X):
        pass

    def backpropagation(self, X, y):
        pass


    def predict(self, X):
        pass

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_activation_derivative(self, x):
        return self.sigmoid_activation(x) * (1 - self.sigmoid_activation(x))
        

    def mse(self, target, prediction):
        return np.square(np.subtract(target, prediction)).mean()