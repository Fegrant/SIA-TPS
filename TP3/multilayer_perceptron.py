import numpy as np
import copy

class MultilayerPerceptron:
    def __init__(self, n_inputs, hidden_layers, n_outputs, momentum=None):
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs
        self.momentum = momentum
        self.init_weights()

    # initialize weights with random values
    def init_weights(self):
        self.weights = np.random.randn(self.n_inputs, self.hidden_layers[0])
        for i in range(1, len(self.hidden_layers)):
            self.weights = np.concatenate((self.weights, np.random.randn(self.hidden_layers[i-1], self.hidden_layers[i])), axis=1)
        self.weights = np.concatenate((self.weights, np.random.randn(self.hidden_layers[-1], self.n_outputs)), axis=1)
        if self.momentum is not None:
            self.prev_weights = copy.deepcopy(self.weights)

    def fit(self, X, y, epochs=1000, learning_rate=0.1, beta=0.1, momentum=None):
        self.beta = beta
        self.momentum = momentum
        
        if self.momentum is not None:
            self.prev_weights = np.zeros(self.weights.shape)

        sum_errors = np.array([], dtype=np.float64)

        # for each epoch train the network on each sample and update weights
        for epoch in range(epochs):
            # Iterate over the samples
            for i in range(X.shape[0]):
                # Feedforward the input and compute the activations
                self.feedforward(X[i])
                # Backpropagate the error and update the weights
                self.backpropagation(X[i], y[i], learning_rate)
            
            # Compute the MSE after each epoch and append it to the list (for plotting purposes, each epoch is a point in the plot)
            sum_errors = np.append(sum_errors, np.sum((y - self.feedforward(X))**2)/X.shape[0])
        


    # feedforward the input through the network and return the output layer
    def feedforward(self, X):
        self.activations = [X]
        for i in range(len(self.hidden_layers) + 1):
            self.activations.append(self.sigmoid_activation(np.dot(self.activations[i], self.weights[i])))
        return self.activations[-1]
        

    # backpropagate the error and update the weights
    def backpropagation(self, X, Y, learning_rate=0.1):
        # Compute the error at the output layer
        output_error = Y - self.activations[-1]

        # Compute the gradient at the output layer
        output_layer_gradient = output_error * self._sigmoid_activation_derivative(self.activations[-1])

        # Compute the gradient at the hidden layers
        hidden_layer_gradients = []
        for i in range(len(self.hidden_layers), 0, -1):
            if i == len(self.hidden_layers):
                gradient = np.dot(output_layer_gradient, self.weights[:, sum(self.hidden_layers[:-1]):].T) * self._sigmoid_activation_derivative(self.activations[i])
            else:
                gradient = np.dot(hidden_layer_gradients[-1], self.weights[:, sum(self.hidden_layers[i:]):sum(self.hidden_layers[i+1:])].T) * self._sigmoid_activation_derivative(self.activations[i])
            hidden_layer_gradients.append(gradient)

        # Reverse the order of the gradients for convenience
        hidden_layer_gradients.reverse()

        # Update the weights using the computed gradients
        for i in range(len(self.hidden_layers)+1):
            if i == 0:
                weight_update = np.dot(X.T, hidden_layer_gradients[i])
            elif i == len(self.hidden_layers):
                weight_update = np.dot(self.activations[-2].T, output_layer_gradient)
            else:
                weight_update = np.dot(self.activations[i-1].T, hidden_layer_gradients[i])
            if self.momentum is not None:
                weight_update = self.momentum * self.prev_weights[:, sum(self.hidden_layers[:i]):sum(self.hidden_layers[:i+1])] + (1 - self.momentum) * weight_update
            self.weights[:, sum(self.hidden_layers[:i]):sum(self.hidden_layers[:i+1])] += weight_update * learning_rate
        if self.momentum is not None:
            self.prev_weights = copy.deepcopy(self.weights)

    
    # update the weights
    def update_weights(self, weight_updates, learning_rate, momentum=None):
        if momentum is not None:
            weight_updates = momentum * self.prev_weights + (1 - momentum) * weight_updates
            self.prev_weights = weight_updates
        self.weights += learning_rate * weight_updates

    def predict(self, X):
        return self.feedforward(X)

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-2*self.beta*x))

    def sigmoid_activation_derivative(self, x):
        return 2*self.beta*self.sigmoid_activation(x) * (1 - self.sigmoid_activation(x))
        

    def mse(self, target, prediction):
        return np.square(np.subtract(target, prediction)).mean()