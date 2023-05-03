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

    def fit(self, X, y, epochs=1000, learning_rate=0.1, beta=0.1, batch_size=1):
        self.beta = beta
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        if n_samples % batch_size != 0:
            n_batches += 1
        
        if self.momentum is not None:
            self.prev_weights = np.zeros(self.weights.shape)

        sum_errors = []

        # for each epoch train the network on each sample and update weights
        for epoch in range(epochs):
            # iterate over each batch, if batch_size is 1, then it's stochastic gradient descent, if batch_size is n_samples, then it's batch gradient descent
            # Osea que si batch_size es 1, es SGD, si es n_samples es BGD, y si es otro valor es mini-batch
            for batch_idx in range(n_samples):
                batch_idx = batch_idx % n_batches
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_samples)
                batch_X = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]

                # feedforward the input through the network
                self.feedforward(batch_X)

                # backpropagate the error and update the weights
                gradients = self.backpropagation(batch_X, batch_y)
                
                # update the weights
                self.update_weights(gradients, learning_rate, self.momentum)
            
            # Compute the MSE after each epoch and append it to the list (for plotting purposes, each epoch is a point in the plot)
            sum_errors.append(self.mse(y, self.feedforward(X)))
        


    # feedforward the input through the network and return the output layer
    def feedforward(self, batch_X):
        n_samples = batch_X.shape[0]
        # Initialize the activations 3d matrix (n_samples, n_neurons, n_layers)
        batch_X_3d = batch_X[np.newaxis, :, :]
        print(batch_X_3d)

        self.activations = batch_X_3d
        # Concatenate the results of each prediction of each layer to the activations 3d matrix (n_samples, n_neurons, n_layers)
        for i in range(n_samples):
            for j in range(len(self.hidden_layers)):
                layer_output = self.sigmoid_activation(np.dot(self.activations[i], self.weights[j]))
                self.activations = np.concatenate((self.activations, layer_output[:, :, np.newaxis]), axis=2)
            output = self.sigmoid_activation(np.dot(self.activations[:, :, -1], self.weights[-1]))
            self.activations = np.concatenate((self.activations, output[:, :, np.newaxis]), axis=2)
        return self.activations[:, :, -1]

    # backpropagate the error and update the weights
    def backpropagation(self, batch_X, batch_y):
        batch_size = batch_X.shape[0]

        # Compute the error at the output layer
        output_error = batch_y - self.activations[-1]

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

        # Compute the weight updates for each batch item
        weight_updates = []
        for j in range(batch_size):
            update = []
            for i in range(len(self.hidden_layers)+1):
                if i == 0:
                    weight_update = np.outer(batch_X[j], hidden_layer_gradients[i][j])
                elif i == len(self.hidden_layers):
                    weight_update = np.outer(self.activations[-2][j], output_layer_gradient[j])
                else:
                    weight_update = np.outer(self.activations[i-1][j], hidden_layer_gradients[i][j])
                update.append(weight_update)
            weight_updates.append(update)

        return weight_updates
    
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