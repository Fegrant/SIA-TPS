import numpy as np
import copy

class MultilayerPerceptron:
    def __init__(self, n_inputs, hidden_layer_amount, n_outputs, momentum=None):
        self.n_inputs = n_inputs
        self.hidden_layer_amount = hidden_layer_amount
        self.n_outputs = n_outputs
        self.momentum = momentum
        self.init_weights()

    # initialize weights with random values
    def init_weights(self):
        self.weights_per_layer = []

        # Initialize the weights of the first hidden layer
        weights = np.random.randn(self.hidden_layer_amount[0], self.n_inputs + 1)
        # weights = np.zeros((self.hidden_layer_amount[0], self.n_inputs + 1))
        self.weights_per_layer.append(weights)

        # Initialize the weights of the next hidden layers
        for i in range(1, len(self.hidden_layer_amount)):
            weights = np.random.randn(self.hidden_layer_amount[i], self.hidden_layer_amount[i-1] + 1)
            # weights = np.zeros((self.hidden_layer_amount[i], self.hidden_layer_amount[i-1] + 1))
            self.weights_per_layer.append(weights)
        
        # Initialize the weights of the output layer
        weights = np.random.randn(self.n_outputs, self.hidden_layer_amount[-1] + 1)
        # weights = np.zeros((self.n_outputs, self.hidden_layer_amount[-1] + 1))
        self.weights_per_layer.append(weights)

        if self.momentum is not None:
            self.prev_weights_per_layer = copy.deepcopy(self.weights_per_layer)


    def fit(self, X, y, epochs=1000, learning_rate=0.1, beta=0.1, momentum=None):
        self.beta = beta
        self.momentum = momentum

        sum_errors = np.array([], dtype=np.float64)

        # for each epoch train the network on each sample and update weights
        for _ in np.arange(10):
            # Iterate over the samples
            for i in np.arange(X.shape[0]):
                # Feedforward the input and compute the activations
                self.feedforward(X[i])
                # Backpropagate the error and update the weights
                self.backpropagation(X[i], y[i], learning_rate)
                if np.array_equal(X[i], np.array([1, 0])):
                    print('Activations: ' + str(self.activations))
                    print('Weights: ' + str(self.weights_per_layer))
                    print()
            
            # Compute the MSE after each epoch and append it to the list (for plotting purposes, each epoch is a point in the plot)
            sum_errors = np.append(sum_errors, self.mse(y, self.predict(X)))


    # feedforward the input through the network and return the output layer
    def feedforward(self, X):
        # Add the bias to the input layer
        # print(X)
        self.activations = [np.insert(X, 0, 1)]

        # print(self.activations)
        # print(self.weights_per_layer)
        # print()

        # Compute the activations of the hidden layers
        for i in np.arange(len(self.hidden_layer_amount)):
            activation = self.sigmoid_activation(np.dot(self.activations[i], self.weights_per_layer[i].T))
            self.activations.append(np.insert(activation, 0, 1))

        # Compute the activations of the output layer
        activation = self.sigmoid_activation(np.dot(self.activations[-1], self.weights_per_layer[-1].T))
        self.activations.append(activation)

        return self.activations[-1]
    

    # feedforward the input through the network and return the output layer
    def predictfeedforward(self, X):
        if X.ndim == 1:
            current_layer_activation = np.insert(X, 0, 1)
            for i in np.arange(len(self.hidden_layer_amount)):
                current_layer_activation = self.sigmoid_activation(np.dot(current_layer_activation, self.weights_per_layer[i].T))
                current_layer_activation = np.insert(current_layer_activation, 0, 1)
            current_layer_activation = self.sigmoid_activation(np.dot(current_layer_activation, self.weights_per_layer[-1].T))
        else:
            current_layer_activation = np.insert(X, 0, 1, axis=1)

            for data_activation in current_layer_activation:
                for i in np.arange(len(self.hidden_layer_amount)):
                    data_activation = self.sigmoid_activation(np.dot(data_activation, self.weights_per_layer[i].T))
                    data_activation = np.insert(data_activation, 0, 1)
                data_activation = self.sigmoid_activation(np.dot(data_activation, self.weights_per_layer[-1].T))

        return current_layer_activation
        

    # backpropagate the error and update the weights
    def backpropagation(self, X, Y, learning_rate=0.1):
        # Compute the error at the output layer
        output_error = Y - self.activations[-1]
        layer_deltas = []

        # Compute the gradient at the output layer
        last_layer_delta = output_error * self.sigmoid_activation_derivative(self.activations[-1])
        print(self.sigmoid_activation_derivative(self.activations[-1]))
        layer_deltas.append(last_layer_delta)

        # Compute the gradient at the hidden layers
        for i in np.arange(len(self.hidden_layer_amount), 0, -1):
            # print(self.weights_per_layer)
            # print(layer_deltas)
            # print(self.weights_per_layer[i])
            output_error = np.dot(self.weights_per_layer[i].T, last_layer_delta)
            # print(output_error)
            # print(self.sigmoid_activation_derivative(self.activations[i]))
            layer_delta = np.dot(np.array(output_error).T, self.sigmoid_activation_derivative(self.activations[i]))
            layer_deltas.append(layer_delta)
            last_layer_delta = copy.deepcopy(layer_delta)

        print(layer_deltas)

        # Update the weights of the output layer
        # print(np.dot(last_layer_delta, self.activations[-2]))
        self.weights_per_layer[-1] += learning_rate * np.dot(last_layer_delta, self.activations[-2])

        # Update the weights of the hidden layers
        for i in np.arange(len(self.hidden_layer_amount), 0, -1):
            delta_w = learning_rate * np.dot(layer_deltas[i], self.activations[i-1])
            self.weights_per_layer[i-1] += delta_w
            if self.momentum is not None:
                self.weights_per_layer[i-1] += self.momentum * self.prev_weights_per_layer[i-1]
                self.prev_weights_per_layer[i-1] = delta_w


    def predict(self, X):
        return self.predictfeedforward(X)

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-2*self.beta*x))

    def sigmoid_activation_derivative(self, x):
        return 2*self.beta*self.sigmoid_activation(x) * (1 - self.sigmoid_activation(x))

    def mse(self, target, prediction):
        return np.square(np.subtract(target, prediction)).mean()