import numpy as np




class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        self.weights = np.zeros(num_inputs + 1)     # Extra for w0 (bias)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    
    def activation(self, input):           # Heaviside predict
        value = np.dot(input, self.weights[1:]) + self.weights[0]
        return np.where(value >= 0, 1, -1)

    def train(self, X, y):
        epochs = 0
        while epochs < self.max_epochs:
            for input, solution in zip(X, y):           # zip(X, y) makes tuples of (input, solution) to iterate them
                prediction = self.activation(input)
                
                delta_w = self.delta_w(solution, input)

                self.weights[0] += delta_w
                self.weights[1:] += delta_w * input
            
            prediction = self.activation(X)
            if np.array_equal(y, prediction):
                print("Convergence reached at epoch", epochs)
                break               # Finish by convergence
            
            if self.error(X, y) == 100:
                print("100% accuracy reached at epoch", epochs)
                break               # Finish by 100% accuracy on predicts
            epochs += 1
        print("Epochs: ", epochs)

    def delta_w(self, y, xi):
        return xi - y
        
    # Returns the accuracy of the perceptron on the given data. X and y must be the same length
    def error(self, X, y):
        prediction = self.activation(X)
        return 100 * (1 - np.mean(prediction == y))       # Checks if all values of prediction and y are equal


class SimpleLinealPerceptron(Perceptron):
    def activation(self, input):                 # input might be a vector or scalar
        value = np.dot(input, self.weights[1:]) + self.weights[0]
        return value
    
    # def train(self, X, y):
    #     epochs = 0
    #     while epochs < self.max_epochs:
    #         for input, solution in zip(X, y):           # zip(X, y) makes tuples of (input, solution) to iterate them
    #             prediction = self.activation(input)

    #             delta_w = self.learning_rate * (solution - prediction)

    #             self.weights[0] += delta_w
    #             self.weights[1:] += delta_w * input
            
    #         mse = self.error(X, y)
    #         if mse < 0.001:    # TODO: Change this to a parameter
    #             print("Convergence reached at epoch", epochs)
    #             break               # Finish by convergence
    
    #         epochs += 1
    #     print("Epochs: ", epochs)
    

    def delta_w(self, y, xi):
        diff = super().delta_w(y, xi)
        return self.learning_rate * diff * xi
    
    # Calculates the mean square error of the perceptron on the given data. X and y must be the same length.
    def error(self, X, y):
        prediction = self.activation(X)
        return np.mean((y - prediction)**2)


class SimpleNonLinealPerceptron(SimpleLinealPerceptron):
    
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100, beta=0.1, activation_func="tanh"):
        super().__init__(num_inputs, learning_rate, max_epochs)
        self.beta = beta
        self.activation_func = activation_func
    
    # tanh Activation function
    def tanh_activation(self, input):
        return np.tanh(self.beta * input)

    # Derivative of tanh activation function
    def tanh_derivative(self, input):
        return self.beta * (1 - np.tanh(self.beta * input)**2)

    # logistic Activation function
    def logistic_activation(self, input):
        return 1 / (1 + np.exp(-2 * self.beta * input))

    # Derivative of logistic activation function
    def logistic_derivative(self, input):
        return 2 * self.beta * self.logistic_activation(input) * (1 - self.logistic_activation(input))
    
    def activation(self, input):
        if self.activation_function == "tanh":
            return self.tanh_activation(input)
        elif self.activation_function == "logistic":
            return self.logistic_activation(input)
        else:
            raise ValueError("Activation function not supported")
    
    def derivative(self, input):
        if self.activation_function == "tanh":
            return self.tanh_derivative(input)
        elif self.activation_function == "logistic":
            return self.logistic_derivative(input)
        else:
            raise ValueError("Activation function not supported")
        
    def delta_w(self, y, xi):
        pass
