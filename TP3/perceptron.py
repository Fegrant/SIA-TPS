import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        self.weights = np.zeros(num_inputs + 1)     # Extra for w0 (umbral) initialized to 0 (higher the umbral more flexible the perceptron)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    
    def heaviside_predict(self, input):                 # input might be a vector or scalar
        value = np.dot(input, self.weights[1:]) + self.weights[0]
        return np.where(value >= 0, 1, -1)
    
    def train(self, X, y):
        epochs = 0
        while epochs < self.max_epochs:
            for input, solution in zip(X, y):           # zip(X, y) makes tuples of (input, solution) to iterate them
                prediction = self.heaviside_predict(input)
                
                error = solution - prediction
                input_approximation = self.learning_rate * error

                self.weights[0] += input_approximation
                self.weights[1:] += input_approximation * input
            
            prediction = self.heaviside_predict(X)
            if np.array_equal(y, prediction):
                print("Convergence reached at epoch", epochs)
                break               # Finish by convergence
            
            if self.perceptron_accuracy(X, y) == 100:
                print("100% accuracy reached at epoch", epochs)
                break               # Finish by 100% accuracy on predicts
            epochs += 1
        print("Epochs: ", epochs)
    
    # Returns the accuracy of the perceptron on the given data. X and y must be the same length
    def perceptron_accuracy(self, X, y):
        prediction = self.heaviside_predict(X)
        return 100 * np.mean(prediction == y)       # Checks if all values of prediction and y are equal