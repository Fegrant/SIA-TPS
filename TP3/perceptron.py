import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01, max_epochs=1000):
        self.weights = np.zeros(num_inputs + 1)     # Extra for w0
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    
    def heaviside_predict(self, input):                 # input might be a vector or scalar
        value = np.dot(input, self.weights[1:]) + self.weights[0]
        return np.where(value >= 0, 1, 0)
    
    def train(self, X, y):
        for epoch in np.arange(self.max_epochs):
            for input, solution in zip(X, y):           # zip(X, y) makes tuples of (input, solution) to iterate them
                prediction = self.heaviside_predict(input)
                
                error = solution - prediction
                input_approximation = self.learning_rate * error

                self.weights[0] += input_approximation
                self.weights[1:] += input_approximation * input
            
            prediction = self.heaviside_predict(X)
            if np.array_equal(y, prediction):
                break               # Finish by convergence
            
            if self.perceptron_accuracy(X, y) == 100:
                break               # Finish by 100% accuracy on predicts
    
    def perceptron_accuracy(self, X, y):
        prediction = self.heaviside_predict(X)
        return 100 * np.mean(prediction == y)       # Checks if all values of prediction and y are equal

# Creation, training and values of and perceptron
and_perceptron = Perceptron(2)

and_X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
and_y = np.array([-1, -1, -1, 1])

and_perceptron.train(and_X, and_y)

print("AND Weights: ", and_perceptron.weights[1:])
print("AND Bias: ", and_perceptron.weights[0])
print("AND Predictions: ", and_perceptron.heaviside_predict(and_X))

print()

# Creation, training and values of xor perceptron
xor_perceptron = Perceptron(2)

xor_X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
xor_y = np.array([1, 1, -1, -1])

xor_perceptron.train(xor_X, xor_y)

print("XOR Weights: ", xor_perceptron.weights[1:])
print("XOR Bias: ", xor_perceptron.weights[0])
print("XOR Predictions: ", xor_perceptron.heaviside_predict(xor_X))
