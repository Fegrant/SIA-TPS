from math import hypot
from typing import Tuple
import numpy as np

# print(parse_csv_file('./europe.csv'))

class Neuron:

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def distance(self, x: np.ndarray):
        if len(self.weights) != len(x):
            raise Exception("Dimensions don't match")
        return np.linalg.norm(x - self.weights)
    
class Kohonen:

    def __init__(self, grid_dimension, radius, learning_rate, epochs):
        self.neurons = [[None for _ in range(grid_dimension)] for _ in range(grid_dimension)]
        self.grid_dimension = grid_dimension
        self.radius = radius
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, inputs: list[list[float]]) -> None:
        for y in range(self.grid_dimension):
            for x in range(self.grid_dimension):
                rand_input = inputs[ y * x % len(inputs)]
                self.neurons[y][x] = Neuron(weights=list(rand_input))

        total_iterations = self.epochs * len(inputs)

        iteration = 0
        r = self.radius
        n = self.learning_rate
        while iteration < total_iterations:

            i = iteration % len(inputs)
            rand_input = inputs[i]

            best_x, best_y = self.find_best_neuron(rand_input)

            for x in range(best_x - r, best_x + r + 1, 1):
                for y in range(best_y - r, best_y + r + 1, 1):
                    if x >= 0 and x < self.grid_dimension and y >= 0 and y < self.grid_dimension and hypot(x - best_x, y - best_y) <= r:
                        self.neurons[y][x].weights += n * (rand_input - self.neurons[y][x].weights)
        
            iteration += 1
            n = (0.7 - self.learning_rate)/total_iterations * iteration + self.learning_rate
            r = int((1-self.radius)/total_iterations * iteration) + self.radius

    def find_best_neuron(self, input: list[float]) -> Tuple[int, int]:
        best_distance = np.inf
        best_coords = None
        for y in range(self.grid_dimension):
            for x in range(self.grid_dimension):
                distance = self.neurons[y][x].distance(input)
                if distance < best_distance:
                    best_distance = distance
                    best_coords = [x,y]
        return best_coords
