from kohonen import Kohonen, Neuron
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
from config import load_kohonen_config


import numpy as np
import pandas

from matplotlib import cm
import matplotlib.pyplot as plt

from utils.parser import parse_csv_file

df = parse_csv_file('./europe.csv')
labels = df["Country"].to_numpy()
df.drop(columns=["Country"], axis=1, inplace=True)
inputs = StandardScaler().fit_transform(df.values)
cells = ["Area","GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"]

config = load_kohonen_config()

grid_dimension = int(config['grid_dimension'])
radius = int(config['radius'])
learning_rate = float(config['learning_rate'])
epochs = int(config['epochs'])

# print(input)
kohonen = Kohonen(grid_dimension, radius, learning_rate, epochs)
kohonen.train(inputs)

def label_plot():
    Xs, Ys = [],[]
    texts = []
    for i in range(len(inputs)):
        x, y = kohonen.find_best_neuron(inputs[i])
        Xs.append(x)
        Ys.append(y)
        texts.append(plt.text(x, y, labels[i]))
    plt.scatter(Xs, Ys)
    adjust_text(texts, only_move={'points':'y', 'text':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.show()

def count_plot():
    heatmap = np.zeros((grid_dimension, grid_dimension))
    for input in inputs:
        x, y = kohonen.find_best_neuron(input)
        heatmap[y][x] += 1
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()

def average_variable_plot():
    count_matrix = np.ones((grid_dimension, grid_dimension))
    variables_matrix = np.zeros((len(inputs), grid_dimension, grid_dimension))

    for input in inputs:
        x, y = kohonen.find_best_neuron(input)
        count_matrix[y][x] += 1
        for var in range(len(cells)):
            variables_matrix [var][y][x] =+ kohonen.neurons[y][x].weights[var]
        
    for var in range(len(cells)):
        variables_matrix[var] = variables_matrix[var] / count_matrix
    
    _, axes = plt.subplots(2, 4)
    for i in range(len(cells)):
        axes[i//4][i%4].imshow(variables_matrix[i])
        axes[i//4][i%4].set_title(cells[i])
    plt.show()

def matrix_plot():
    heatmap = np.zeros((grid_dimension, grid_dimension))
    for x in range(grid_dimension):
        for y in range(grid_dimension):
            locals_weights = kohonen.neurons[y][x].weights
            average_neighbour_dist = 0
            valid_neighbours = 0
            for neighbour_x in range(x-1, x+2):
                for neighbour_y in range(y-1, y+2):
                    if neighbour_x >= 0 and neighbour_x < grid_dimension and neighbour_y >= 0 and neighbour_y < grid_dimension:
                        average_neighbour_dist += kohonen.neurons[neighbour_y][neighbour_x].distance(locals_weights)
                        valid_neighbours += 1

            heatmap[y][x] = average_neighbour_dist / valid_neighbours
    print(f"Average: {sum(heatmap[y][x] for y in range(grid_dimension) for x in range(grid_dimension)) / grid_dimension**2}")

    plt.imshow(heatmap, cmap= cm.gray)
    plt.colorbar()
    plt.show()

label_plot()
count_plot()
matrix_plot()
average_variable_plot()

