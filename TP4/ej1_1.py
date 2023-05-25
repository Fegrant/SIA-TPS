from kohonen import Kohonen
from sklearn.preprocessing import StandardScaler
from config import load_kohonen_config

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from utils.parser import parse_csv_file

df = parse_csv_file('./inputs/europe.csv')
labels = df["Country"].to_numpy()
df.drop(columns=["Country"], axis=1, inplace=True)
cells = list(df.columns)
inputs = StandardScaler().fit_transform(df.values)

config = load_kohonen_config()

grid_dimension = int(config['grid_dimension'])
radius = int(config['radius'])
learning_rate = float(config['learning_rate'])
epochs = int(config['epochs'])

# print(input)
kohonen = Kohonen(grid_dimension, radius, learning_rate, epochs)
kohonen.train(inputs)
n = inputs.shape[0]

def biplot(score, coeff, names, labels=None):
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot countries
    ax.scatter(xs * scalex, ys * scaley, color='b', alpha=0.5)
    for i in np.arange(len(names)):
        ax.annotate(names[i], (xs[i] * scalex + 0.015, ys[i] * scaley), color='blue') # country names
    
    # plot variables
    for i in range(n):
        ax.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5) # variables
        ax.text(coeff[i,0] * 1.05, coeff[i,1] * 1.05, labels[i], color='g', ha='center', va='center')
    
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)

# def label_plot():
    # Xs, Ys = [],[]
    # for i in range(len(inputs)):
    #     x, y = kohonen.find_best_neuron(inputs[i])
    #     Xs.append(x)
    #     Ys.append(y)

    # scalex = 1.0 / (max(Xs) - min(Xs))
    # scaley = 1.0 / (max(Ys) - min(Ys))

    # scaled_Xs = [x * scalex for x in Xs]
    # scaled_Ys = [y * scaley for y in Ys]

    # plt.scatter(scaled_Xs, scaled_Ys)
    # for i in np.arange(len(labels)):
    #     plt.annotate(labels[i], (scaled_Xs[i], scaled_Ys[i]), color = 'blue')

    # # Plot arrows (biplot)
    # for i in range(len(cells)):
    #     plt.arrow(0, 0, inputs[i, 0], inputs[i, 1], color='r', alpha=0.5)
    #     plt.text(inputs[i, 0] * 1.15, inputs[i, 1] * 1.15, cells[i], color='g', ha='center', va='center')

    # plt.show()

def count_plot():
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = np.zeros((grid_dimension, grid_dimension))
    texts = [[[] for x in np.arange(kohonen.grid_dimension)] for y in np.arange(kohonen.grid_dimension)]

    for i in np.arange(len(inputs)):
        x, y = kohonen.find_best_neuron(inputs[i])
        heatmap[y][x] += 1
        texts[y][x].append(labels[i])
        # plt.text(x - 0.25, y, labels[i])

    for y in np.arange(kohonen.grid_dimension):
        for x in np.arange(kohonen.grid_dimension):
            country_amount = len(texts[y][x])
            for i in np.arange(country_amount):
                txt = plt.text(x - 0.35, y + (country_amount / 2) * 0.1 - i * 0.1, texts[y][x][i], color='#fff', size='large', fontweight='bold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='#000')])
            # plt.text(x - 0.2, y, texts[y][x])

    plt.imshow(heatmap, cmap='inferno')
    plt.colorbar()
    plt.xticks(np.arange(grid_dimension))
    plt.yticks(np.arange(grid_dimension))
    plt.show()

def average_variable_plot():
    count_matrix = np.ones((grid_dimension, grid_dimension))
    variables_matrix = np.zeros((len(cells), grid_dimension, grid_dimension))

    for input in inputs:
        x, y = kohonen.find_best_neuron(input)
        count_matrix[y][x] += 1
        for var in range(len(cells)):
            variables_matrix[var][y][x] += kohonen.neurons[y][x].weights[var]
        
    for var in range(len(cells)):
        variables_matrix[var] /= count_matrix
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 12))
    for i in range(len(cells)):
        axes[i//4][i%4].imshow(variables_matrix[i], cmap='inferno')
        axes[i//4][i%4].set_title(cells[i])
        axes[i//4][i%4].set_xticks(np.arange(grid_dimension))
        axes[i//4][i%4].set_yticks(np.arange(grid_dimension))
    
    fig.delaxes(axes[1][3])
    plt.show()

def matrix_plot():
    fig, ax = plt.subplots(figsize=(10, 10))
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

# label_plot()
count_plot()
matrix_plot()
average_variable_plot()

