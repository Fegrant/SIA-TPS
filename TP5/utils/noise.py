import numpy as np

def add_noise(matrix, noise):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if np.random.rand() < noise:
                matrix[i][j] = 1 if matrix[i][j] == 0 else 0
    return matrix