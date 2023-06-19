import numpy as np

# add noise to a flatten matrix
def add_noise(letters, noise):
    for i in range(len(letters)):
        for j in range(len(letters[i])):
            if np.random.rand() < noise:
                letters[i][j] = 1 if letters[i][j] == 0 else 0
        
    return letters