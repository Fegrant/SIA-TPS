import numpy as np

# Returns the normalized data between a and b, where a is the minimum and b is the maximum
# Use this function to normalize the data while trainig non-lineal perceptrons
def feature_scaling(data: np.array, a, b):
    min = np.min(data)
    max = np.max(data)
    return a + (data - min) * (b - a) / (max - min)
    