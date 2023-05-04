import pandas as pd
import numpy as np

def parse_csv_file(file: str):
    df = pd.read_csv(file)
    return df.to_numpy()

def parse_txt_file_3c(file: str):
    df = pd.read_csv(file, header=None, sep=' ', usecols=[0, 1, 2, 3, 4])
    read_matrix = df.to_numpy()
    read_matrix = np.split(read_matrix, 10)
    parsed_matrix = []
    for i in np.arange(10):
        num_matrix = np.zeros(10)
        num_matrix[i] = 1
        parsed_matrix.append([])
        parsed_matrix[i].append(read_matrix[0])
        parsed_matrix[i].append(num_matrix)
    return parsed_matrix

def parse_txt_file_3b(file: str):
    df = pd.read_csv(file, header=None, sep=' ', usecols=[0, 1, 2, 3, 4])
    read_matrix = df.to_numpy()
    read_matrix = np.split(read_matrix, 10)
    parsed_matrix = []
    for i in np.arange(10):
        parsed_matrix.append((read_matrix[0], i % 2))
    return parsed_matrix
