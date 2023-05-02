import pandas as pd
import numpy as np

def parse_csv_file(file: str):
    df = pd.read_csv(file)
    return df.to_numpy()

def parse_txt_file(file: str):
    df = pd.read_csv(file, header=None, sep=' ', usecols=[0, 1, 2, 3, 4])
    read_matrix = df.to_numpy()
    read_matrix = np.split(read_matrix, 10)
    parsed_matrix = np.array([])
    for i in np.arange(10):
        parsed_matrix.append((read_matrix[0], i))
    print(parsed_matrix)
    return parsed_matrix
