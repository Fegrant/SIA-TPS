import pandas as pd
import numpy as np

def parse_csv_file(file: str):
    df = pd.read_csv(file)
    return df.to_numpy()

def parse_txt_file(file: str):
    df = pd.read_csv(file, header=None, sep=' ', usecols=[0, 1, 2, 3, 4])
    arr = df.to_numpy()
    return np.split(arr, 10)
