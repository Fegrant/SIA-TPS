import pandas as pd
import numpy as np

def parse_csv_file(file: str):
    df = pd.read_csv(file)
    return df