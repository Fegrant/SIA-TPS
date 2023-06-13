import numpy as np
import json

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    print('Loaded data', data)
    return data