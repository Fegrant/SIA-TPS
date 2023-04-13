import numpy as np
from color import Color

def generate_initial_population(population_size):
    return [ Color(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(population_size)]