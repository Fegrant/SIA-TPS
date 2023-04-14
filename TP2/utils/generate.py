import numpy as np
from chromosome import Chromosome

def generate_random_palette(size: int):
    return [ (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(size)]


# TODO generate_initial_population
def generate_initial_population(population_size: int):
    return 