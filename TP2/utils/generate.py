import numpy as np
import random
from chromosome import Chromosome
from utils.config import Config

def generate_random_palette(size: int):
    return [ (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(size)]

def generate_initial_population(population_size: int):
    return [ Chromosome(np.random.rand(Config.get_palette_color_amount())) for _ in range(population_size) ]