import random
import copy
import numpy as np

from chromosome import Chromosome
from utils.config import Config

Config.load_from_json("config.json")

palette_color_amount = Config.get_palette_color_amount()

def crossover_one_point(parent1: Chromosome, parent2: Chromosome):
    point = random.randint(0, palette_color_amount)

    # child1 = Chromosome([0 for _ in range(1, palette_color_amount)])
    # child2 = Chromosome([0 for _ in range(1, palette_color_amount)])

    # child1.set_gens(np.append(parent1.get_gens[:point], parent2.get_gens[point:]))
    # child2.set_gens(np.append(parent1.get_gens[:point], parent2.get_gens[point:]))

    child1 = Chromosome(copy.deepcopy(parent1.get_gens))
    child2 = Chromosome(copy.deepcopy(parent2.get_gens))

    for i in range(point, palette_color_amount):
        child1.gens[i] = parent2.gens[i]
        child2.gens[i] = parent1.gens[i]

    return child1, child2

def crossover_uniform(parent1: Chromosome, parent2: Chromosome):
    child1 = Chromosome(copy.deepcopy(parent1.get_gens))
    child2 = Chromosome(copy.deepcopy(parent2.get_gens))
    # child1 = Chromosome([0 for _ in range(1, palette_color_amount)])
    # child2 = Chromosome([0 for _ in range(1, palette_color_amount)])

    # TODO: Buscar si se puede usar deepcopy de hijo a padre para evitar tener que copiar valores
    for i in range(palette_color_amount):
        if random.uniform(0, 1) > 0.5:
        #     child1.gens[i] = parent1.gens[i]
        #     child2.gens[i] = parent2.gens[i]
        # else:
            child1.gens[i] = parent2.gens[i]
            child2.gens[i] = parent1.gens[i]
    return child1, child2