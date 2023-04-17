import random
import numpy as np

from utils.config import Config
from chromosome import Chromosome

# La funcion de mutacion en este caso es aleatoria, 
# se se elige un valor aleatorio para una posición aleatoria del cromosoma. 
# Por ejemplo, podrías cambiar el valor de un componente RGB por un valor aleatorio entre 0 y 255.

Config.load_from_json("config.json")

mutation_probability = Config.mutation["probability"]
palette_color_amount = Config.get_palette_color_amount()

mutation_methods = {
    "one_gene": (lambda individual: gene_mutation(individual)),
    "limited": (lambda individual: limited_multiple_gene_mutation(individual)),
    "uniform": (lambda individual: uniform_multiple_gene_mutation(individual)),
    "complete": (lambda individual: complete_mutation(individual))
}

def gene_mutation(individual: Chromosome):
    if random.uniform(0, 1) <= mutation_probability:
        gene_index = random.sample(range(palette_color_amount), k=1)
        individual.gens[gene_index] = random.uniform(0, 1)
    return individual


def limited_multiple_gene_mutation(individual: Chromosome):
    mutate_amount = Config.mutation["limited"]["amount"]
    if random.uniform(0, 1) <= mutation_probability:
        gens_pos = random.sample(range(palette_color_amount), k=mutate_amount)
        for pos in range(gens_pos):
            individual.gens[pos] = random.uniform(0, 1)
    return individual


def uniform_multiple_gene_mutation(individual: Chromosome):
    for pos in range(palette_color_amount):
        if random.uniform(0, 1) <= mutation_probability:
            individual.gens[pos] = random.uniform(0, 1)
    return individual


def complete_mutation(individual: Chromosome):
    if random.uniform(0, 1) <= mutation_probability:
        individual.set_gens([random.uniform(0, 1) for _ in range(palette_color_amount)])
    return individual