import random
import numpy as np

from utils.config import Config
from utils.converters import proportion_to_rgb
from chromosome import Chromosome
from fitness import calculate_fitness

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
        # Must recalculate fitness and color on mutation
        individual.fitness = calculate_fitness(individual.gens)
        individual.color = proportion_to_rgb(individual.gens)
    return individual


def limited_multiple_gene_mutation(individual: Chromosome):
    mutate_amount = 2
    if random.uniform(0, 1) <= mutation_probability:
        gens_pos = random.sample(range(palette_color_amount), k=mutate_amount)
        for pos in gens_pos:
            individual.gens[pos] = random.uniform(0, 1)
        # Must recalculate fitness and color on mutation
        individual.fitness = calculate_fitness(individual.gens)
        individual.color = proportion_to_rgb(individual.gens)
    return individual


def uniform_multiple_gene_mutation(individual: Chromosome):
    has_mutated = False
    for pos in range(palette_color_amount):
        if random.uniform(0, 1) <= mutation_probability:
            individual.gens[pos] = random.uniform(0, 1)
            has_mutated = True
    if has_mutated:
        # Must recalculate fitness and color on mutation
        individual.fitness = calculate_fitness(individual.gens)
        individual.color = proportion_to_rgb(individual.gens)
    return individual


def complete_mutation(individual: Chromosome):
    if random.uniform(0, 1) <= mutation_probability:
        individual.gens = [random.uniform(0, 1) for _ in range(palette_color_amount)]
        # Must recalculate fitness and color on mutation
        individual.fitness = calculate_fitness(individual.gens)
        individual.color = proportion_to_rgb(individual.gens)
    return individual