import random

from utils.config import Config
from chromosome import Chromosome

# La funcion de mutacion en este caso es aleatoria, 
# se se elige un valor aleatorio para una posición aleatoria del cromosoma. 
# Por ejemplo, podrías cambiar el valor de un componente RGB por un valor aleatorio entre 0 y 255.

Config.load_from_json("config.json")

mutation_probability = Config.mutation["probability"]
palette_color_amount = Config.get_palette_color_amount()

def gene_mutation(individual: Chromosome, gene_index: int):
    if random.uniform(0, 1) <= mutation_probability:
        individual.gens[gene_index] = random.random()
    return


def limited_multiple_gene_mutation(individual: Chromosome, gens_to_mutate: int):
    if random.uniform(0, 1) <= mutation_probability:
        gens_pos = random.sample(range(3), gens_to_mutate)
        for pos in range(gens_pos):
            individual.gens[pos] = random.random()
    return


def uniform_multiple_gene_mutation(individual: Chromosome):
    for pos in range(3):
        if random.uniform(0, 1) <= mutation_probability:
            individual.gens[pos] = random.random()
    return


def complete_mutation(individual: Chromosome):
    if random.uniform(0, 1) <= mutation_probability:
        individual.set_gens(random.random() for _ in range(palette_color_amount))
