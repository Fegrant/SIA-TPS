import random
from color import Color

# La funcion de mutacion en este caso es aleatoria, 
# se se elige un valor aleatorio para una posición aleatoria del cromosoma. 
# Por ejemplo, podrías cambiar el valor de un componente RGB por un valor aleatorio entre 0 y 255.

def gene_mutation(individual: Color, mutation_probability: float, gene: int):
    if random.uniform(0, 1) <= mutation_probability:
        new_locus = random.randint(0, 255)
        individual.rgb[gene] = new_locus
    return 


def limited_multiple_gene_mutation(individual: Color, gens_to_mutate: int, mutation_probability: float):
    if random.uniform(0, 1) <= mutation_probability:
        gens_pos = random.sample(range(3), gens_to_mutate)
        for pos in range(gens_pos):
            new_locus = random.randint(0, 255)
            individual.rgb[pos] = new_locus
    return


def uniform_multiple_gene_mutation(individual: Color, mutation_probability: float):
    for pos in range(3):
        if random.uniform(0, 1) <= mutation_probability:
            new_locus = random.randint(0, 255)
            individual.rgb[pos] = new_locus
    return



def complete_mutation(individual: Color, mutation_probability: float):
    if random.uniform(0, 1) <= mutation_probability:
        individual.rgb = [random.randint(0, 255) for _ in range(3)]