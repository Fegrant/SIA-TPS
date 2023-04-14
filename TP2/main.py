import numpy as np

from fitness import calculate_fitness
from selection import select_elite, select_roulette
from utils.generate import generate_initial_population
from cross import crossover_one_point
from genetic_algorithm import GeneticAlgorithm
from utils.config import Config

Config.load_from_json('config.json')


def euclidean(a, b):
    return np.linalg.norm(a - b)

def diversity(population):
    unique_chromosomes = set(tuple(chromosome) for chromosome in population)
    return len(unique_chromosomes) / len(population)

generic_algorithm = GeneticAlgorithm(Config.max_population_size, Config.select_amount_per_generation, Config.num_generations, Config.mutation_probability)
best_chromosome = generic_algorithm.run(Config.color_objective, Config.palette, Config.num_generations)


# Ejemplo de uso
palette = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
target_color = np.array([0.5, 0.5, 0])


mixed_color = np.dot(best_chromosome, palette)
print(f'Proporciones de la paleta a utilizar: {best_chromosome}')
print(f'Similitud con el color objetivo: {euclidean(target_color, mixed_color)}')
