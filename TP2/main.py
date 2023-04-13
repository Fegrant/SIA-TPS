import random
import numpy as np

from color import color
from fitness import calculate_fitness
from selection import select_elite, select_roulette
from utils.generate import generate_initial_population

def euclidean(a, b):
    return np.linalg.norm(a - b)

def crossover_one_point(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def genetic_algorithm(target_color, palette=None, population_size=100, num_elite=5, num_generations=100, mutation_probability=0.05):
    
    if palette is None:
        population = generate_initial_population(population_size)
    
    for i in range(num_generations):
        elite = select_elite(population, num_elite)
        new_population = elite
        while len(new_population) < population_size:
            parent1, parent2 = select_roulette(population, 2)
            child1, child2 = crossover_one_point(parent1, parent2)
            # child1 = mutate_swap(child1, mutation_probability)
            # child2 = mutate_swap(child2, mutation_probability)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
    best_chromosome = max(population, key=lambda x: calculate_fitness(x, target_color, palette))
    return best_chromosome

# Ejemplo de uso
palette = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
target_color = np.array([0.5, 0.5, 0])
best_chromosome = genetic_algorithm(target_color, palette)
mixed_color = np.dot(best_chromosome, palette)
print(f'Proporciones de la paleta a utilizar: {best_chromosome}')
print(f'Similitud con el color objetivo: {euclidean(target_color, mixed_color)}')
