import numpy as np

from fitness import calculate_fitness

def select_elite(population, num_elite):
    sorted_population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True)
    return sorted_population[:num_elite]

def select_roulette(population, num_selected):
    fitness_scores = [calculate_fitness(chromosome) for chromosome in population]
    fitness_sum = sum(fitness_scores)
    probabilities = [fitness / fitness_sum for fitness in fitness_scores]
    selected_indices = np.random.choice(len(population), num_selected, p=probabilities)
    return [population[i] for i in selected_indices]