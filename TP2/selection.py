import numpy as np
import random

from fitness import calculate_fitness
from utils.config import Config

select_methods = {
    "elite": (lambda population, num_selected: select_elite(population, num_selected)),
    "roulette": (lambda population, num_selected: select_roulette(population, num_selected)),
    "universal": (lambda population, num_selected: select_universal(population, num_selected)),
    "deterministic_tournament": (lambda population, num_selected: select_deterministic_tournament(population, num_selected))
}

def select_elite(population, num_selected):
    sorted_population = sorted(population, key=lambda x: calculate_fitness(x.get_gens()), reverse=True)
    return sorted_population[:num_selected]

def select_roulette(population, num_selected):
    fitness_scores = [calculate_fitness(chromosome.get_gens()) for chromosome in population]
    fitness_sum = sum(fitness_scores)
    probabilities = [fitness / fitness_sum for fitness in fitness_scores]
    cumulative_probabilities = np.cumsum(probabilities)
    selected_indices = []
    for i in range (num_selected):
        point = random.uniform(0, 1)
        index = np.searchsorted(cumulative_probabilities, point)
        selected_indices.append(index)
    return [population[i] for i in selected_indices]

def select_universal(population, num_selected):
    fitness_scores = [calculate_fitness(chromosome.get_gens()) for chromosome in population]
    fitness_sum = sum(fitness_scores)
    probabilities = [fitness / fitness_sum for fitness in fitness_scores]
    cumulative_probabilities = np.cumsum(probabilities)
    selected_indices = []
    for i in range(num_selected - 1):
        start_point = np.random.uniform(0, 1)
        point = start_point + i / num_selected
        index = np.searchsorted(cumulative_probabilities, point)
        selected_indices.append(index)
    return [population[i-1] for i in selected_indices]

# TODO: tournament_size = M, la cantidad de individuos a elegir de los N disponibles en la poblacion. La pregunta es, que valor tiene M?
def select_deterministic_tournament(population, num_selected, tournament_size):
    selected_indices = []
    while len(selected_indices) < num_selected:
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament = [population[i] for i in tournament_indices]
        tournament_fitnesses = [calculate_fitness(chromosome.get_gens()) for chromosome in tournament]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected_indices.append(winner_index)
    return [population[i] for i in selected_indices]
