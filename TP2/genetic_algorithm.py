from math import log
import random
from time import time
import numpy as np
from datetime import datetime

import sys
import os
import csv

from utils.config import Config
from utils.generate import generate_initial_population
from utils.converters import proportion_to_rgb
from fitness import calculate_fitness

from selection import select_methods
from cross import cross_methods
from mutation import mutation_methods

Config.load_from_json('config.json')

new_gen_selects = {
    "use_all": (lambda old_population, children: fill_all(old_population, children)),
    "new_over_actual": (lambda old_population, children: fill_new_over_actual(old_population, children))
}

class GeneticAlgorithm:
    def run(self):
        (directory, filename) = create_output_file()
        generations_data = []
        generation = 0
        population = generate_initial_population(Config.max_population_size)

        while Config.max_generations is not None and generation < Config.max_generations:
            cross_seed = random.uniform(0, 1)
            children = []
            if cross_seed <= Config.cross_over['probability']:
                parents = select_methods[Config.selections['parents']['name']](population, Config.selections['parents']['amount'])

                parent_amount = len(parents)
                parent_pairs = random.sample(range(0, parent_amount), k=parent_amount)
                for i in np.arange(0, parent_amount, 2):
                    new_children = cross_methods[Config.cross_over['name']](parents[parent_pairs[i]], parents[parent_pairs[i+1]])
                    children.extend(new_children)
        
            new_population = population + children
            for individual in new_population:
                mutation_seed = random.uniform(0, 1)
                if mutation_seed <= Config.mutation['probability']:
                    individual = mutation_methods[Config.mutation['name']](individual)
                    individual.fitness = calculate_fitness(individual.get_gens())
                    individual.color = proportion_to_rgb(individual.get_gens())
            
            new_population = new_gen_selects[Config.implementation](population, children)
            population = new_population

            min_fitness = population[0].fitness
            max_fitness = population[0].fitness
            avg_fitness = 0.0

            indiv_proportions = {}

            for indiv in population:
                if indiv.color not in indiv_proportions:
                    indiv_proportions[indiv.color] = 0
                indiv_proportions[indiv.color] += 1 / Config.max_population_size
                if indiv.fitness < min_fitness:
                    min_fitness = indiv.fitness
                if indiv.fitness > max_fitness:
                    max_fitness = indiv.fitness
                avg_fitness += indiv.fitness
            
            avg_fitness /= Config.max_population_size
            diversity = shannon_diversity(indiv_proportions)

            generations_data.append({
                'generation': generation,
                'min_fitness': min_fitness,
                'avg_fitness': avg_fitness,
                'max_fitness': max_fitness,
                # 'max_fitness': calculate_fitness(max(population, key=lambda chromosome: calculate_fitness(chromosome.get_gens())).get_gens()),
                'diversity': diversity
            })
            generation += 1
        
        with open(os.path.join(directory, filename), 'w', encoding='UTF-8', newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(['generation', 'min_fitness', 'avg_fitness', 'max_fitness', 'diversity'])
            for gen_data in generations_data:
                csvwriter.writerow(gen_data.values())
        
        # TODO: Finish execution on acceptable solution or by structure (same values over multiple generations)
        return max(population, key=lambda chromosome: calculate_fitness(chromosome.get_gens()))

def shannon_diversity(indiv_proportions):
    diversity = 0
    for indiv in indiv_proportions:
        diversity -= indiv_proportions[indiv] * log(indiv_proportions[indiv])
    return diversity

def create_output_file():
    argc = len(sys.argv)
    sub_directory = ''
    if argc == 2:
        sub_directory = sys.argv[1]
    
    # Options are:
    # 'i': implementations
    # 'c': crosses
    # 'm': mutations
    # 's': selections
    # default: other

    root_directory = 'results'
    directory = root_directory
    filename = ''
    timestamp = datetime.now().timestamp()

    if not os.path.exists('results'):
        os.makedirs('results')

    match sub_directory:
        case 'i':
            directory += '/implementations'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = '{}_{}.csv'.format(Config.implementation, timestamp)
        case 'c':
            directory += '/crosses'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = '{}-{}_{}.csv'.format(Config.cross_over['name'], Config.cross_over['probability'], timestamp)
        case 'm':
            directory += '/mutations'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = '{}-{}_{}.csv'.format(Config.mutation['name'], Config.mutation['probability'], timestamp)
        case 's':
            directory += '/selections'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = '{}-{}_{}-{}_{}.csv'.format(Config.selections['parents']['name'], Config.selections['parents']['amount'], Config.selections['new_gen']['name'], Config.selections['parents']['amount'], timestamp)
        case _:
            directory += '/other'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = 'other_{}.csv'.format(timestamp)
    
    return (directory, filename)


def fill_all(old_population, children):
    population = old_population + children
    return select_methods[Config.selections['new_gen']['name']](population, Config.max_population_size)

    
def fill_new_over_actual(old_population, children):
    new_population = select_methods[Config.selections['new_gen']['name']](children, Config.selections['new_gen']['amount'])
    new_population_amount = len(new_population)
    if new_population_amount < Config.max_population_size:
        new_population.append(select_methods[Config.selections['new_gen']['name']](old_population, Config.selections['new_gen']['amount'] - new_population_amount))
    return new_population

