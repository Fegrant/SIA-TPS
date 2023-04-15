from audioop import cross
import random
import numpy as np

from utils.config import Config
from utils.generate import generate_initial_population
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
            
            new_population = new_gen_selects[Config.implementation](population, children)
            population = new_population
            generation += 1
        
        return max(population, key=lambda chromosome: calculate_fitness(chromosome.get_gens()))

        # if self.break_condition == "generations":
        #     return self.run_generations(target_color)

        # population = generate_initial_population(self.population_size)
        # for i in range(self.num_generations):
        #     elite = select_elite(population, num_elite)
        #     new_population = elite
        #     while len(new_population) < population_size:
        #         parent1, parent2 = select_roulette(population, 2)
        #         child1, child2 = crossover_one_point(parent1, parent2)
        #         # child1 = mutate_swap(child1, mutation_probability)
        #         # child2 = mutate_swap(child2, mutation_probability)
        #         new_population.append(child1)
        #         new_population.append(child2)
        #     population = new_population
        # best_chromosome = max(population, key=lambda x: calculate_fitness(x, target_color, palette))
        # return best_chromosome
    
def fill_all(old_population, children):
    population = old_population + children
    return select_methods[Config.selections['new_gen']['name']](population, Config.max_population_size)

    
def fill_new_over_actual(old_population, children):
    new_population = select_methods[Config.selections['new_gen']['name']](children, Config.selections['new_gen']['amount'])
    new_population_amount = len(new_population)
    if new_population_amount < Config.max_population_size:
        new_population.append(select_methods[Config.selections['new_gen']['name']](old_population, Config.selections['new_gen']['amount'] - new_population_amount))
    return new_population

