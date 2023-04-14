from utils.generate import generate_initial_population

class GeneticAlgorithm:
    def __init__(self, palette, color_objective, selection, max_population_size, break_condition, cross_over, mutation, select_amount_per_generation):
        self.palette = palette
        self.color_objective = color_objective
        self.selection = selection
        self.max_population_size = max_population_size
        self.break_condition = break_condition
        self.cross_over = cross_over
        self.mutation = mutation
        self.select_amount_per_generation = select_amount_per_generation

    def run(self, target_color):
        if self.break_condition == "generations":
            return self.run_generations(target_color)

        population = generate_initial_population(self.population_size)
        for i in range(self.num_generations):
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