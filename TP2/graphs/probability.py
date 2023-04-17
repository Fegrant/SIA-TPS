# fitness vs crossover probability

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# fitness vs crossover probability

def fitness_vs_crossover_probability():
    probability('./results/crosses/uniform-')
    

# fitness vs mutation probability
def fitness_vs_mutation_probability():
    probability('./results/mutations/complete-')
        

def probability(plot_name):
    # Create empty lists to store data for all probabilities
    crossover_probabilities = []
    min_fitnesses = []
    avg_fitnesses = []
    max_fitnesses = []
    sem_min_fitnesses = []
    sem_avg_fitnesses = []
    sem_max_fitnesses = []

    for i in range(0, 11):
        crossover_probability = i/10
        crossover_probabilities.append(crossover_probability)
        if i == 0:
            crossover_probability = i
        if i == 10:
            crossover_probability = 1
        filename = '{}{}.csv'.format(plot_name, crossover_probability)
        df = pd.read_csv(filename)

        # Create a new dataframe selecting the run and his last generation
        grouped = df.groupby('run').last()
        
        # Get the min max and average fitness for all run and last generation and its standard error
        min_fitnesses.append(grouped["min_fitness"].mean())
        avg_fitnesses.append(grouped["avg_fitness"].mean())
        max_fitnesses.append(grouped["max_fitness"].mean())
        sem_min_fitnesses.append(grouped["min_fitness"].sem())
        sem_avg_fitnesses.append(grouped["avg_fitness"].sem())
        sem_max_fitnesses.append(grouped["max_fitness"].sem())

    # Plot data for all probabilities
    plt.errorbar(crossover_probabilities, min_fitnesses, yerr=sem_min_fitnesses, label="Minimum Fitness")
    plt.errorbar(crossover_probabilities, avg_fitnesses, yerr=sem_avg_fitnesses, label="Average Fitness")
    plt.errorbar(crossover_probabilities, max_fitnesses, yerr=sem_max_fitnesses, label="Maximum Fitness")

    plt.legend()
    if plot_name == './results/crosses/uniform-':
        plt.xlabel("Crossover Probability")
        plt.title("Fitness vs. Crossover Probability")
    else:
        plt.xlabel("Mutation Probability")
        plt.title("Fitness vs. Mutation Probability")
        
    plt.ylabel("Fitness")
    plt.show()

if __name__ == '__main__':
    probability('./results/mutations/complete-')
