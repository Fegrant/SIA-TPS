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
    for i in range(0, 11):
        j = i/10
        if i == 0:
            j = i
        if i == 10:
            j = 1
        filename = '{}{}.csv'.format(plot_name, j)
        df = pd.read_csv(filename)

        # Add a column to identify the last generation for each run
        df["last_generation"] = df.groupby("run")["generation"].transform("max")

        # Group by "last_generation" column and calculate the mean and standard deviation of the fitness values
        grouped = df.groupby("last_generation").agg({"min_fitness": "mean", "avg_fitness": "mean", "max_fitness": "mean"})

        # Calculate the standard error of the mean for each fitness value
        grouped["sem_min_fitness"] = df.groupby("last_generation")["min_fitness"].sem()
        grouped["sem_avg_fitness"] = df.groupby("last_generation")["avg_fitness"].sem()
        grouped["sem_max_fitness"] = df.groupby("last_generation")["max_fitness"].sem()

        # Reset the index of the resulting DataFrame
        grouped = grouped.reset_index()

        plt.errorbar(grouped["last_generation"], grouped["min_fitness"], yerr=grouped["sem_min_fitness"], label="Minimum Fitness")
        plt.errorbar(grouped["last_generation"], grouped["avg_fitness"], yerr=grouped["sem_avg_fitness"], label="Average Fitness")
        plt.errorbar(grouped["last_generation"], grouped["max_fitness"], yerr=grouped["sem_max_fitness"], label="Maximum Fitness")

        plt.legend()
        if plot_name == '../results/crosses/uniform-':
            plt.xlabel("Crossover Probability")
            plt.title("Fitness vs. Crossover")
        else:
            plt.xlabel("Mutation Probability")
            plt.title("Fitness vs. Mutation")
        

        plt.ylabel("Fitness")
        plt.show()

if __name__ == '__main__':
    fitness_vs_mutation_probability()




