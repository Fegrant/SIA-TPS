# Fitness vs method used
import pandas as pd
import matplotlib.pyplot as plt

stochastic_files = [
    './results/selections/elite-250_elite-250.csv',
    './results/selections/roulette-250_roulette-250.csv',
    './results/selections/universal-250_universal-250.csv',
    './results/selections/deterministic_tournament-250_deterministic_tournament-250.csv'
]

def fitness_by_stochastic_method():
    for file in stochastic_files:
        df = pd.read_csv(file)
        groupby_run = df.groupby('run')

fitness_by_stochastic_method()

#df1 = pd.read_csv('./results/selections/use_all.csv')
#df1 = pd.read_csv('./results/selections/use_all.csv')
#df1 = pd.read_csv('./results/selections/use_all.csv')
#df1 = pd.read_csv('./results/selections/use_all.csv')
