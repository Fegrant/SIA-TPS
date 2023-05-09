import pandas as pd
import matplotlib.pyplot as plt

from statistics import mean, stdev

df = pd.read_csv('ej2_sets_proportions.csv')
x = df['test_proportion'].unique() * 100

x_ticks = [i for i in range(10, 100, 10)]
width = 0.3

groupby_method = df.groupby('method')

### Lineal ###
for name_method, grouped_method in groupby_method:
    plt.xlabel("Test set %", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title("Error en ambos sets (" + name_method + ")")
    plt.errorbar(x, grouped_method['test_error'], yerr=grouped_method['test_error_std'], marker='o', label='Test')
    plt.errorbar(x, grouped_method['train_error'], yerr=grouped_method['train_error_std'], marker='o', label ='Training')
    
    plt.yscale('log')
    plt.xticks(x_ticks)
    plt.legend()
    plt.show()