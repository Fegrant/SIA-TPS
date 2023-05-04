import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_rate_3b():
    fig, ax = plt.subplots()
    df = pd.read_csv('csvs/learning_rates_3b.csv')
    ax.errorbar(df['learning_rate'], df['error'], yerr=df['error_std'], linestyle='dotted', fmt='o', capsize=4, label='Modelo 3B')

    ax.set_ylabel('Error medio', fontsize=12)
    ax.set_xlabel('Tasa de aprendizaje', fontsize=12)

    plt.legend()
    plt.show()


def plot_learning_rate_3c():
    fig, ax = plt.subplots()
    df = pd.read_csv('csvs/learning_rates_3c.csv')
    ax.errorbar(df['learning_rate'], df['error'], yerr=df['error_std'], linestyle='dotted', fmt='o', capsize=4, label='Modelo 3C', color='tab:orange')

    ax.set_ylabel('Error medio', fontsize=12)
    ax.set_xlabel('Tasa de aprendizaje', fontsize=12)
    plt.legend()
    plt.show()


def plot_inner_layer():
    fig, ax = plt.subplots()
    df1 = pd.read_csv('csvs/inner_layer_3b.csv')
    df2 = pd.read_csv('csvs/inner_layer_3c.csv')

    ax.errorbar(df1['perceptron_amount'], df1['error'], yerr=df1['error_std'], linestyle='dotted', fmt='o', capsize=4, label='Modelo 3B')
    ax.errorbar(df2['perceptron_amount'], df2['error'], yerr=df2['error_std'], linestyle='dotted', fmt='o', capsize=4, label='Modelo 3C')

    ax.set_ylabel('Error medio', fontsize=12)
    ax.set_xlabel('Cantidad de perceptrones en capa interna', fontsize=12)
    plt.legend()
    plt.show()


def plot_inner_layers():
    fig, ax = plt.subplots()
    df1 = pd.read_csv('csvs/inner_layers_3b.csv')
    df2 = pd.read_csv('csvs/inner_layers_3c.csv')

    ax.errorbar(df1['perceptron_amount'], df1['error'], yerr=df1['error_std'], linestyle='dotted', fmt='o', capsize=4, label='Modelo 3B')
    ax.errorbar(df2['perceptron_amount'], df2['error'], yerr=df2['error_std'], linestyle='dotted', fmt='o', capsize=4, label='Modelo 3C')

    ax.set_ylabel('Error medio', fontsize=12)
    ax.set_xlabel('Cantidad de perceptrones en capa interna', fontsize=12)
    plt.legend()
    plt.show()


def plot_noise():
    fig, ax = plt.subplots()
    df = pd.read_csv('csvs/noise_data_3c.csv')

    ax.errorbar(df['noise_rate'], df['error'], yerr=df['error_std'], linestyle='dotted', fmt='o', capsize=4, color='tab:orange')

    ax.set_ylabel('Error medio', fontsize=12)
    ax.set_xlabel('Tasa de ruido', fontsize=12)
    plt.legend()
    plt.show()


def plot_generalization():
    fig, ax = plt.subplots()

    x = [0, 1]
    width = 0.25
    multiplier = 0

    data = [
        ['Modelo 3B','no',0.03719360897512953,0.0031722432937496773],
        ['Modelo 3B','yes',0.5480272061422325,0.19220319654158508],
        ['Modelo 3C','no',0.22941458717851743,0.06630437116390327],
        ['Modelo 3C','yes',1.0,0.0]
    ]

    i = 0
    for j in range(len(data)):
        offset = (j+i) * width
        if data[j][1] == 'no':
            color = 'tab:blue'
        else:
            color = 'tab:orange'
        bar = ax.bar(offset, data[j][2], width, label=data[j][0], color=color)
        ax.errorbar(offset, data[j][2], yerr=data[j][3], fmt='o', color='black')
        if j == 1:
            i += 1
    
    ax.set_ylabel('Error medio', fontsize=12)
    plt.xticks([])
    plt.show()


plot_generalization()
