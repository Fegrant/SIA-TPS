from oja import OjaPerceptron
from sklearn.preprocessing import StandardScaler
from config import load_oja_config
from matplotlib import pyplot as plt
import numpy as np


from utils.parser import parse_csv_file

df = parse_csv_file('./inputs/europe.csv')
countries = df["Country"].to_numpy()
df.drop(columns=["Country"], axis=1, inplace=True)
cells = list(df.columns)
inputs = StandardScaler().fit_transform(df.values)

config = load_oja_config()

epochs = int(config['epochs'])
learning_rate = float(config['learning_rate'])

oja = OjaPerceptron(inputs, learning_rate)
pca1 = oja.train(epochs)

print(f"Oja eigenvector that builds PC1:\n {pca1}")
countries_pca1 = [np.inner(pca1,inputs[i]) for i in range(len(inputs))]
libray_pca1 = [0.12487390183337656,-0.5005058583604993,0.4065181548118897,-0.4828733253002008,0.18811161613179747,-0.475703553912758,0.27165582007504635]
countries_library_pca1 = [-np.inner(libray_pca1,inputs[i]) for i in range(len(inputs))]
fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12, 10))
bar1 = ax1.bar(countries,countries_pca1)
bar2 = ax2.bar(countries,countries_library_pca1)
ax1.set_ylabel('PCA1')
ax1.set_title('PCA1 per country using Oja')
ax2.set_ylabel('PCA1')
ax2.set_title('PCA1 per country using Sklearn')
ax1.set_xticks(range(len(countries)))
ax2.set_xticks(range(len(countries)))
ax1.set_xticklabels(countries, rotation=90)
ax2.set_xticklabels(countries, rotation=90)
plt.show()