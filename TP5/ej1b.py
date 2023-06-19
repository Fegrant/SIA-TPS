from autoencoder import MultilayerPerceptron

from utils.noise import add_noise
from utils.parser import *
from config import load_config_multilayer
import numpy as np
import matplotlib.pyplot as plt

hidden_layers = [28, 22, 17, 10]

autoencoder = MultilayerPerceptron([35] + hidden_layers + [2] + hidden_layers[::-1] + [35], momentum=None)
letters, labels = load_data_as_bin_array('inputs/font.json')

denoising_letters = []
aux = add_noise(letters, 0.1)
denoising_letters.append(np.copy(aux))

autoencoder.train()