from variational_autoencoder import VariationalAutoencoder
from utils.parser import *
from utils.print_letter import *
from utils.plots import biplot, biplot_with_new_letter

import numpy as np
import matplotlib.pyplot as plt


hidden_layers = [28, 22, 17, 10]

vae = VariationalAutoencoder([35] + hidden_layers + [2] + hidden_layers[::-1] + [35], momentum=None)
letters, labels = load_data_as_bin_array('inputs/font.json')

vae.train(letters, 20000, 0.00005)

new_letter = vae.generate_samples(1)

print_letter(new_letter)
