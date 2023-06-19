from autoencoder import MultilayerPerceptron

from utils.noise import add_noise
from utils.parser import *
from config import load_config_multilayer
import numpy as np
import matplotlib.pyplot as plt
import copy


def is_same_letter(originals: list[float], predictions: list[float], max_errors=4):
    wrong_letters = []
    wrong_predictions = []
    for i in range(len(originals)):
        errors = 0
        letter = originals[i]
        letter_pred = predictions[i]
        for j in range(len(letter)):
            if letter[j] != int(letter_pred[j]):
                errors += 1
                if errors > max_errors:
                    wrong_letters.append(i)
                    wrong_predictions.append(letter_pred)
                    break
    return wrong_letters, wrong_predictions


hidden_layers = []

autoencoder = MultilayerPerceptron([35] + hidden_layers + [20] + hidden_layers[::-1] + [35], momentum=None)
letters, labels = load_data_as_bin_array('inputs/font.json')

# original_letters = copy.deepcopy(letters)

# print(letters)

noise_letters = add_noise(copy.deepcopy(letters), 0.1)

autoencoder.train(np.array(noise_letters), letters, 60000, 0.0005)


# Test set of noising letters and predict with the autoencoder latent space

test_noise = add_noise(copy.deepcopy(letters), 0.1)

wrong_letters, predictions = is_same_letter(letters, np.around(autoencoder.predict(test_noise), 0))


print("Wrong letters amount: {}".format(len(wrong_letters)))
for i in range(len(wrong_letters)):
    print("Wrong letter: {}".format(labels[wrong_letters[i]]))
    print(letters[wrong_letters[i]].reshape(7,5).astype(int))
    print(predictions[i].reshape(7,5).astype(int))
    print()

