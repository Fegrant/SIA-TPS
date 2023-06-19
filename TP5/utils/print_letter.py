import matplotlib.pyplot as plt
import numpy as np


def print_letter(letter, index):
    monocromatic_cmap = plt.get_cmap('binary')
    plt.imshow(letter)
    plt.savefig("letter_{}.png".format(index))

def print_noise_letters(original_letters, noise_letters, predictions):
    fig, axes = plt.subplots(3, original_letters.shape[0], figsize=(7,5))
    for i in range(original_letters.shape[0]):
        axes[0, i].set_yticklabels([])
        axes[0, i].set_xticklabels([])
        axes[1, i].set_yticklabels([])
        axes[1, i].set_xticklabels([])
        axes[2, i].set_yticklabels([])
        axes[2, i].set_xticklabels([])
        axes[0, i].imshow(original_letters[i].reshape(7,5))
        axes[1, i].imshow(noise_letters[i].reshape(7,5))
        axes[2, i].imshow(predictions[i].reshape(7,5))
        axes[0, 15].set_title("Original")
        axes[1, 15].set_title("Noise")
        axes[2, 15].set_title("Prediction")
    plt.show()

def print_letters(original_letters, predictions):
    monocromatic_cmap = plt.get_cmap('binary')
    fig, axes = plt.subplots(2, original_letters.shape[0], figsize=(7,5))
    for i in range(original_letters.shape[0]):
        axes[0, i].set_yticklabels([])
        axes[0, i].set_xticklabels([])
        axes[1, i].set_yticklabels([])
        axes[1, i].set_xticklabels([])
        # axes[0, i].axis('off')
        # axes[1, i].axis('off')
        axes[0, i].imshow(original_letters[i].reshape(7,5), cmap=monocromatic_cmap)
        axes[1, i].imshow(predictions[i].reshape(7,5), cmap=monocromatic_cmap)
        axes[0, 15].set_title("Original")
        axes[1, 15].set_title("Prediction")
        
    plt.show()

