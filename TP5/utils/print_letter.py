import matplotlib.pyplot as plt


def print_letter(letter):
    monocromatic_cmap = plt.get_cmap('binary')
    plt.imshow(letter, cmap=monocromatic_cmap)
    plt.show()
