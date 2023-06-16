from multilayer_perceptron import MultilayerPerceptron

from utils.parser import *
from config import load_config_multilayer
import numpy as np
import matplotlib.pyplot as plt

def is_same_letter(originals: list[float], predictions: list[float], max_errors=1):
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

# letters, labels = load_data('inputs/font.json')
# monocromatic_cmap = plt.get_cmap('binary')
# plt.imshow(to_bin_array(letters[2]), cmap=monocromatic_cmap)
# plt.show()

autoencoder = MultilayerPerceptron([35, 22, 10, 2, 10, 22, 35], 0.9)
letters, labels = load_data_as_bin_array('inputs/font.json')
# print(letters)
autoencoder.train(letters, letters, 20000, 0.005)

# print("a: ", letters[1].reshape(7, 5))
# print("a_pred: ", np.around(autoencoder.predict(letters[1]).reshape(7, 5), 3))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[1]), letters[1])))
# print(autoencoder.predict(letters[1]))

wrong_letters, predictions = is_same_letter(letters, np.around(autoencoder.predict(letters), 0))

print(wrong_letters)

# if len(wrong_letters) >= 0:
#     for i in range(len(wrong_letters)):
#         print('Wrong letter: {}'.format(labels[wrong_letters[i]]))
#         print('Original:\n{}'.format(letters[wrong_letters[i]].reshape(7, 5)))
#         print('Predicted:\n{}'.format(predictions[i].reshape(7, 5)))
#         print()
# else:
#     print('All letters were guessed correctly')

# print("b prediction")
# print("b: ", autoencoder.predict(letters[2]))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[2]), letters[2])))

# print("c prediction")
# print("c: ", autoencoder.predict(letters[3]))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[3]), letters[3])))

# print("d prediction")
# print("d: ", autoencoder.predict(letters[4]))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[4]), letters[4])))
