from config import load_hopfield_config
from utils.parser import parse_combined_matrix
import numpy as np
from hopfield import Hopfield


input_file = './inputs/letters.txt'

# Parse the input file
matrix_dict = parse_combined_matrix(input_file)

# Print the matrix for each letter
# for letter, matrix in matrix_dict.items():
#     print(f'Matrix for letter {letter}:')
#     print(np.array(matrix))
#     print()

#load the hopfield config
config = load_hopfield_config()

# test the hopfield algorithm
# hopfield = Hopfield([matrix_dict['A'], matrix_dict['B'], matrix_dict['C'], matrix_dict['D']])
hopfield = Hopfield([np.array(matrix_dict['A']),np.array( matrix_dict['B']), np.array(matrix_dict['C']), np.array(matrix_dict['D'])])

# print("Weights:")
# print(hopfield.weights)
# print(hopfield.patterns)
# print("Dimension:")
# print(hopfield.dimension)
matrix_test = matrix_dict['A']
matrix_test[0][0] = -1
print(hopfield.train(matrix_test, 10))




