from utils.parser import parse_combined_matrix
import numpy as np
from hopfield import Hopfield


input_file = './inputs/letters.txt'

# Parse the input file
matrix_dict = parse_combined_matrix(input_file)

# Print the matrix for each letter
for letter, matrix in matrix_dict.items():
    print(f'Matrix for letter {letter}:')
    print(np.array(matrix))
    print()

# test the hopfield algorithm
hopfield = Hopfield([np.array(matrix_dict['A']),np.array( matrix_dict['B']), np.array(matrix_dict['C']), np.array(matrix_dict['D'])])
print(hopfield.weights)
print(hopfield.patterns)
print(hopfield.dimension)
print(hopfield.train(matrix_dict['A'], 10))




