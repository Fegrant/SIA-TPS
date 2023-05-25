from utils.parser import parse_combined_matrix
import numpy as np

input_file = './inputs/letters.txt'

# Parse the input file
matrix_dict = parse_combined_matrix(input_file)

# Print the matrix for each letter
for letter, matrix in matrix_dict.items():
    print(f'Matrix for letter {letter}:')
    print(np.array(matrix))
    print()



