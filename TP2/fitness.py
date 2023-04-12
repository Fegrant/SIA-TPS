import numpy as np
from color import color

def calculate_fitness(target_color: color, individual: color):
    # Calculate Euclidean distance between the solution color and the desired color
    distance = np.linalg.norm(target_color.rgb - individual.rgb)
    # Return inverse distance as fitness score
    return 1 / (1 + distance)
    
