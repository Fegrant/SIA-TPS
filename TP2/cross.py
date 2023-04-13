import random
import numpy as np

from color import Color

def crossover_one_point(parent1: Color, parent2: Color):
    point = random.randint(0, 2)

    child1 = Color(0, 0, 0)
    child2 = Color(0, 0, 0)
    child1.set_rgb(*parent1.get_rgb()[:point], *parent2.get_rgb()[point:])
    child2.set_rgb(*parent2.get_rgb()[:point], *parent1.get_rgb()[point:])
    
    return child1, child2

def crossover_uniform(parent1: Color, parent2: Color):
    child1 = Color(0, 0, 0)
    child2 = Color(0, 0, 0)
    for i in range(3):
        if random.uniform(0, 1) <= 0.5:
            child1.rgb[i] = parent1.rgb[i]
            child2.rgb[i] = parent2.rgb[i]
        else:
            child1.rgb[i] = parent2.rgb[i]
            child2.rgb[i] = parent1.rgb[i]
    return child1, child2