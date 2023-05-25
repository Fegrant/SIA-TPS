import numpy as np


class Hopfield:
    def __init__(self, query_pattern) -> None:
        precalc= np.outer(query_pattern, query_pattern) # !! not true we have to compute all patterns but the query pattern
        self.previous_state = None
        self.state = query_pattern
        upper_triangle = np.triu(precalc, k=1)  # Generating upper triangle
        self.weights = upper_triangle + upper_triangle.T  # Adding upper triangle to its transpose

    def update(self):
        stable = False
        limit = 100
        while (not stable)&(limit>0):
          self.previous_state = self.state
          self.state = np.sign(self.weights @ self.state)
          limit -= 1
          if(np.sign()):
              stable = True
        return self.state
    
        

