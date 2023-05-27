import numpy as np

class Hopfield:
    def __init__(self, patterns : list[np.ndarray]):
        self.patterns = patterns # [[1 -1  1  1 ... 1 (25)], ... 4 patterns]
        self.dimension = len(patterns[0]) # 25
        self.weights = np.zeros((self.dimension, self.dimension)) # 25x25
        for pattern in self.patterns:
            self.weights += np.dot(pattern.T, pattern) # 25x25
            
        # self.weights = np.dot(self.patterns.T, self.patterns) # 25x25
        np.fill_diagonal(self.weights, 0) # the diagonal is 0 (no self-connection)
    
    def train(self, untrained_pattern, epochs=1):
        states = []
        state = np.array(untrained_pattern) 
        previous_state = np.array(untrained_pattern)
        untrained_pattern = np.array(untrained_pattern)
        states.append(state)
        energies = []
        energies.append(self.energy(state))

        i = 0
        # stop when two consecutive states are equals or when the number of epochs is reached
        while i < epochs and not np.array_equal(state, previous_state):
            energies.append(self.energy(state))
            state = self.update(state)
            states.append(state)
            previous_state = states[-2]
            i += 1
        
        # if the state is a pattern, return True
        for pattern in self.patterns:
            if np.array_equal(pattern, state):
                return True, state, energies
        return False, state, energies # spurious state (not a pattern)
        


    # calculate the energy of the current state
    def energy(self, states):
        h = 0
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                h += self.weights[i][j] * states[i] * states[j]
        return -h
    
    # update the states of the network
    def update(self, state):
        new_state = np.zeros(self.dimension)
        for i in range(self.dimension):
            new_state[i] = np.sign(np.dot(self.weights[i], state))
        return new_state
    
        

