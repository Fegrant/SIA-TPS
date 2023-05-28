import numpy as np

class Hopfield:
    def __init__(self, patterns:list[np.ndarray]):
        self.patterns = patterns # [[1 -1  1  1 ... 1 (25)], ... 4 patterns]
        self.dimension = len(patterns[0].flatten()) # 25
        self.weights = np.zeros((self.dimension, self.dimension)) # matrix 25x25
        # for pattern in self.patterns:
        #     vector= np.array(pattern).flatten()
        #     self.weights += np.dot(vector.T, vector) # 25x25
        K=np.zeros((self.dimension,len(self.patterns)))
        K=np.dstack([pattern.flatten() for pattern in self.patterns])
        K=K[0]
        print(K.shape)
        self.weights=np.dot(K, K.T) 
        # self.weights = np.dot(self.patterns.T, self.patterns) # 25x25
        np.fill_diagonal(self.weights, 0) # the diagonal is 0 (no self-connection)
        self.weights=np.divide(self.weights, self.dimension) # divide by the dimension of the patterns (25)
    
    def train(self, untrained_pattern, epochs=1):
        states = []
        untrained_pattern = np.array(untrained_pattern)
        state = untrained_pattern
        previous_state = np.zeros(self.dimension)
        states.append(state)
        energies = []
        # energies.append(self.energy(state))

        i = 0
        # stop when two consecutive states are equals or when the number of epochs is reached
        while i < epochs and not np.array_equal(state, previous_state):
            # energies.append(self.energy(state))
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
    # def energy(self, states):
    #     h = 0
    #     for i in range(self.dimension):
    #         for j in range(i+1, self.dimension):
    #             h += self.weights[i][j] * states[i] * states[j]
    #     return -h
    
    # update the states of the network
    def update(self, state:np.ndarray):
        new_state = np.zeros(self.dimension)
        temp = state.flatten()
        print(temp.T.shape)
        for i in range(self.dimension):
            new_state[i] = np.dot(self.weights[i], temp)
        new_state = np.sign(new_state)
        new_state = new_state.reshape(5,5)
        return new_state
    
        

