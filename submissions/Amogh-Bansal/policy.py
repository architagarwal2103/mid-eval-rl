import numpy as np

c = 0.025

class Policy:


    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms        
        self.horizon = horizon      
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        
        means = self.sums/self.counts
        bonus = np.sqrt(c*np.log(self.t)/self.counts)
        ucb = means + bonus
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.t += 1
        self.counts[arm] += 1
        self.sums[arm] += reward