import numpy as np

class Policy:

    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms

        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.c = 0.05    # Parameter

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm]==0:
                return arm
        
        total = np.sum(self.counts)
        ucb_values = self.values + np.sqrt(self.c*np.log10(total)/self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm]+=1
        n=self.counts[arm]
        val=self.values[arm]
        self.values[arm]+=(reward-val)/n