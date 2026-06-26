import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.t = 0
        
        # Track purely empirical data to adapt to any random seed's CTR scale
        self.pulls = np.zeros(n_arms)
        self.clicks = np.zeros(n_arms)

    def select_arm(self):
        # Phase 1: Round-Robin Baseline
        # Give every ad exactly one chance so we don't divide by zero later.
        if self.t < self.n_arms:
            return self.t

        # Phase 2: UCB-Tuned
        means = self.clicks / self.pulls
        
        # Calculate the exact empirical variance per ad (p * (1-p))
        variance = means * (1.0 - means)
        
        # UCB-Tuned calculates a specific variance upper bound
        V = variance + np.sqrt(2.0 * np.log(self.t) / self.pulls)
        
        # Standard UCB-Tuned exploration bonus (capped at max Bernoulli variance of 0.25)
        bonus = np.sqrt((np.log(self.t) / self.pulls) * np.minimum(0.25, V))
        
        # Phase 3: The Squared Clock Factor
        # t=1000 (20% done) -> factor is 0.64 (Still carefully exploring)
        # t=2500 (50% done) -> factor is 0.25 (Aggressively committing)
        # t=4500 (90% done) -> factor is 0.01 (Pure exploitation, no wasted clicks)
        clock_factor = ((self.horizon - self.t) / self.horizon) ** 2
        
        # Final Score = Empirical Mean + (Tuned Bonus * Shrinking Clock)
        scores = means + (bonus * clock_factor)
        
        return int(np.argmax(scores))

    def update(self, arm, reward):
        # Update trackers and increment the universal clock
        self.pulls[arm] += 1
        self.clicks[arm] += reward
        self.t += 1