# policy.py — UCB1 warm-start + Thompson Sampling + greedy finale
#
# Algorithm:
#   Phase 1 (first 5% = 250 steps): UCB1
#     → Rapidly eliminates clearly bad arms with principled optimism bonus
#   Phase 2 (250 → 4600 steps): Thompson Sampling (Beta-Bernoulli posterior)
#     → Bayesian exploration — uncertain arms occasionally sample high and get tried
#     → Naturally focuses on good arms as evidence accumulates
#   Phase 3 (last 8% = 400 steps): Pure greedy on posterior mean
#     → No more exploration needed; cash in all remaining impressions on best arm
#
# Prior: Beta(1.0, 18.0) encodes "ads have ~5% CTR on average" — pessimistic
# enough to make UCB/Thompson explore genuinely uncertain arms, not just lucky ones.

import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms  = n_arms
        self.horizon = horizon
        self.t       = 0

        # Beta prior: pessimistic on CTR, encourages real exploration
        self.alpha = np.ones(n_arms)         # successes  + prior
        self.beta  = np.full(n_arms, 18.0)   # failures   + prior

        # For UCB1 phase
        self.counts  = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)

        self.ucb_end      = max(n_arms, int(horizon * 0.05))   # end of UCB phase
        self.greedy_start = int(horizon * 0.92)                # start of greedy phase

    def select_arm(self):
        t = self.t

        # Phase 1: UCB1 — optimistic under uncertainty
        if t < self.ucb_end:
            if t < self.n_arms:
                return t  # pull each arm exactly once to initialise
            avg   = self.rewards / np.maximum(self.counts, 1e-9)
            bonus = np.sqrt(2.0 * np.log(t + 1) / np.maximum(self.counts, 1e-9))
            return int(np.argmax(avg + bonus))

        # Phase 3: Greedy on posterior mean — exploit what we know
        if t >= self.greedy_start:
            return int(np.argmax(self.alpha / (self.alpha + self.beta)))

        # Phase 2: Thompson Sampling
        return int(np.argmax(np.random.beta(self.alpha, self.beta)))

    def update(self, arm, reward):
        self.t += 1
        self.counts[arm]  += 1
        self.rewards[arm] += reward
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm]  += 1
