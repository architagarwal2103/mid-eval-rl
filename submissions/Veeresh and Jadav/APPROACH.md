

## The Problem

We have 10 ads, each with a hidden fixed click-through rate (CTR). We get 5,000
impressions total. Every time we show an ad, we observe a Bernoulli reward — click (1)
or no-click (0). The goal is to maximise total clicks over the 5,000 impressions,
averaged across 10 fixed random seeds.

The core difficulty: the top ads have **very close CTRs**, so telling the best ad
apart from the runner-up requires careful, sustained exploration — not just a quick
initial sweep.

---

## Algorithms Considered

### 1. ε-Greedy
The simplest approach: explore randomly with probability ε, exploit the current best
arm otherwise. The problem is that a fixed ε explores forever — even in round 4,999
when there's nothing left to learn. We ruled this out early.

### 2. UCB1
Upper Confidence Bound adds a bonus term `sqrt(2 * log(t) / n_i)` to each arm's
empirical mean, picking the arm with the highest optimistic estimate. This is
theoretically strong, but UCB1's confidence interval doesn't account for the actual
variance of a Bernoulli arm — it uses the same bonus regardless of whether the arm's
mean is 0.5 (high variance) or 0.05 (low variance).

### 3. Thompson Sampling
Draw one sample from each arm's Beta posterior (conjugate to Bernoulli), pick the
argmax. Elegant and empirically strong, but a single sample per arm can be unlucky
early on, causing premature commitment to a suboptimal arm.

### 4. UCB-Tuned ← **Final Choice**
UCB-Tuned improves on UCB1 by plugging in the **empirical variance** of each arm
instead of the worst-case 0.25 bound:

```
V_i = variance_i + sqrt(2 * log(t) / n_i)
bonus_i = sqrt((log(t) / n_i) * min(0.25, V_i))
score_i = mean_i + bonus_i
```

This gives tighter, more honest confidence intervals. Arms with low variance (clearly
bad or clearly good) get smaller bonuses; uncertain arms in the middle get larger ones.

---

## Our Key Innovation: The Squared Clock Factor

Standard UCB-Tuned explores throughout the entire horizon, even in the final rounds
where exploration can't be cashed in. We added a **horizon-aware decay** multiplied
onto the exploration bonus:

```python
clock_factor = ((horizon - t) / horizon) ** 2
score_i = mean_i + bonus_i * clock_factor
```

The effect:
| Progress | Clock Factor | Behaviour |
|----------|-------------|-----------|
| t = 500  (10%) | 0.81 | Strong exploration |
| t = 1000 (20%) | 0.64 | Careful exploration |
| t = 2500 (50%) | 0.25 | Shifting to exploitation |
| t = 4500 (90%) | 0.01 | Near-pure exploitation |
| t = 4999 (≈100%) | ≈0  | Pure exploitation |

The squaring (exponent 2) makes the transition smooth but aggressive — exploration
collapses fast in the second half of the run, ensuring the impressions we spent
learning are fully exploited.

---

## Full Algorithm

```
1. Warm-up: pull each arm once in round-robin (rounds 0–9)
   → ensures no division-by-zero and gives every arm a real first observation

2. Main phase (rounds 10–4999):
   means    = clicks / pulls
   variance = means * (1 - means)
   V        = variance + sqrt(2 * log(t) / pulls)
   bonus    = sqrt((log(t) / pulls) * min(0.25, V))
   clock    = ((horizon - t) / horizon)^2
   score    = means + bonus * clock
   → pull arm with highest score

3. Update: increment pulls[arm] and clicks[arm] on each reward
```

No hyperparameters to tune. No hard-coded arm indices or CTR values.

---

## Results

Evaluated on the live leaderboard (10 fixed seeds, 5,000 impressions, 10 ads):

| Metric | Value |
|--------|-------|
| Average clicks | **394.3** |
| Regret | **34.4** |
| CTR | **7.89%** |


The impression distribution confirmed correct identification: the single best ad
(Ad 3) received ~2,900 of 5,000 impressions on average, with the clock factor
successfully concentrating exploitation in the second half of the run.

The cumulative clicks curve tracked closely against the oracle (always-best-ad)
baseline, with the gap opening only during the early exploration phase and
essentially closing by t ≈ 1,500.

---

## What We Tried That Didn't Help

- **Larger exponent (cube instead of square):** Committed too early, hurt performance
  on seeds where the best arm took longer to identify.
- **Hard commit cutoff (pure greedy after t = 4800):** Slightly worse than the smooth
  decay; abrupt transitions caused edge cases on some seeds.
- **Optimistic Thompson Sampling (K=10 samples per arm):** Comparable performance to
  UCB-Tuned + clock but slightly more variance across seeds.

---

## Dependencies

Only `numpy` and the Python standard library. No internet, no file I/O, no extra packages.
