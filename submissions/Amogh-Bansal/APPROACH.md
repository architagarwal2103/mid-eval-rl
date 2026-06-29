# Approach — UCB1 with a tuned exploration constant

## 1. Understanding the problem

I am handed an ad slot with **10 ads**, **5,000 impressions**, and a hidden,
fixed click-through-rate (CTR) per ad. Each impression I pick one ad and the user
clicks (reward = 1) or not (reward = 0). The score is total clicks averaged over
10 fixed seeds.

This is a stationary Bernoulli multi-armed bandit, and the whole difficulty is
the **explore vs exploit tradeoff**:

- *Exploit* the ad that looks best so far and cash in clicks, but risk crowning
  the wrong ad after too little data.
- *Explore* uncertain ads to learn their true CTR, but waste impressions on ads
  that turn out to be weak.

I measure performance by **regret**: clicks lost versus an oracle that always
shows the single best ad. On these CTRs the best ad clicks about 11% of the
time, so the oracle scores ~550 clicks and random scores ~336. Lower regret =
more clicks.


## 2. What I chose, and why

I use **UCB1**, which follows the principle of "optimism in the face of
uncertainty." Instead of judging an ad by its observed average, I judge it by the
best value it could *plausibly* have given how little I know, and play the most
optimistic ad each round:

$$\text{UCB}_i = \underbrace{\bar{x}_i}_{\text{exploitation}} + \underbrace{\sqrt{\frac{c\,\ln t}{n_i}}}_{\text{exploration bonus}}$$

- $\bar{x}_i$ is ad $i$'s average reward so far, which pulls me toward known good ads.
- The bonus is large when an ad is barely tested ($n_i$ small) and shrinks to
  zero as I gather data. The $\ln t$ term gently keeps abandoned ads eligible
  for a recheck without ever exploding.

I chose UCB1 over the alternatives for two reasons:

1. **It explores the *uncertain* arms specifically**, not uniformly at random
   like $\varepsilon$-greedy. That is exactly what this problem needs, where the
   hard part is resolving the close top arms.
2. **It is deterministic**, so the same history always gives the same choice,
   making it reproducible and easy to reason about.

I initialize by showing each ad once (10 warm-up rounds) so no ad starts with a
divide-by-zero in the bonus term.

## 3. Tuning the exploration constant

UCB1's textbook constant is $c = 2$, which exists to guarantee a worst-case
regret bound. This problem is not worst-case: the weak ads are clearly weak, so
the default bonus keeps re-checking known losers and bleeds impressions. I swept
$c$ on the same 10 seeds used in the local test:

| c       | Avg clicks | Regret | Gap closed (random→optimal) |
|---------|-----------:|-------:|----------------------------:|
| 2.0     | 382.9      | 164.0  | 22%                         |
| 1.0     | 408.2      | 141.7  | 34%                         |
| 0.5     | 439.8      | 116.9  | 48%                         |
| 0.1     | 493.4      |  59.9  | 74%                         |
| 0.05    | 511.4      |  43.9  | 82%                         |
| **0.025** | **523.2**  | **30.9** | **87%**                  |

The trend is **monotonic**: every reduction in $c$ improved the score. I
submitted **c = 0.025**.

The *reason* smaller $c$ wins is that optimism-based exploration is calibrated for a worst case that does not occur here. When the bad arms are easy to identify, less exploration means faster commitment to a strong arm, and fewer impressions wasted retesting the losers.

Driving $c$ this low effectively turns UCB1 into near greedy behavior, which works
here because the weak ads separate quickly and the seeds are fixed. If the
situation changed (more arms, genuinely close CTRs across the board, or different
seeds), I would need to retune this parameter, since such low exploration would
risk committing to the wrong arm.

## 4. Results

- **Submitted policy:** UCB1, $c = 0.025$.
- **Local practice score:** 523.2 clicks (regret 30.9), closing ~87% of the
  random→optimal gap.
- **Website leaderboard (seperate hidden CTRs):** Clicks: 409.9, CTR: 8.20%, Regret: 18.1, **Rank: 2nd** 

## 5. What I'd do next

One improvement is **KL-UCB**, which replaces the square root bonus with an information theoretic confidence bound that is provably near optimal for Bernoulli rewards. It targets exploration tightly *without* a constant I have to tune by hand.

## How to reproduce

```bash
cd starter
python local_test.py "../submissions/Amogh-Bansal/policy.py"
```