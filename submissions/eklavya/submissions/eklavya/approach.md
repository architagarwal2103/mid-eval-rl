# APPROACH.md — Ad Bandit Challenge

## Team
Eklavya

## The Problem
We have 10 ads with hidden click-through-rates (CTRs). Over 5,000 impressions,
we need to figure out which ad is best and show it as often as possible.
This is a classic **Bernoulli multi-armed bandit** — a pure explore vs exploit problem.

## Our Algorithm: UCB1 → Thompson Sampling → Greedy

We split the 5,000 impressions into three phases:

### Phase 1: UCB1 (first 250 steps = 5%)
**Upper Confidence Bound** — instead of picking the ad with the best *average* 
click rate seen so far, we pick the ad with the best *plausible* click rate:

```
score(arm) = average_ctr(arm) + sqrt(2 * log(t) / n_pulls(arm))
```

The bonus term is large when an arm hasn't been tried much, shrinks as we gather 
evidence. This forces us to try uncertain arms rather than getting stuck early.
We also pull each arm once before UCB kicks in (initialisation).

### Phase 2: Thompson Sampling (steps 250–4600 = main phase)
We maintain a **Beta distribution** as our belief about each ad's CTR.
Every round:
1. Sample one value from each ad's Beta distribution
2. Show the ad with the highest sample

Ads we're uncertain about occasionally sample high → we explore them.
Ads that have clearly failed sample low → we stop wasting impressions.
As evidence piles up, the distributions tighten around the true CTR.

**Prior choice:** We used `Beta(1, 18)` which encodes a prior belief that ads 
have ~5% CTR. This pessimistic prior makes the algorithm explore genuinely 
uncertain arms rather than getting fooled by lucky early clicks.

### Phase 3: Pure Greedy (last 400 steps = 8%)
Since we know the horizon, we can stop exploring completely near the end.
We compute the **posterior mean** `alpha / (alpha + beta)` for each arm 
and exploit the best one for the remaining impressions.

## Why This Works

The key insight is **using the clock**: exploration early, exploitation late.
- UCB1 quickly rules out clearly bad ads in the first ~250 impressions
- Thompson Sampling efficiently handles the close-competition phase
- Greedy ending wastes zero impressions on exploration when we can't benefit

## Results

| Policy | Clicks (avg/10 seeds) | Gap closed vs random |
|--------|----------------------|---------------------|
| Random baseline | 336.3 | 0% |
| Vanilla Thompson Sampling | 477.3 | 66% |
| **Our UCB + Thompson + Greedy** | **509.1** | **81%** |
| Optimal (oracle) | 550.0 | 100% |

## What We Tried

1. **Pure random** → 336 clicks, obvious baseline
2. **Vanilla Thompson Sampling** (Beta(1,1) prior) → 477 clicks / 66% gap
3. **Adding UCB1 warm-start** → pushed to ~495 clicks
4. **Tuning the prior** to Beta(1,18) → better than uniform Beta(1,1)
5. **Adding greedy finale** in last 8% → squeezed more clicks at end
6. **Grid-searched** prior parameters and phase boundaries → 509 clicks / 81%

## Key Concepts We Learned

- **Explore-exploit tradeoff**: spending impressions on bad ads costs you clicks
- **Regret**: clicks you missed vs. always showing the best ad (we got ~46 regret)
- **Beta-Bernoulli model**: a clean Bayesian way to track click probability beliefs
- **UCB1**: proven algorithm with logarithmic regret guarantees
- **Thompson Sampling**: works well empirically, Bayesian and elegant

## What Could Be Even Better

- **KL-UCB**: tighter confidence bounds for Bernoulli rewards
- **Bayes-UCB**: use quantiles of the posterior instead of samples
- Adaptive prior based on observed click rates early in the run
