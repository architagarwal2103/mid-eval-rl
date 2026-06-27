Approach- Team "Suhani and Antara"

1. Understanding the problem:
10 ads, 5,000 impressions, hidden fixed click-rates, Bernoulli rewards. Score is total clicks averaged over 10 seeds, so we need a policy that finds the best ad quickly and stops wasting impressions once it's confident. The problem statement warns the top ads have very close click-rates on purpose, so the hard part isn't finding a decent ad,jjjj it's telling the best ad apart from the runner-up without overspending on exploration.

2. What we chose and why:
We started from the given baseline, plain Thompson Sampling with a uniform 'Beta(1, 1)' prior, redrawing one random sample per ad each round and playing the highest. This explores well early but has no sense of "the clock", it treats round 1 and round 4,999 the same way.

Our first improvement made the policy clock-aware: blend a random Thompson
sample with the plain running average, weighted by how far through the
5,000 rounds we are ('progress = (t / horizon) ** exponent'). Early on, trust
the random sample (explore); late on, trust the average (exploit). This
already beat the baseline clearly.

We then refined this further with two changes:

- Lowered the exploitation exponent. We first tried steepening it
  (exponent = 2.0), expecting faster commitment to help instead it made things worse, because it started trusting the average too early, before there was enough evidence. This told us to push the exponent down instead, which we kept doing until it stopped helping (final value: '0.2').
- Switched the prior from uniform to Jeffreys ('Beta(0.5, 0.5)')** and added a forced warm-up, every ad is guaranteed 5 shows before the policy trusts any comparison between them. Since the top ads are deliberately close, leaving the first few looks entirely to random sampling risks unluckily under-trying the actual best ad early on. A guaranteed minimum exposure protects against that.

3. What else we tried:
Variant                       Local avg clicks             Notes
Thompson Sampling             477.3 (regret 72.3, 66%      Explores well but has no sense 
(given baseline, uniform      gap closed)                  of how much tim is life, treats 
prior)                                                     round 1 and round 4999
                                                           identically

Initial close aware           512.6(regret 41.3, 82% gap   Blending a random sample 
Thompson Sampling (Jeffreys   closed)                      with the running average, 
prior, exponent=1.5)                                       weighted by progress through 
                                                           horizon, already beat the 
                                                           baseline clearly

Clock aware TS, steeper       497.9(regret 51.7, 76% gap   Worse than the initial version,                                            (exponent=2.0)                closed)                      shifted to pure exploitation  
                                                           too early costs clicks before  
                                                           there is enough evidence to  
                                                           commit confidently 

Final: Jeffery prior+         526.7(regret 27.6, 89% gap   Best result, guaranteeing   
forced warm up (5/arm)        closed)                      every ad a fair early look 
+ gentle exponent(0.2)                                     (important since the top ads  
                                                           are so close) combined with 
                                                           leaning toward exploration for longer overall.

4. Results:
 - Local practice score: 526.7 clicks, regret 27.6 (closed ~89% of the
  random→optimal gap).
 - Best website leaderboard score: 409.8, rank #3

5. What we'd do next:
Tune the warm-up length ('min_rounds') more precisely we picked 5 somewhat heuristically and didn't have time to sweep it properly. We'd also like to try scaling the warm-up with 'n_arms' and 'horizon' directly, so it generalizes better to bandit instances with a different number of arms or horizon, rather than using a fixed constant.

6. How to reproduce:
cd starter
python local_test.py "../submissions/Antara and Suhani/policy.py"


