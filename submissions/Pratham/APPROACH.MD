# Approach — Team "Pratham" (Breaking Bank)

## 1. The problem
This is a standard **mulit-armed bandit** problem :
- The scenario is that there are 10 ads that can be shown to a user. (No. of arms = 10)
- Each ad has a hidden Click-Through-Rate (CTR) which is the probability that a user will click on it (reward = 1)
- There are a total of 5000 impressions (Horizon = 5000)
- This will be carried out over 10 seeds and the average clicks is the final score.

## 2. What we chose and why
**Upper Confidence Bound** was the algorithm with the best score:
- This algorithm focuses both on the mean but also on its confidence level.
- The more an ad is showed, the more confident we are of its expected probability being accurate.
- So, the ad less explored is more likely to have a high "score" which increases chances of picking that ad.

This algorithm balances exploration and exploitation aspect.
- Exploitation : Always the ad with best score is picked.
- Exploration : Ensures that less confident scores are more likely to be picked until their mean gets settled.

` UCB_score = mean + √( c * log(t) / N) `

where
- mean = the experimental probability of the ad
- c = parameter
- t = time or total no. of ads shown
- N = no. of times that particular ad was shown

We also force one pull of every ad such that no ad starts with a blind estimate.
(Also, so that there is no dividing by 0)

Another algorithm implemented was MOSS (Minimax Optimal Strategy in the Stochastic case) which is a variant of UCB. Like UCB, it also balances the explorationa and exploitation, but it adjusts the exploration bonus.

` score = mean + √( c* max (0, log( horizon / (n * N))) / N) `

where
- mean = the experimental probability of the ad
- c = parameter
- horizon = total no of ads to be shown = 5000
- n = no. of ads that can be picked = 10
- N = no. of times that particular arm was pulled

## 3. The Local Test Used

The initial local test provided had fixed CTR and less seeds to run the policy on.
So, the local test was changed to 100 seeds and the CTR were randomised to make it more fair and prevent biased results.

Due to the nature of the code for randomness, the CTR for a particular ad in a particular seed number remains same. So, the clicks with optimal and random policy remain same.

- ` Optimal = 862.9 `
- ` Random = 435.1 `

## 4. What else we tried

| Variant                               | Local Avg. Clicks | Gap Closed | Notes                                                         |
|---------------------------------------|------------------:|-----------:|---------------------------------------------------------------|
| Fixed ε-greedy (ε = 0.1)              | 758.0             | 75%        | Never stops exploring.                                        |
| Decaying ε-greedy (ε = c/t)           | 794.4             | 84%        | Best parameter was **c = 45**.                                         |
| Decaying ε-greedy (ε = c/√t)          | 788.5             | 83%        | Best parameter was **c = 3**.                                          |
| Explore-Then-Commit (41 pulls/arm)    | 792.6             | 84%        | Performed better than expected since the horizon and *n* were known. |
| Thompson Sampling                     | 777.3             | 80%        | Performed worse than expected on the local benchmark.          |
| Thompson Sampling (Modified Prior)    | 789.6             | 83%        | Lower initial prior improved performance.                     |
| **Upper Confidence Bound (UCB)**      | **812.1**         | **88%**    | Best among the algorithms taught.                   |
| **MOSS**                              | **820.9**         | **90%**    | Highest score on the local benchmark.                         |

## 5. Results

From the table, it might seem that the MOSS is better than UCB. But when submitting both of the policies to the LeaderBoard, it was found that the UCB had a greater score than MOSS. (This could be either due to wrong parameter, unlucky test cases or wrong implementation in MOSS).

- ` UCB = 399.5 `
- ` MOSS = 397.0 `

So, UCB was the best algorithm found

- Local practice score: **812 clicks** (closed **88%** of the random→optimal gap).
- Best website leaderboard score: **399.5**, rank **6**.

## 6. What we'd do next
We can add other features to the scenario similar to the variations of the Bandit problem.
Then, we can test our policies against them to see their performance.