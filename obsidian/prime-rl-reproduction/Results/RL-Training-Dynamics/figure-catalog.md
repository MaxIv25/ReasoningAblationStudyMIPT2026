# Figure catalog — RL training dynamics

## `accuracy-and-length.{pdf,png}`

- **Purpose:** compare correctness dynamics and response-length confounding.
- **Data:** TRL log dictionaries; 18 GRPO, 1 PRIME and 10 DPO-Z snapshots.
- **Caption requirements:** state single seed, incomplete trajectories, no
  smoothing; dotted orange line marks PRIME termination after step 9.
- **Notice:** GRPO and DPO-Z lengths converge near 4.5K tokens; accuracy remains
  noisy and currently does not establish a winner.
- **Decision:** wait for matched held-out evaluations.

## `common-training-metrics.{pdf,png}`

- **Purpose:** expose truncation, non-informative groups, gradient scale and
  runtime differences.
- **Data:** same snapshots; percentages are raw fractions multiplied by 100.
- **Caption requirements:** step times are affected by different shared-GPU
  contention; no error bands because independent repeats do not exist.
- **Notice:** PRIME is operationally much slower and its only logged snapshot
  has higher zero-std fraction; evidence is too short for a trend claim.
- **Decision:** fix official-style refill, then collect a continuous PRIME curve.

## `optimization-signals.{pdf,png}`

- **Purpose:** detect instability within each method.
- **Data:** reported policy loss, reward std and actor learning rate.
- **Caption requirements:** loss magnitudes/signs are not comparable across
  different advantage estimators.
- **Notice:** available snapshots are finite; no NaN/collapse is visible.
- **Decision:** use this plot for monitoring, not method ranking.
