# Statistical appendix — partial RL training dynamics

## Comparison unit

- Unit plotted: one logged optimizer-step batch aggregate.
- Run/seed count: one run and seed per method.
- Primary plotted metric: correctness accuracy on generated train trajectories;
  higher is better, but it is not held-out generalization.
- Supporting metrics: completion length, truncation fraction, zero-reward-std
  group fraction, actor grad norm, step time, policy loss and reward std.

## Descriptive sample sizes

| Method | Logged points | Latest logged step | Run status at snapshot |
|---|---:|---:|---|
| Vanilla GRPO | 18 | 90 | running |
| PRIME | 1 | 5 | failed after step 9 |
| GRPO + DPO-Z | 10 | 50 | running |

Raw parsed values are in [[figures/metrics-snapshot.csv]]; parser summary is in
[[figures/summary.json]]. No smoothing or normalization was applied.

## Inferential statistics

Inferential comparison is blocked. Optimizer-step aggregates within one run are
serially dependent and cannot substitute for independent seeds. With `n=1`
run per method, between-method standard errors, 95% CI, hypothesis tests and
standardized effect sizes would be fabricated. Multiple-comparison correction
therefore does not apply at this stage.

Promotion gate: complete comparable runs, run held-out evaluation from matched
checkpoints, then repeat promising methods for at least three seeds. Statistical
analysis should use seed-level held-out metrics as the comparison unit.
