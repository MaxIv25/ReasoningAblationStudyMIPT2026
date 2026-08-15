---
type: result
status: partial-running
date: 14-08-2026
experiment_line: prime-reproduction
tags: [grpo, prime, dpo-z, training-dynamics, partial-result]
---
# RL training dynamics: GRPO, PRIME и DPO-Z

Сравнение отвечает на вопрос: отличаются ли текущие training dynamics трёх
методов при общей SFT initialization, dataset, correctness reward и actor
optimizer contract. Это **partial descriptive snapshot**, а не финальное
сравнение качества.

## Main comparison

![[figures/accuracy-and-length.png]]

**Purpose.** Сопоставить train-batch accuracy и длину trajectories на общей
оси optimizer steps.

**Observation.** Vanilla GRPO и DPO-Z в доступной части trajectory остаются в
сходном диапазоне accuracy при большой batch-to-batch вариативности. Их mean
completion length к последним snapshot почти совпадает: 4492 и 4543 tokens.
У PRIME доступна только одна logged точка на step 5; run остановился после step
9, поэтому видим не PRIME curve, а один ранний measurement.

**Interpretation.** Сейчас нет evidence, что DPO-Z повышает accuracy относительно
GRPO: runs находятся на разных шагах, имеют один seed и ещё не оценены на
held-out benchmarks. Сходная длина снижает риск очевидного confound через
разную verbosity между двумя продолжающимися runs.

**Implication.** Дождаться одинаковых checkpoints и сравнивать GSM8K/MATH eval;
не выбирать метод по текущей train accuracy.

## Common operational metrics

![[figures/common-training-metrics.png]]

PRIME значительно медленнее на один optimizer step: единственный logged step
занял 996 s против 300 s у GRPO и 294 s у DPO-Z в последних snapshot. Это
ожидаемо из-за PRM/reference passes и refill, но точное отношение также
confounded разной GPU contention. Truncation в показанных последних batches
составляет 2.5% / 5.4% / 4.4% для GRPO / PRIME / DPO-Z соответственно.

## Optimization signals

![[figures/optimization-signals.png]]

Policy loss нельзя сравнивать по абсолютному значению между GRPO, PRIME/RLOO и
DPO-Z: advantage definitions различаются, а DPO-Z baseline не обязан давать
нулевой mean advantage. График нужен для поиска NaN, jumps и collapse внутри
каждого метода, а не для ranking методов.

## Latest exact snapshot

| Method | Logged step | Accuracy | Mean length | 16K clipped | Zero-std groups | Grad norm | Step time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla GRPO | 90 | 0.6906 | 4492 | 2.50% | 5.00% | 0.1183 | 300.4 s |
| PRIME | 5 | 0.5398 | 4980 | 5.42% | 18.29% | 0.0833 | 996.4 s |
| GRPO + DPO-Z | 50 | 0.6813 | 4543 | 4.38% | 15.00% | 0.0666 | 294.1 s |

Snapshot time: vanilla `10:38:59 MSK`, DPO-Z `10:37:45 MSK`; PRIME log final
timestamp `07:49:08 MSK`, 14-08-2026. Source data:
[[figures/metrics-snapshot.csv]]. Plot generator:
`scripts/plot_rl_training_dynamics.py`.

## Boundary of evidence

- Один seed (`42`), поэтому CI, significance tests и between-run effect sizes
  не определены.
- Runs имеют разный progress: 90 / 9 failed / 50 completed steps на момент
  snapshot.
- Каждая точка — aggregate одного train batch, не независимый sample и не
  held-out evaluation.
- PRIME имеет одну logged metrics point и цензурирован operational failure.
- GPU contention различается; step-time comparison не является чистым
  algorithmic benchmark.

Связано: [[../../Experiments/PRIME-Reproduction]],
[[../../Experiments/GRPO-DPO-Z-GPU7]], [[../../Knowledge/Research-Ideas]].
