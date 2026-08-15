---
title: Completed models — GSM8K and MATH-500 maj@8
date: 2026-08-15
status: running
host: h200
gpu: 3
code_revision: 849e4f3bf1681c564f5b17e9b055ec2c4dd382b4
tags:
  - prime
  - grpo
  - evaluation
  - maj8
---

# Completed models — GSM8K and MATH-500 maj@8

Связанные заметки: [[GRPO-DPO-Z-GPU7]], [[../Results/RL-Training-Dynamics/analysis-report]], [[../00-Hub]].

## Goal

Проверить, является ли нестабильная frozen-validation accuracy в основном
шумом `G=1`, и сравнить только завершённые модели на прежних test splits:
GSM8K test и MATH-500 test. Primary metric — `maj@8`.

> [!warning] Evidence gate
> До завершения всех пяти jobs partial JSON/traces не считаются результатом.
> PRIME и текущий GRPO `LR=5e-6` исключены, потому что training не завершён.

## Comparison set

| Label | Artifact | Completion evidence |
|---|---|---|
| Base | `Qwen/Qwen3.5-0.8B-Base` snapshot `dc7cdfe2...` | immutable HF checkpoint |
| SFT | `outputs/sft_lora_r64_16k_two_epochs_merged` | final merged two-epoch SFT |
| Vanilla GRPO, decay | `outputs/grpo_vanilla_full_gpu1_val20_postsync_r4` | `187/187` |
| Vanilla GRPO, constant | `outputs/grpo_vanilla_constant_full_gpu1_val20_postsync_r1` | `187/187` |
| GRPO + DPO-Z cosine | `outputs/grpo_dpo_z_full_gpu7_val20_postsync_r4` | `187/187` |

RL adapters загружаются поверх одного и того же merged SFT. Smoke/profiling
adapters не входят в comparison.

## Evaluation contract

- Datasets: GSM8K test `n=1319`; MATH-500 test `n=500`, только локальный
  Hugging Face cache.
- Sampling: `num_samples=8`, `temperature=1`, `top_p=1`, `top_k=0`,
  explicit seed `42`, max completion `16384`.
- Base получает raw problem + instruction. SFT/RL получают зафиксированный
  assistant prefix `<think>\n`, соответствующий SFT/RL training format.
- Metric: majority vote по normalized boxed answers; дополнительно сохраняются
  format compliance, average words и каждый individual verifier outcome.
- Code: detached worktree commit
  `849e4f3bf1681c564f5b17e9b055ec2c4dd382b4`.
- Entrypoint: `scripts/eval_completed_maj8_h200.sh`.
- GPU: physical H200 GPU3; `CUDA_VISIBLE_DEVICES=3`; vLLM memory fraction
  `.30`; CPU/BLAS threads ограничены двумя.

Один sampling seed позволяет paired task-level comparison и bootstrap по
задачам, но не даёт seed-level uncertainty обучения. После завершения
`results-analysis` должен явно разделить эти два уровня.

## Raw artifacts

- Metrics: `results/maj8_test_20260815/<model>.json`.
- Full traces: `results/maj8_test_20260815/<model>.traces.jsonl.gz`.
- Каждый JSON содержит sampling config и SHA-256 compressed trace artifact.
- Runtime log: `logs/eval_completed_maj8_g3_20260815.log`.
- tmux: `eval_completed_maj8_g3_20260815`.

## Gates and current status

- CPU unit test full-trace writer: `1 passed`.
- GPU smoke: final DPO-Z LoRA loaded; 2 problems × 2 samples on both datasets;
  all 8 trajectories written and hashed; GPU memory released after exit.
- Full queue started at `20:04 UTC` on 15-08-2026.
- Current job: Base, GSM8K. First stabilized progress estimate was roughly
  2–3 h for its GSM8K portion; total queue ETA is not frozen until the Base
  GSM8K long-tail distribution is visible.

## Next step

После clean finish проверить row counts (`(1319 + 500) * 8 = 14552` на
модель), trace hashes, отсутствие missing/duplicate samples и построить
paired `maj@8` comparison с task-bootstrap confidence intervals.
