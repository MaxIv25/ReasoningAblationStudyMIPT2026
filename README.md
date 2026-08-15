# PRIME RL reproduction — Qwen3.5-0.8B

Текущая цель проекта — test-backed воспроизведение original PRIME поверх нового LoRA-SFT checkpoint `Qwen3.5-0.8B-Base`, а затем изолированные проверки четырёх research ideas. Старые SFT/GRPO эксперименты сохранены как legacy context и не являются evidence для новой линии.

> Статус: GPU-validated research implementation. CPU suite проходит
> (`89 passed`), LoRA SFT завершён и merged, exact token-chunked GRPO/PRIME
> math, save/resume и post-reload HF↔vLLM weight sync проверены. Vanilla GRPO
> и DPO-Z прошли substantive single-GPU runs; PRIME reproduction и frozen
> `maj@8` evaluation ещё не завершены, поэтому paper-level claims пока нет.

## Что реализовано

- Author-aligned PRIME semantics: `K=4`, accuracy filter `[0.2, 0.8]`, refill до полного batch, pre-update PRM reward (`update=after`), separate outcome/process RLOO, reverse cumulative returns, global process normalization, PRM `beta=0.05`, `grad_clip=10`, `weight_decay=0`.
- Exact memory-bounded log-probabilities для actor, PRM и reference: `lm_head` вычисляется token chunks с activation checkpointing вместо materialization `[B,T,V]`.
- Text-only Qwen3.5 loading без vision encoder и vLLM `language_model_only`.
- Optional LoRA отдельно для actor и PRM; faithful profile оставляет full-parameter training.
- DPO-Z baseline в ordinary GRPO и PRIME: `beta * logmeanexp(R / beta)`, leave-one-out, без последующего centering, которое уничтожило бы baseline.
- Четыре независимых PRIME extensions: DPO-Z, gradual/reliability-gated process reward, gauge-complete prompt calibration, PRM-guided candidate selection.

## Профили

| Config | Назначение |
|---|---|
| `configs/prime_faithful.yaml` | Literal effective public RLOO semantics, 3K, full tuning, `K=4`; resource-scaled batch |
| `configs/prime_research_16k.yaml` | Declared-intent `rm_coef=5`, 16K, LoRA actor/PRM, bounded-memory log-probs |
| `configs/grpo_dpo_z.yaml` | Ordinary GRPO with DPO-Z advantage |
| `configs/prime_idea_*.yaml` | По одному overlay на каждую из четырёх идей |

Idea overlays наследуют `prime_research_16k.yaml`; их наличие не означает, что ablations уже запускались. Public launch задаёт `rm_coef=5`, но literal RLOO path не применяет его к decomposed `rm_scores`; поэтому faithful и declared-intent profiles разведены явно.

## Memory contract

У Qwen3.5 vocabulary size `248,320`. Один BF16 tensor logits для `T=16,384`, `B=1` занимает примерно 8.14 GB; вместе с FP32 normalization intermediates peak ещё выше. `memory.logprob_chunk_tokens=256` ограничивает raw vocab tensor примерно 127 MB на chunk (фактический peak projection + FP32 normalization — менее 0.4 GB).

`vllm_gpu_memory_utilization`, sleep mode, gradient checkpointing, sequential
PRM/reference placement и LoRA снижают память. Vanilla GRPO и DPO-Z используют
exact policy loss с token chunks по 256 позиций; Liger отключён, потому что при
microbatch 1 он не режет 16K sequence по токенам. Реальные 16K GRPO runs
работают примерно в 30–40 GiB собственного GPU footprint; конкретный reserve
зависит от vLLM KV-cache profile и совместного использования карты.

LoRA здесь нормальна как memory/speed ablation: при `r=16` у PRM около 10.8M trainable parameters (1.42%). Но faithful comparison должен включать full tuning, потому что LoRA меняет optimization hypothesis.

## Environment

Зависимости pinned в `pyproject.toml` и `uv.lock`:

```bash
uv sync --frozen
uv run pytest -q
```

На H200 выбран существующий environment `~/opt_project/venv`. В нём `torch
2.10.0`, `transformers 5.8.1`, `trl 1.2.0`, `peft 0.19.1`, `liger-kernel
0.7.0`, `vllm 0.19.1`, `flash-linear-attention 0.5.0`. TRL предупреждает, что
заявленная совместимость заканчивается на vLLM 0.18.0; поэтому каждый run
использует fail-closed post-reload sync canary. Real-GPU paired diagnostic
подтвердил расхождение trajectories после разных actor updates.

## Запуск

Проверка без training:

```bash
~/opt_project/venv/bin/python -m pytest -q tests
```

После окончания SFT сначала merge adapter:

```bash
~/opt_project/venv/bin/python -m scripts.merge_text_lora \
  --base Qwen/Qwen3.5-0.8B-Base \
  --adapter outputs/sft_lora_r64_16k_two_epochs \
  --output outputs/sft_lora_r64_16k_two_epochs_merged
```

После проверки свободной GPU и отдельного подтверждения expensive run доступны
двухшаговые smoke-варианты с обязательным явным GPU:

```bash
scripts/run_rl_smoke.sh vanilla <gpu-id>
scripts/run_rl_smoke.sh dpo_z <gpu-id>
scripts/run_rl_smoke.sh prime <gpu-id>
```

Launcher ограничивает CPU threads, проверяет dataset/merged model и пишет
`logs/<variant>_smoke.log`; full training автоматически не запускается.

## Структура

```text
configs/                         faithful, 16K, GRPO and idea profiles
src/train_prime.py               PRIME trainer
src/train_grpo.py                GRPO/DAPO/Dr.GRPO + DPO-Z
src/rl/prime_core.py             filtering, baselines, returns, schedules
src/rl/chunked_logprobs.py       bounded-memory exact selected log-probs
src/rl/prompt_calibration.py     gauge-complete PRM objective
src/rl/guided_search.py          explicit off-policy candidate selection
scripts/run_rl_smoke.sh          explicit-GPU two-step smoke launcher
tests/                           synthetic parity and memory contracts
obsidian/prime-rl-reproduction/  canonical project knowledge base
```

Obsidian vault path `/home/maxim/Obsidian/Research/Projects/prime-rl-reproduction` оставлен symlink на repo-local notes.

## Research integrity

Старые checkpoints удалены намеренно и не восстанавливаются. Ни failed/unfinished run, ни unit tests не считаются experimental evidence. Каждый будущий результат должен сохранять commit, config, seed, dataset/model revision, environment и raw metrics.
