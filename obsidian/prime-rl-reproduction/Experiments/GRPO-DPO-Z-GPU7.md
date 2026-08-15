---
type: experiment
status: training-complete-evaluation-pending
date: 14-08-2026
experiment_line: grpo-dpo-z
model: Qwen3.5-0.8B-SFT-LoRA
tags: [grpo, dpo-z, lora, runtime-gate]
---
# GRPO with DPO-Z baseline — GPU 7

> [!danger] Scientific status
> Исходные vanilla/DPO-Z full runs остановлены и не являются evidence для
> сравнения методов: rollout backend генерировал из исходного SFT checkpoint.
> Checkpoints сохранены только как debugging artifacts.

Это launch/runtime evidence, не итоговый экспериментальный результат.

## Run contract

- PID `2326981`, tmux `grpo_dpo_z_full_gpu7`, physical GPU 7.
- Output `outputs/grpo_dpo_z_full_gpu7`; raw log
  `logs/grpo_dpo_z_full_gpu7.log`; VRAM trace
  `logs/grpo_dpo_z_full_gpu7_memory.csv`.
- Merged two-epoch SFT actor; LoRA `r=16`; 1500 prompts из
  `data/grpo_prime_calibrated_v3`.
- `G=8`, 64 trajectories/generation, actor microbatch `8`, accumulation `8`,
  187 optimizer steps, LR `1.5e-6`, 16K completion cap, exact policy-logit
  chunks по 256 tokens.
- Reward совпадает с vanilla GRPO: correctness weight `1`, format weight `0`.
  DPO-Z меняет только advantage baseline: `beta_Z=.1`, leave-one-out, без
  group-std scaling.
- GPU-specific deviation: colocated vLLM reservation `.20` вместо `.25` из-за
  44.2 GiB free перед запуском. Наш generation footprint около 31.2 GiB;
  минимальный наблюдавшийся остаток карты около 12.6 GiB.

## Gates and current evidence

- CPU regression suite перед запуском: `38 passed`.
- Двухшаговый smoke завершён: runtime `459.50 s`, loss `0.4142`; step 1
  accuracy `.75`, mean length `3568`; step 2 accuracy `.25`, mean length
  `5094`. DPO-Z advantages finite и ненулевые.
- Первая smoke-попытка была failed plumbing run: FlashInfer JIT не нашёл
  `ninja` в `PATH`. Повторный launch добавил project venv/bin в `PATH`; code и
  scientific hyperparameters не менялись.
- Старый HF↔vLLM canary показывал `max_diff=0` и `changed=True`, но проверял
  resident tensor до вызова generation. Это оказалось недостаточным и ниже
  заменено post-reload gate. Шаг 1 завершён за `8:38`.
- После двух шагов tqdm ETA около 21.7 h; оно нестабильно из-за length tails и
  чужой нагрузки на GPU 7. OOM/NaN не наблюдались.

## Provenance

- Git HEAD `d5fd1a6ea268d77ef71a803e8f07532960810428`, dirty/uncommitted research
  worktree; seed `42`.
- Config SHA256 prefixes: `grpo_base=c87d6565`, `vanilla=53856264`,
  `dpo_z=aedb3c1a`, `gpu7=b7c6665a`.
- Dataset Arrow SHA256 prefix: `ea960ea1`.
- Environment: PyTorch `2.10.0+cu128`, Transformers `5.8.1`, TRL `1.2.0`,
  vLLM `0.19.1`. TRL официально предупреждает о поддержке vLLM только до
  `0.18.0`, поэтому реальные sync canary gates обязательны.

Связано: [[../Knowledge/Research-Ideas]],
[[../Knowledge/Verifier-and-vLLM-Sync]], [[PRIME-Reproduction]].

## Postmortem: identical rollouts and vLLM sleep-mode reload — 14-08-2026

### Symptom and invalidated runs

- Во всех 22 общих completion snapshots исходных full runs, steps `5..110`,
  совпали `1408/1408` completion strings row-by-row. Prompts и rewards также
  совпали; advantages различались.
- При этом checkpoint-75 policies не были одинаковыми: все 372 LoRA tensors
  различались, а effective `scale * B @ A` имел global L2 difference `0.049`.
  Различия переживали BF16 merge. Следовательно, loss менял actor, но rollout
  backend его не использовал.
- Vanilla остановлен на step `184/187`, DPO-Z — на `114/187`. Их метрики и
  графики нельзя интерпретировать как сравнение algorithms.

### Root cause

Installed TRL `1.2.0` в colocate sleep mode выполнял:

1. `sync_weights()` — merge текущей LoRA и загрузка в resident vLLM;
2. внутри `generate()` — `wake_up(tags=["weights"])` и
   `collective_rpc("reload_weights")`;
3. vLLM `0.19.1` обрабатывал `reload_weights` загрузкой original model path с
   диска, тем самым перезаписывая current actor непосредственно перед sampling.

Старый canary стоял между пунктами 1 и 2 и поэтому давал true-positive sync,
не доказывая, что generation использует эти weights.

### Fix and regression gates

- При sleep mode ранний trainer sync теперь deferred. После каждого
  `reload_weights` выполняется ровно один verified policy sync; canary сравнивает
  tensor уже в окончательном pre-generation resident state.
- Перед PEFT wrapping явно вызывается `set_seed(args.seed)`, потому что TRL
  создаёт LoRA до позднего seed внутри `Trainer.__init__`.
- Regression tests воспроизводят disk reload старых weights и проверяют
  post-reload resync. Полный CPU suite: `80 passed`.

### Paired real-GPU diagnostic

Config files: `grpo_vanilla_sync_diagnostic.yaml` и
`grpo_dpo_z_sync_diagnostic.yaml`. Один и тот же physical GPU 4 использован
последовательно; seed `42`, production LR `1.5e-6`, `G=8`, one prompt-group per
optimizer step, 3 steps, completion cap 2048, token-logit chunks 256.

| Gate | Vanilla | DPO-Z |
|---|---:|---:|
| Runtime | 23.87 s | 24.55 s |
| Post-reload canary checks | 3/3 | 3/3 |
| Resident max diff | 0 | 0 |
| Step 1 trajectories equal across methods | 8/8 | 8/8 |
| Step 2 trajectories equal across methods | 0/8 | 0/8 |
| Step 3 trajectories equal across methods | 0/8 | 0/8 |

Prompts во всех paired steps совпали. После checkpoint-1 все 186 `lora_A`
tensors побитово равны между методами (`max_abs_diff=0`), то есть LoRA subspace
одинаков; все 186 `lora_B` различаются, global L2 difference `0.0031466`.

Raw artifacts:

- `outputs/grpo_vanilla_sync_diagnostic_fix1`;
- `outputs/grpo_dpo_z_sync_diagnostic_fix1`;
- `logs/grpo_vanilla_sync_diagnostic_fix1.log`;
- `logs/grpo_dpo_z_sync_diagnostic_fix1.log`.

Conclusion: sync bug устранён для GRPO real-GPU path. Это runtime validation,
не оценка качества DPO-Z. Новые substantive runs должны начинаться с нуля из
одного seeded actor; старые full runs не resume.

## Vanilla GRPO scheduler ablation — 15-08-2026

### Hypothesis and falsification

Текущий vanilla baseline использует `lr=1.5e-6`, cosine decay и `5%` warmup;
к концу 187-step run LR практически обнуляется. Проверяем, улучшает ли
постоянный LR обучение LoRA actor на том же коротком on-policy horizon.
Гипотеза опровергается, если constant не улучшает frozen-val trajectory и
финальный `maj@8` относительно cosine baseline либо ухудшает stability.

### Single-variable contract

- Control: `outputs/grpo_vanilla_full_gpu1_val20_postsync_r4`, завершённый
  cosine run, `warmup_ratio=.05`.
- Treatment config: `configs/grpo_vanilla_constant_with_val.yaml`;
  `lr_scheduler_type=constant`, `warmup_ratio=0`, peak LR неизменён
  (`1.5e-6`).
- Не меняются actor initialization/seed `42`, LoRA `r=16`, 1500 prompts,
  `G=8`, 64 trajectories/update, microbatch `8`, accumulation `8`, reward,
  sampling (`T=1`, `top_p=1`, `top_k=0`), 16K cap, validation split и cadence.
- Contract-test сравнивает полностью resolved configs после удаления только
  `run_name`, `lr_scheduler_type` и `warmup_ratio`: `13 passed` вместе с
  config-profile suite на H200.

### Launch

- Status: `running`; start `15-08-2026 11:38 MSK` on physical GPU 1.
- tmux: `grpo_vanilla_constant_val20_g1_r1`.
- Output: `outputs/grpo_vanilla_constant_full_gpu1_val20_postsync_r1`.
- Raw log: `logs/grpo_vanilla_constant_full_gpu1_val20_postsync_r1.log`.
- Config SHA256: `b8490fe0f4c4b56d24e1e0fc9a81484950605f724af8a67404e8828b0a681158`.
- Git HEAD: `d5fd1a6ea268d77ef71a803e8f07532960810428`; worktree dirty, поэтому
  config hash и raw log являются обязательной частью provenance.

Primary decision metric после завершения: fresh frozen-val `maj@8`; текущая
on-training validation с `num_generations=1` служит noisy trajectory metric и
сама по себе не является final comparison.

### Constant-scheduler LR ablation: `5e-6`

- Hypothesis: при фиксированном constant scheduler повышение actor LR с
  `1.5e-6` до `5e-6` даст более заметное улучшение за 187 on-policy steps без
  collapse/нестабильности.
- Config: `configs/grpo_vanilla_constant_lr5e6_with_val.yaml`; resolved-config
  test доказывает, что относительно constant `1.5e-6` меняется только
  `learning_rate`. Config/profile tests: `14 passed`.
- Status: `running`; start `15-08-2026 11:54 MSK` on physical GPU 4.
- tmux: `grpo_vanilla_constant_lr5e6_val20_g4_r1`.
- Output: `outputs/grpo_vanilla_constant_lr5e6_full_gpu4_val20_postsync_r1`.
- Raw log: `logs/grpo_vanilla_constant_lr5e6_full_gpu4_val20_postsync_r1.log`.
- Config SHA256: `6ae4fc3a62403fb163dbddda5d36a3b7c49d00d44e0fe1607d85c87a0d1406d8`.
- Resource deviation: перед launch GPU 4 уже имела `74.25 GiB` foreign
  allocation и `100%` utilization; свободно было `68.91 GiB`. После vLLM init
  свободно около `31 GiB`. Поэтому wall-clock throughput нельзя сравнивать с
  GPU 1, но model-quality contract остаётся тем же.

Stop/failure criteria: OOM/NaN, sync-canary mismatch, sustained zero actor
updates или явная instability в reward/length/clip metrics. Primary selection
между LR проводится только после clean finish и одинакового frozen-val
`maj@8`, а не по noisy `G=1` validation.

### Runtime checkpoint — 15-08-2026 12:32 MSK

Все три активных GRPO processes и tmux sessions живы; traceback/OOM/NaN и
sync-canary mismatch не обнаружены.

| Run | Progress | Latest diagnostic | Sync | GPU free |
|---|---:|---|---|---:|
| vanilla constant `1.5e-6` | step 18/187 | step-15 accuracy `.6406`, mean length `4989`, clipped `.05`, grad norm `.0987` | changed, max diff `0` | 20.25 GiB |
| vanilla constant `5e-6` | step 9/187 | step-5 accuracy `.6594`, mean length `5273`, clipped `.0563`, grad norm `.1074` | changed, max diff `0` | 30.17 GiB |
| DPO-Z cosine `1.5e-6` | step 129/187 | val@120 accuracy `.65`; step-125 accuracy `.65` | changed, max diff `0` | 13.39 GiB |

Обе constant ветки дали exact-identical start validation: accuracy `.68`, mean
length `6004`, clipped ratio `.14`. Это подтверждает одинаковую initial policy
и sampler state. Ранние training-batch accuracy не являются evidence в пользу
одного LR; первый paired post-update val будет на step 20.

### Runtime checkpoint — 15-08-2026 18:05 MSK

#### DPO-Z retry4: clean training finish

- Full `187/187` завершён `17:54 MSK`; runtime `17:59:53`, train loss
  `.24446`.
- Финальный post-reload canary: `changed=true`, resident/source max diff `0`.
- Final adapter сохранён (`43.34 MB`), checkpoints `150` и `187` присутствуют.
- Frozen `G=1` validation: start `.68`, затем
  `.61/.65/.65/.56/.61/.65/.64/.62/.63/.65` на steps
  `20/40/60/80/100/120/140/160/180/187`. Final mean length `6534`, clipped
  ratio `.14`.
- Scientific status: training complete, but method comparison remains pending
  fresh frozen-val `maj@8`; noisy final `G=1` равен cosine vanilla (`.65`) и
  сам по себе не показывает улучшения.

#### Vanilla constant `1.5e-6`: running

- Step `142/187`; latest canary healthy (`changed=true`, max diff `0`), no
  OOM/NaN. Checkpoint `75` существует.
- Frozen `G=1` validation through step 140:
  `.68/.65/.62/.67/.59/.59/.62/.60` at steps
  `0/20/40/60/80/100/120/140`.
- Latest train metric at step 140: accuracy `.6406`, mean length `5108`,
  clipped `.0719`, grad norm `.1094`, clip ratio `0`.

#### Vanilla constant `5e-6`: failed resource run

- Run crashed at generation start after completed step `44`, `14:36 MSK`.
- Exact failure: CUDA OOM in vLLM `cumem_allocator` while executing
  `wake_up(tags=["kv_cache"])`; preceding step-44 sync canary was healthy
  (`changed=true`, max diff `0`). No NaN, gradient or policy-loss failure.
- GPU 4 была shared/100%-utilized до launch; поэтому это resource failure, не
  evidence против `lr=5e-6`.
- `save_steps=75`, поэтому adapter checkpoint отсутствует и resume невозможен.
  Rollout traces through step 40 и sync log сохранены.
- Partial frozen val была promising but non-conclusive: `.68` start, `.68` at
  step 20, `.69` at step 40; для вывода нужен clean restart from initial actor
  на карте с достаточным запасом и более частый recoverable checkpoint cadence.
