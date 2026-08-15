---
type: experiment
status: implementation-validation
experiment_line: prime-reproduction
model: Qwen/Qwen3.5-0.8B
updated: 12-08-2026
tags: [prime, reproduction, parity-test]
---
# PRIME reproduction

## Goal

Получить test-backed end-to-end implementation original PRIME semantics на `Qwen/Qwen3.5-0.8B`, используя существующий repository, но новую clean branch, новый pinned environment и новые model states.

## Motivation from papers and prior results

- [[Papers/Process Reinforcement through Implicit Rewards]] оставляет часть training semantics только в code.
- [[Knowledge/Legacy-Audit]] показывает, что старые runs завершались с известными correctness gaps.
- [[Knowledge/Official-Implementation-Gaps]] отделяет original target от maintained recipe.

## Reuse versus rewrite

Переиспользовать:

- model loading и Qwen3.5 compatibility fixes;
- verifier/data/evaluation;
- logging и checkpoint infrastructure;
- H200 offload lessons.

Выделить заново в небольшие pure/testable components:

- prompt-group filtering и refill;
- implicit reward construction;
- separate-source RLOO;
- process normalization;
- PRM update ordering;
- state save/resume contract.

## Frozen original semantics

- `K=4`.
- Accuracy filter inclusive `[.2,.8]` до PRM/actor update.
- `max_prompt_length=1024`, `max_response_length=3072`.
- `single-forward`: PRM score до update текущего batch.
- Actor KL coefficient `0`.
- Actor LR `5e-7`; PRM LR `1e-6`.
- PRM `beta=.05`, `grad_clip=10`, `weight_decay=0`.
- Outcome/process returns рассчитываются отдельно через RLOO.
- Original batch normalization process returns включена.
- Truncation group filter реализован и протестирован, но выключен для literal published script unless explicitly enabled.

## Unit/parity gates

1. При `π_PRM == π_ref` все implicit token rewards равны нулю.
2. Synthetic correct/incorrect labels дают тот же PRM loss и gradient direction, что official worker.
3. Для известных `K=4` rewards RLOO и discounted returns совпадают с official code.
4. Filter удаляет prompt group целиком и никогда не обучает PRM на rejected samples.
5. Refill не смешивает responses разных prompts и сохраняет contiguous group layout.
6. `single-forward` test доказывает, что текущий actor advantage не использует post-update PRM.
7. Checkpoint/resume восстанавливает следующую update trajectory.

## Qwen3.5 smoke gates

- Post-trained checkpoint без `-Instruct` suffix загружается в text-only mode.
- Thinking mode включён явно и проверен на rendered prompt.
- Policy, PRM и reference используют одинаковые response token ids/masks.
- Vision encoder не создаёт неожиданный memory/gradient path.
- Один update имеет finite loss и non-zero actor/PRM gradients.
- Strict verifier принимает `dfrac/frac`, но отвергает tuple/set comma collapse и потерю numeric-base annotation.
- Colocated vLLM resident tensor совпадает с фактически переданным TRL tensor и меняется после optimizer update.

## Required metrics

- filter acceptance и pass-rate histogram;
- truncation rate;
- outcome/process sequence returns до и после normalization;
- PRM loss, ranking/AUC или correct-vs-incorrect margin;
- actor loss, KL, entropy и grad norms;
- response length distribution;
- exact counts generated/accepted/updated;
- runtime и peak GPU memory per phase.

## Promotion criteria

Результат считается reproduction только если:

- parity tests проходят;
- environment/config/seed/data/model/code revisions сохранены;
- short run проходит save/resume;
- нет silent zero gradients, NaN, systematic truncation или раннего collapse;
- raw metrics доступны, а не только poster/aggregates.

## Deferred hypotheses

После promotion проверить в порядке из [[Knowledge/Research-Ideas]]: gradual process weighting, corrected leave-one-out DPO-Z, затем — только после reformulation — prompt calibration alternative вместо текущего `Z(x)` loss.



## Implementation status — 12-08-2026

- Ветка: `feat/prime-faithful-memory`.
- Author-aligned core реализует pre-update reward (`update=after`), inclusive accuracy filter, refill до полного batch, separate RLOO, reverse-return normalization, PRM `beta=.05`, `grad_clip=10`, `weight_decay=0`.
- Actor/PRM/reference log-probabilities вычисляются через exact checkpointed token chunks; при `chunk_tokens=256` vocab tensor для Qwen уменьшается примерно с 8.14 GB до 127 MB.
- Full tuning остаётся faithful default; LoRA actor/PRM — отдельный research profile. Для `r=16` trainable PRM parameters: 10,822,656 (около 1.42%).
- `Qwen/Qwen3.5-0.8B` скачан и прошёл offline text-only load; unit suite: 31 passed. Отдельно проверены raw implicit reward без `beta`, official PRM BCE, exact chunked forward/backward и refill.
- Не пройдены: actual GPU forward/backward, vLLM generation, save/resume и short end-to-end run. Все H200 были заняты, поэтому experiment status остаётся implementation-validation, не reproduced.
- Environment warning: TRL 1.2.0 заявляет vLLM support до 0.18.0, установлен vLLM 0.19.1, необходим generation smoke до training.
- Legacy monkey patch `IS_NVIDIA_HOPPER=False` удалён: возможная kernel incompatibility должна проявляться ошибкой, а не скрытым численно неверным backward.

## GRPO VRAM acceptance budget — 12-08-2026

Статус: `planned`; ниже аналитический budget, не измеренный результат.

Target для ordinary GRPO: peak allocated VRAM не более 30 GiB на одной GPU при `Qwen/Qwen3.5-0.8B`, completion 16,384 и `K=8`.

- Text model: 763,215,680 parameters; BF16 weights около 1.42 GiB.
- Full tuning persistent model/gradient/Adam states: около 8.5 GiB без activations и временных optimizer buffers; full tuning поэтому не имеет надёжного запаса под 30 GiB.
- LoRA `r=16`: около 10.8M trainable parameters; model + LoRA gradients/Adam states около 1.55 GiB до activations.
- Qwen3.5 имеет шесть full-attention layers. BF16 KV для 17,408 tokens оценивается примерно в 204 MiB на sequence; 64 одновременно длинных sequences — около 12.75 GiB только full-attention KV. Gated DeltaNet states и runtime workspace добавляются отдельно.
- На H200 `vllm_gpu_memory_utilization=0.12` означает около 17 GiB, а не 45 GiB; значение зависит от полного объёма конкретной GPU.

Предварительный envelope:

| Фаза | LoRA, текущий aggressive config | Рекомендуемый safe profile |
|---|---:|---:|
| vLLM generation | 20–26 GiB | 16–22 GiB |
| actor forward/backward | 8–16 GiB | 7–14 GiB |
| общий ожидаемый peak | 22–28 GiB | 18–25 GiB |

Safe profile: LoRA actor `r=16`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=64`, `generation_batch_size=64`, `vllm_gpu_memory_utilization=0.12`, exact token-chunked policy loss (`256` positions), gradient checkpointing и sleep mode. `max_prompt_length=1024`, `max_model_len=17408`; итоговый memory gate всё ещё требует GPU smoke.

Acceptance gate перед training: короткий generation + one-update smoke должен показать `torch.cuda.max_memory_allocated < 27 GiB`, оставляя не менее 3 GiB запаса до hard target 30 GiB. Отдельно записать NVML peak/reserved memory, поскольку allocator-reserved и allocated значения различаются.

## Proposed data, batch and step protocol — 12-08-2026

Статус: `planned`; sampling/training не запускались.

### Empirical prior from the legacy poster

Старый `Qwen3.5-0.8B-Base` имел maj@8 при `temperature=0.6`: GSM8K 72.6, MATH-500 64.8, MATH-Hard 38.7. GRPO/DR-GRPO/DAPO использовали около 1.5K train prompts из GSM8K (8%) и MATH L2/L3/L4 (25/50/17%) и прошли 187 rollout steps. Это соответствует примерно восьми prompts на update и показывает, что source mixture работал для старого checkpoint.

Legacy PRIME set был выбран по `4/8` correct rollouts у custom SFT policy. Он остаётся useful candidate pool, но не frozen train set для official post-trained `Qwen/Qwen3.5-0.8B`: capability и sampling policy изменились. `4/8` также является шумной оценкой true pass rate.

### Candidate and calibration protocol

1. Основной candidate pool: только `openai/gsm8k` split `train` и original MATH split `train` (levels 1–5). Не использовать GSM8K test, MATH-500 или MATH-Hard как training data.
2. Сначала повторно probe старые 1.5K prompts новым post-trained checkpoint при training sampling (`temperature=1`, тот же top-p/top-k и response cap), не начиная training. Затем при необходимости добрать harder/easier prompts из полного GSM8K+MATH train pool.
3. Использовать минимум 8 samples/prompt для прямой оценки mixed-group rate; предпочтительно 16, а для boundary prompts довести total до 32.
4. Не выбирать только exact 50%. Собрать устойчивую смесь по estimated pass rate: примерно 20% hard (`p≈.1–.3`), 60% medium (`.3–.65`), 20% easy (`.65–.85`). По мере улучшения hard tail становится новым medium. Source ratio не фиксировать заранее; вероятно GSM8K потребуется только 5–15%, если post-trained policy решает его слишком уверенно.
5. Зафиксировать один deduplicated train set для GRPO/DAPO/PRIME comparison; online filtering/refill остаётся частью конкретного algorithm, а не различием datasets. DAPO-Math-17K оставить для поздней external-data ablation, а не смешивать с первым clean baseline.
6. Удалить exact/normalized/fuzzy overlaps с GSM8K test, MATH-500, MATH-Hard и AIME evaluation. Хранить canonical ground-truth answer из исходного dataset, а не извлекать label из первой сгенерированной correct trace.

Acceptance target: informative ordinary-GRPO groups (`1–7` correct из 8) не менее 70%; для PRIME strict groups (`2–6` correct из 8) желательно не менее 50% до refill.

### Prompt length

`max_prompt_length=1024` считать default, а не предположением: измерить tokens после official chat template и generation prefix. Принять 1024, если truncation равен нулю либо меньше 0.5% и p99 имеет запас. Длинные outliers лучше отфильтровать; повышать до 1536/2048 только при содержательной потере задач. `max_model_len` должен быть не меньше prompt cap + completion cap, то есть минимум 17,408 для 1024 + 16,384.

### Batches and steps

Для 30-GiB LoRA profile:

- `K=8` responses/prompt для research comparison; faithful PRIME reproduction отдельно сохраняет `K=4`.
- generation sub-batch: 4 prompts = 32 completions.
- actor micro-batch: 1 completion.
- effective optimizer batch: 8 prompts = 64 completions; `gradient_accumulation_steps=64`.
- Начать с `num_iterations=1` для basic GRPO. Для controlled GRPO-vs-DAPO comparison использовать одинаковый `num_iterations=2` у всех compared methods, иначе clipping DAPO на первой итерации почти не активен.

При effective batch 8 prompts:

| Milestone | Unique prompts | Rollout steps | Назначение |
|---|---:|---:|---|
| smoke | fixed 32–80 | 5–10 | correctness/memory only |
| legacy-scale pilot | 1,500 | 188 | сопоставление со старым проектом |
| first substantive run | 3,000 | 375 | достаточно для первого решения |
| main comparison | 8,000 | 1,000 | более надёжный method comparison |

Promising methods повторять на 3 seeds; не тратить seeds на configuration smoke. Eval каждые 50 rollout steps, early stop при трёх evaluation points без улучшения либо при росте truncation/length и падении informative-group rate.

### Reward comparability

Для чистого сравнения PRIME с ordinary GRPO основной reward должен быть один и тот же — correctness. Текущий ordinary GRPO добавляет format reward с весом 0.5, а PRIME использует только accuracy; это confound и должно быть исправлено до ablation. Format compliance следует сначала логировать как metric, а не оптимизировать отдельной наградой.

## Active LoRA-16K execution profile — 13-08-2026

Статус: `planned`, memory numbers аналитические, GPU peak ещё не измерен.

- Active experimental baseline: `prime_research_16k.yaml`; literal
  `prime_faithful.yaml` остаётся semantic reference, но не planned long run.
- Actor и PRM: LoRA `r=16`, `alpha=32`, `dropout=0`, `all-linear`; reference
  frozen. Actor/PRM LR `1e-5`, weight decay `0`, BF16.
- Context: prompt cap `1024`, completion cap `16384`, vLLM max model length
  `17408`; `K=8`, generation batch `64`, actor microbatch `1`, accumulation
  `64` = `8 prompts / 64 completions` на update.
- PRIME: implicit-reward `beta=.05`, RLOO, `gamma=1`, accuracy filter
  `[.2,.8]`, refill, PRM ref batch `2`, PRM grad batch `1`.
- Memory/speed: token-logit chunks `512`; policy/PRM/reference остаются на GPU
  (`cpu_offload_* = false`) во избежание повторных CPU↔GPU transfers. vLLM
  colocate pool `0.12` от H200 = около `16.85 GiB`, включая vLLM weights/cache.
- Estimated peak: `23–27 GiB`; conservative fragmentation/workspace envelope
  `до 30–32 GiB`. Target acceptance остаётся `<30 GiB` measured peak; при
  превышении первым fallback будет вернуть auxiliary CPU offload или chunks
  `256`, не уменьшать context вслепую.

Prompt-length audit существующих datasets с exact Qwen tokenizer:

| Dataset | n | p99 | max | >1024 |
|---|---:|---:|---:|---:|
| `grpo_easy_3k` | 3000 | 410 | 1131 | 1 (0.033%) |
| `grpo_easy_1500` | 1500 | 381 | 1131 | 1 (0.067%) |
| `grpo_prime_calibrated` | 1500 | 333 | 808 | 0 |
| `grpo_prime_calibrated_v2` | 1500 | 323 | 798 | 0 |
| `grpo_prime_calibrated_v3` | 1500 | 280 | 810 | 0 |

Decision: оставить `max_prompt_length=1024`; единственный outlier исключить при
формировании нового post-trained train set. После новой calibration повторить
audit; acceptance — `<0.5%` outliers и большой запас у p99.

## Fresh post-trained difficulty probe protocol — 13-08-2026

Статус: `prepared`; GPU generation не запускалась, потому что все 8 H200 были
заняты при preflight.

Старые `grpo_prime_calibrated*` не использовать для train selection или нового
evidence: они размечены pass rate другой custom SFT policy. Можно переиспользовать
только source-loading/parser code для immutable `GSM8K train` и original
`MATH train`.

### Stage A: unbiased stratum probe

- Model: frozen `Qwen/Qwen3.5-0.8B` post-trained.
- Strata: `GSM8K` и `MATH Level 1–5`; внутри каждого MATH level — одинаковое
  число задач из каждого из 7 subjects.
- Sample: 70 prompts на MATH level (10/subject) и 100 GSM8K = 450 prompts.
- Rollouts: `K=8`, seed manifest фиксирован; всего 3600 completions.
- Sampling совпадает с training: temperature `1`, top-p `1`, top-k `0`, prompt
  cap `1024`, completion cap `16384`, strict frozen verifier.
- Probe prompts исключаются из последующего train set. GSM8K test, MATH-500,
  MATH-Hard и другие eval sets никогда не участвуют в selection.

Для каждой stratum и level×subject cell записать accuracy, count-correct
histogram, GRPO informative-group fraction `P(1≤C≤7)`, PRIME accepted-group
fraction `P(2≤C≤6)` для `K=8`, all-zero/all-one rates, truncation и response
lengths. Uncertainty считать cluster bootstrap по prompts, не считать 8 rollouts
независимыми observations.

### Stage B: train-set construction

Выбирать strata/levels, а не individual prompts. Starting criterion:

- GRPO informative groups желательно `≥70%`;
- PRIME accepted groups желательно `≥40–50%` до refill;
- слишком лёгкие/невозможные strata исключать либо оставлять малой anchor-долей;
- внутри выбранных MATH levels сохранять subject balance;
- 3000 train prompts семплировать fresh из оставшейся части original train.

Конкретные MATH levels и доля GSM8K заранее не фиксируются. Гипотеза
опровергается, если ни одна комбинация level strata не даёт достаточно mixed
groups при training sampling; тогда потребуется per-prompt adaptive curriculum,
но не перенос старых SFT labels.


### Prepared run

- Entrypoint: `scripts/probe_posttrained_difficulty.py`.
- Persistent server artifact: `~/opt_project/probe_runs/qwen35_08b_gsm100_math350_seed42/`.
- Frozen model revision: `Qwen/Qwen3.5-0.8B` commit
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Manifest: 450 unique prompts — 100 GSM8K и по 70 для MATH L1–L5; 10 задач
  на каждый `level × subject`. Максимальная длина prompt после official chat
  template — 910 tokens; превышений cap 1024 в manifest нет.
- Source preprocessing отбросил 4 MATH examples без parseable boxed answer и
  7 MATH prompts длиннее 1024 до stratified sampling.
- Raw results сохраняются после каждого request batch в append-only JSONL;
  resume использует именно frozen manifest. Summary включает accuracy,
  prompt-cluster bootstrap CI, GRPO/PRIME usable-group rates, truncation и lengths.
- Unit tests: 2 passed локально и в pinned server environment; lint passed.

Preflight snapshot: все GPU показывали 100% utilization; запуск на частично
свободную, но активно вычисляющую карту отклонён. Следующий шаг — сначала 6-prompt
smoke на одной действительно idle H200 с `gpu_memory_utilization=.18`,
`max_num_seqs=8` и полным 16K cap. Только после проверки outputs/verifier/VRAM
запустить full 450-prompt probe на той же одной карте.

User decision: дальнейшие GPU checks остановлены; smoke перенесён на завтра.

### Full difficulty probe stop — 13-08-2026

Статус: `stopped_by_user`, не finished result. Однокарточный non-thinking 8K
probe был остановлен по запросу пользователя. Append-only raw JSONL содержит
64/450 полностью завершённых prompts (512 completions); незавершённый batch не
записан. Tmux, parent process и наш vLLM EngineCore завершены. GPU 4 вернулась к
56,047 MiB, занятым единственным чужим PID. Run можно продолжить через `--resume`.

## Qwen3.5-0.8B-Base LoRA SFT bootstrap — 14-08-2026

Статус: `running`; финальных SFT/eval metrics ещё нет.

### Goal and contract

- Научить base model assistant/reasoning format на 20K DeepSeek-R1 traces из
  `OpenR1-Math-220K`, не оптимизируя prompt tokens.
- Model: `Qwen/Qwen3.5-0.8B-Base`; context `16384`; LoRA `r=64`, `alpha=128`,
  dropout `.05`, `all-linear`; BF16; two epochs.
- Exact completion-only CE: prompt labels `-100`; generic
  `LigerFusedLinearCrossEntropyLoss` получает final hidden states и `lm_head`
  напрямую, поэтому full tensor `[batch, tokens, 248K vocab]` не создаётся.
- Main config: `configs/sft_lora_r64_16k_two_epochs.yaml`; data:
  `data/openr1_20k` (`19000 train / 1000 eval`); checkpoint at every epoch.

### Memory/speed diagnosis

Worst-case smoke использовал 64 самых длинных 16K trajectories. Проверенные
результаты:

| Config | Outcome |
|---|---|
| batch 16, gradient checkpointing, 30 GiB | OOM |
| batch 16, gradient checkpointing, 35 GiB | OOM in backward; required additional 8 GiB |
| batch 8, accumulation 2, checkpointing, SDPA | passed; peak `31.206 GiB`; second step `44 s` на сильно занятой GPU 7 |
| batch 4/2/1, no checkpointing, 35 GiB | все OOM в forward; batch 1 reached `34.93 GiB` before requesting another 112 MiB |
| batch 8, accumulation 2, checkpointing, FlashAttention-2 | passed; steps `19.9/15.6 s`; peak `25.881 GiB`; loss `.802→.727` |
| batch 8, accumulation 2, FA2, LoRA dropout 0 | passed; steps `18.5/14.5 s`; peak `22.694 GiB` |
| batch 16, accumulation 1, FA2, LoRA dropout .05 | OOM in backward at `43.21 GiB`, requested additional `3.5 GiB` |
| batch 16, accumulation 1, FA2, LoRA dropout 0 | passed; post-JIT second step about `12.6 s`; peak `43.114 GiB`; driver footprint about `47.4 GB` |

Interpretation: прежний VRAM blow-up от vocab logits устранён. Оставшийся
dominant cost без checkpointing — activations/recurrent states Qwen3.5 Gated
DeltaNet. FlashAttention-2 даёт полезное ускорение и снижает measured peak; final
profile selected by user: micro/effective batch `16`, checkpointing, FA2, fused
CE, LoRA dropout `0`, allocator cap `45 GiB`. Это использует почти весь текущий
свободный запас общей GPU; выбор сделан осознанно ради скорости.

### Active run

- Первый batch-8 full run (PID `1573955`) остановлен после 7 steps для
  batch-size sweep; checkpoint/result он не дал.
- Physical GPU: `1`; active batch-16 Python PID: `1601506`.
- Log: `logs/sft_lora_r64_16k_two_epochs_fa2_b16.log`.
- Active driver footprint `47,392 MiB`; на момент запуска общий запас GPU был
  около `3.2 GiB`. Первые full steps нестабильны (`17.9–54 s`) из-за новых
  compiled shapes и contention; ETA до steady-state не фиксировать.
- GPU 1 shared with foreign processes; do not attribute total GPU utilization or
  total used memory to this run.

Next step: wait for epoch-1 checkpoint, assess actual ETA/loss, then merge the
adapter, run a small correctness/pass-rate probe on fresh-policy generations,
and only after a non-degenerate reward signal perform vanilla GRPO smoke.

## Implementation audit and active SFT — 14-08-2026

Статус reproduction остаётся `implementation-validation`: CPU numerical and
contract suite проходит (`49 passed`), но реальный Qwen3.5 + LoRA + colocated
vLLM generation/update ещё не прошёл GPU smoke. TRL `1.2.0` предупреждает, что
официально поддерживает vLLM только до `0.18.0`, тогда как environment содержит
`0.19.1`; это runtime gate, а не доказанная несовместимость.

Literal original launch подтверждён: `run_prime_main.sh` не переопределяет
`filter_truncated`, default YAML равен `False`. Поэтому faithful reference
оставляет whole-group truncation filter выключенным; включённый вариант —
отдельная ablation.

Обнаружена неоднозначность `rm_coef=5`: launch config задаёт коэффициент, но
public RLOO path вычисляет decomposed `rm_scores`/`gt_scores` и не использует
умноженный composite `all`, поэтому коэффициент фактически dead в literal code.
Текущий research implementation активирует `5 × process reward`, то есть
воспроизводит declared intent, но не literal effective bug. До baseline run
нужно сохранить два явно названных профиля либо выбрать один target и записать
deviation; exact parity одновременно с active `rm_coef=5` утверждать нельзя.

PRM final-save исправлен: actor output теперь сопровождается `prm/model.pt`,
`optimizer.pt`, `state.pt` и optional prompt-calibration head даже если
`max_steps < save_steps`; round-trip regression test проходит.

Активный LoRA SFT: PID `1601506`, physical GPU 1, step `450/2376`, elapsed
`1:08:43`, cumulative `9.16 s/step`; OOM/traceback нет. Прогноз epoch 1 около
`03:15 MSK`, обеих эпох около `06:20 MSK`. Это прогноз running trajectory, не
готовый результат; checkpoint на момент записи отсутствует.


### Token-bounded ordinary GRPO and DPO-Z

Vanilla GRPO и ordinary DPO-Z теперь используют один exact policy-loss path,
который проецирует final hidden states в vocabulary чанками по `256` token
positions. Full `[B,T,V]` logits не сохраняются. DPO-Z меняет только sequence
advantages; chunked loss потребляет их без повторной нормализации или изменения.

Toy full-vocabulary parity проверяет loss и gradients embedding, backbone и
`lm_head`; отдельный trainer-level test проверяет тот же contract через
`MemoryBoundedGRPOTrainer`, а composition test фиксирует DPO-Z generation hook и
memory-bounded `compute_loss`. Full CPU suite: `62 passed`. Runtime acceptance
остаётся двухшаговый GPU smoke с measured peak VRAM и vLLM sync canary.

## Official-style refill fix and GPU smoke — 14-08-2026

Статус: `plumbing smoke passed`; это не quality result и не проверка полного
16K memory envelope.

Причина остановки первого PRIME run после step 9 установлена точно: local
adapter ограничивал сбор валидного batch значением `max_refill_rounds=8`, тогда
как public PRIME продолжает брать prompts до заполнения batch или исчерпания
dataloader. Исправленный single-GPU adapter использует детерминированную
перестановку train set без replacement, исключает prompts исходного batch и
завершается ошибкой только после фактического исчерпания доступного pool.
Regression воспроизводит восемь невалидных раундов и успешный девятый; отдельный
тест проверяет отсутствие повторов и exhaustion. Targeted PRIME/config suite:
`35 passed`.

Research profile с `G=8` использует filter `[0.1, 0.9]`: допустимы `1--7`
successes, что является тем же дискретным non-unanimous criterion, что public
`G=4` с `[0.2, 0.8]`. Faithful profile оставляет авторские `G=4` и
`[0.2, 0.8]`. Для early recovery research checkpoints сохраняются каждые
5 steps вместе с PRM state.

Короткий smoke `prime_research_refill_smoke.yaml` выполнен на physical GPU 4:
2 optimizer steps, `G=8`, completion cap `2048`, active filter/refill, actor и
PRM LoRA. Runtime `50.005 s`; `train_loss=-0.006437`; accuracy `.375/.25`;
PRM loss `.6931/.6987`. Обе исходные группы были информативными, поэтому
`refill_rounds=0`; сам повторный refill остаётся покрыт deterministic CPU
regression. vLLM canary после первого update: synced tensor изменился,
resident tensor совпал с переданным (`max_abs_difference=0`). Checkpoints 1 и
2 содержат actor adapter/optimizer и полный `prm/{model,optimizer,state}`;
final PRM также сохранён. После завершения GPU 4: `0 MiB`, `0%`.

PyTorch report после smoke: `31.66 GiB allocated`, `35.66 GiB reserved`.
Следовательно, plumbing и hard limit `40 GiB` проходят, но старый soft gate
`<27--30 GiB` не выполнен даже при 2K completion cap; перенос этого числа на
16K запрещён. Перед full 16K run нужен отдельный measured memory gate либо
осознанное принятие большего бюджета.

Две подготовительные попытки не являются training evidence: первая завершилась
до vLLM из-за несовместимого `expandable_segments` с memory pool; вторая дошла
до FlashInfer JIT и обнаружила, что `ninja` package установлен, но его executable
не входит в detached-tmux `PATH`. Финальный launcher явно добавляет project
`venv/bin`; `ninja==1.13.0` зафиксирован как runtime dependency.

## vLLM sleep-mode correction — 14-08-2026

Предыдущий двухшаговый PRIME smoke больше не доказывает, что rollout policy
обновлялась: его canary проверял resident tensor до последующего
`reload_weights` внутри TRL generation. PRM/actor backward, checkpoints и
memory measurements остаются валидными как plumbing evidence, но generated
trajectories могли принадлежать original SFT policy.

Shared colocate fix теперь deferred ранний actor sync и выполняет verified
resync сразу после `reload_weights`; перед actor PEFT wrapping также установлен
explicit seed. Regression unit test и реальный paired GRPO smoke подтверждают
post-reload semantics. PRIME должен пройти отдельный короткий smoke с новым
gate до возобновления full run; текущий status остаётся
`implementation-validation`, не reproduction.
