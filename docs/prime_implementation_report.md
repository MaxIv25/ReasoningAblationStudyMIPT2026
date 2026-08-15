# Отчёт по реализации PRIME

Дата аудита: 12-08-2026
Ветка: `feat/prime-faithful-memory`
Target: `Qwen/Qwen3.5-0.8B` post-trained
Статус: **implementation-validation**, пока не end-to-end reproduction.

## Итог

Старый репозиторий можно было исправить без полного переписывания: сохранены data/reward/evaluation и CLI-пути, а алгоритмически опасная часть выделена в небольшие тестируемые модули. Faithful baseline и четыре расширения разделены конфигами; research flags выключены по умолчанию.

Нельзя пока утверждать, что PRIME воспроизведён: unit/parity contracts проходят, checkpoint модели загружается, но actual H200 forward/backward, vLLM generation, save/resume и short update ещё не выполнены. Все GPU во время проверки были заняты.

## Что приведено к original PRIME

| Компонент | Реализация | Проверка |
|---|---|---|
| Rollouts | `K=4`, prompt 1024, response 3072 | config contract |
| Solvability filter | inclusive accuracy `[0.2, 0.8]`, whole prompt group | unit test |
| Refill | rejected groups заменяются до исходного числа valid groups | synthetic refill test; single-GPU only |
| Update order | actor получает process reward от pre-update PRM (`update=after`) | порядок зафиксирован в trainer; integration test ещё нужен |
| Implicit reward | raw `log π_prm − log π_ref`, без `β` | numeric unit test |
| PRM loss | sequence BCE с `β=0.05` только внутри logit | numeric equivalence test |
| Actor rewards | outcome и process baseline считаются отдельно | pure-core tests |
| Baseline | RLOO отдельно для outcome и process | pure-core tests |
| Process normalization | global max absolute reverse cumulative return | numeric unit test |
| Returns | reverse discounted returns, затем masked whitening | numeric unit tests |
| Optimizers | actor LR `5e-7`, PRM LR `1e-6`, PRM WD `0`, clip `10` | faithful config |
| Actor KL | `0` | faithful config |
| Truncation filter | whole-group filter реализован, но выключен как в published launch | config + unit test |

Канонический baseline: `configs/prime_faithful.yaml`. Длинный LoRA-профиль: `configs/prime_research_16k.yaml`.

## Memory path

У модели `V=248,320`. Полный BF16 tensor logits для `B=1`, `T=16,384`:

`1 × 16,384 × 248,320 × 2 bytes ≈ 8.14 GB`.

Это только один tensor; FP32 `log_softmax` и сохранённые intermediates увеличивают peak. Старый `log_prob_micro_batch_size` делит trajectories, но не tokens одной trajectory, поэтому проблему 16K не решает.

Новый path в `src/rl/chunked_logprobs.py`:

- получает hidden states без полного output logits;
- проецирует в vocabulary по `256` tokens;
- сразу выбирает log-prob target token;
- использует activation checkpointing для vocab projection при backward;
- даёт exact log-prob, а не sampled/approximate softmax.

Raw vocab tensor на chunk — около `127 MB`; с FP32 normalization ожидаемый локальный projection peak меньше `0.4 GB`. Forward и backward совпадают с full projection в synthetic test.

Для PRIME `use_liger_kernel=false`, потому что PRIME advantages имеют форму `(B,T)`, а текущий Liger 0.7 fused GRPO interface ожидает sequence advantages. Для ordinary GRPO/DPO-Z остаётся Liger fused loss, где advantages `(B,)`.

Target `20–30 GB`, максимум `40 GB` — пока **не измеренный результат**. Ему помогают sequential actor/PRM/reference placement, CPU offload optimizer/model state, vLLM sleep mode, gradient checkpointing и optional LoRA. Реальный peak должен быть измерен по фазам на H200.

## LoRA

LoRA допустима и полезна для быстрых ablations. При `r=16` post-trained PRM имеет `10,822,656` trainable из `763,215,680` параметров, около `1.42%`. Это резко сокращает gradients, optimizer state и checkpoint size, но не vocabulary projection и не KV/cache автоматически.

Поэтому:

- faithful reference оставлен full-parameter;
- `prime_research_16k.yaml` включает LoRA actor + PRM;
- LoRA и full tuning нельзя смешивать в одной строке сравнения без отдельной ablation.

## Research extensions

Все расширения opt-in и имеют отдельный overlay.

1. **DPO-Z baseline** — реализован в ordinary GRPO и PRIME как `β logmeanexp(R/β)`, по умолчанию leave-one-out. Он заменяет group centering; повторное центрирование не применяется. Текущий Monte Carlo estimator использует actor samples, а не независимые samples из reference policy — это ограничение надо явно указывать.
2. **Gradual process reward** — `constant`, linear warmup или reliability gate по correct-vs-incorrect pair ordering. Идея перспективна как стабилизация раннего noisy PRM; reliability metric необходимо валидировать held-out.
3. **Z(x)-calibrated PRM** — реализована осторожная версия: prompt-dependent intercept, L2 regularization и optional within-prompt ranking. Это gauge-complete calibration objective, но не доказанное восстановление истинного DPO partition function, поскольку ratio reward сам по себе не идентифицирует prompt constant.
4. **PRM-guided candidates** — candidate-pool selection поддерживает stochastic proposal с weights; deterministic top-k разрешается только явным `allow_biased_update`. Это не token-level beam search. Настоящий guided beam потребует учёта behavior policy на каждом branching step.

Наиболее перспективный и чистый первый ablation: gradual/reliability-gated process weight. DPO-Z теоретически интересен, но его эффект зависит от estimator distribution и исчезает при неосторожном повторном centering. Prompt calibration — более рискованная, но потенциально содержательная гипотеза. Guided search — самая дорогая и методологически опасная из четырёх из-за off-policy bias.

## Environment и модель

Выбран существующий H200 environment `~/opt_project/venv`:

- Python 3.11;
- torch 2.10.0+cu128;
- transformers 5.8.1;
- TRL 1.2.0;
- PEFT 0.19.1;
- Liger 0.7.0;
- vLLM 0.19.1;
- flash-linear-attention 0.5.0;
- tilelang 0.1.9.

`Qwen/Qwen3.5-0.8B` скачан через high-performance HF Xet, без ONNX/GGUF/duplicate original artifacts; cache footprint около 1.7 GB. Offline text-only load возвращает `Qwen3_5ForCausalLM`, hidden size 1024 и vocabulary 248,320. LoRA wrapping также проходит.

Удалён старый monkey patch `fla.utils.IS_NVIDIA_HOPPER=False`: он обходил защиту kernel library и мог молча разрешить численно некорректный backward. Faithful path больше не скрывает kernel incompatibility.

## Verification

Выполнено:

- `ruff check` изменённого PRIME/GRPO кода: pass;
- `ruff format --check`: pass;
- `py_compile`: pass;
- `git diff --check`: pass;
- `pytest`: **23 passed**, 19 dependency warnings;
- post-trained offline text-only load: pass;
- LoRA construction/count: pass;
- `uv lock --check`: pass.

Открытые gates:

1. vLLM language-only generation smoke;
2. one-batch actor + PRM forward/backward с finite/non-zero gradients;
3. peak GPU memory по generation / actor / PRM / reference phases;
4. checkpoint/save/resume equivalence;
5. tiny end-to-end update на подготовленном dataset;
6. после этого — короткий faithful run и только затем ablations.

TRL 1.2.0 предупреждает, что официально поддерживает vLLM только до 0.18.0, тогда как установлен 0.19.1. Понижать vLLM без smoke нельзя: новая версия может быть нужна Qwen3.5. Сначала нужен фактический compatibility test.

## Не затронуто

- Полный training не запускался.
- Серверный working repository не перезаписывался; validation делалась в `/tmp/maxiv25_prime_codex`.
- Commit/push не выполнялись.
- Пользовательские untracked result JSON сохранены без изменений.
- Удалённые старые checkpoints не восстанавливались и не используются как evidence.
