---
type: knowledge
status: active
updated: 14-08-2026
tags: [verifier, vllm, trl, reward-correctness, synchronization]
---
# Verifier correctness and vLLM synchronization

## Решение

- Training path использует `vllm_mode=colocate`, не `trl vllm-serve`.
- Online reward работает fail-closed: strict `Math-Verify`, strict boxed extraction
  в caller, без permissive string fallback.
- Каждый colocated sync проверяется в окончательном pre-generation state — после sleep-mode `reload_weights`. Успешное сообщение `sync_weights` до generation не считается доказательством.

## Воспроизведённые ошибки старого verifier

Старый `verify_answer` после неудачного symbolic parsing безусловно удалял все
запятые. Локальный differential probe подтвердил false positives:

| Gold | Prediction | Старый verifier | Исправленный verifier |
|---|---|---:|---:|
| `(1,2)` | `(12)` | correct | incorrect |
| `\{1,2\}` | `\{12\}` | correct | incorrect |
| `10001_2` | `10001` | correct | incorrect |
| `10001_2` | `10001_3` | correct | incorrect |

При этом formatting-equivalent `\frac{1}{2}` / `\dfrac{1}{2}` и настоящий
thousands separator `16384` / `16,384` должны приниматься.

Причины:

1. `replace(",", "")` стирал структуру tuple/set.
2. Общий `math_verify.parse()` мог потерять numeric-base annotation, поэтому
   требовался отдельный structural guard.
3. Последний string fallback превращал parse failure в потенциальный reward,
   что особенно опасно для online RL и verifier gaming.

Новый профиль взят из проверенной реализации `prompt-policy-transfer` и
согласован с рекомендациями
[Math-Verify](https://github.com/huggingface/Math-Verify): strict LaTeX
normalization для reward, `malformed_operators=False`, `nits=False`,
`equations=False`, `fallback_mode=no_fallback`. Запятые удаляются только по
шаблону настоящего thousands separator. Numeric-base annotation проверяется до
symbolic comparison.

## Почему не `trl vllm-serve`

Старый server-mode run имел симптом: trainer логировал sync, но generation
оставалась от старой policy. Git history фиксирует 87.5–100% filtered prompt
groups и переход на colocate (`2ec4466`).

В установленном TRL server endpoint `/update_named_param/` отправляет worker
command как `fire_and_forget` и сразу отвечает `Request received`. Поэтому HTTP
acknowledgement и trainer log доказывают постановку команды в очередь, но не
успешный `model.load_weights`. Server mode также требует trainer и server на
разных CUDA devices согласно
[официальной TRL vLLM integration](https://huggingface.co/docs/trl/main/en/vllm_integration).
Для single-H200 memory target это лишняя GPU и отдельный NCCL failure surface.

Colocate не означает автоматическую корректность. Поэтому перенесён
non-intrusive canary из `prompt-policy-transfer`:

1. при sleep mode deferred ранний trainer sync;
2. ждёт `wake_up` и `collective_rpc("reload_weights")` внутри generation;
3. сразу после disk reload выполняет ровно один current-policy sync;
4. временно перехватывает resident `llm_model.load_weights`;
5. сохраняет representative projection tensor, фактически переданный штатным TRL sync;
6. вызывает original `load_weights` ровно один раз;
7. сравнивает переданный tensor с resident vLLM parameter;
8. отслеживает изменение source actor tensor между optimizer steps; отдельный zero-signal step допустим, но восемь последовательных steps без изменения source считаются stalled training;
9. пишет append-only `vllm_sync_checks.jsonl` в run output и падает fail-closed при resident mismatch или persistent stall.

Canary не выполняет дополнительный PEFT `merge/unmerge`: такой intrusive checker ранее сам создавал BF16 drift.

Причина нового ordering — подтверждённая несовместимость TRL `1.2.0` с установленным vLLM `0.19.1`: TRL синхронизировал merged LoRA, но затем `generate()` вызывал `reload_weights`; vLLM загружал original model path с диска и перезаписывал actor. Старый canary стоял до reload и не ловил ошибку.

## Acceptance gates перед training

- CPU regression suite проходит.
- На GPU event 0: `resident_matches_synced_tensor=true`.
- После первого ненулевого optimizer update: `synced_tensor_changed_since_previous=true`, resident match сохраняется и событие записано после `reload_weights`.
- Отдельно проверить `grad_norm > 0`; stale policy и zero-gradient являются разными failure modes.
- Paired diagnostic: initial trajectories одинаковы, а trajectories и fixed logits после разных optimizer updates расходятся.
- Сохранить `vllm_sync_checks.jsonl`, config, environment versions и raw rollout metrics вместе с run.

## Текущий статус

- Реализация добавлена для ordinary GRPO, DPO-Z GRPO и PRIME.
- Regression tests и полный CPU suite `80 passed` в pinned H200 environment.
- Paired Qwen3.5 real-GPU smoke: post-reload canary `3/3` для vanilla и DPO-Z; step 1 trajectories `8/8` equal, step 2/3 `0/8` equal при одинаковых prompts.
- Старые full runs invalidated; PRIME требует отдельного post-fix GPU smoke.
