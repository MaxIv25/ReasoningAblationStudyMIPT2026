---
type: synthesis
status: active
updated: 07-08-2026
tags: [prime, reproduction, project-overview]
---
# Project overview

## Что изучает проект

Проект исследует online RL для reasoning LLM с dense token-level signal из implicit process reward model. Ближайшая цель не в заявлении нового метода, а в получении надёжной реализации PRIME на компактной готовой instruct/post-trained модели.

## Research object

- Policy, PRM и frozen reference инициализируются одним checkpoint `Qwen/Qwen3.5-0.8B`.
- Outcome verifier даёт response-level correct/incorrect labels.
- Implicit PRM обучается online на этих labels.
- Token-level score строится из log-ratio `log π_PRM - log π_ref`.
- Outcome и process returns объединяются после отдельного RLOO.

## Граница первой milestone

Первая milestone отвечает на один вопрос: **можно ли воспроизвести original PRIME semantics end-to-end на Qwen3.5-0.8B так, чтобы каждый скрытый engineering choice был проверяемым?**

Она не включает:

- новый SFT;
- полноценное сравнение с GRPO/DAPO;
- проверку трёх авторских extensions;
- попытку воспроизвести paper benchmark numbers на другой модели и данных.

## Reproduction и empirical replication

Это разные уровни:

1. **Algorithm/code-path reproduction**: численная parity формул, filtering, update order, optimizer и checkpoint lifecycle.
2. **Empirical replication**: воспроизведение заявленных gains на исходной модели, данных и hardware scale.

На `Qwen/Qwen3.5-0.8B` реалистична первая цель и новая model-specific проверка поведения. Она не является прямой репликацией чисел Eurus-2-7B-PRIME.

## Стратегия работы с кодом

Начинать новый repository с пустого листа не требуется. Существующий проект уже содержит полезные Qwen3.5 loaders, data/eval pipeline, verifier, logging, calibration и checkpoint fixes. Новый старт означает:

- новая branch и pinned environment;
- отсутствие наследуемых model checkpoints;
- новые parity tests;
- selective rewrite критического PRIME core;
- запрет считать старые метрики доказательством корректности нового path.

## Почему Qwen3.5-0.8B — нетривиальная адаптация

- Это post-trained multimodal checkpoint с architecture class `Qwen3_5ForConditionalGeneration`, а не обычный text-only `AutoModelForCausalLM`.
- Thinking mode существует, но для 0.8B выключен по умолчанию.
- Официальный PRIME recipe не был опубликован для этого checkpoint.
- Значит, даже faithful algorithm semantics требуют отдельной проверки model loading, chat template, token alignment и rollout engine compatibility.

## Связи

- Метод: [[Papers/Process Reinforcement through Implicit Rewards]]
- Основа implicit reward: [[Papers/Free Process Rewards without Process Labels]]
- Точный runbook: [[Experiments/PRIME-Reproduction]]
- Старые артефакты и negative results: [[Knowledge/Legacy-Audit]]
