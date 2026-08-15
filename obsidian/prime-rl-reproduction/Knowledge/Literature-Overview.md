---
type: synthesis
status: active
updated: 07-08-2026
tags: [prime, literature, llm-rl]
---
# Необходимая литература

## Обязательна для первой milestone

1. [[Papers/Process Reinforcement through Implicit Rewards]] — алгоритм PRIME, decomposition outcome/process returns, online PRM и main ablations.
2. [[Papers/Free Process Rewards without Process Labels]] — теоретическая и эмпирическая основа implicit PRM как likelihood-ratio model.
3. [Original PRIME code](https://github.com/PRIME-RL/PRIME) — фактическая semantics filtering, RLOO, normalization и update order.
4. [Maintained verl-recipe/prime](https://github.com/verl-project/verl-recipe/tree/main/prime) — современная инженерная реализация, но изменённый algorithm target.
5. [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) — post-trained checkpoint, thinking mode и architecture constraints.

## Нужна после faithful reproduction

### [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

Источник GRPO. Для будущего baseline важны group-relative normalization, отсутствие critic и связь с PPO objective. Не является каноном для первой PRIME implementation milestone, поскольку original PRIME main recipe использует RLOO.

### [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)

Важен для будущего сравнения: Clip-Higher, dynamic sampling, token-level policy gradient loss и overlong reward shaping. DAPO не должен незаметно менять PRIME reproduction protocol; его mechanics добавляются как отдельный baseline.

### [DAPO official repository](https://github.com/BytedTsinghua-SIA/DAPO)

Primary implementation reference для будущего baseline. Dataset/verifier/scale значительно отличаются от текущего 0.8B setting.

## Вывод для проекта

На первой стадии literature review не должен раздувать scope. Достаточно понять PRIME + Implicit PRM и зафиксировать original code semantics. GRPO/DAPO остаются проверенными источниками для следующей стадии, но experiment plan для них пока не создаётся.
