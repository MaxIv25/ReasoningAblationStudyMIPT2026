---
type: paper
status: read
year: 2024
arxiv: "2412.01981"
url: "https://arxiv.org/abs/2412.01981"
updated: 07-08-2026
tags: [implicit-prm, process-reward, reward-modeling]
---
# Free Process Rewards without Process Labels

## Citation

Lifan Yuan, Wendi Li, Huayu Chen, et al. *Free Process Rewards without Process Labels*. arXiv:2412.01981, 2024.

## Claim

Implicit PRM можно получить без step-level labels: outcome reward parameterized через log-likelihood ratio policy и reference, после чего token-wise decomposition ratio даёт process scores.

## Method

Основной объект:

$$
q_\phi(x,y)=\log\frac{\pi_\phi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=\sum_t\log\frac{\pi_\phi(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}.
$$

ORM обучается на response-level labels с CE/DPO-подобными objectives, а отдельные token contributions используются как process reward.

## Evidence

Авторы показывают, что implicit PRM может быть data-efficient, работает при сильном label imbalance и не обязательно выигрывает от добавления Math-Shepherd process labels. Для проекта важнее механизм, чем benchmark numbers.

## Limitation, важное для авторских идей

Additive prompt-dependent constant reward не восстанавливается из одного likelihood ratio. Кроме того,

$$
\mathbb{E}_{y\sim\pi_{\mathrm{ref}}}
\exp\left(\log\frac{\pi_\phi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right)=1.
$$

Поэтому `log mean exp(q)` на reference samples не даёт ненулевой `Z(x)` calibration. Это ключевая проблема старой гипотезы `Z(x)`-calibrated PRM loss.

## Direct relevance to repository

- Определяет PRM BCE/log-ratio unit tests.
- Требует точного token alignment policy/PRM/reference.
- Объясняет, почему user idea `Z(x)` нужно переформулировать как heuristic intercept, если она останется в проекте.

## Relation to other papers

- [[Papers/Process Reinforcement through Implicit Rewards]] переносит implicit PRM в online RL.
