---
type: paper
status: read
year: 2025
arxiv: "2502.01456"
url: "https://arxiv.org/abs/2502.01456"
code: "https://github.com/PRIME-RL/PRIME"
updated: 07-08-2026
tags: [prime, process-reward, online-rl]
---
# Process Reinforcement through Implicit Rewards

## Citation

Ganqu Cui, Lifan Yuan, Zefan Wang, et al. *Process Reinforcement through Implicit Rewards*. arXiv:2502.01456, 2025.

## Claim

PRIME даёт dense token-level rewards без process labels: implicit PRM обучается online на policy rollouts с outcome labels, а затем его likelihood ratios используются как process signal для actor.

## Method

Policy `πθ`, PRM `πφ` и frozen reference `πref` инициализируются одним checkpoint. Для каждого prompt генерируется group responses, outcome verifier помечает их, PRM обучается на response-level labels, а token score задаётся как

$$
r_\phi(y_t)=\beta\left[
\log\pi_\phi(y_t\mid x,y_{<t})-
\log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
\right].
$$

Outcome и process returns считаются отдельно и затем суммируются. Main implementation использует RLOO и online accuracy filtering.

## Evidence

Статья сообщает существенное улучшение reasoning benchmarks для Eurus-2-7B-PRIME и показывает совместимость process rewards с разными advantage estimators. Для этого проекта эти числа являются evidence метода на исходной setup, но не target numbers для Qwen3.5-0.8B.

## Implementation details, критичные для reproduction

- main script: 256 prompts × 4 rollouts;
- prompt length 1024, response length 3072;
- accuracy filtering `[.2,.8]` до updates;
- actor LR `5e-7`, PRM LR `1e-6`, `β=.05`;
- actor KL coefficient `0`;
- separate-source RLOO;
- process reward batch normalization;
- original default `single-forward`: текущий actor использует pre-update PRM score.

## Limitations

- Paper equation не раскрывает все code-level normalizations и coefficients.
- Filtering и batch refill существенно меняют effective training distribution.
- Original repository и maintained recipe теперь расходятся.
- Reported hardware/data/model scale не переносится напрямую на Qwen3.5-0.8B.
- Dense reward не устраняет reward hacking автоматически; online update только снижает часть distribution-shift risk.

## Direct relevance to repository

Является canonical algorithm source. Любая custom implementation считается корректной только после numerical parity с original code или после явного документирования divergence. См. [[Knowledge/Official-Implementation-Gaps]] и [[Experiments/PRIME-Reproduction]].

## Relation to other papers

- [[Papers/Free Process Rewards without Process Labels]] — исходная likelihood-ratio construction implicit PRM.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO baseline, не main estimator original PRIME.
- [DAPO](https://arxiv.org/abs/2503.14476) — более поздний outcome-only RL recipe для будущего comparison.
