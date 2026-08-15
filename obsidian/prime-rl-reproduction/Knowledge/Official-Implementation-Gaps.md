---
type: synthesis
status: active
updated: 07-08-2026
tags: [prime, implementation-audit, reproducibility]
---
# PRIME: paper, original code и maintained recipe

## Три разных объекта

1. [PRIME paper v2](https://arxiv.org/html/2502.01456v2).
2. [Исходный PRIME repository](https://github.com/PRIME-RL/PRIME), commit `18ad596f`.
3. [Поддерживаемый verl-recipe/prime](https://github.com/verl-project/verl-recipe/tree/main/prime), который меняет часть semantics.

Для первой reproduction milestone каноном считаются paper + original repository. Maintained recipe нужен как инженерный reference и отдельный target.

## Original published recipe

- 256 prompts × 4 rollouts.
- `max_prompt_length=1024`, `max_response_length=3072`.
- Accuracy filter: inclusive `[0.2, 0.8]` до PRM/actor update.
- Actor LR `5e-7`, PRM LR `1e-6`.
- `beta_train=0.05`, PRM `grad_clip=10`, `weight_decay=0`.
- Actor KL coefficient `0`.
- `single-forward`: process rewards текущего batch получены до обновления PRM; update влияет на следующий batch.
- Outcome и process reward sources получают отдельные RLOO baselines/returns.
- Process rewards дополнительно нормализуются по максимальному absolute reverse cumulative return в batch.
- Reward coefficients в code/config имеют масштаб порядка `5`, хотя equation в статье записана как простая сумма.

## Nuance: truncation filtering

Исходный README описывает удаление всей prompt group, если хотя бы один response truncated. Однако published `run_prime_main.sh` не включает `filter_truncated`; YAML default — `False`. Поэтому:

- literal published run: truncation group filter выключен;
- parity harness: semantics фильтра должны быть протестированы;
- включение фильтра — отдельный явный config choice, а не скрытый default.

## Maintained recipe не эквивалентен original

В текущем `verl-recipe/prime` встречаются:

- `update=before`: PRM сначала обновляется, затем rewards пересчитываются;
- actor KL reward penalty около `.001`;
- включённый truncation filtering;
- oversampling factor `4` и новый selection behavior;
- исключение последнего response token из process score;
- иные entropy/default settings.

Это разумные engineering changes, но их нельзя смешивать с original baseline.

## Критические gaps старого custom trainer

| Компонент | Official original | Legacy custom code |
|---|---|---|
| Accuracy filtering | До PRM/actor updates, `[.2,.8]`, group removal/refill | PRM обучается на полном batch; после этого actor advantages all-correct/all-wrong groups обнуляются |
| Truncation | Optional whole-group rejection | Token masking; нет official group/refill behavior |
| PRM order | Pre-update reward (`update=after`) | PRM update, затем reward от обновлённого PRM |
| PRM optimizer | clip `10`, weight decay `0` | clip `1`, weight decay `.01` |
| Lengths / K | prompt `1024`, response `3072`, K=4 | response до 16K, K=8; prompt cap не эквивалентен |
| Actor LR | `5e-7` | обычно `1.5e-6` через inherited config |
| Batch scale | 256 prompts | обычно 8 prompts |

## Главный вывод

Старый repo можно доисправить, но нельзя продолжать добавлять patches в монолит без numerical contract. Нужно сохранить reusable infrastructure и заменить/выделить filtering, reward construction, RLOO и update ordering в маленькие тестируемые компоненты. См. [[Experiments/PRIME-Reproduction]].


## `rm_coef=5`: declared intent versus literal RLOO execution

`run_prime_main.sh` задаёт `reward_model.rm_coef=5`, а `main_ppo.py` умножает
на него composite reward `all`. Однако original `compute_rloo_returns` считает
returns только для decomposed keys `rm_scores` и `gt_scores` и пропускает
`all`; raw `rm_scores` до этого не умножаются. Следовательно, в опубликованной
RLOO execution коэффициент выглядит operationally dead.

Это создаёт два честных reproduction targets:

- `literal-code`: process coefficient фактически `1`, включая public bug;
- `declared-intent`: применить заданный `rm_coef=5` к process source.

Результаты этих профилей нельзя смешивать или называть одной exact reproduction.
