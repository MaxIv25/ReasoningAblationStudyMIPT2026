# PRIME RL reproduction — plan

## Текущий operational state — 14-08-2026

- [x] Зафиксировать unseen validation set и одинаковый sampling protocol для
  Vanilla GRPO, DPO-Z и PRIME.
- [x] Пройти GPU validation smoke для всех трёх методов до и после первого
  actor update.
- [ ] Дождаться завершения новых full runs; до этого не промотировать метрики
  в `Results/`.
- [ ] После full runs выполнить primary final evaluation как `maj@8` на том же
  sampler; official Qwen thinking preset оставить отдельным secondary protocol.

## Приоритет 0: зафиксировать reproduction target

- [x] Разделить original PRIME и современный maintained `verl-recipe`.
- [x] Зафиксировать original PRIME commit `18ad596f08d487bb546d80d738d99ec697bd2e75`.
- [x] Выбрать `Qwen/Qwen3.5-0.8B` как post-trained starting checkpoint.
- [x] Создать clean reproduction branch от legacy commit `d5fd1a6`; новый repo не создавать.
- [x] Выделить pure/testable PRIME core; legacy implementation сохранена в Git history, entrypoint заменён исправленным trainer.
- [x] Зафиксировать зависимости в `pyproject.toml` и `uv.lock`; выбран server env `~/opt_project/venv`.

## Приоритет 1: deterministic parity harness

До H200 training реализовать маленькие synthetic tests для `K=4`:

- [x] process reward равен нулю при `PRM == reference`;
- [ ] `single-forward`: текущий batch получает pre-update PRM reward, update влияет только на следующий batch;
- [x] accuracy filter принимает prompt groups только при pass rate в `[0.2, 0.8]`;
- [x] optional truncation filter удаляет всю group, если хотя бы один response truncated;
- [x] oversampling/refill сохраняет целевой prompt batch size;
- [x] outcome и process rewards проходят отдельные RLOO baselines и discounted returns;
- [x] process reward batch normalization совпадает с official code;
- [x] PRM BCE и EOS mask совпадают численно; `beta=0.05`, `grad_clip=10`, `weight_decay=0` зафиксированы config contract;
- [ ] save/resume восстанавливает actor, PRM, optimizers, schedulers и global step.

## Приоритет 2: Qwen3.5 compatibility smoke

- [x] Проверить post-trained Qwen3.5 text-only loading без vision encoder overhead.
- [ ] Проверить официальный chat template и явно включить thinking mode; у 0.8B он по умолчанию выключен.
- [ ] Проверить одинаковую tokenization/log-prob alignment для policy, PRM и reference.
- [x] Заменить permissive answer verifier на strict Math-Verify profile с structural guards.
- [x] Подключить fail-closed resident-weight sync canary к GRPO и PRIME colocate paths.
- [x] На one-update GPU smoke подтвердить resident match и изменение synced tensor после optimizer step.
- [x] Ограничить prompt/response lengths reproduction protocol, не использовать нативные 262K модели.
- [ ] Один forward/backward/update на synthetic batch: finite loss, non-zero gradients, no shape drift.

## Приоритет 2.5: fresh post-trained data probe

- [x] Исключить старые SFT-policy calibration labels из нового evidence/train selection.
- [ ] Создать immutable probe manifest: MATH L1–L5 по 70 subject-balanced prompts + 100 GSM8K train.
- [ ] Проверить отсутствие eval overlap и исключить probe prompts из будущего train.
- [ ] После GPU approval сгенерировать `K=8` при exact training sampling и strict verifier.
- [ ] Выбрать MATH levels/GSM8K share по mixed-group fraction, не по individual prompt pass rate.
- [ ] Fresh-семплировать 3000 train prompts из оставшегося original train pool.

## Приоритет 3: короткий end-to-end run

Только после подтверждения пользователя и проверки свободной GPU:

- [ ] малый фиксированный dataset, seed и frozen verifier;
- [ ] логировать PRM loss/AUC, process/outcome returns до whitening, actor KL, entropy, grad norms, filter acceptance, truncation rate и response length;
- [ ] проверить checkpoint/resume на середине run;
- [ ] сохранить config, commit, environment, raw metrics и exact dataset revision.

## Go/no-go для полного reproduction run

- Все parity tests проходят.
- Первая process reward при `PRM == reference` численно нулевая.
- Filter/refill происходит до PRM и actor updates.
- PRM начинает различать correct/incorrect rollouts без exploding reward scale.
- Actor получает конечный ненулевой gradient; нет silent zeroing большинства batch.
- Resume даёт ту же следующую update trajectory в пределах ожидаемой nondeterminism.
- Smoke run не демонстрирует ранний length/entropy collapse.

## После reproduction

Порядок дальнейших исследований: gradual process weighting → frozen implicit-PRM-guided search ablation → только при положительном результате guided stochastic proposal с behavior correction → исправленный leave-one-out DPO-Z → gauge-complete PRM вместо буквального `Z(x)` loss. GRPO/DAPO comparison и полный ablation plan проектируются отдельно.
