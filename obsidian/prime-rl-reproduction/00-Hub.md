---
type: project
status: active
phase: validation-enabled-full-runs
model: Qwen/Qwen3.5-0.8B
legacy_repo: git@github.com:MaxIv25/ReasoningAblationStudyMIPT2026.git
legacy_commit: d5fd1a6ea268d77ef71a803e8f07532960810428
official_prime_commit: 18ad596f08d487bb546d80d738d99ec697bd2e75
host: h200
updated: 14-08-2026
tags: [prime, llm-rl, process-reward, reproduction, qwen3-5]
---
# PRIME RL reproduction — project hub

## Главная цель

С нуля получить проверенное end-to-end воспроизведение PRIME на готовой post-trained модели `Qwen/Qwen3.5-0.8B`, без отдельного SFT-этапа. До достижения official parity собственные расширения, GRPO/DAPO comparison и полный ablation plan не реализуются.

> [!important] Что здесь означает reproduction
> Первый target — семантика статьи и исходного репозитория PRIME: `single-forward`, reward от PRM до его update на текущем batch, actor KL `0`, исходный RLOO и normalization. Современный `verl-recipe/prime` рассматривается как отдельная, изменённая реализация, а не как незаметная замена исходного алгоритма.

## Почему нужен чистый старт

- Старый custom trainer технически запускался, но не совпадал с official PRIME в filtering, PRM update order, optimizer settings и data/length protocol.
- Единственный сильный исторический PRIME evaluation был выполнен до позднейших исправлений RLOO, advantage normalization и checkpointing, поэтому не подтверждает текущий код.
- Старые checkpoints намеренно удалены с H200 как неудачные и занимающие диск. Новый проект не должен использовать их как starting point.
- Среда старого проекта не pinned и содержит потенциально несовместимые версии `trl` и `vllm`.

## Текущие решения

- Модель: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B), официальный post-trained checkpoint. Отдельного официального `-Instruct` checkpoint не найдено.
- `Qwen/Qwen3.5-0.8B-Base` не используется: model card называет его pre-trained-only.
- Qwen3.5 имеет multimodal `Qwen3_5ForConditionalGeneration`; text-only PRIME path обязан пройти отдельный compatibility smoke test.
- Новый repo не нужен. Работа идёт в ветке `feat/prime-faithful-memory` существующего `ReasoningAblationStudyMIPT2026`; legacy trainer заменён test-backed implementation, а его история остаётся доступна через Git.
- Никакой дорогой training job не запускается без отдельного подтверждения.
- Obsidian project memory теперь хранится в repo: `obsidian/prime-rl-reproduction`; старый vault path является symlink.
- Post-trained `Qwen/Qwen3.5-0.8B` скачан на H200 и проходит text-only load smoke; vision encoder не загружается.
- CPU regression suite: `85 passed`. Post-reload vLLM sync и validation path
  прошли real-GPU smoke для Vanilla GRPO, DPO-Z и PRIME: после update canary
  видит `changed=True`. Это runtime evidence, не оценка качества методов.
- Frozen validation split: `data/grpo_validation_unseen_100_v1`, 100 unseen
  prompts, exact train overlap `0`. Validation идёт на start и каждые 20 steps
  при training sampler `T=1, top_p=1, top_k=0, presence_penalty=0`, по одной
  generation на prompt. `eval batch=20` делит 100 prompts без остатка и избегает
  batch-unweighted смещения внутренних TRL GRPO metrics.
- Full-run retry4 для Vanilla / PRIME / DPO-Z запущен с нуля на GPU 1/3/7.
  Exact val100 baseline завершён: accuracy `.68`, mean length `6004`, clipped
  ratio `.14` у всех трёх методов. Vanilla и DPO-Z прошли step 2 с ненулевым
  actor update и post-reload `changed=true`; PRIME также завершил refill и
  подтвердил `changed=true` на step 2. Runs активны, но незавершённые результаты
  не являются evidence.

## Четыре авторские гипотезы — только после reproduction

1. `DPO-Z advantage baseline`.
2. Постепенное включение process reward.
3. `Z(x)`-calibrated PRM loss.
4. Online implicit-PRM-guided rollout search.

Предварительная оценка находится в [[Knowledge/Research-Ideas]]. Эти гипотезы не считаются подтверждёнными результатами старого проекта.

## Канонические заметки

- [[Knowledge/Project-Overview|Постановка и границы проекта]]
- [[Knowledge/Official-Implementation-Gaps|Статья, original code и maintained recipe: расхождения]]
- [[Knowledge/Legacy-Audit|Аудит старого локального и серверного проекта]]
- [[Knowledge/Verifier-and-vLLM-Sync|Verifier correctness и vLLM synchronization]]
- [[Knowledge/Research-Ideas|Оценка четырёх авторских идей]]
- [[Knowledge/Literature-Overview|Необходимая литература]]
- [[Experiments/PRIME-Reproduction|Reproducibility contract и acceptance gates]]
- [[Results/RL-Training-Dynamics/analysis-report|GRPO / PRIME / DPO-Z: training dynamics]]
- [[Papers/Process Reinforcement through Implicit Rewards|PRIME]]
- [[Papers/Free Process Rewards without Process Labels|Implicit PRM]]
- [[01-Plan|Активный план]]

## Код и источники

- Код локально: `/home/maxim/opt_project/ReasoningAblationStudyMIPT2026`
- Активная ветка: `feat/prime-faithful-memory`
- Legacy repo на H200: `/data/users/maxiv25/opt_project/ReasoningAblationStudyMIPT2026`
- Официальный PRIME clone: `/home/maxim/opt_project/PRIME_official`, commit `18ad596f`
- Poster source: `/home/maxim/opt_project/poster/poster.tex`
