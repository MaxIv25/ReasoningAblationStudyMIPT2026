---
type: synthesis
status: historical
updated: 07-08-2026
tags: [prime, legacy, negative-results, audit]
---
# Legacy project audit

## Provenance

- Local repo: `/home/maxim/opt_project/ReasoningAblationStudyMIPT2026`
- H200 repo: `/data/users/maxiv25/opt_project/ReasoningAblationStudyMIPT2026`
- GitHub: `git@github.com:MaxIv25/ReasoningAblationStudyMIPT2026.git`
- Local/server `main`: `d5fd1a6ea268d77ef71a803e8f07532960810428`, совпадает с `origin/main`.
- Parent folder `/home/maxim/opt_project` сам не является Git repo.
- Official clone: `/home/maxim/opt_project/PRIME_official`, clean current main `18ad596f`.

## Главные source artifacts

- Poster: `/home/maxim/opt_project/poster/poster.pdf`
- Poster source и три идеи: `/home/maxim/opt_project/poster/poster.tex:288`
- PRIME review: `/home/maxim/opt_project/stage2_grpo/PRIME_report.md`
- Theory/extensions: `/home/maxim/opt_project/stage2_grpo/PRIME_theory_and_extensions.md`
- Claude Q&A: `/home/maxim/opt_project/stage2_grpo/faq_deep_questions.md`

Последние три документа — secondary narrative, не raw experimental evidence.

## Что можно переиспользовать

- Qwen3.5 multimodal/text loading fixes.
- Math verifier и data preparation/calibration.
- Evaluation и maj@8 tooling.
- Logging/metrics export.
- Checkpoint/resume fixes для policy и PRM.
- H200 memory/offload knowledge.

## Что нельзя считать подтверждённым

Исторический `exp5_5b` evaluation сообщал GSM8K `76.12`, MATH-500 `69.4`, MATH-Hard `45.17`, но checkpoint получен до позднейших correctness fixes. Эти числа сохраняются как history, не как доказательство faithful PRIME.

Другие negative results:

- первый PRIME run терял около 85–90% prompts после фильтрации и иногда имел zero gradients;
- 4B run демонстрировал collapse: длина росла примерно `4.8K → 16.2K`, accuracy падала примерно `.72 → .0125`, format reward → `0`;
- серверный 4B run технически дошёл до `187/187`, но занимал около 38.6 часа и часто повторял PRM batches после OOM; algorithmic parity при этом отсутствовала;
- DPO-Z run не завершён и не имеет evaluation;
- curriculum run имеет неполную локальную audit trail;
- Z-calibrated loss implemented, но не имеет валидного завершённого evaluation.

## Checkpoints и environment

- Старые PRIME checkpoints намеренно удалены пользователем ради диска, так как реализации считались неудачными.
- Это не потеря положительного evidence: старые checkpoints не должны становиться starting point нового path.
- Текущий H200 environment не pinned: Python `3.11.15`, torch `2.10.0`, transformers `5.8.1`, trl `1.2.0`, vLLM `0.19.1`, PEFT `0.19.1`; в логах есть предупреждение о несовместимости TRL с установленным vLLM.
- Calibration datasets зависят от старых SFT checkpoints и не должны переноситься на `Qwen/Qwen3.5-0.8B` без новой probing/calibration.

## Operational note

На read-only snapshot 07-08-2026 все восемь H200 были заняты. Ничего не запускалось; это transient state, а не blocker проекта.
