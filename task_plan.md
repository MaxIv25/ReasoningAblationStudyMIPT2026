# Task Plan: faithful PRIME, memory-safe RL, and research extensions

## Goal

Bring the repository to a testable author-faithful PRIME baseline, then add isolated memory and research extensions for Qwen/Qwen3.5-0.8B without mixing them into the baseline.

## Phases

- [x] Phase 1: Audit the local trainer, official PRIME code, configs, and environment
- [x] Phase 2: Add failing unit/contract tests for published PRIME semantics and DPO-Z
- [x] Phase 3: Implement the faithful PRIME core and wire it into the trainer
- [x] Phase 4: Add memory-safe log-probability computation and optional LoRA
- [x] Phase 5: Add DPO-Z to GRPO and four independently switchable PRIME ideas
- [x] Phase 6: Move the project Obsidian knowledge base into the repository
- [ ] Phase 7: Complete GPU integration, memory profiling, and save/resume verification
- [x] Phase 8: Restart constant-LR `5e-6` GRPO safely on H200 GPU 7
- [x] Phase 9: Audit, commit, and push the reproducible research worktree
- [x] Phase 10: Bootstrap a thread-limited environment and artifacts on `ssh opt`
- [ ] Phase 11: Validate and launch constant-LR `1e-5` GRPO on `ssh opt`
- [ ] Phase 12: Record multi-server provenance, runtime gates, and handoff

## Key Questions

1. Which differences from the official code change the algorithm rather than only infrastructure?
2. Which installed TRL/Transformers/Liger APIs can avoid materializing `[batch, sequence, vocab]` logits?
3. How should DPO-Z be defined so it does not cancel under the usual group-mean/RLOO centering?
4. Which parts of PRM-guided generation can be implemented correctly with behavior-policy correction?
5. Can the full actor + PRM + reference setup fit under the requested memory budget with LoRA and fused losses?

## Decisions Made

- Keep two explicit profiles: `faithful_prime` (published defaults) and `research_16k` (long-context extensions).
- Preserve full-parameter training as the reproduction reference; make LoRA an optional experimental/storage-efficient mode.
- Do not start expensive H200 training during implementation; use unit and smoke tests first.
- Do not treat old runs or deleted checkpoints as evidence.
- Use `~/opt_project/venv` on H200 for validation; it already contains the required Qwen/TRL/PEFT/Liger stack.
- Use explicit checkpointed token chunks for dense PRIME advantages because Liger GRPO currently accepts sequence-level `(B,)` advantages only.
- Treat the failed GPU-4 `5e-6` run as a resource failure: restart from the
  initial SFT actor, use the successful GPU-7 `.20` vLLM memory profile, and
  save every 20 steps for recovery.
- Use `constant 1e-5` as the next LR sweep point on `ssh opt`; keep algorithmic
  settings matched and document server/resource-only deviations separately.
- Limit setup/build parallelism on `ssh opt` (`OMP_NUM_THREADS=2`,
  `MAX_JOBS=2`, one concurrent build) and avoid source-building FlashAttention.

## Errors Encountered

- Local sandbox failed with `bwrap: setting up uid map: Permission denied`; read-only commands are being retried with approved elevated execution.
- `apply_patch` intermittently failed with a bwrap loopback error; the affected one-file edit was applied from a generated unified diff.
- The selected H200 environment had no pytest; installed only `pytest==9.0.2` plus its small dependencies with uv.
- The selected environment warns that TRL 1.2.0 supports vLLM through 0.18.0 while the environment has vLLM 0.19.1; generation smoke testing remains required.
- The first server pytest retry could not import `src`; fixed reproducibly with `pythonpath = ["."]` in `pyproject.toml`.
- Removed the legacy Hopper guard override (`IS_NVIDIA_HOPPER=False`), which could silently allow an incorrect FLA backward.
- GPU-4 constant `5e-6` failed after step 44 while waking vLLM KV cache; no
  checkpoint existed because `save_steps=75`.
- Первый secret-scan command имел shell quoting error; повторный scan с
  отдельными безопасными regex завершён, credential material не найден.
- Первый direct-transfer checkpoint command не использовал `scp -r`; повторный
  tracked transfer корректно продолжает копирование каталога без удаления partial artifact.
- Одна orchestration-сессия прервалась во время долгого `scp -3`; сам transfer
  был перезапущен как отдельная наблюдаемая PTY-сессия и показывает монотонный progress.
- Первый A100 smoke завершился до model load: FLA 0.5.0 импортировал TileLang
  до проверки `FLA_TILELANG=0`, а bundled TVM из TileLang 0.1.9 конфликтовал
  с `tvm_ffi` при регистрации `ffi.Tensor`. Minimal import воспроизводил
  failure 2/2 и проходил 2/2 без TileLang; dependency сделана Hopper-only extra.
- Две попытки `uv lock --offline` не нашли registry metadata в локальных
  caches, а локальный online resolver завис на network metadata. Lock был
  безопасно сгенерирован online на `ssh opt` за 6.94 s с concurrency=2.

## Status

**Phase 11 active** — custom merged SFT and both datasets match H200 SHA-256
manifests on `ssh opt`. The A100 default environment excludes optional
TileLang, passes the original Qwen3.5 import loop twice and the full 91-test
suite with two threads. Re-run the two-step GPU smoke before the full `1e-5`.
