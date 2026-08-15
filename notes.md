# Notes: PRIME reproduction and extensions

## Scope

- Target model: `Qwen/Qwen3.5-0.8B` (post-trained model).
- First milestone: reproduce the authors' PRIME implementation before testing extensions.
- Later long-context target: up to 16,384 completion tokens.
- Desired peak memory: 20–30 GB per GPU, with 40 GB as an upper bound including PRM.

## Initial audit findings

- The current trainer updates the PRM before computing actor process rewards; the published PRIME path uses the pre-update PRM for the current rollout batch.
- The current online filtering zeros actor advantages after PRM training; the author implementation filters/refills the rollout batch before both updates.
- Current PRM optimizer defaults differ from the published recipe (`weight_decay=0.01`, gradient clipping 1 instead of `weight_decay=0`, clipping 10).
- Current experimental defaults (8 rollouts and long completions) must not be presented as faithful published defaults (4 rollouts, 3,072 response tokens).
- The existing DPO-Z formula uses the inverse temperature in the wrong place relative to `beta * log E exp(R / beta)`.
- A prompt-constant DPO-Z baseline cancels if followed by ordinary within-group centering; it must replace that baseline or affect prompt weighting.
- LoRA reduces trainable gradients and optimizer states but does not solve vocabulary-logit or activation memory by itself.
- The official `log_prob_micro_batch_size` chunks trajectories, not tokens within one trajectory; it cannot prevent a single 16K × 248K logits tensor.
- TRL/Liger exposes a chunked fused GRPO loss, but Liger 0.7.0 accepts sequence-level `(B,)` advantages. PRIME uses token-level `(B,T)` advantages, so the policy path needs a dedicated checkpointed token-chunk projection.
- The H200 environment selected for this project is `~/opt_project/venv`: torch 2.10.0, transformers 5.8.1, TRL 1.2.0, PEFT 0.19.1, Liger 0.7.0, vLLM 0.19.1, FLA 0.5.0.
- TRL 1.2.0 emits a compatibility warning for vLLM 0.19.1 (documented support ends at 0.18.0). Do not launch training until generation compatibility is smoke-tested or the environment is pinned compatibly.
- The validated suite passes 23 tests, including exact chunked/full forward and backward equivalence, raw implicit reward, official PRM BCE, config isolation, and refill.
- The post-trained checkpoint is cached on H200 (~1.7 GB after excluding duplicate formats), loads offline as text-only `Qwen3_5ForCausalLM`, and wraps with LoRA.
- The legacy `IS_NVIDIA_HOPPER=False` monkey patch was removed; a kernel compatibility error is preferable to silently incorrect gradients.

## Research ideas to preserve behind independent flags

1. **DPO-Z baseline** for actor advantages, using an explicit estimator and no subsequent centering that would cancel it.
2. **Gradual/reliability-gated process reward**, ramping or gating its actor weight.
3. **Z(x)-calibrated PRM loss**, reformulated as a learnable prompt intercept plus absolute and within-prompt calibration objectives rather than claiming an identifiable partition function from ratios alone.
4. **PRM-guided stochastic rollout proposal**, with behavior-policy correction; deterministic top-k/beam trajectories must not be treated as on-policy samples.

## Verification contract

- Unit tests for filtering, separate outcome/process RLOO, reverse cumulative process return, normalization, DPO-Z, and idea isolation.
- Configuration tests for faithful published defaults versus 16K research defaults.
- A tiny forward/backward smoke test for the selected-logprob path before any training run.
- Every eventual result must record revision, config, seed, dataset version, environment, and raw metrics.

## Multi-server deployment — 15-08-2026

- H200 GPU 7 had 46.3 GiB free around a stable foreign workload. The restarted
  constant `5e-6` run uses the DPO-Z-proven `.20` vLLM reservation and saves
  every 20 steps; after vLLM initialization it retained ~15.4 GiB free.
- `ssh opt` is `brain-lab.mipt.ru`: 32 CPU cores, 196 GiB RAM, 4×A100 80GB.
  GPU 1 was completely free (81.15 GiB); `/home` had 457 GiB free.
- Required non-Git artifacts are small enough to transfer directly: merged SFT
  checkpoint 1.5 GiB, RL train dataset 708 KiB, validation split 44 KiB.
- Opt system Python is 3.10 and `uv` was not in PATH; project lock requires
  Python 3.11. Setup must be user-local and build-limited.
- The transferred merged SFT and both RL datasets match their H200 SHA-256
  manifests byte-for-byte.
- FLA 0.5.0 evaluates `TileLangBackend.is_available()` before
  `is_enabled()`, so `FLA_TILELANG=0` still imports TileLang. TileLang 0.1.9
  bundled TVM conflicts with `tvm_ffi` on the A100 environment. Keeping
  TileLang as an optional Hopper extra restores the Qwen3.5 import and leaves
  the A100 on the Triton/Torch fallback path.
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for colocated
  vLLM sleep mode: its CuMemAllocator memory pool asserts that expandable
  segments are unsupported.
- A100 16K gate evidence: one actor microbatch of 8 trajectories completed in
  74.76 s; train completion mean/max 2951/6823, validation reached the 16384
  cap, post-update vLLM canary changed with max diff zero, and vLLM-awake own
  footprint was about 31 GiB.
