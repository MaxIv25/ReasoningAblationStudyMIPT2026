#!/bin/bash
# ============================================================
# Download models for SFT Ablation Study
# ============================================================
# Uses hf_transfer for faster downloads and optional HF mirror
#
# Usage:
#   bash scripts/download_models.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_models.sh
# ============================================================

set -e

# Enable hf_transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Uncomment if you need a mirror (e.g., for slow connections in Russia/China):
# export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "Downloading models for SFT Ablation Study"
echo "=========================================="
echo "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
echo "HF_TRANSFER=enabled"
echo ""

# ── Install hf_transfer if not present ────────────────────────
pip install -q hf_transfer 2>/dev/null || true

# ── Download Qwen3.5-4B-Base (student) ────────────────────────
echo "[1/2] Downloading Qwen/Qwen3.5-4B-Base (~8GB)..."
huggingface-cli download Qwen/Qwen3.5-4B-Base

# ── Download Qwen3.5-35B-A3B (teacher) ───────────────────────
# Uncomment if not already downloaded:
# echo "[2/2] Downloading Qwen/Qwen3.5-35B-A3B (~70GB)..."
# huggingface-cli download Qwen/Qwen3.5-35B-A3B

echo ""
echo "=========================================="
echo "Download complete!"
echo "Models are in HF cache: ~/.cache/huggingface/hub/"
echo ""
echo "Verify with:"
echo "  ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/"
echo "=========================================="
