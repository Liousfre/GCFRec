#!/usr/bin/env bash
# Minimal GCFRec example. Run from the src directory.
set -euo pipefail

DATASET="${DATASET:-baby}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2025}"

python main.py \
  --dataset "${DATASET}" \
  --model gcfrec \
  --device "${DEVICE}" \
  --random_seed "${SEED}" \
  --gcf_fusion_type gate \
  --gcf_gate_mode scalar \
  --gate_init_bias 4.0
