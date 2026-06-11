#!/usr/bin/env bash
# =============================================================================
# run_safe.sh - auto-restart loop for running explain_vit_attention safely
#
# Purpose:
#   Even if the GPU crashes (CUDA error: unknown error / black screen) under
#   heavy settings such as 768px, this script automatically restarts the
#   process and completes the run from where it stopped, via --resume.
#
#   Once a CUDA context is corrupted it cannot be recovered within the same
#   process, so killing and restarting the process is the only reliable
#   recovery. This script does exactly that.
#
# Usage:
#   bash scripts/run_safe.sh
#   (Edit the variables below for your environment, or override via env vars)
#       RENDERS=... EMB=... IDS=... OUT=... bash scripts/run_safe.sh
# =============================================================================
set -u

# Move to the project root (one level up from this script) so relative paths stay stable
cd "$(dirname "$0")/.." || exit 1

# --- Run parameters (overridable via env vars; defaults match the original example) ---
RENDERS="${RENDERS:-../MultiView3D-DINOv2/data/renders}"
EMB="${EMB:-../MultiView3D-DINOv2/data/embeddings/embeddings.npy}"
IDS="${IDS:-../MultiView3D-DINOv2/data/embeddings/ids.txt}"
OUT="${OUT:-results/explain}"
IMAGE_SIZE="${IMAGE_SIZE:-768}"   # keep 768px
CROP_SIZE="${CROP_SIZE:-768}"
NUM_SHOW="${NUM_SHOW:-12}"
COOLDOWN="${COOLDOWN:-5}"         # cool-down seconds after each specimen (thermal / power relief)
VRAM_FRACTION="${VRAM_FRACTION:-0.9}"  # cap process VRAM to this fraction; exceeding it raises OOM instead of hanging
COOLDOWN_AFTER_CRASH="${COOLDOWN_AFTER_CRASH:-30}"  # seconds to let the GPU cool down after a crash
MAX_RETRIES="${MAX_RETRIES:-1000}"

# Reduce VRAM fragmentation (helps prevent hangs)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Enable only when debugging (slower, but reports the exact error location)
# export CUDA_LAUNCH_BLOCKING=1

attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "[run_safe] ===== attempt #${attempt} $(date '+%Y-%m-%d %H:%M:%S') ====="

  python3 -m src.explain_vit_attention \
    --renders "${RENDERS}" \
    --emb "${EMB}" \
    --ids "${IDS}" \
    --out "${OUT}" \
    --image-size "${IMAGE_SIZE}" \
    --crop-size "${CROP_SIZE}" \
    --num-show "${NUM_SHOW}" \
    --resume \
    --cooldown "${COOLDOWN}" \
    --vram-fraction "${VRAM_FRACTION}"

  code=$?
  if [ "${code}" -eq 0 ]; then
    echo "[run_safe] Finished successfully (all specimens processed)."
    break
  fi

  echo "[run_safe] Stopped with exit code ${code}. Cooling the GPU for ${COOLDOWN_AFTER_CRASH}s, then restarting (resuming via --resume)..."
  sleep "${COOLDOWN_AFTER_CRASH}"

  if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
    echo "[run_safe] Reached the maximum retry count ${MAX_RETRIES}; aborting." >&2
    exit 1
  fi
done
