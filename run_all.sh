#!/usr/bin/env bash
# =============================================================================
# End-to-end pipeline for the H100: generate (PNG) -> train -> evaluate.
#
# Usage:
#   ./run_all.sh                 # full run
#   ./run_all.sh --smoke         # tiny end-to-end sanity run (few images/epochs)
#   DEVICE=cuda N_WORKERS=16 ./run_all.sh
#   FORCE_GENERATE=1 ./run_all.sh   # regenerate the dataset even if it exists
#
# Env vars:
#   DEVICE          torch device (default: cuda)
#   N_WORKERS       dataloader workers (default: 8)
#   FORCE_GENERATE  set to 1 to regenerate the dataset even if dataset/ exists
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda}"
N_WORKERS="${N_WORKERS:-8}"
FORCE_GENERATE="${FORCE_GENERATE:-0}"

SMOKE=""
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE="--smoke" ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

echo "=== device=$DEVICE  workers=$N_WORKERS  smoke=${SMOKE:-no} ==="

# --- 1. Generate the synthetic dataset (uint8 PNG) ---------------------------
if [[ -f dataset/manifest.csv && "$FORCE_GENERATE" != "1" ]]; then
  echo "[generate] dataset/manifest.csv exists -> skipping (FORCE_GENERATE=1 to override)"
else
  echo "[generate] building synthetic dataset (PNG) ..."
  uv run python src/generate_dataset.py --config configs/dataset.yaml
fi

# --- 2. Train ----------------------------------------------------------------
echo "[train] starting ..."
uv run python src/train.py --config configs/train.yaml \
  --device "$DEVICE" --num-workers "$N_WORKERS" $SMOKE

# --- 3. Evaluate -------------------------------------------------------------
echo "[evaluate] scoring val set ..."
uv run python src/evaluate.py --config configs/eval.yaml --device "$DEVICE"

echo "=== done ==="
