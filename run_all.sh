#!/bin/bash
# Full GMsFEM-NO pipeline: offline bases -> three operators (in parallel) -> online diffusion & Richards, f = 1.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs
GPUS=(${GPUS:-0 1 2})

$PY scripts/make_dataset.py --workers "${WORKERS:-30}"

$PY scripts/train.py --kind full   --gpu "${GPUS[0]}" > logs/train_full.out   2>&1 &
$PY scripts/train.py --kind half   --gpu "${GPUS[1]}" > logs/train_half.out   2>&1 &
$PY scripts/train.py --kind corner --gpu "${GPUS[2]}" > logs/train_corner.out 2>&1 &
wait

$PY scripts/evaluate.py --gpu "${GPUS[0]}" --problem diffusion --f 1.0
$PY scripts/evaluate.py --gpu "${GPUS[0]}" --problem richards  --f 1.0
