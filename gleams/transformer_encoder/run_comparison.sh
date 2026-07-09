#!/usr/bin/env bash
# One-command CNN-vs-transformer holdout comparison.
#
# Mints a fresh timestamped run folder under COMPARISONS_ROOT, then runs the CNN
# evaluator (gleams TF env), the transformer evaluator (base torch env), and the
# report into that same folder. Outputs: <run>/comparison.csv and comparison.png,
# each stamped with the CNN weights + transformer checkpoint used.
#
# Usage:
#   ./run_comparison.sh [TRANSFORMER_CKPT] [CNN_HDF5]
#
# Defaults: the checkpoint scoped in compare_transformer.py and the CNN weights
# in compare_cnn.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PY=/home/saartje/miniconda3/bin/python
GLEAMS_PY=/home/saartje/miniconda3/envs/gleams/bin/python

TF_CKPT="${1:-}"
CNN_HDF5="${2:-}"

cd "$HERE"

# Mint one run folder both evaluators write into.
RUN_DIR="$("$BASE_PY" -c 'import compare_common as c; print(c.new_run_dir())')"
echo "=== comparison run dir: $RUN_DIR ==="

echo "=== [1/3] CNN (gleams env) ==="
CNN_ARGS=(--run-dir "$RUN_DIR")
[ -n "$CNN_HDF5" ] && CNN_ARGS+=(--model "$CNN_HDF5")
TF_CPP_MIN_LOG_LEVEL=3 "$GLEAMS_PY" compare_cnn.py "${CNN_ARGS[@]}"

echo "=== [2/3] Transformer (base env) ==="
TF_ARGS=(--run-dir "$RUN_DIR")
[ -n "$TF_CKPT" ] && TF_ARGS+=(--checkpoint "$TF_CKPT")
"$BASE_PY" compare_transformer.py "${TF_ARGS[@]}"

echo "=== [3/3] Report ==="
"$BASE_PY" compare_report.py --run-dir "$RUN_DIR"

echo "=== done: $RUN_DIR ==="
