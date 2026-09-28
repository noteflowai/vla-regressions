#!/usr/bin/env bash
# Pilot: X-VLA on LIBERO-10, 10 tasks x 10 states x 5 repeats per variant.
# Resumable: rerunning skips episodes already in the output.
set -euo pipefail
cd "$(dirname "$0")"
. /tmp/vlareg/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/vlareg/libero_cfg MUJOCO_GL=egl
for variant in "$@"; do
  python perstate_eval.py --variant "$variant" --suite libero_10 --tasks 0-9 --states 0-9 \
    --repeats 5 --batch-size 5 --seed 1000 --out "results/pilot/libero_10-$variant.jsonl" \
    2>&1 | grep --line-buffered -E "^task|quantized|Error|Traceback" | tee -a "results/pilot/libero_10-$variant.log"
done
