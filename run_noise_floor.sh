#!/usr/bin/env bash
# Noise floor: the fp32 policy again on the same states (scene seed 1000), new policy noise.
set -euo pipefail
cd "$(dirname "$0")"
. /tmp/vlareg/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/vlareg/libero_cfg MUJOCO_GL=egl
python perstate_eval.py --variant fp32 --suite libero_10 --tasks 0-9 --states 0-9 \
  --repeats 5 --batch-size 5 --seed 2000 --scene-seed 1000 --out results/pilot/libero_10-fp32-seed2000.jsonl \
  2>&1 | grep --line-buffered -E "^task|Error|Traceback" | tee -a results/pilot/libero_10-fp32-seed2000.log
