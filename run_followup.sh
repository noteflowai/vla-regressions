#!/usr/bin/env bash
# Follow-up on single flagged states: more rollouts with fresh policy seeds, same scene seeds.
#   ./run_followup.sh TASK STATE VARIANT...   (writes results/followup/libero_10-t<TASK>s<STATE>-<variant>.jsonl)
set -euo pipefail
cd "$(dirname "$0")"
. /tmp/vlareg/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/vlareg/libero_cfg MUJOCO_GL=egl
task=$1 state=$2; shift 2
for variant in "$@"; do
  python perstate_eval.py --policy lerobot/xvla-libero --control-mode absolute --variant "$variant" \
    --suite libero_10 --tasks "$task" --states "$state" --repeats 20 --batch-size 5 --seed 3000 --scene-seed 1000 \
    --out "results/followup/libero_10-t${task}s${state}-$variant.jsonl" \
    2>&1 | grep --line-buffered -E "^task|Error|Traceback"
done
