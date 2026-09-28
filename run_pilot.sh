#!/usr/bin/env bash
# Pilot on LIBERO-10, 10 tasks x 10 states x 5 repeats per variant. X-VLA by default;
# POLICY, CONTROL and TAG (output name prefix) select another model, e.g.
#   POLICY=HuggingFaceVLA/smolvla_libero CONTROL=relative TAG=smolvla- ./run_pilot.sh chunk1
# Resumable: rerunning skips episodes already in the output.
set -euo pipefail
cd "$(dirname "$0")"
. /tmp/vlareg/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/vlareg/libero_cfg MUJOCO_GL=egl
POLICY=${POLICY:-lerobot/xvla-libero} CONTROL=${CONTROL:-absolute} TAG=${TAG:-}
for variant in "$@"; do
  python perstate_eval.py --policy "$POLICY" --control-mode "$CONTROL" --variant "$variant" \
    --suite libero_10 --tasks 0-9 --states 0-9 --repeats 5 --batch-size 5 --seed 1000 \
    --out "results/pilot/libero_10-$TAG$variant.jsonl" \
    2>&1 | grep --line-buffered -E "^task|quantized|Error|Traceback" | tee -a "results/pilot/libero_10-$TAG$variant.log"
done
