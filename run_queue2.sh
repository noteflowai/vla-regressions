#!/usr/bin/env bash
# Second round on X-VLA: scene noise against policy noise, then stronger updates.
set -euo pipefail
cd "$(dirname "$0")"
# Same policy seeds as the FP32 run, new scene seeds: how much of the noise is the scene.
SCENE_SEED=2000 TAG=-scene2000 ./run_pilot.sh fp32
./run_pilot.sh w3 steps2 bf16
echo "queue2 done $(date -Is)"
