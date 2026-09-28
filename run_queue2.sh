#!/usr/bin/env bash
# Second round: stronger X-VLA updates, then SmolVLA's action-chunk length.
set -euo pipefail
cd "$(dirname "$0")"
./run_pilot.sh w3 steps2
POLICY=HuggingFaceVLA/smolvla_libero CONTROL=relative TAG=smolvla- ./run_pilot.sh chunk1 chunk10 chunk50
echo "queue2 done"
