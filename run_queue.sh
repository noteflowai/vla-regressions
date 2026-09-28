#!/usr/bin/env bash
# Everything queued for the pilot, detached from the terminal: ./run_queue.sh >> results/pilot/queue.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
./run_pilot.sh fp32 w4
./run_noise_floor.sh
echo "queue done $(date -Is)"
