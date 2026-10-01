"""Synthetic CPU-only process fixture; never exposed by the native CLI."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/confirmation-20261001"))
from side_worker import run_request


class FixtureBackend:
    def load(self, side, context, folder, batch_id):
        return {"fixture": True}

    def run_episode(self, side, identity, folder, deadline):
        import numpy as np
        from paired_collection import atomic_json
        atomic_json(folder / "raw.json", {"test_fixture": True, "no_policy_or_simulator": True})
        atomic_json(folder / "action-000.json", {"action": [[.1] * 7]})
        (folder / "transitions.jsonl").write_text('{"step":0,"success":true}\n')
        for name in ("input-000.npz", "terminal.npz"):
            np.savez_compressed(folder / name, camera=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                                simulator_state=np.ones(7))
        digest = "a" * 64
        return {**{key: identity[key] for key in ("task", "init_state", "scene_seed", "policy_seed")},
                "success": True, "infrastructure_error": None, "steps": 1,
                "terminated": True, "truncated": False,
                "environment_terminated": True, "environment_truncated": False,
                "collector_truncated": False, "reset_state_exact": True, "reset_inputs_exact": True,
                "initial_state_sha256": digest, "initial_input_sha256": digest,
                "pipeline_sha256": identity[side + "_pipeline_sha256"],
                "evaluator_sha256": identity["evaluator_sha256"], "elapsed_seconds": .01}

    def close(self):
        self.cpu_release_report = {"test_fixture": True}


parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--fixture-mode", choices=("normal", "nonzero", "hang", "hang_before"),
                    default="normal")
args = parser.parse_args()
if args.fixture_mode == "hang_before":
    time.sleep(30)
run_request(args.request, backend_factory=FixtureBackend)
if args.fixture_mode == "nonzero":
    sys.exit(7)
if args.fixture_mode == "hang":
    time.sleep(30)
