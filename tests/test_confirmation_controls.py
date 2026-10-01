"""Exercise precision and retained-evidence controls without model or GPU imports."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "experiments/confirmation-20261001"
sys.path.insert(0, str(HERE))
precision = importlib.import_module("precision")
runner = importlib.import_module("run_confirmation")


class ConfirmationControlsTests(unittest.TestCase):
    def test_precision_is_configured_before_load_and_preserves_cadence(self):
        cfg = SimpleNamespace(type="xvla", dtype="float32", num_denoising_steps=10,
                              chunk_size=30, n_action_steps=30, n_obs_steps=1, compile_model=True)
        pipeline = {"family": "xvla", "update": "bf16", "precision": "bfloat16"}
        changed = precision.configure_precision(cfg, pipeline)
        self.assertEqual(changed.dtype, "bfloat16")
        self.assertFalse(changed.compile_model)
        self.assertEqual(cfg.dtype, "float32")
        for field in ("num_denoising_steps", "chunk_size", "n_action_steps", "n_obs_steps"):
            bad = SimpleNamespace(**vars(cfg))
            setattr(bad, field, 2)
            with self.assertRaises(ValueError):
                precision.configure_precision(bad, pipeline)
        with self.assertRaises(ValueError):
            precision.configure_precision(cfg, {**pipeline, "precision": "float32"})

    def test_actual_dtype_mismatch_is_rejected(self):
        class Parameter:
            def __init__(self, dtype):
                self.dtype = "torch." + dtype
            def is_floating_point(self):
                return True
            def numel(self):
                return 7
        policy = SimpleNamespace(parameters=lambda: [Parameter("bfloat16")])
        self.assertEqual(precision.verify_precision(policy, "bfloat16"), {"bfloat16": 7})
        policy.parameters = lambda: [Parameter("bfloat16"), Parameter("float32")]
        with self.assertRaises(ValueError):
            precision.verify_precision(policy, "bfloat16")

    def test_budget_cannot_be_replenished_or_accept_invalid_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.json"
            with self.assertRaises(ValueError):
                runner.read_budget(path)
            for charge in (-1, float("nan"), True, 43201):
                path.write_text(json.dumps({"limit_seconds": 43200, "reservations": {
                    "previous": {"status": "finalized", "charged_seconds": charge}}}))
                with self.assertRaises(ValueError):
                    runner.read_budget(path)
            path.write_text(json.dumps({"limit_seconds": 43200, "reservations": {}}))
            with self.assertRaises(ValueError):
                runner.read_budget(path)
            path.write_text(json.dumps({"limit_seconds": 43200, "reservations": {
                "failed": {"status": "finalized", "charged_seconds": 3995}}}))
            self.assertEqual(runner.read_budget(path), 39205)

    def test_matching_success_is_not_sufficient_for_reload_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = [Path(tmp) / side for side in ("old", "new")]
            record = {}
            for side, folder in zip(("old", "new"), folders):
                folder.mkdir()
                record[side] = dict(success=True, steps=1, terminated=True, truncated=False,
                                    environment_terminated=True, environment_truncated=False,
                                    collector_truncated=False, raw_folder=str(folder))
                (folder / "transitions.jsonl").write_text('{"step":0,"success":true}\n')
                (folder / "action-000.json").write_text(json.dumps({"action": [[.1] * 7]}))
                for name in ("input-000.npz", "terminal.npz"):
                    np.savez_compressed(folder / name, camera=np.zeros((2, 4, 4, 3)),
                                        simulator_state=np.ones(7))
            runner.compare_control_pair(record)
            np.savez_compressed(folders[1] / "input-000.npz",
                                camera=np.ones((2, 4, 4, 3)), simulator_state=np.ones(7))
            with self.assertRaisesRegex(ValueError, "observations"):
                runner.compare_control_pair(record)
            np.savez_compressed(folders[1] / "input-000.npz",
                                camera=np.zeros((2, 4, 4, 3)), simulator_state=np.ones(7))
            (folders[1] / "action-000.json").write_text(json.dumps({"action": [[.2] * 7]}))
            with self.assertRaisesRegex(ValueError, "actions"):
                runner.compare_control_pair(record)

    def test_camera_rounding_is_bounded_and_excluded_from_prediction_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = [Path(tmp) / side for side in ("old", "new")]
            record = {}
            camera = np.zeros((1, 360, 360, 3), dtype=np.uint8)
            for side, folder in zip(("old", "new"), folders):
                folder.mkdir()
                record[side] = dict(success=True, steps=2, terminated=True, truncated=False,
                                    environment_terminated=True, environment_truncated=False,
                                    collector_truncated=False, raw_folder=str(folder))
                (folder / "transitions.jsonl").write_text('{"step":0}\n{"step":1}\n')
                for step in range(2):
                    (folder / f"action-{step:03d}.json").write_text(
                        json.dumps({"action": [[.1] * 7]}))
                for name in ("input-000.npz", "input-001.npz", "terminal.npz"):
                    np.savez_compressed(folder / name, **{
                        "observation/pixels/image2": camera, "simulator_state": np.ones(7)})
            # A one-level difference between predictions is retained as a diagnostic.
            changed = camera.copy()
            changed[0, 0, 0, 0] = 1
            np.savez_compressed(folders[1] / "input-001.npz", **{
                "observation/pixels/image2": changed, "simulator_state": np.ones(7)})
            diagnostics = runner.compare_control_pair(record)
            self.assertEqual(diagnostics[0]["different_channels"], 1)
            # The identical perturbation at a prediction or terminal frame is forbidden.
            for name in ("input-000.npz", "terminal.npz"):
                np.savez_compressed(folders[1] / name, **{
                    "observation/pixels/image2": changed, "simulator_state": np.ones(7)})
                with self.assertRaisesRegex(ValueError, "observations"):
                    runner.compare_control_pair(record)
                np.savez_compressed(folders[1] / name, **{
                    "observation/pixels/image2": camera, "simulator_state": np.ones(7)})
            # Off-cadence magnitude and changed-channel fraction both remain bounded.
            for bad in (np.full_like(camera, 1), np.full_like(camera, 2)):
                np.savez_compressed(folders[1] / "input-001.npz", **{
                    "observation/pixels/image2": bad, "simulator_state": np.ones(7)})
                with self.assertRaisesRegex(ValueError, "rounding bound"):
                    runner.compare_control_pair(record)
            np.savez_compressed(folders[1] / "input-001.npz", **{
                "observation/pixels/image2": camera, "simulator_state": np.ones(7) + 1e-12})
            with self.assertRaisesRegex(ValueError, "simulator states"):
                runner.compare_control_pair(record)


@unittest.skipUnless(os.environ.get("NATIVE_KERNEL_ROOT"), "Set NATIVE_KERNEL_ROOT for adapter checks")
class SerialProducerTests(unittest.TestCase):
    def setUp(self):
        runner.install_runtime(Path(os.environ["NATIVE_KERNEL_ROOT"]))
        self.native = importlib.import_module("native_collection")
        self.adapter = importlib.import_module("serial_backend")
        self.paired = importlib.import_module("paired_collection")

    def test_independent_reload_follows_frozen_order_and_preserves_attempts(self):
        from design import pair_plan
        digest = "a" * 64
        context = dict(seed=2026100110, states=[{"task": 3, "init_state": 3,
                                               "scene_seed": 10007306}],
                       protocol={"pair_plan": pair_plan("primary")},
                       protocol_sha256=digest, evaluator_sha256=digest,
                       old_pipeline_sha256=digest, new_pipeline_sha256=digest)
        calls = []
        class Backend:
            def load(self, side, context, folder, batch_id):
                calls.append(("load", side))
                return {}
            def run_episode(self, side, identity, folder, deadline):
                calls.append(("episode", side))
                (folder / "raw.json").write_text('{"test_fixture":true}\n')
                return {**{key: identity[key] for key in
                           ("task", "init_state", "scene_seed", "policy_seed")},
                        "success": True, "infrastructure_error": None, "steps": 1,
                        "reset_state_exact": True, "reset_inputs_exact": True,
                        "initial_state_sha256": digest, "initial_input_sha256": digest,
                        "pipeline_sha256": digest, "evaluator_sha256": digest}
            def close(self):
                calls.append(("close", None))
        with tempfile.TemporaryDirectory() as tmp:
            producer = self.adapter.SerialPairProducer(Path(tmp) / "raw", context, Backend(), 60)
            cache = self.native.NativePairCache(Path(tmp) / "pairs", context)
            self.assertEqual(cache.get_many([(0, 0)], producer.produce_many), [(True, True)])
            order = context["protocol"]["pair_plan"][0]["side_order"]
            self.assertEqual(calls, [(kind, side if kind != "close" else None)
                                     for side in order for kind in ("load", "episode", "close")])
            cache.get_many([(0, 0)], producer.produce_many)
            self.assertEqual(len(calls), 6)  # Retained pair is verified, never recollected.
            identity = cache.identity(0, 0)
            with self.assertRaises(RuntimeError):
                producer.produce_many([identity])
            with self.assertRaises(ValueError):
                producer.produce_many([cache.identity(0, 1), cache.identity(0, 2)])
            # A changed raw file cannot remain a valid cached native pair.
            episode = Path(tmp) / "raw/episode-0-0-old/raw.json"
            episode.write_text('{"tampered":true}\n')
            with self.assertRaises(ValueError):
                cache.get_many([(0, 0)], producer.produce_many)

    def test_loader_failure_releases_policy_and_blocks_blind_retry(self):
        from design import pair_plan
        digest = "b" * 64
        context = dict(seed=2026100110, states=[{"task": 3, "init_state": 3,
                                               "scene_seed": 10007306}],
                       protocol={"pair_plan": pair_plan("primary")},
                       protocol_sha256=digest, evaluator_sha256=digest,
                       old_pipeline_sha256=digest, new_pipeline_sha256=digest)
        closed = []
        class FailingBackend:
            def load(self, *args):
                raise RuntimeError("fixture construction failure")
            def close(self):
                closed.append(True)
        with tempfile.TemporaryDirectory() as tmp:
            producer = self.adapter.SerialPairProducer(Path(tmp) / "raw", context, FailingBackend(), 60)
            cache = self.native.NativePairCache(Path(tmp) / "pairs", context)
            with self.assertRaisesRegex(RuntimeError, "construction failure"):
                cache.get_many([(0, 0)], producer.produce_many)
            self.assertEqual(closed, [True])
            with self.assertRaisesRegex(RuntimeError, "Unresolved"):
                cache.get_many([(0, 0)], producer.produce_many)
            self.assertEqual(closed, [True])

    def test_control_verification_requires_raw_evidence_and_clean_integer_exit(self):
        from design import CONTROL_SEED, pair_plan
        digest = "c" * 64
        pipeline = {"family": "xvla", "precision": "float32", "update": "baseline_reload"}
        context = dict(seed=CONTROL_SEED, states=[{"task": 3, "init_state": 3,
                                                 "scene_seed": 10007306}],
                       protocol={"mode": "control", "pair_plan": pair_plan("control")},
                       evaluator={"sources": {}}, old_pipeline=pipeline, new_pipeline=pipeline,
                       protocol_sha256=digest, evaluator_sha256=digest,
                       old_pipeline_sha256=digest, new_pipeline_sha256=digest)
        primary = {**context, "new_pipeline": {**pipeline, "precision": "bfloat16", "update": "bf16"}}
        class Backend:
            def load(self, *args):
                return {}
            def run_episode(self, side, identity, folder, deadline):
                (folder / "transitions.jsonl").write_text('{"step":0,"success":true}\n')
                (folder / "action-000.json").write_text(json.dumps({"action": [[.1] * 7]}))
                for name in ("input-000.npz", "terminal.npz"):
                    np.savez_compressed(folder / name, camera=np.zeros((2, 4, 4, 3)),
                                        simulator_state=np.ones(7))
                return {**{key: identity[key] for key in
                           ("task", "init_state", "scene_seed", "policy_seed")},
                        "success": True, "infrastructure_error": None, "steps": 1,
                        "terminated": True, "truncated": False, "environment_terminated": True,
                        "environment_truncated": False, "collector_truncated": False,
                        "reset_state_exact": True, "reset_inputs_exact": True,
                        "initial_state_sha256": digest, "initial_input_sha256": digest,
                        "pipeline_sha256": digest, "evaluator_sha256": digest}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.paired.atomic_json(folder / "context.json", context)
            producer = self.adapter.SerialPairProducer(folder / "raw", context, Backend(), 60)
            cache = self.native.NativePairCache(folder / "pairs", context)
            for repeat in range(2):
                cache.get_many([(0, repeat)], producer.produce_many)
            launch = folder / "launches/fixture"
            launch.mkdir(parents=True)
            summary_path = launch / "summary.json"
            self.paired.atomic_json(summary_path, {
                "status": "control_completed", "control_prediction_and_physics_exact_match": True,
                "off_cadence_camera_diagnostics": [[], []],
                "completed_pairs": 2, "pair_elapsed_seconds": [2, 3],
                "context_sha256": self.native.object_digest(context)})
            receipt = {
                "status": "finished", "context_sha256": self.native.object_digest(context),
                "execution": {"status": "completed", "clean_process_exit": True,
                              "worker_exit_code": 0,
                              "summary_sha256": self.native.digest_file(summary_path)}}
            receipt_path = launch / "receipt.json"
            self.paired.atomic_json(receipt_path, receipt)
            verified = runner.verified_control(folder, primary)
            self.assertEqual(verified["estimated_primary_wall_seconds"], 361)
            for bad_exit in (True, 1, None):
                broken = json.loads(json.dumps(receipt))
                broken["execution"]["worker_exit_code"] = bad_exit
                self.paired.atomic_json(receipt_path, broken)
                with self.assertRaisesRegex(ValueError, "lifecycle"):
                    runner.verified_control(folder, primary)
            self.paired.atomic_json(receipt_path, receipt)
            summary_path.write_text(summary_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "lifecycle"):
                runner.verified_control(folder, primary)


if __name__ == "__main__":
    unittest.main()
