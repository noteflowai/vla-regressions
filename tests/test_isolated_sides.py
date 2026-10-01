"""Actual owned subprocess controls, without Torch, model loading or simulation."""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "experiments/confirmation-20261001"
sys.path.insert(0, str(HERE))
FIXTURE = Path(__file__).parent / "fixtures/isolated_side.py"


@unittest.skipUnless(os.environ.get("NATIVE_KERNEL_ROOT"), "Set NATIVE_KERNEL_ROOT for process checks")
class IsolatedSideTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Path(os.environ["NATIVE_KERNEL_ROOT"]).resolve()
        import run_confirmation
        run_confirmation.install_runtime(self.runtime)
        self.native = importlib.import_module("native_collection")
        self.isolated = importlib.import_module("isolated_sides")
        self.paired = importlib.import_module("paired_collection")
        from design import pair_plan
        digest = "a" * 64
        self.context = {
            "seed": 2026100111, "states": [{"task": 3, "init_state": 3, "scene_seed": 10007306}],
            "protocol": {"pair_plan": pair_plan("control")},
            "protocol_sha256": digest, "evaluator_sha256": digest,
            "old_pipeline_sha256": digest, "new_pipeline_sha256": digest,
            "evaluator": {"model_lifecycle": "exec_owned_worker_per_side"},
        }

    def producer(self, folder, mode="normal", wall_budget=60, **kwargs):
        class FixtureProducer(self.isolated.IsolatedSideProducer):
            def command(self, request_path):
                return [sys.executable, "-B", str(FIXTURE), "--request", str(request_path),
                        "--fixture-mode", mode]
        return FixtureProducer(folder, self.context, self.runtime, wall_budget=wall_budget, **kwargs)

    def test_one_fresh_exec_per_side_with_verified_retained_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(self.isolated, "resource_probe",
                             return_value={"test_fixture": True, "allowed": True}):
            root = Path(tmp)
            producer = self.producer(root / "raw")
            cache = self.native.NativePairCache(root / "pairs", self.context)
            self.assertEqual(cache.get_many([(0, 0)], producer.produce_many), [(True, True)])
            record = json.loads(cache.path(0, 0).read_text())
            self.isolated.verify_isolated_pair(record)
            events = [json.loads(line) for line in (root / "raw/physical-events.jsonl").read_text().splitlines()]
            reaped = [e for e in events if e["phase"] == "side_process_reaped"]
            self.assertEqual(len({e["execution"]["worker_pid"] for e in reaped}), 2)
            self.assertTrue(all(e["execution"]["clean_process_exit"] is True for e in reaped))
            self.assertTrue(all(e["parent_torch_imported"] is False for e in reaped))
            order = self.context["protocol"]["pair_plan"][0]["side_order"]
            self.assertEqual([e["side"] for e in reaped], order)
            first_reap = next(i for i, e in enumerate(events) if e["phase"] == "side_process_reaped")
            second_start = [i for i, e in enumerate(events) if e["phase"] == "side_process_starting"][1]
            self.assertLess(first_reap, second_start)
            # Replay validates evidence and does not create another worker.
            self.assertEqual(cache.get(0, 0, producer=None), (True, True))
            self.assertEqual(len(list((root / "raw/side-workers").iterdir())), 2)
            receipt_path = Path(record["old"]["raw_folder"]) / "side-process.json"
            receipt_path.write_text("{}")
            with self.assertRaises(ValueError):
                self.native.NativePairCache(root / "pairs", self.context).get(0, 0, producer=None)

    def test_saved_success_with_nonzero_exit_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(self.isolated, "resource_probe",
                             return_value={"test_fixture": True, "allowed": True}):
            root = Path(tmp)
            producer = self.producer(root / "raw", mode="nonzero")
            cache = self.native.NativePairCache(root / "pairs", self.context)
            with self.assertRaisesRegex(RuntimeError, "clean owned-worker exit"):
                cache.get_many([(0, 0)], producer.produce_many)
            receipt = json.loads(next((root / "raw/side-workers").glob("*/receipt.json")).read_text())
            self.assertEqual(receipt["execution"]["worker_exit_code"], 7)
            self.assertFalse(cache.path(0, 0).exists())
            self.assertEqual(len(list((root / "raw").glob("episode-*/episode.json"))), 1)
            self.assertEqual(len(list((root / "raw/side-workers").iterdir())), 1)

    def test_saved_success_cannot_override_teardown_timeout(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(self.isolated, "resource_probe",
                             return_value={"test_fixture": True, "allowed": True}):
            root = Path(tmp)
            producer = self.producer(root / "raw", mode="hang",
                                     teardown_budget=.5, terminate_grace=.2)
            cache = self.native.NativePairCache(root / "pairs", self.context)
            with self.assertRaisesRegex(RuntimeError, "clean owned-worker exit"):
                cache.get_many([(0, 0)], producer.produce_many)
            receipt = json.loads(next((root / "raw/side-workers").glob("*/receipt.json")).read_text())
            self.assertEqual(receipt["execution"]["timeout_phase"], "teardown")
            self.assertFalse(receipt["execution"]["clean_process_exit"])
            self.assertFalse(cache.path(0, 0).exists())

    def test_collection_timeout_without_summary_starts_no_second_side(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(self.isolated, "resource_probe",
                             return_value={"test_fixture": True, "allowed": True}):
            root = Path(tmp)
            producer = self.producer(root / "raw", mode="hang_before", wall_budget=.5,
                                     teardown_budget=.5, terminate_grace=.2)
            cache = self.native.NativePairCache(root / "pairs", self.context)
            with self.assertRaisesRegex(RuntimeError, "clean owned-worker exit"):
                cache.get_many([(0, 0)], producer.produce_many)
            receipt = json.loads(next((root / "raw/side-workers").glob("*/receipt.json")).read_text())
            self.assertEqual(receipt["execution"]["timeout_phase"], "collection_or_initialization")
            self.assertFalse(cache.path(0, 0).exists())
            self.assertEqual(len(list((root / "raw/side-workers").iterdir())), 1)

    def test_failed_resource_admission_creates_no_worker(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(self.isolated, "resource_probe", return_value={"allowed": False}):
            producer = self.producer(Path(tmp) / "raw")
            identity = producer.cache_identity.identity(0, 0)
            with self.assertRaisesRegex(RuntimeError, "current resource headroom"):
                producer.run_side("old", identity, self.native.object_digest([identity]),
                                  time.monotonic() + 20)
            work = next((producer.folder / "side-workers").iterdir())
            self.assertTrue((work / "admission.json").is_file())
            self.assertFalse((work / "receipt.json").exists())

    def test_expired_outer_deadline_starts_no_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = self.producer(Path(tmp) / "raw")
            identity = producer.cache_identity.identity(0, 0)
            with self.assertRaises(TimeoutError):
                producer.run_side("old", identity, self.native.object_digest([identity]),
                                  time.monotonic() - 1)
            self.assertFalse((producer.folder / "side-workers").exists())

    def test_boolean_exit_and_mismatched_request_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            request = {"identity": {"fixture": True}, "side": "old"}
            summary = {"status": "side_completed", "identity": request["identity"], "side": "old",
                       "request_sha256": self.native.object_digest(request)}
            self.paired.atomic_json(path, summary)
            receipt = {"status": "finished", "worker_pid": 1234,
                       "request_sha256": self.native.object_digest(request), "execution": {
                "worker_pid": 1234,
                "status": "completed", "clean_process_exit": True, "worker_exit_code": 0,
                "summary_sha256": self.native.digest_file(path)}}
            self.isolated.require_clean_side(receipt, summary, request, path)
            for value in (True, False, 1, None):
                broken = deepcopy(receipt)
                broken["execution"]["worker_exit_code"] = value
                with self.assertRaises(RuntimeError):
                    self.isolated.require_clean_side(broken, summary, request, path)
            with self.assertRaises(RuntimeError):
                self.isolated.require_clean_side(receipt, summary, {**request, "side": "new"}, path)

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open"), "Linux pidfd required")
    def test_hard_parent_death_terminates_only_its_owned_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holder_code = (
                "import sys,json;from pathlib import Path;"
                "sys.path.insert(0,sys.argv[1]);from health_supervisor import supervise;"
                "root=Path(sys.argv[2]);"
                "supervise([sys.executable,'-c','import time;time.sleep(30)'],"
                "root/'summary.json',root/'worker.log',20,1,"
                "on_started=lambda pid:(root/'pid.json').write_text(json.dumps(pid)))")
            holder = subprocess.Popen([sys.executable, "-c", holder_code,
                                       str(self.runtime / "experiments"), str(root)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pidfd = None
            try:
                deadline = time.monotonic() + 10
                while not (root / "pid.json").exists() and time.monotonic() < deadline:
                    if holder.poll() is not None:
                        self.fail("Owned supervisor exited before starting its fixture")
                    time.sleep(.02)
                child_pid = json.loads((root / "pid.json").read_text())
                pidfd = os.pidfd_open(child_pid)
                holder.kill()
                holder.wait(timeout=5)
                self.assertTrue(select.select([pidfd], [], [], 5)[0],
                                "Owned worker outlived its hard-killed parent")
            finally:
                if holder.poll() is None:
                    holder.kill()
                    holder.wait(timeout=5)
                if pidfd is not None:
                    if not select.select([pidfd], [], [], 0)[0]:
                        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    os.close(pidfd)


if __name__ == "__main__":
    unittest.main()
