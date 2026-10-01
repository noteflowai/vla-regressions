"""One exec-created, supervised Linux worker per side; the parent loads no model."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from health_supervisor import supervise, resource_probe
from native_collection import digest_file, object_digest
from paired_collection import atomic_json
from serial_backend import SerialPairProducer


def side_summary(path):
    try:
        data = json.loads(path.read_text())
        return data if data.get("status") in ("side_completed", "side_error") else None
    except (OSError, ValueError):
        return None


def require_clean_side(receipt, summary, request, summary_path):
    execution = receipt["execution"]
    if (receipt["status"] != "finished" or execution["status"] != "completed"
            or receipt["request_sha256"] != object_digest(request)
            or receipt["worker_pid"] != execution["worker_pid"]
            or execution["clean_process_exit"] is not True
            or type(execution["worker_exit_code"]) is not int
            or execution["worker_exit_code"] != 0
            or execution["summary_sha256"] != digest_file(summary_path)
            or summary is None or summary.get("status") != "side_completed"
            or summary.get("request_sha256") != object_digest(request)
            or summary.get("identity") != request["identity"]
            or summary.get("side") != request["side"]):
        raise RuntimeError("A saved side requires matching evidence and clean owned-worker exit")


LIFECYCLE_FILES = ("side-process.json", "side-request.json", "side-summary.json",
                   "native-side-episode.json")


def verify_isolated_pair(record):
    for side in ("old", "new"):
        episode = record[side]
        path = Path(episode["raw_folder"])
        request = json.loads((path / "side-request.json").read_text())
        summary = side_summary(path / "side-summary.json")
        receipt = json.loads((path / "side-process.json").read_text())
        if (request["identity"] != record["identity"] or request["side"] != side
                or object_digest(request["context"]) != record["identity"]["context_sha256"]
                or request["raw_folder"] != str(path.parent)):
            raise ValueError("Saved side request belongs to another frozen pair")
        require_clean_side(receipt, summary, request, path / "side-summary.json")
        original_path = path / "native-side-episode.json"
        original = json.loads(original_path.read_text())
        restored = {**episode, "raw_evidence": {
            name: digest for name, digest in episode["raw_evidence"].items()
            if name not in LIFECYCLE_FILES}}
        if (summary["episode_sha256"] != digest_file(original_path) or original != restored
                or any(name not in episode["raw_evidence"] for name in LIFECYCLE_FILES)):
            raise ValueError("Saved native side differs from its adopted lifecycle evidence")


class IsolatedSideProducer(SerialPairProducer):
    def __init__(self, folder, context, runtime_root, wall_budget,
                 teardown_budget=120, terminate_grace=10):
        super().__init__(folder, context, None, wall_budget)
        self.runtime_root = Path(runtime_root).resolve()
        self.teardown_budget = teardown_budget
        self.terminate_grace = terminate_grace

    def command(self, request_path):
        return [sys.executable, "-u", str(Path(__file__).with_name("side_worker.py")),
                "--request", str(request_path)]

    def run_side(self, side, identity, batch_id, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Declared native physical budget exhausted")
        work = self.folder / "side-workers" / f"{batch_id}-{side}"
        work.mkdir(parents=True, exist_ok=False)
        admission = resource_probe()
        atomic_json(work / "admission.json", admission)
        if admission.get("allowed") is not True:
            raise RuntimeError("Owned side worker requires current resource headroom")
        request = {
            "context": deepcopy(self.context), "identity": deepcopy(identity), "side": side,
            "batch_id": batch_id, "raw_folder": str(self.folder),
            "runtime_root": str(self.runtime_root), "deadline_monotonic": deadline,
        }
        request_path, summary_path = work / "request.json", work / "summary.json"
        atomic_json(request_path, request)
        receipt_path = work / "receipt.json"
        receipt = {"status": "starting", "request_sha256": object_digest(request),
                   "parent_pid": os.getpid()}
        atomic_json(receipt_path, receipt)
        def started(pid):
            receipt.update(status="running", worker_pid=pid)
            atomic_json(receipt_path, receipt)
        self.event("side_process_starting", side=side, batch=batch_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Declared native physical budget exhausted before side exec")
        execution = supervise(
            self.command(request_path), summary_path, work / "worker.log",
            remaining, self.teardown_budget, terminate_grace=self.terminate_grace,
            on_started=started, summary_reader=side_summary)
        receipt.update(status="finished", execution=execution)
        atomic_json(receipt_path, receipt)
        summary = side_summary(summary_path)
        self.event("side_process_reaped", side=side, batch=batch_id,
                   execution=execution, resource_probe=resource_probe(),
                   parent_torch_imported="torch" in sys.modules)
        require_clean_side(receipt, summary, request, summary_path)
        path = self.folder / f"episode-{identity['state_index']}-{identity['repeat']}-{side}"
        episode = json.loads((path / "episode.json").read_text())
        attempt = json.loads((path / "attempt.json").read_text())
        if (summary["episode_sha256"] != digest_file(path / "episode.json")
                or attempt["status"] != "completed" or attempt["identity"] != identity
                or attempt["side"] != side
                or attempt["episode_sha256"] != summary["episode_sha256"]):
            raise ValueError("Side summary differs from its durable native episode")
        # Adopt lifecycle evidence only after the exact child has exited normally.
        atomic_json(path / "native-side-episode.json", episode)
        atomic_json(path / "side-request.json", request)
        atomic_json(path / "side-summary.json", summary)
        atomic_json(path / "side-process.json", receipt)
        for name in LIFECYCLE_FILES:
            episode["raw_evidence"][name] = digest_file(path / name)
        atomic_json(path / "episode.json", episode)
        atomic_json(path / "attempt.json", {**attempt,
                    "episode_sha256": digest_file(path / "episode.json")})
        return episode
