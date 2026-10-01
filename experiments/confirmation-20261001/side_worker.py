"""Internal one-side worker. No standalone native retry or public fixture mode."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

from run_confirmation import install_runtime


def run_request(request_path, backend_factory=None):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    install_runtime(Path(request["runtime_root"]))
    from native_collection import digest_file, object_digest
    from paired_collection import atomic_json
    from serial_backend import SerialPairProducer, SerialPrecisionBackend
    if backend_factory is None:
        backend_factory = SerialPrecisionBackend
    receipt = json.loads((request_path.parent / "receipt.json").read_text())
    if (receipt["status"] not in ("starting", "running")
            or receipt["parent_pid"] != os.getppid()
            or receipt.get("worker_pid", os.getpid()) != os.getpid()
            or receipt["request_sha256"] != object_digest(request)
            or request["side"] not in ("old", "new")
            or not 0 < request["deadline_monotonic"] - time.monotonic() <= 43200):
        raise ValueError("Side worker requires a live, bounded supervising request")
    def interrupted(signum, frame):
        raise TimeoutError("Owned side worker interrupted")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    producer = SerialPairProducer(
        request["raw_folder"], request["context"], backend_factory(),
        min(43200, request["deadline_monotonic"] - time.monotonic()))
    identity = request["identity"]
    if (identity != producer.cache_identity.identity(identity["state_index"], identity["repeat"])
            or request["batch_id"] != object_digest([identity])):
        raise ValueError("Side identity differs from the frozen pair plan")
    summary_path = request_path.parent / "summary.json"
    common = {"request_sha256": object_digest(request), "identity": identity,
              "side": request["side"]}
    try:
        episode = producer.run_side(request["side"], identity, request["batch_id"],
                                    request["deadline_monotonic"])
        atomic_json(summary_path, {**common, "status": "side_completed",
                    "episode_sha256": digest_file(Path(episode["raw_folder"]) / "episode.json")})
    except BaseException as error:
        atomic_json(summary_path, {**common, "status": "side_error",
                    "error": type(error).__name__ + ": " + str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    run_request(parser.parse_args().request)
