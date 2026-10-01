"""Prepare or supervise the frozen first-paper serial confirmation protocol."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from design import (ALPHA, CONTROL_PAIRS, CONTROL_SEED, PRIMARY_PAIRS, PRIMARY_SEED,
                    STATE, final_analysis, pair_plan)

HERE = Path(__file__).resolve().parent
CORE_HASHES = {
    "native_rollout.py": "8395afb993f8c553e7a68f33cd3ccd283cd556b319de86c52f1bbf5de4fcddd2",
    "native_collection.py": "d078849bde271965220ac06d051f5842d8fabac13ec911e936880dc0f0d5e7e2",
    "health_supervisor.py": "ce13850adadbe7d31b2c6ee3937b82607944986d4d368dc2b8ff3db13617dabd",
    "paired_collection.py": "07778354ab090262291b1b8cdbf3ddda5855e04cdc48941cd3c78b9e22ca44c2",
    "run_native_collection.py": "4cf1cc4733de7b1c772a2527df3029c8509e9d955b7ff4a33e3ea674b2c3ac49",
}


def install_runtime(root):
    from hashlib import sha256
    for name, expected in CORE_HASHES.items():
        if sha256((root / "experiments" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned native kernel changed: " + name)
    sys.path.insert(0, str(root / "experiments"))
    for key, value in {
        "MUJOCO_GL": "egl", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
        "LIBERO_CONFIG_PATH": "/tmp/vlareg/libero_cfg",
    }.items():
        os.environ.setdefault(key, value)


def freeze(args):
    from native_collection import freeze_context, digest_file, object_digest
    # Read-only verification of health/reset witnesses; does not load a policy.
    base = freeze_context("xvla", "baseline_reload", args.health_receipt)
    suite_path = args.runtime_root / "protocols/closedloop-suite-001/manifest.json"
    suite = json.loads(suite_path.read_text())
    inventory = next(item for item in suite["inventory"] if item["task"] == STATE["task"])
    evaluator = deepcopy(base["evaluator"])
    receipt = json.loads(args.health_receipt.read_text())
    health_manifest = Path(receipt["output"]) / "manifest.json"
    audit_manifest = Path(evaluator["reset_audit"]["summary_path"]).parent / "manifest.json"
    for manifest_path in (health_manifest, audit_manifest):
        for name, expected in json.loads(manifest_path.read_text())["sources"].items():
            if digest_file(name) != expected:
                raise ValueError("Health/reset witness source changed: " + name)
            evaluator["sources"][name] = expected
    for path in sorted(HERE.glob("*.py")) + [HERE / "PROTOCOL.md"]:
        evaluator["sources"][str(path)] = digest_file(path)
    # The upstream rollout imports this original first-paper environment constructor.
    constructor = Path("/home/dcvuser/work/vla-regressions/perstate_eval.py")
    evaluator["sources"][str(constructor)] = digest_file(constructor)
    evaluator["sources"][str(constructor.parent / "quant.py")] = digest_file(
        constructor.parent / "quant.py")
    for path_key, hash_key in (("init_file", "init_file_sha256"), ("bddl_file", "bddl_sha256")):
        evaluator["sources"][inventory[path_key]] = inventory[hash_key]
    evaluator.update(environment_lanes=1, reload_every_pair=True,
                     cudnn_benchmark=False, cudnn_deterministic=True,
                     phase_profiling=args.profile_episodes)
    pipeline = {**base["old_pipeline"], "precision": "float32", "n_obs_steps": 1}
    new = {**pipeline, "update": "bf16", "precision": "bfloat16"} if args.mode == "primary" else pipeline
    protocol = {
        "id": "selected-xvla-bf16-serial-confirmation-20261001-v3",
        "mode": args.mode, "alpha": ALPHA, "primary_pairs": PRIMARY_PAIRS,
        "required_pairs": PRIMARY_PAIRS if args.mode == "primary" else CONTROL_PAIRS,
        "state": {**STATE, "task_name": inventory["name"]},
        "health": base["protocol"]["health"],
        "reset_audit": evaluator["reset_audit"],
        "pair_plan": pair_plan(args.mode),
        "document_sha256": digest_file(HERE / "PROTOCOL.md"),
        "design_sha256": digest_file(HERE / "design.py"),
        "shared_physical_budget_seconds": 43200,
        "scope": "New serial evaluator for one historically selected state; no historical "
                 "bit-exact replay, suite prevalence or second-paper efficacy claim.",
    }
    return {
        "family": "xvla", "update": "bf16" if args.mode == "primary" else "baseline_reload",
        "seed": PRIMARY_SEED if args.mode == "primary" else CONTROL_SEED,
        "states": [protocol["state"]], "protocol": protocol, "evaluator": evaluator,
        "old_pipeline": pipeline, "new_pipeline": new,
        "protocol_sha256": object_digest(protocol), "evaluator_sha256": object_digest(evaluator),
        "old_pipeline_sha256": object_digest(pipeline), "new_pipeline_sha256": object_digest(new),
    }


def read_summary(path):
    try:
        result = json.loads(path.read_text())
        if result.get("status") in ("control_completed", "confirmation_completed", "native_error"):
            return result
    except (OSError, ValueError):
        pass
    return None


def read_budget(path):
    if not path.is_file():
        raise ValueError("The existing shared physical ledger is required; never initialize one")
    state = json.loads(path.read_text())
    if state["limit_seconds"] != 43200 or not state["reservations"]:
        raise ValueError("Expected the existing consumed shared 12-hour budget")
    for item in state["reservations"].values():
        if (item["status"] not in ("reserved", "finalized")
                or type(item["charged_seconds"]) not in (int, float)
                or not math.isfinite(item["charged_seconds"])
                or not 0 <= item["charged_seconds"] <= 43200):
            raise ValueError("Invalid shared budget entry")
    remaining = 43200 - sum(item["charged_seconds"] for item in state["reservations"].values())
    if remaining < 0:
        raise ValueError("Overspent physical ledger")
    return remaining


def compare_control_pair(record):
    """Exact physics/actions/prediction inputs; bound and report off-cadence RGB rounding."""
    import numpy as np
    old, new = record["old"], record["new"]
    if any(old[key] != new[key] for key in (
            "success", "steps", "terminated", "truncated", "environment_terminated",
            "environment_truncated", "collector_truncated")):
        raise ValueError("Independent FP32 reload outcomes differ")
    folders = [Path(side["raw_folder"]) for side in (old, new)]
    if (folders[0] / "transitions.jsonl").read_bytes() != (folders[1] / "transitions.jsonl").read_bytes():
        raise ValueError("Independent FP32 reload transitions differ")
    for step in range(old["steps"]):
        actions = [json.loads((folder / f"action-{step:03d}.json").read_text())["action"]
                   for folder in folders]
        if actions[0] != actions[1]:
            raise ValueError("Independent FP32 reload actions differ")
    camera_diagnostics = []
    for name in [f"input-{step:03d}.npz" for step in range(old["steps"])] + ["terminal.npz"]:
        prediction_input = name == "terminal.npz" or int(name[6:9]) % 30 == 0
        with np.load(folders[0] / name, allow_pickle=False) as a, np.load(
                folders[1] / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError("Independent FP32 reload observation fields differ")
            for key in a.files:
                x, y = a[key], b[key]
                if x.shape != y.shape or x.dtype != y.dtype:
                    raise ValueError("Independent FP32 reload observation schema differs")
                if np.array_equal(x, y):
                    continue
                if (prediction_input or not key.startswith("observation/pixels/")
                        or x.dtype != np.uint8 or x.shape != (1, 360, 360, 3)):
                    raise ValueError("Independent FP32 reload observations or simulator states differ")
                changed = int(np.count_nonzero(x != y))
                max_difference = int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max())
                if max_difference > 1 or changed / x.size > 1e-4:
                    raise ValueError("Off-cadence camera variation exceeds declared rounding bound")
                camera_diagnostics.append({"file": name, "field": key,
                                           "different_channels": changed,
                                           "total_channels": int(x.size),
                                           "maximum_absolute_difference": max_difference})
    return camera_diagnostics


def verified_control(folder, primary_context):
    from native_collection import NativePairCache, digest_file, object_digest
    context = json.loads((folder / "context.json").read_text())
    if (context["evaluator"] != primary_context["evaluator"]
            or context["old_pipeline"] != primary_context["old_pipeline"]
            or context["new_pipeline"] != context["old_pipeline"]
            or context["protocol"]["mode"] != "control"
            or context["seed"] != CONTROL_SEED
            or context["protocol"]["pair_plan"] != pair_plan("control")
            or context["states"] != primary_context["states"]):
        raise ValueError("Control pipeline, seeds or frozen evaluator differs")
    launches = sorted((folder / "launches").glob("*"))
    if len(launches) != 1:
        raise ValueError("Exactly one clean control launch is required")
    summary_path = launches[0] / "summary.json"
    receipt = json.loads((launches[0] / "receipt.json").read_text())
    execution = receipt["execution"]
    if (receipt.get("status") != "finished"
            or receipt.get("context_sha256") != object_digest(context)
            or execution.get("status") != "completed"
            or execution.get("clean_process_exit") is not True
            or type(execution.get("worker_exit_code")) is not int
            or execution["worker_exit_code"] != 0
            or execution.get("summary_sha256") != digest_file(summary_path)):
        raise ValueError("Control requires unchanged summary and clean process lifecycle")
    result = read_summary(summary_path)
    if (result is None or result["context_sha256"] != object_digest(context)
            or result["status"] != "control_completed"
            or result["control_prediction_and_physics_exact_match"] is not True
            or result["completed_pairs"] != CONTROL_PAIRS
            or len(result["pair_elapsed_seconds"]) != CONTROL_PAIRS):
        raise ValueError("Complete exact-match engineering control is required")
    cache = NativePairCache(folder / "pairs", context)
    observed_diagnostics = []
    for repeat in range(CONTROL_PAIRS):
        if not cache.path(0, repeat).is_file():
            raise ValueError("Missing control pair")
        cache.get(0, repeat, producer=None)
        observed_diagnostics.append(compare_control_pair(json.loads(cache.path(0, repeat).read_text())))
    if result.get("off_cadence_camera_diagnostics") != observed_diagnostics:
        raise ValueError("Control rendering diagnostics differ from retained raw evidence")
    durations = result["pair_elapsed_seconds"]
    if any(type(t) not in (float, int) or not math.isfinite(t) or not 0 < t <= 43200
           for t in durations):
        raise ValueError("Invalid engineering control timing")
    return {"context_sha256": object_digest(context), "summary_sha256": digest_file(summary_path),
            "estimated_primary_wall_seconds": int(max(durations) * PRIMARY_PAIRS * 1.5 + 1)}


def worker(args, context):
    from native_collection import NativePairCache, object_digest
    from paired_collection import atomic_json
    from serial_backend import SerialPairProducer, SerialPrecisionBackend
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    def interrupted(signum, frame):
        raise TimeoutError("Native confirmation worker interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    producer = None
    started = time.monotonic()
    values, durations, camera_diagnostics = [], [], []
    try:
        producer = SerialPairProducer(args.output / "raw", context, SerialPrecisionBackend(),
                                      wall_budget=args.wall_budget)
        cache = NativePairCache(args.output / "pairs", context)
        for planned in context["protocol"]["pair_plan"]:
            if shutil.disk_usage(args.output).free < 2 * 1024 ** 3:
                raise RuntimeError("Native disk reserve exhausted")
            before = time.monotonic()
            repeat = planned["repeat"]
            if cache.identity(0, repeat)["policy_seed"] != planned["policy_seed"]:
                raise ValueError("Actual policy seed differs from published pair plan")
            pair = cache.get_many([(0, repeat)], producer.produce_many)[0]
            durations.append(time.monotonic() - before)
            if args.mode == "control":
                camera_diagnostics.append(compare_control_pair(
                    json.loads(cache.path(0, repeat).read_text())))
            values.append(pair)
        result = {
            "status": "control_completed" if args.mode == "control" else "confirmation_completed",
            "completed_pairs": len(values), "pair_elapsed_seconds": durations,
            "control_prediction_and_physics_exact_match": True if args.mode == "control" else None,
            "off_cadence_camera_diagnostics": camera_diagnostics if args.mode == "control" else None,
            "analysis": final_analysis(values) if args.mode == "primary" else None,
            "context_sha256": object_digest(context),
            "worker_elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.launch / "summary.json", result)
    except BaseException as error:
        atomic_json(args.launch / "summary.json", {
            "status": "native_error", "completed_pairs": len(values),
            "context_sha256": object_digest(context),
            "error": type(error).__name__ + ": " + str(error),
            "worker_elapsed_seconds": time.monotonic() - started, "analysis": None})
        raise
    finally:
        if producer is not None:
            producer.backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--health-receipt", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "primary"), required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--wall-budget", type=int, default=2400)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--profile-episodes", action="store_true",
                        help="Freeze opt-in inclusive phase timings; requires a new control context")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--launch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("runtime_root", "health_receipt", "budget_ledger", "output", "control", "launch"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    install_runtime(args.runtime_root)
    from native_collection import digest_file, object_digest
    from paired_collection import atomic_json
    from run_native_collection import PhysicalBudget
    from health_supervisor import resource_probe, supervise
    if args.budget_ledger != (args.runtime_root / "runs/native-feasibility-budget-001.json").resolve():
        raise ValueError("Use the audited runtime's canonical shared physical ledger")
    remaining = read_budget(args.budget_ledger)
    context = freeze(args)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "context.json"
    if marker.exists() and json.loads(marker.read_text()) != context:
        raise ValueError("Frozen context changed; preserve old cohort")
    if not marker.exists():
        atomic_json(marker, context)
    if args.worker:
        if args.launch is None or not (args.launch / "receipt.json").is_file():
            raise ValueError("Worker requires a durable supervising receipt")
        worker(args, context)
        return 0
    probe = resource_probe()
    required_disk = context["protocol"]["required_pairs"] * 2 * 128 * 1024 ** 2 + 8 * 1024 ** 3
    probe.update(disk_free_bytes=shutil.disk_usage(args.output).free,
                 disk_required_bytes=required_disk, remaining_shared_budget_seconds=remaining)
    probe["allowed"] = probe["allowed"] and probe["disk_free_bytes"] >= required_disk
    atomic_json(args.output / "latest-readiness.json", {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "context_sha256": object_digest(context), "resource_probe": probe,
        "native_episodes_started": 0, "prepare_only": args.prepare_only})
    if args.prepare_only or not probe["allowed"]:
        print(json.dumps({"status": "prepared" if args.prepare_only else "resource_blocked",
                          "probe": probe, "context_sha256": object_digest(context),
                          "native_episodes_started": 0}))
        return 0 if args.prepare_only else 3
    control = None
    if args.mode == "primary":
        if args.control is None:
            raise ValueError("Primary launch requires the completed independent-reload control")
        control = verified_control(args.control, context)
        args.wall_budget = control["estimated_primary_wall_seconds"]
    if not 0 < args.wall_budget <= 43200 - 135:
        raise ValueError("Invalid bounded native wall time or infeasible primary forecast")
    reserved = args.wall_budget + 135
    if reserved > remaining:
        raise RuntimeError("Insufficient remaining shared budget; do not reduce fixed sample size")
    # A launch, including a failed one, is never silently retried in this cohort.
    with (args.output / "collector.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list((args.output / "launches").glob("*")) or (args.output / "pairs").exists():
            raise RuntimeError("Prior physical attempt requires explicit reconciliation; no retry")
        launch = args.output / "launches" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        launch.mkdir(parents=True, exist_ok=False)
        budget = PhysicalBudget(args.budget_ledger)
        budget.reserve(str(launch), reserved)
        receipt = {"status": "starting", "context_sha256": object_digest(context),
                   "mode": args.mode, "resource_probe": probe, "control": control,
                   "budget_reservation": str(launch), "reserved_seconds": reserved,
                   "supervisor_sha256": digest_file(args.runtime_root / "experiments/health_supervisor.py")}
        atomic_json(launch / "receipt.json", receipt)
        def record_started(pid):
            receipt.update(status="running", worker_pid=pid)
            atomic_json(launch / "receipt.json", receipt)
        command = [
            "nice", "-n", "10", sys.executable, "-u", str(Path(__file__).resolve()),
            "--runtime-root", str(args.runtime_root), "--health-receipt", str(args.health_receipt),
            "--budget-ledger", str(args.budget_ledger), "--output", str(args.output),
            "--mode", args.mode, "--wall-budget", str(args.wall_budget),
            "--worker", "--launch", str(launch)]
        if args.profile_episodes:
            command.append("--profile-episodes")
        result = supervise(command, launch / "summary.json", launch / "worker.log",
                           args.wall_budget, 120, on_started=record_started,
                           summary_reader=read_summary)
        receipt.update(status="finished", execution=result)
        atomic_json(launch / "receipt.json", receipt)
        if result["elapsed_seconds"] <= reserved:
            budget.finish(str(launch), result["elapsed_seconds"])
        summary = read_summary(launch / "summary.json")
        passed = (result["clean_process_exit"] is True and result["worker_exit_code"] == 0
                  and result["status"] == "completed" and summary is not None
                  and summary["status"] != "native_error"
                  and summary["context_sha256"] == object_digest(context))
        print(json.dumps({"native_collection_completed": passed, "lifecycle": result, "summary": summary}))
        return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
