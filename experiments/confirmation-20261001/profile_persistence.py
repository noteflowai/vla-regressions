"""CPU-only durable NPZ replay benchmark; never loads a policy or edits input data."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import zipfile

import numpy as np

NATIVE_WRITER_SHA256 = "8395afb993f8c553e7a68f33cd3ccd283cd556b319de86c52f1bbf5de4fcddd2"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def candidate_writer(path, arrays, compression):
    """Keep file fsync, atomic replace, directory fsync and the NPZ array schema."""
    temporary = path.with_suffix(".tmp")
    phases = {}
    before = time.perf_counter()
    with temporary.open("wb") as handle:
        if compression == "stored":
            np.savez(handle, **arrays)
        else:
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for name, array in arrays.items():
                    with archive.open(name + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, array, allow_pickle=False)
        phases["encode_write_seconds"] = time.perf_counter() - before
        before = time.perf_counter()
        handle.flush()
        os.fsync(handle.fileno())
        phases["file_flush_fsync_seconds"] = time.perf_counter() - before
    before = time.perf_counter()
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    phases["replace_directory_fsync_seconds"] = time.perf_counter() - before
    return phases


def verify(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError("NPZ fields changed")
        for key, array in expected.items():
            actual = saved[key]
            if (actual.shape != array.shape or actual.dtype != array.dtype
                    or not np.array_equal(actual, array, equal_nan=True)):
                raise ValueError("NPZ round trip changed: " + key)


def summarize(rows):
    result = {}
    for mode in ("native_compressed", "deflate_level_1", "stored"):
        selected = [row for row in rows if row["mode"] == mode]
        result[mode] = {
            "writes": len(selected),
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in selected),
            "maximum_wall_seconds": max(row["wall_seconds"] for row in selected),
            "median_process_cpu_seconds": statistics.median(
                row["process_cpu_seconds"] for row in selected),
            "mean_output_bytes": statistics.mean(row["output_bytes"] for row in selected),
            "all_array_round_trips_verified": all(row["verified"] for row in selected),
        }
        # This is sample-based storage planning, not a guaranteed size bound.
        result[mode]["sample_based_80_pair_520_step_npz_bytes"] = int(
            result[mode]["mean_output_bytes"] * 80 * 2 * 521)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026100113)
    parser.add_argument("--concurrent-native-collection", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 2 <= args.samples_per_episode <= 32:
        parser.error("Replay is bounded to 1–10 repeats and 2–32 snapshots per episode")
    writer_source = args.runtime_root.resolve() / "experiments/native_rollout.py"
    if digest(writer_source) != NATIVE_WRITER_SHA256:
        raise ValueError("Native writer source differs from the audited version")
    sys.path.insert(0, str(writer_source.parent))
    native_writer = importlib.import_module("native_rollout").save_arrays
    selected, sources = [], []
    for episode in args.episode:
        episode = episode.resolve()
        files = sorted(episode.glob("input-*.npz")) + [episode / "terminal.npz"]
        if len(files) < args.samples_per_episode or not all(path.is_file() for path in files):
            raise ValueError("Complete saved episode required")
        indices = np.linspace(0, len(files) - 1, args.samples_per_episode, dtype=int)
        for index in indices:
            path = files[int(index)]
            selected.append(path)
            sources.append({"path": str(path), "sha256": digest(path),
                            "bytes": path.stat().st_size})
    args.output = args.output.resolve()
    # Never place benchmark output inside a preserved raw episode.
    if any(args.output.is_relative_to(path.resolve()) for path in args.episode):
        raise ValueError("Benchmark output must be outside the preserved raw episodes")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    randomizer = random.Random(args.seed)
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="npz-replay-", dir=args.output) as temporary:
        target = Path(temporary) / "snapshot.npz"
        for path in selected:
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            for repeat in range(args.repeats):
                modes = ["native_compressed", "deflate_level_1", "stored"]
                randomizer.shuffle(modes)
                for mode in modes:
                    cpu_before, wall_before = time.process_time(), time.perf_counter()
                    if mode == "native_compressed":
                        native_writer(target, arrays)
                        phases = None
                    else:
                        phases = candidate_writer(target, arrays, mode)
                    wall, cpu = time.perf_counter() - wall_before, time.process_time() - cpu_before
                    verify(target, arrays)
                    rows.append({"source": str(path), "repeat": repeat, "mode": mode,
                                 "wall_seconds": wall, "process_cpu_seconds": cpu,
                                 "output_bytes": target.stat().st_size, "verified": True,
                                 "phases": phases})
                    target.unlink()
    for source in sources:
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("Preserved raw input changed during the benchmark")
    result = {
        "status": "completed", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_source_sha256": digest(Path(__file__)),
        "native_writer_source_sha256": NATIVE_WRITER_SHA256,
        "numpy_version": np.__version__, "seed": args.seed, "sources": sources,
        "concurrent_native_collection": args.concurrent_native_collection,
        "summary": summarize(rows), "rows": rows,
        "scope": "CPU-only replay of retained observations; no policy, simulation or native "
                 "episode. Timing excludes input reading and verification. Warm page cache, "
                 "sampled frames and concurrent jobs limit extrapolation. Candidate formats "
                 "are not adopted by the frozen collector. Storage projection includes NPZ "
                 "only and is a sample mean, not an admission bound.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
