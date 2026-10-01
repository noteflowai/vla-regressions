"""Opt-in inclusive wall/CPU timers; no CUDA synchronization or extra model calls."""
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import wraps
import statistics
import time


class PhaseTimers:
    def __init__(self):
        self.rows = []
        self.counts = {}
        self.stack = ExitStack()

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            index = self.counts.get(name, 0)
            self.counts[name] = index + 1
            cpu, wall = time.process_time(), time.perf_counter()
            failed = True
            try:
                result = function(*args, **kwargs)
                failed = False
                return result
            finally:
                self.rows.append({
                    "phase": name, "call_index": index, "failed": failed,
                    "wall_seconds": time.perf_counter() - wall,
                    "process_cpu_seconds": time.process_time() - cpu,
                })
        return measured

    def patch(self, owner, attribute, name, function=None):
        # Restore the object's own attribute dictionary, including inherited methods.
        had_own = attribute in vars(owner)
        own = vars(owner).get(attribute)
        original = getattr(owner, attribute)
        setattr(owner, attribute, self.wrap(name, original if function is None else function))
        def restore():
            if had_own:
                setattr(owner, attribute, own)
            else:
                delattr(owner, attribute)
        self.stack.callback(restore)
        return original

    def close(self):
        self.stack.close()

    def summary(self):
        result = {}
        for name in sorted(self.counts):
            rows = [row for row in self.rows if row["phase"] == name]
            result[name] = {
                "calls": len(rows), "failed_calls": sum(row["failed"] for row in rows),
                "total_inclusive_wall_seconds": sum(row["wall_seconds"] for row in rows),
                "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
                "total_process_cpu_seconds": sum(row["process_cpu_seconds"] for row in rows),
            }
        return result


def profile_episode(backend, run, folder):
    import native_rollout
    import run_closedloop_health as h
    from paired_collection import atomic_json

    timers = PhaseTimers()
    started = datetime.now(timezone.utc).isoformat()
    status = "error"
    try:
        for owner, attribute, name in (
                (native_rollout, "save_arrays", "snapshot_persistence"),
                (native_rollout, "atomic_json", "metadata_persistence"),
                (h, "preprocess_observation", "observation_conversion"),
                (h, "sim_state", "simulator_state_capture"),
                (backend, "pre", "policy_processor"),
                (backend, "env_pre", "environment_preprocessor"),
                (backend, "post", "action_postprocessor"),
                (backend, "env_post", "environment_postprocessor"),
                (backend.policy, "select_action", "policy_select_action")):
            timers.patch(owner, attribute, name)
        original_vector = h.gym.vector.SyncVectorEnv
        def construct_vector(*args, **kwargs):
            vector = original_vector(*args, **kwargs)
            for attribute, name in (("step", "simulator_step"),
                                    ("reset", "canonical_reset"),
                                    ("close", "environment_close")):
                timers.patch(vector, attribute, name)
            return vector
        timers.patch(h.gym.vector, "SyncVectorEnv", "environment_construction",
                     function=construct_vector)
        result = run()
        steps = result["steps"]
        expected = {
            "snapshot_persistence": steps + 1, "policy_select_action": steps,
            "simulator_step": steps, "observation_conversion": steps,
            "policy_processor": steps, "environment_preprocessor": steps,
            "action_postprocessor": steps, "environment_postprocessor": steps,
            "canonical_reset": 2, "environment_construction": 1, "environment_close": 1,
        }
        if (any(timers.counts.get(name) != count for name, count in expected.items())
                or any(row["failed"] for row in timers.rows)):
            raise ValueError("Incomplete or failed native phase timing")
        status = "completed"
        return result
    finally:
        timers.close()
        atomic_json(folder / "phase-profile.json", {
            "status": status, "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "summary": timers.summary(), "rows": timers.rows,
            "scope": "Inclusive host wall/process CPU timing. Nested phases overlap; do not "
                     "sum phase totals. CUDA work can complete in a later phase; these are "
                     "not synchronized GPU kernel timings. Policy call index zero starts "
                     "after reset; refresh is expected at indices divisible by 30. No extra "
                     "prediction, state read, RNG draw or CUDA synchronization is performed.",
        })
