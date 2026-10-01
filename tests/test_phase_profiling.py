import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "phase_profiling", Path(__file__).resolve().parents[1]
    / "experiments/confirmation-20261001/phase_profiling.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PhaseProfilingTests(unittest.TestCase):
    def test_wrapper_preserves_argument_and_result_identity(self):
        timers = module.PhaseTimers()
        marker = object()
        received = []
        def operation(value, *, label):
            received.append((value, label))
            return marker
        measured = timers.wrap("operation", operation)
        self.assertIs(measured(marker, label="once"), marker)
        self.assertEqual(received, [(marker, "once")])
        self.assertEqual(timers.summary()["operation"]["calls"], 1)
        self.assertEqual(timers.summary()["operation"]["failed_calls"], 0)

    def test_error_identity_and_inherited_method_are_restored(self):
        error = RuntimeError("preserve this exception")
        class Owner:
            def method(self):
                raise error
        owner = Owner()
        timers = module.PhaseTimers()
        timers.patch(owner, "method", "operation")
        try:
            with self.assertRaises(RuntimeError) as raised:
                owner.method()
            self.assertIs(raised.exception, error)
        finally:
            timers.close()
        self.assertNotIn("method", vars(owner))
        self.assertIs(owner.method.__func__, Owner.method)
        self.assertEqual(timers.summary()["operation"]["failed_calls"], 1)

    def test_own_attribute_and_nested_timers_restore_in_reverse_order(self):
        class Owner:
            pass
        owner = Owner()
        marker = object()
        original = lambda: marker
        owner.method = original
        timers = module.PhaseTimers()
        timers.patch(owner, "method", "inner")
        timers.patch(owner, "method", "outer")
        self.assertIs(owner.method(), marker)
        timers.close()
        self.assertIs(owner.method, original)
        self.assertEqual(set(timers.summary()), {"inner", "outer"})
        self.assertGreaterEqual(timers.summary()["outer"]["total_inclusive_wall_seconds"],
                                timers.summary()["inner"]["total_inclusive_wall_seconds"])

    def test_episode_hooks_preserve_operations_restore_and_record_errors(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                calls, reports, vectors = [], [], []
                marker = object()
                error = RuntimeError("native episode failure")
                def operation(name):
                    def call(*args, **kwargs):
                        calls.append(name)
                        return marker
                    return call
                class Vector:
                    def __init__(self):
                        vectors.append(self)
                    reset = operation("reset")
                    step = operation("step")
                    close = operation("close")
                class Policy:
                    select_action = operation("select_action")
                native = ModuleType("native_rollout")
                native.save_arrays = operation("save_arrays")
                native.atomic_json = operation("metadata")
                health = ModuleType("run_closedloop_health")
                health.preprocess_observation = operation("observation_conversion")
                health.sim_state = operation("state_capture")
                health.gym = SimpleNamespace(vector=SimpleNamespace(SyncVectorEnv=Vector))
                paired = ModuleType("paired_collection")
                paired.atomic_json = lambda path, report: reports.append((path, report))
                backend = SimpleNamespace(
                    policy=Policy(), pre=operation("pre"), env_pre=operation("env_pre"),
                    post=operation("post"), env_post=operation("env_post"))
                original_pre = backend.pre
                original_save = native.save_arrays
                expected_calls = [
                    "reset", "reset", "save_arrays", "state_capture", "observation_conversion",
                    "env_pre", "pre", "select_action", "post", "env_post", "metadata",
                    "step", "save_arrays", "close"]
                def run():
                    vector = health.gym.vector.SyncVectorEnv()
                    try:
                        vector.reset()
                        vector.reset()
                        native.save_arrays()
                        health.sim_state()
                        health.preprocess_observation()
                        backend.pre(backend.env_pre())
                        self.assertIs(backend.policy.select_action(), marker)
                        backend.env_post(backend.post())
                        native.atomic_json()
                        self.assertIs(vector.step(), marker)
                        native.save_arrays()
                        if fail:
                            raise error
                        return {"steps": 1}
                    finally:
                        vector.close()
                with patch.dict(sys.modules, {
                        "native_rollout": native, "run_closedloop_health": health,
                        "paired_collection": paired}):
                    if fail:
                        with self.assertRaises(RuntimeError) as raised:
                            module.profile_episode(backend, run, Path("/unused"))
                        self.assertIs(raised.exception, error)
                    else:
                        self.assertEqual(module.profile_episode(
                            backend, run, Path("/unused")), {"steps": 1})
                self.assertEqual(calls, expected_calls)
                self.assertIs(health.gym.vector.SyncVectorEnv, Vector)
                self.assertIs(backend.pre, original_pre)
                self.assertIs(native.save_arrays, original_save)
                self.assertNotIn("select_action", vars(backend.policy))
                self.assertNotIn("step", vars(vectors[0]))
                self.assertEqual(reports[0][1]["status"], "error" if fail else "completed")
                self.assertEqual(reports[0][1]["summary"]["simulator_step"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
