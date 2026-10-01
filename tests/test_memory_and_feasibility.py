import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1]
        / "experiments/confirmation-20261001" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


memory = load("memory_hygiene")
feasibility = load("feasibility")


class MemoryAndFeasibilityTests(unittest.TestCase):
    def test_cleanup_uses_own_allocator_and_records_both_snapshots(self):
        calls = []
        def trim(pad):
            calls.append(pad)
            return 1
        with patch.object(memory.ctypes, "CDLL", return_value=SimpleNamespace(malloc_trim=trim)) as library, \
                patch.object(memory, "snapshot", side_effect=[{"rss": 100}, {"rss": 50}]):
            result = memory.trim_own_cpu_allocator()
        library.assert_called_once_with(None)
        self.assertEqual(calls, [0])
        self.assertEqual(result["before"], {"rss": 100})
        self.assertEqual(result["after"], {"rss": 50})

    def test_missing_allocator_is_not_silently_reported_as_released(self):
        with patch.object(memory.ctypes, "CDLL", return_value=SimpleNamespace()):
            with self.assertRaises(RuntimeError):
                memory.trim_own_cpu_allocator()

    def test_full_horizon_forecast_separates_fixed_overhead_and_does_not_shrink_sample(self):
        pair = [{"steps": 260, "elapsed_seconds": 26},
                {"steps": 260, "elapsed_seconds": 26}]
        result = feasibility.primary_forecast([72], [pair])
        self.assertEqual(result["maximum_observed_pair_fixed_overhead_seconds"], 20)
        self.assertEqual(result["full_horizon_pair_seconds_before_margin"], 124)
        self.assertEqual(result["estimated_primary_wall_seconds"], 14881)
        self.assertEqual(result["primary_pairs"], 80)
        self.assertGreater(result["estimated_primary_wall_seconds"], 72 * 80 * 1.5)

    def test_invalid_native_timing_is_rejected(self):
        for steps, elapsed in ((True, 1), (521, 1), (10, float("nan")), (10, -1)):
            with self.subTest(steps=steps, elapsed=elapsed):
                with self.assertRaises(ValueError):
                    feasibility.primary_forecast(
                        [10], [[{"steps": steps, "elapsed_seconds": elapsed}] * 2])
        with self.assertRaises(ValueError):
            feasibility.primary_forecast(
                [1], [[{"steps": 520, "elapsed_seconds": 1}] * 2])
        with self.assertRaises(ValueError):
            feasibility.primary_forecast(
                [3], [[{"steps": 520, "elapsed_seconds": 1}] * 2], primary_pairs=20)


if __name__ == "__main__":
    unittest.main()
