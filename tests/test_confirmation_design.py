"""Independent enumerations and design safeguards for the prospective experiment."""
import importlib.util
from itertools import product
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "experiments/confirmation-20261001/design.py"
spec = importlib.util.spec_from_file_location("confirmation_design", path)
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


class ConfirmationDesignTests(unittest.TestCase):
    def test_power_matches_direct_trial_enumeration(self):
        for n, h, g in ((5, .7, .1), (6, .3, .2), (7, .2, .2), (8, .4, .0)):
            total = 0.0
            for trials in product(range(3), repeat=n):
                harmful, beneficial = trials.count(0), trials.count(1)
                # Calculate the null tail independently using integer combinatorics.
                from math import comb
                m = harmful + beneficial
                tail = sum(comb(m, k) for k in range(harmful, m + 1)) / 2 ** m
                if tail <= .05:
                    probability = h ** harmful * g ** beneficial * (1 - h - g) ** trials.count(2)
                    total += probability
            self.assertAlmostEqual(design.exact_power(n, h, g), total, places=12)

    def test_null_size_and_known_exact_tail(self):
        for n in (5, 20, 80):
            for r in (.1, .5, 1.0):
                self.assertLessEqual(design.exact_power(n, r / 2, r / 2), .05 + 1e-14)
        self.assertEqual(design.paired_pvalue(0, 0), 1.0)
        self.assertEqual(design.paired_pvalue(15, 1), 17 / 65536)
        self.assertEqual(design.exact_power(4, 1, 0), 0.0)
        self.assertEqual(design.exact_power(5, 1, 0), 1.0)

    def test_invalid_planning_inputs(self):
        for args in ((0, .1, .2), (True, .1, .2), (5, .7, .4),
                     (5, float("nan"), .1), (5, -.1, .2), (5, .2, .1, 0)):
            with self.assertRaises(ValueError):
                design.exact_power(*args)

    def test_seed_streams_are_distinct_and_order_is_balanced(self):
        primary, control = design.pair_plan("primary"), design.pair_plan("control")
        seeds = [p["policy_seed"] for p in primary + control]
        self.assertEqual(len(seeds), len(set(seeds)))
        for plan in (primary, control):
            self.assertEqual(sum(p["side_order"][0] == "old" for p in plan), len(plan) // 2)
        self.assertEqual(primary, design.pair_plan("primary"))

    def test_incomplete_cohort_cannot_produce_primary_inference(self):
        for rows in ([], [(True, False)] * 79, [(True, False)] * 81,
                     [(True, False)] * 79 + [(None, False)]):
            with self.assertRaises(ValueError):
                design.final_analysis(rows)
        result = design.final_analysis([(True, False)] * 80)
        self.assertEqual(result["one_sided_exact_p"], 2 ** -80)
        self.assertEqual(result["coarse_block_sign_p"], 2 ** -8)
        self.assertEqual(result["block_success_differences"], [10] * 8)
        self.assertLess(result["paired_loss_95pct_interval"][0], 1)
        self.assertEqual(result["paired_loss_95pct_interval"][1], 1)
        null_result = design.final_analysis([(True, True)] * 80)
        self.assertLess(null_result["paired_loss_95pct_interval"][0], 0)
        self.assertGreater(null_result["paired_loss_95pct_interval"][1], 0)


if __name__ == "__main__":
    unittest.main()
