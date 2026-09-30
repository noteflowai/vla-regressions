import json
from pathlib import Path
import tempfile
import unittest
from scipy.stats import binomtest
from analyze import by, fisher_less, load, pair_rows, paired_pvalue, prob_worse_paired


class PairedAnalysisTests(unittest.TestCase):
    def test_five_discordant_pairs_cannot_have_independent_sample_pvalue(self):
        self.assertEqual(paired_pvalue(5, 0), 1/32)
        self.assertLess(fisher_less(0, 5, 5, 5), paired_pvalue(5, 0))
        # A negative-correlation null can realize all five harmful pairs with
        # probability 1/32, so the smaller Fisher value is not generally valid.

    def test_followup_matches_independent_exact_binomial_reference(self):
        self.assertAlmostEqual(paired_pvalue(15, 1),
                               binomtest(15, 16, .5, alternative="greater").pvalue)

    def test_no_discordance_is_uninformative(self):
        self.assertEqual(paired_pvalue(0, 0), 1.)
        self.assertEqual(prob_worse_paired(0, 0), .5)

    def test_multiplicity_does_not_flag_five_repeat_extremes(self):
        self.assertEqual(by([1/32]*7+[1.]*93, .1), set())

    def test_incomplete_pairs_and_undeclared_seed_changes_fail(self):
        row = {"repeat": 0, "seed": 1000}
        with self.assertRaises(ValueError):
            pair_rows([row], [{"repeat": 1, "seed": 1000}])
        with self.assertRaises(ValueError):
            pair_rows([row], [{"repeat": 0, "seed": 2000}])
        self.assertEqual(len(pair_rows([row], [{"repeat": 0, "seed": 2000}], True)), 1)

    def test_duplicate_or_non_boolean_records_fail(self):
        row = {"task": 0, "init_state": 0, "scene_seed": 1, "repeat": 0,
               "seed": 1000, "success": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"episodes.jsonl"
            path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
            with self.assertRaises(ValueError):
                load(path)
            row["success"] = 1
            path.write_text(json.dumps(row)+"\n")
            with self.assertRaises(ValueError):
                load(path)


if __name__ == "__main__":
    unittest.main()
