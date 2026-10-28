"""Tests for scripts/predict_classifier.py. Temp models and CSVs only; Qt and network never touched."""
import contextlib
import io
import unittest

from app.dictionary import load_entries
from scripts import predict_classifier as cli
from scripts import train_classifier as tc
from tests.test_classifier import TempDirTest, make_rows, write_csv


class CliCase(TempDirTest):
    def setUp(self):
        super().setUp()
        self.train_rows = make_rows()
        self.run_quiet(self.train_rows)  # writes self.data (training CSV) and self.model
        self.heldout = self.dir / "heldout.csv"

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main([*args, "--model", str(self.model), "--training-data", str(self.data),
                                 "--heldout", str(self.heldout)])
            except SystemExit as e:  # argparse errors
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def novel_rows(self, status=tc.REVIEWED, n=3):
        # disjoint vocabulary from the training fixture, so provably unseen
        return [{"term": "Toxic", "text": f"novel{i}a novel{i}b novel{i}c novel{i}d", "label": lb,
                 "review_status": status} for i, lb in enumerate(tc.LABELS[:n])]


class Predict(CliCase):
    def test_single_example_prints_label_all_scores_and_disclaimer(self):
        code, out, err = self.cli("--term", "Toxic", "--text", "w0x0 w0x1 w0x2")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("predicted: " + tc.LABELS[0], out)
        for label in tc.LABELS:
            self.assertIn(label, out)
        self.assertIn("not guarantees", out)

    def test_multiple_examples_in_order(self):
        code, out, _ = self.cli("--term", "Toxic", "--text", "w0x0 w0x1", "--term", "Trauma", "--text", "w2x0 w2x1")
        self.assertEqual(code, 0)
        self.assertLess(out.index("[1]"), out.index("[2]"))
        self.assertIn("term: Trauma", out)

    def test_file_input(self):
        path = self.dir / "in.csv"
        path.write_text("term,text\nToxic,w1x0 w1x1\nToxic,w2x0 w2x1\n", encoding="utf-8")
        code, out, _ = self.cli("--file", str(path))
        self.assertEqual(code, 0)
        self.assertIn("[2]", out)

    def test_bad_inputs_exit_nonzero_without_traceback(self):
        path = self.dir / "bad.csv"
        path.write_text("x,y\n1,2\n", encoding="utf-8")
        for args in (("--term", "Toxic"), ("--file", str(path)), ("--file", str(self.dir / "nope.csv")),
                     ("--term", "Toxic", "--text", "  ")):
            with self.subTest(args):
                code, _, err = self.cli(*args)
                self.assertEqual(code, 1)
                self.assertTrue(err.startswith("ERROR:"), err)

    def test_no_arguments_is_a_usage_error(self):
        self.assertEqual(self.cli()[0], 2)

    def test_missing_or_invalid_model(self):
        self.model.unlink()
        code, _, err = self.cli("--term", "Toxic", "--text", "w0x0")
        self.assertEqual(code, 1)
        self.assertIn("not found", err)
        self.model.write_bytes(b"junk")
        code, _, err = self.cli("--term", "Toxic", "--text", "w0x0")
        self.assertEqual(code, 1)
        self.assertIn("could not be read", err)

    def test_familiarity_is_reported_honestly(self):
        seen = self.train_rows[0]["text"]
        _, out, _ = self.cli("--term", "Toxic", "--text", seen.upper())  # case change is still "seen"
        self.assertIn("SEEN IN TRAINING", out)
        _, out, _ = self.cli("--term", "Toxic", "--text", "something entirely different here")
        self.assertIn("not in training", out)
        self.assertNotIn("SEEN IN TRAINING", out)

    def test_unknown_term_is_flagged(self):
        _, out, _ = self.cli("--term", "Banana", "--text", "w0x0 w0x1")
        self.assertIn("not a dictionary term", out)

    def test_deterministic_output(self):
        args = ("--term", "Toxic", "--text", "w1x1 w2x2")
        self.assertEqual(self.cli(*args), self.cli(*args))


class Evaluate(CliCase):
    def test_scores_only_reviewed_unseen_rows(self):
        write_csv(self.heldout, self.novel_rows()
                  + [{**self.novel_rows()[0], "text": "pending pending words", "review_status": tc.NEEDS_REVIEW}])
        code, out, err = self.cli("--evaluate")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("reviewed: 3", out)
        self.assertIn("needs_review (ignored): 1", out)
        self.assertIn("Evaluated on 3 reviewed, unseen rows", out)
        self.assertIn("Confusion matrix", out)
        self.assertIn("do not show", out)

    def test_rows_seen_in_training_are_excluded_not_scored(self):
        leaked = {**self.train_rows[0], "text": self.train_rows[0]["text"] + "."}  # same words
        write_csv(self.heldout, [leaked] + self.novel_rows())
        code, out, _ = self.cli("--evaluate")
        self.assertEqual(code, 0)
        self.assertIn("EXCLUDED", out)
        self.assertIn("Evaluated on 3 reviewed, unseen rows", out)

    def test_all_leaked_fails(self):
        write_csv(self.heldout, self.train_rows[:5])
        code, _, err = self.cli("--evaluate")
        self.assertEqual(code, 1)
        self.assertIn("No usable held-out rows", err)

    def test_only_unreviewed_fails_with_clear_message(self):
        write_csv(self.heldout, self.novel_rows(status=tc.NEEDS_REVIEW))
        code, _, err = self.cli("--evaluate")
        self.assertEqual(code, 1)
        self.assertIn("Review rows", err)

    def test_invalid_heldout_file_fails(self):
        write_csv(self.heldout, [{**self.novel_rows()[0], "label": "MAYBE"}])
        code, _, err = self.cli("--evaluate")
        self.assertEqual(code, 1)
        self.assertIn("invalid label", err)

    def test_missing_training_data_refuses_to_claim_unseen(self):
        write_csv(self.heldout, self.novel_rows())
        self.data.unlink()
        code, _, err = self.cli("--evaluate")
        self.assertEqual(code, 1)
        self.assertIn("Training data is needed", err)

    def test_missing_labels_warned(self):
        write_csv(self.heldout, self.novel_rows(n=1))
        _, out, _ = self.cli("--evaluate")
        self.assertIn("WARNING: no held-out examples for", out)


class ShippedHeldoutCandidates(unittest.TestCase):
    def test_valid_and_not_near_duplicates_of_training(self):
        rows = tc.load_rows(cli.HELDOUT_PATH)
        tc.validate(rows, {e.term for e in load_entries()})
        refs = [r["text"] for r in tc.reviewed_rows(tc.load_rows(tc.DATA_PATH))]
        for r in rows:
            self.assertLess(tc.closest(r["text"], refs)[0], tc.SIMILARITY, r["text"])
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
