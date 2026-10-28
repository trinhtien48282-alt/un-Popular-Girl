"""Regression tests: original-CSV row numbers in predict_classifier, and the training coverage report."""
import contextlib
import io
import unittest

from scripts import predict_classifier as cli
from scripts import train_classifier as tc
from tests.test_classifier import TempDirTest, make_rows, write_csv
from tests.test_predict_cli import CliCase

NEW, MISUSE, PLAUSIBLE = tc.LABELS[0], tc.LABELS[2], tc.LABELS[1]  # NEEDS CONTEXT, POSSIBLE MISUSE, PLAUSIBLE USAGE
assert (NEW, MISUSE, PLAUSIBLE) == ("NEEDS CONTEXT", "POSSIBLE MISUSE", "PLAUSIBLE USAGE")


def row(term, label, status=tc.REVIEWED, text=None):
    return {"term": term, "text": text or f"{term} {label} {status} sample", "label": label, "review_status": status}


class OriginalRowNumbers(CliCase):
    """needs_review rows come first in the file, so reviewed-list index != CSV row."""

    def setUp(self):
        TempDirTest.setUp(self)
        unreviewed = [{**r, "text": f"pending{i} pending{i}b pending{i}c", "review_status": tc.NEEDS_REVIEW}
                      for i, r in enumerate(make_rows(per_label=1))]  # 3 rows -> CSV rows 2-4
        self.train_rows = unreviewed + make_rows()
        self.run_quiet(self.train_rows)
        self.heldout = self.dir / "heldout.csv"
        self.target = 7  # index within make_rows()
        self.expected_row = 3 + self.target + 2  # 3 unreviewed rows + data row offset (header = row 1)
        self.target_text = make_rows()[self.target]["text"]

    def test_training_texts_carry_original_row_numbers(self):
        refs = cli._training_texts(self.data)
        self.assertEqual(refs[0][0], 5)  # first reviewed row is CSV row 5, not 2
        self.assertEqual(len(refs), 30)
        self.assertEqual(refs[self.target], (self.expected_row, self.target_text))

    def test_predict_reports_original_row(self):
        code, out, _ = self.cli("--term", "Toxic", "--text", self.target_text)
        self.assertEqual(code, 0)
        self.assertIn(f"reviewed training row {self.expected_row}\n", out)
        self.assertNotIn(f"reviewed training row {self.target + 2}\n", out)

    def test_evaluate_exclusion_reports_original_row(self):
        write_csv(self.heldout, [{**make_rows()[self.target], "text": self.target_text + "."},
                                 *self.novel_rows()])
        code, out, _ = self.cli("--evaluate")
        self.assertEqual(code, 0)
        self.assertIn(f"to training row {self.expected_row})", out)

    def test_row_number_matches_the_file_itself(self):
        # Independent check: read the CSV file's own lines (valid here: no field has a line break).
        lines = self.data.read_text(encoding="utf-8").splitlines()
        self.assertIn(self.target_text, lines[self.expected_row - 1])  # row N is line N

    def test_missing_training_file_still_degrades(self):
        self.assertIsNone(cli._training_texts(self.dir / "nope.csv"))
        self.assertIn("unavailable", cli._familiarity("x", None))


class CoverageReport(TempDirTest):
    def lines(self, rows, known=None):
        return tc.coverage_lines(rows, known)

    def find(self, lines, prefix):
        return next(line for line in lines if line.strip().startswith(prefix))

    def test_counts_only_reviewed_rows_per_term_and_label(self):
        rows = [row("Alpha", MISUSE), row("Alpha", MISUSE, text="alpha two"), row("Alpha", PLAUSIBLE),
                row("Alpha", NEW, status=tc.NEEDS_REVIEW)]
        line = self.find(self.lines(rows), "Alpha")
        cols = line.split()
        self.assertEqual(cols[1:4], ["0", "1", "2"])  # NEEDS CONTEXT, PLAUSIBLE USAGE, POSSIBLE MISUSE
        self.assertEqual(cols[4], "1")                # one unreviewed row
        self.assertIn("no reviewed: NEEDS CONTEXT", line)

    def test_complete_term_has_no_gap_text(self):
        rows = [row("Beta", label) for label in tc.LABELS]
        self.assertNotIn("no reviewed", self.find(self.lines(rows), "Beta"))
        self.assertNotIn("NO REVIEWED", self.find(self.lines(rows), "Beta"))

    def test_term_with_only_unreviewed_rows_is_flagged(self):
        line = self.find(self.lines([row("Gamma", NEW, status=tc.NEEDS_REVIEW)]), "Gamma")
        self.assertIn("NO REVIEWED ROWS", line)

    def test_dictionary_term_absent_from_csv_is_flagged(self):
        line = self.find(self.lines([row("Alpha", MISUSE)], known={"Alpha", "Ghost"}), "Ghost")
        self.assertIn("NO REVIEWED ROWS", line)

    def test_label_with_no_reviewed_rows_is_flagged(self):
        out = "\n".join(self.lines([row("Alpha", MISUSE), row("Beta", PLAUSIBLE)]))
        self.assertIn(f"{NEW}: NO reviewed rows", out)

    def test_label_totals_show_where_rows_come_from(self):
        rows = [row("Alpha", NEW, text=f"a{i}") for i in range(3)] + [row("Beta", NEW)]
        out = "\n".join(self.lines(rows))
        self.assertIn(f"{NEW}: 4 reviewed; most rows from one term: Alpha (3 of 4)", out)

    def test_makes_no_quality_claim_and_applies_no_minimum(self):
        out = "\n".join(self.lines([row("Alpha", MISUSE)]))
        self.assertIn("gaps, not data quality", out)
        self.assertNotIn("insufficient", out.lower())

    def test_run_prints_coverage_and_training_behaviour_is_unchanged(self):
        write_csv(self.data, make_rows())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pipe = tc.run(self.data, self.model, known_terms={"Toxic", "Absent term"})
        text = out.getvalue()
        self.assertIn("Coverage by term", text)
        self.assertIn("Absent term", text)
        self.assertIn("NO REVIEWED ROWS", text)  # flagged, but training still proceeded
        self.assertTrue(self.model.exists())
        self.assertEqual(pipe.named_steps["clf"].random_state, tc.SEED)

    def test_validate_only_still_prints_coverage(self):
        write_csv(self.data, make_rows())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tc.run(self.data, self.model, validate_only=True)
        self.assertIn("Coverage by term", out.getvalue())
        self.assertFalse(self.model.exists())


if __name__ == "__main__":
    unittest.main()
