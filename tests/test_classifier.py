"""Tests for the ML prototype. Small synthetic fixtures in temp dirs; Qt and network never touched.

unittest style like the rest of tests/, so `unittest` and `pytest` both run it.
"""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.classifier import Classifier, ClassifierError, build_input
from app.dictionary import load_entries
from app.models import Category
from scripts import train_classifier as tc


def make_rows(per_label=10, status=tc.REVIEWED):
    """Cleanly separable synthetic rows: each label has its own vocabulary, and
    consecutive rows share 3 of 9 words (Jaccard 0.33), below the grouping threshold."""
    rows = []
    for li, label in enumerate(tc.LABELS):
        words = [f"w{li}x{k}" for k in range(per_label * 3 + 6)]
        for i in range(per_label):
            rows.append({"term": "Toxic", "text": " ".join(words[i * 3:i * 3 + 6]),
                         "label": label, "review_status": status})
    return rows


def write_csv(path, rows, fields=tc.COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data, self.model = self.dir / "t.csv", self.dir / "m" / "model.joblib"

    def run_quiet(self, rows, **kw):
        write_csv(self.data, rows)
        with contextlib.redirect_stdout(io.StringIO()):
            return tc.run(self.data, self.model, **kw)


class DatasetValidation(TempDirTest):
    def test_good_rows_pass(self):
        write_csv(self.data, make_rows())
        tc.validate(tc.load_rows(self.data))

    def test_missing_file(self):
        with self.assertRaisesRegex(tc.DatasetError, "not found"):
            tc.load_rows(self.dir / "nope.csv")

    def test_missing_column(self):
        write_csv(self.data, make_rows(), fields=("term", "text", "label"))
        with self.assertRaisesRegex(tc.DatasetError, "review_status"):
            tc.load_rows(self.data)

    def test_each_problem_is_reported(self):
        base = make_rows(per_label=2)
        cases = {
            "'text' is empty": {**base[0], "text": ""},
            "invalid label": {**base[0], "label": "MAYBE"},
            "review_status must be": {**base[0], "review_status": "yes"},
            "unknown term": {**base[0], "term": "Nonsense"},
        }
        for msg, bad in cases.items():
            with self.subTest(msg), self.assertRaisesRegex(tc.DatasetError, msg):
                tc.validate([bad], known_terms={"Toxic"})

    def test_duplicates_and_conflicts(self):
        a = make_rows(per_label=1)[0]
        with self.assertRaisesRegex(tc.DatasetError, "duplicate of row 2"):
            tc.validate([a, {**a, "text": a["text"].upper()}])  # case-insensitive
        other = next(l for l in tc.LABELS if l != a["label"])
        with self.assertRaisesRegex(tc.DatasetError, "conflicting label"):
            tc.validate([a, {**a, "label": other}])

    def test_unreviewed_rows_are_filtered(self):
        rows = make_rows(per_label=1) + make_rows(per_label=1, status=tc.NEEDS_REVIEW)
        self.assertEqual(len(tc.reviewed_rows(rows)), 3)

    def test_shipped_dataset_is_valid(self):
        rows = tc.load_rows(tc.DATA_PATH)
        tc.validate(rows, {e.term for e in load_entries()})
        self.assertTrue(tc.reviewed_rows(rows))


class LeakageSafeguards(unittest.TestCase):
    def test_near_duplicates_share_a_group(self):
        g = tc.group_ids([
            "He disagreed with me about the movie ending. That's gaslighting.",
            "He disagreed with me about the movie. That's gaslighting.",
            "The factory dumped toxic waste into the river.",
        ])
        self.assertEqual(g[0], g[1])
        self.assertNotEqual(g[0], g[2])

    def test_no_group_spans_train_and_test(self):
        rows = make_rows()
        # add a near-duplicate twin of every row; twins must land on the same side
        rows += [{**r, "text": r["text"] + " extra"} for r in rows]
        y = [r["label"] for r in rows]
        groups = tc.group_ids([r["text"] for r in rows])
        splits = tc.grouped_splits(y, groups)
        self.assertEqual(len(splits), tc.FOLDS)
        for train, test in splits:
            self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})

    def test_splits_reproducible(self):
        rows = make_rows()
        y = [r["label"] for r in rows]
        groups = tc.group_ids([r["text"] for r in rows])
        a, b = tc.grouped_splits(y, groups), tc.grouped_splits(y, groups)
        for (t1, e1), (t2, e2) in zip(a, b):
            self.assertEqual(list(e1), list(e2))


class Training(TempDirTest):
    def test_unreviewed_rows_never_train(self):
        rows = make_rows() + [{**make_rows(per_label=1, status=tc.NEEDS_REVIEW)[0],
                               "text": "zzunreviewedtoken zzunreviewedtoken"}]
        pipe = self.run_quiet(rows)
        vocab = pipe.named_steps["tfidf"].vocabulary_
        self.assertNotIn("zzunreviewedtoken", vocab)
        self.assertIn("w0x0", vocab)

    def test_too_little_reviewed_data_fails(self):
        with self.assertRaisesRegex(tc.DatasetError, "Not enough reviewed data"):
            self.run_quiet(make_rows(per_label=3))
        self.assertFalse(self.model.exists())

    def test_missing_class_fails(self):
        rows = [r for r in make_rows() if r["label"] != Category.NEEDS_CONTEXT.value]
        with self.assertRaisesRegex(tc.DatasetError, "NEEDS CONTEXT: 0"):
            self.run_quiet(rows)

    def test_only_unreviewed_data_fails(self):
        with self.assertRaisesRegex(tc.DatasetError, "Not enough reviewed data"):
            self.run_quiet(make_rows(status=tc.NEEDS_REVIEW))

    def test_pipeline_is_saved_whole_and_reloads(self):
        pipe = self.run_quiet(make_rows())
        saved = joblib.load(self.model)
        self.assertIsInstance(saved, Pipeline)
        self.assertEqual([n for n, _ in saved.steps], ["tfidf", "clf"])
        text = [build_input("Toxic", "w0x0 w0x1 w0x2")]
        self.assertEqual(list(saved.predict_proba(text)[0]), list(pipe.predict_proba(text)[0]))

    def test_validate_and_evaluate_only_save_nothing(self):
        for flag in ("validate_only", "evaluate_only"):
            with self.subTest(flag):
                self.assertIsNone(self.run_quiet(make_rows(), **{flag: True}))
                self.assertFalse(self.model.exists())

    def test_reproducible(self):
        a = self.run_quiet(make_rows())
        b = self.run_quiet(make_rows())
        text = [build_input("Toxic", "w1x3 w1x4 w2x0")]
        self.assertEqual(list(a.predict_proba(text)[0]), list(b.predict_proba(text)[0]))

    def test_report_shows_counts_matrix_and_caveat(self):
        write_csv(self.data, make_rows())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tc.run(self.data, self.model, evaluate_only=True)
        text = out.getvalue()
        for needle in ("reviewed: 30", "precision", "Confusion matrix", "macro-F1", "do not show"):
            self.assertIn(needle, text)


class Inference(TempDirTest):
    def setUp(self):
        super().setUp()
        self.rows = make_rows()
        self.run_quiet(self.rows)
        self.clf = Classifier.load(self.model)

    def test_predicts_clearly_separable_reviewed_examples(self):
        # Fixture vocab is disjoint per label, so this is justified (not true of real sentences).
        for r in self.rows:
            self.assertEqual(self.clf.predict(r["term"], r["text"]).label.value, r["label"])

    def test_probabilities_are_a_distribution_over_categories(self):
        p = self.clf.predict("Toxic", "w0x0 w0x1")
        self.assertEqual(set(p.probabilities), set(Category))
        self.assertAlmostEqual(sum(p.probabilities.values()), 1.0)

    def test_deterministic(self):
        self.assertEqual(self.clf.predict("Toxic", "w1x1 w2x2"), self.clf.predict("Toxic", "w1x1 w2x2"))

    def test_min_confidence_is_off_by_default_and_falls_back_to_needs_context(self):
        text = "w0x0 w0x1"
        plain = self.clf.predict("Toxic", text)
        self.assertEqual(self.clf.predict("Toxic", text, min_confidence=0.0).label, plain.label)
        self.assertEqual(self.clf.predict("Toxic", text, min_confidence=1.01).label, Category.NEEDS_CONTEXT)

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            self.clf.predict("Toxic", "  ")


class ModelFileProblems(TempDirTest):
    def test_missing_file(self):
        with self.assertRaisesRegex(ClassifierError, "not found"):
            Classifier.load(self.dir / "nope.joblib")

    def test_garbage_file(self):
        bad = self.dir / "bad.joblib"
        bad.write_bytes(b"this is not a model")
        with self.assertRaisesRegex(ClassifierError, "could not be read"):
            Classifier.load(bad)

    def test_not_a_pipeline(self):
        path = self.dir / "dict.joblib"
        joblib.dump({"a": 1}, path)
        with self.assertRaisesRegex(ClassifierError, "Pipeline"):
            Classifier.load(path)

    def test_foreign_labels_rejected(self):
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
        pipe.fit(["aa bb", "cc dd", "aa cc", "bb dd"], ["X", "Y", "X", "Y"])
        path = self.dir / "foreign.joblib"
        joblib.dump(pipe, path)
        with self.assertRaisesRegex(ClassifierError, "unknown labels"):
            Classifier.load(path)


if __name__ == "__main__":
    unittest.main()
