"""Tests for scripts/generate_candidates.py and scripts/promote_candidates.py.

Everything happens in temp directories: the real data files are never read or written here,
except test_builtin_templates_match_the_real_dictionary, which only reads data/terminology.json.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from app.dictionary import load_entries
from scripts import generate_candidates as gen
from scripts import promote_candidates as pro
from scripts import train_classifier as tc
from tests.test_classifier import write_csv

TERMS = {"Toxic": ["toxic"], "Trauma": ["trauma"], "Gaslighting": ["gaslighting"], "Zebra": ["zebra"]}
N_TEMPLATES = sum(len(gen.TEMPLATES[t]) for t in ("Toxic", "Trauma", "Gaslighting"))  # Zebra has none


def entry(term, patterns):
    return {"id": term.lower(), "term": term, "term_category": "Test", "definition": "d", "guidance": "g",
            "misuse_examples": ["a"], "legitimate_examples": ["b"], "patterns": patterns,
            "misuse_clues": [], "plausible_clues": [], "humor": None, "source_title": None, "source_url": None}


def row(term, text, label, status=tc.REVIEWED):
    return {"term": term, "text": text, "label": label, "review_status": status}


def cand(term, text, label="", status=tc.NEEDS_REVIEW, suggested="NEEDS CONTEXT", context="ambiguous"):
    return {**row(term, text, label, status), "suggested_label": suggested, "context": context}


def toxic_template(context):
    return next(t for c, _label, t in gen.TEMPLATES["Toxic"] if c == context)


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.terms = self.dir / "terminology.json"
        self.terms.write_text(json.dumps([entry(t, p) for t, p in TERMS.items()]), encoding="utf-8")
        self.training = self.dir / "training.csv"
        self.heldout = self.dir / "heldout.csv"
        self.out = self.dir / "candidates.csv"
        write_csv(self.training, [row("Toxic", "A training sentence about nothing in particular.", "NEEDS CONTEXT")])
        write_csv(self.heldout, [row("Trauma", "A held-out sentence that stays separate.", "NEEDS CONTEXT", tc.NEEDS_REVIEW)])

    def call(self, main, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def gen(self, *extra, output=None):
        return self.call(gen.main, "--output", str(output or self.out), "--training", str(self.training),
                         "--heldout", str(self.heldout), "--terminology", str(self.terms), *extra)

    def bytes(self, path):
        return Path(path).read_bytes()


class Generation(Base):
    def test_generates_schema_compatible_needs_review_rows_with_empty_labels(self):
        code, out, err = self.gen()
        self.assertEqual((code, err), (0, ""))
        self.assertIn(f"Wrote {N_TEMPLATES} new candidate(s)", out)
        rows = gen.read_candidates(self.out)  # also checks the exact header
        self.assertEqual(len(rows), N_TEMPLATES)
        for r in rows:
            self.assertEqual(r["review_status"], tc.NEEDS_REVIEW)
            self.assertEqual(r["label"], "")
            self.assertIn(r["suggested_label"], tc.LABELS)
            self.assertIn(r["context"], gen.CONTEXTS)
        self.assertEqual(len({r["text"].lower() for r in rows}), len(rows))

    def test_training_loader_reads_it_but_validation_refuses_unlabelled_rows(self):
        self.gen()
        rows = tc.load_rows(self.out)  # same four core columns
        self.assertEqual(tc.reviewed_rows(rows), [])  # nothing is reviewed
        with self.assertRaisesRegex(tc.DatasetError, "'label' is empty"):
            tc.validate(rows)  # so it cannot be trained on by accident

    def test_covers_each_context_and_skips_terms_without_templates(self):
        _, out, _ = self.gen()
        contexts = {r["context"] for r in gen.read_candidates(self.out)}
        self.assertEqual(contexts, set(gen.CONTEXTS))
        self.assertIn("Terms with no templates (skipped): Zebra", out)
        self.assertNotIn("Zebra", {r["term"] for r in gen.read_candidates(self.out)})

    def test_builtin_templates_match_the_real_dictionary(self):
        pool, report = gen.build_pool(load_entries())
        self.assertEqual(report["dropped"], [])
        self.assertEqual(report["unknown_template_terms"], [])
        self.assertTrue(pool)
        for term, templates in gen.TEMPLATES.items():
            for context, label, _text in templates:
                self.assertIn(context, gen.CONTEXTS, term)
                self.assertIn(label, tc.LABELS, term)

    def test_template_missing_its_term_is_dropped_not_written(self):
        pool, report = gen.build_pool([type("E", (), {"term": "Toxic", "patterns": ("zzz",)})()])
        self.assertEqual(pool, [])
        self.assertEqual(len(report["dropped"]), len(gen.TEMPLATES["Toxic"]))

    def test_dry_run_writes_nothing(self):
        code, out, _ = self.gen("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("Would write", out)
        self.assertFalse(self.out.exists())


class Protection(Base):
    def test_never_modifies_training_or_heldout(self):
        before = (self.bytes(self.training), self.bytes(self.heldout))
        self.gen()
        self.gen()
        self.assertEqual((self.bytes(self.training), self.bytes(self.heldout)), before)

    def test_refuses_to_write_into_training_or_heldout(self):
        for target in (self.training, self.heldout):
            with self.subTest(target.name):
                before = self.bytes(target)
                code, _, err = self.gen(output=target)
                self.assertEqual(code, 1)
                self.assertIn("Refusing to write", err)
                self.assertEqual(self.bytes(target), before)

    def test_existing_candidate_rows_are_preserved_byte_for_byte(self):
        write_csv(self.out, [cand("Toxic", "My own hand-written candidate.", "PLAUSIBLE USAGE", tc.REVIEWED)], fields=gen.COLUMNS)
        before = self.bytes(self.out)
        self.gen()
        self.assertTrue(self.bytes(self.out).startswith(before))
        self.assertEqual(len(gen.read_candidates(self.out)), 1 + N_TEMPLATES)

    def test_append_to_file_without_trailing_newline(self):
        write_csv(self.out, [cand("Toxic", "No trailing newline here.")], fields=gen.COLUMNS)
        self.out.write_bytes(self.bytes(self.out).rstrip(b"\r\n"))
        self.gen()
        self.assertEqual(len(gen.read_candidates(self.out)), 1 + N_TEMPLATES)

    def test_unexpected_header_is_refused_and_untouched(self):
        self.out.write_text("a,b\n1,2\n", encoding="utf-8")
        before = self.bytes(self.out)
        code, _, err = self.gen()
        self.assertEqual(code, 1)
        self.assertIn("unexpected header", err)
        self.assertEqual(self.bytes(self.out), before)


class Duplicates(Base):
    def test_exact_duplicate_of_training_ignoring_case_and_spaces(self):
        text = toxic_template("academic")
        write_csv(self.training, [row("Toxic", "  " + text.upper() + "  ", "PLAUSIBLE USAGE")])
        _, out, _ = self.gen()
        self.assertIn("Skipped 1 exact duplicate(s): 1 in training", out)
        self.assertNotIn(text.lower(), {r["text"].lower() for r in gen.read_candidates(self.out)})
        self.assertEqual(len(gen.read_candidates(self.out)), N_TEMPLATES - 1)

    def test_exact_duplicate_of_heldout(self):
        write_csv(self.heldout, [row("Toxic", toxic_template("negation"), "NEEDS CONTEXT", tc.NEEDS_REVIEW)])
        _, out, _ = self.gen()
        self.assertIn("1 in held-out", out)
        self.assertEqual(len(gen.read_candidates(self.out)), N_TEMPLATES - 1)

    def test_rejected_candidates_are_not_regenerated(self):
        text = toxic_template("casual")
        write_csv(self.out, [cand("Toxic", text, status=gen.REJECTED)], fields=gen.COLUMNS)
        _, out, _ = self.gen()
        self.assertIn("1 in earlier candidates", out)
        self.assertEqual(sum(r["text"] == text for r in gen.read_candidates(self.out)), 1)

    def test_repeated_runs_do_not_accumulate_duplicates(self):
        self.gen()
        first = self.bytes(self.out)
        _, out, _ = self.gen()
        self.assertIn("Wrote 0 new candidate(s)", out)
        self.assertIn(f"Skipped {N_TEMPLATES} exact duplicate(s)", out)
        self.assertEqual(self.bytes(self.out), first)

    def test_word_overlap_warning_is_reported_but_row_is_still_written(self):
        text = toxic_template("academic")
        write_csv(self.training, [row("Toxic", text + " indeed", "PLAUSIBLE USAGE")])  # not exact, nearly the same words
        _, out, _ = self.gen()
        self.assertIn("Word-overlap warnings: 1", out)
        self.assertIn("not a semantic check", out)
        self.assertIn(text, {r["text"] for r in gen.read_candidates(self.out)})

    def test_does_not_claim_semantic_detection(self):
        _, out, _ = self.gen()
        self.assertIn("not paraphrases", out)


class MissingOrEmptyInputs(Base):
    def test_missing_training_and_heldout_are_warnings(self):
        self.training.unlink()
        self.heldout.unlink()
        code, out, _ = self.gen()
        self.assertEqual(code, 0)
        self.assertEqual(out.count("WARNING"), 2)
        self.assertEqual(len(gen.read_candidates(self.out)), N_TEMPLATES)
        self.assertFalse(self.training.exists())  # not created either

    def test_empty_files_are_warnings_too(self):
        self.training.write_bytes(b"")
        self.out.write_bytes(b"")  # empty candidates file = no candidates yet
        code, out, _ = self.gen()
        self.assertEqual(code, 0)
        self.assertIn("missing or empty", out)
        self.assertEqual(len(gen.read_candidates(self.out)), N_TEMPLATES)

    def test_missing_terminology_is_an_error(self):
        self.terms.unlink()
        code, _, err = self.gen()
        self.assertEqual(code, 1)
        self.assertIn("not found", err)
        self.assertFalse(self.out.exists())

    def test_malformed_training_file_is_an_error(self):
        self.training.write_text("x,y\n1,2\n", encoding="utf-8")
        code, _, err = self.gen()
        self.assertEqual(code, 1)
        self.assertIn("missing column", err)
        self.assertFalse(self.out.exists())


class Coverage(Base):
    def test_reports_reviewed_counts_candidate_counts_and_gaps(self):
        write_csv(self.out, [cand("Toxic", "Pending one.", suggested="POSSIBLE MISUSE"),
                             cand("Toxic", "Approved one.", "PLAUSIBLE USAGE", tc.REVIEWED),
                             cand("Toxic", "Rejected one.", status=gen.REJECTED)], fields=gen.COLUMNS)
        before = self.bytes(self.out)
        code, out, _ = self.gen("--coverage-only")
        self.assertEqual(code, 0)
        self.assertIn("Coverage by term", out)
        self.assertIn("Toxic: 1 pending, 1 approved (not yet in training), 1 rejected", out)
        self.assertIn("Zebra: 0 pending", out)
        self.assertIn("no reviewed PLAUSIBLE USAGE (0 pending suggest it)", out)
        self.assertIn("NO REVIEWED ROWS", out)  # Zebra, Gaslighting and Trauma have no reviewed training rows
        self.assertIn("no minimum implied", out)
        self.assertEqual(self.bytes(self.out), before)  # coverage-only writes nothing

    def test_gap_line_counts_pending_candidates_that_suggest_the_label(self):
        lines = gen.candidate_lines(
            [row("Toxic", "t1", "NEEDS CONTEXT")],
            [cand("Toxic", "c1", suggested="POSSIBLE MISUSE"), cand("Toxic", "c2", suggested="POSSIBLE MISUSE")],
            ["Toxic"])
        toxic = next(line for line in lines if line.startswith("  Toxic:"))
        self.assertIn("no reviewed POSSIBLE MISUSE (2 pending suggest it)", toxic)
        self.assertNotIn("no reviewed NEEDS CONTEXT", toxic)

    def test_coverage_makes_no_quality_claim(self):
        _, out, _ = self.gen()
        self.assertIn("gaps, not quality", out)
        self.assertNotIn("sufficient", out.lower().replace("insufficient", ""))


class Promotion(Base):
    def setUp(self):
        super().setUp()
        self.cands = [
            cand("Toxic", "Approved and fine.", "PLAUSIBLE USAGE", tc.REVIEWED),
            cand("Trauma", "Approved with a changed label.", "NEEDS CONTEXT", tc.REVIEWED, suggested="POSSIBLE MISUSE"),
            cand("Toxic", "Still waiting for a human."),
            cand("Toxic", "Rejected by a human.", status=gen.REJECTED),
        ]
        write_csv(self.out, self.cands, fields=gen.COLUMNS)

    def pro(self, *extra):
        return self.call(pro.main, "--candidates", str(self.out), "--training", str(self.training),
                         "--heldout", str(self.heldout), "--terminology", str(self.terms), *extra)

    def test_dry_run_is_the_default(self):
        before = self.bytes(self.training)
        code, out, _ = self.pro()
        self.assertEqual(code, 0)
        self.assertIn("Dry run", out)
        self.assertIn("2 approved candidate(s) ready; 1 still needs_review", out)
        self.assertEqual(self.bytes(self.training), before)

    def test_apply_appends_only_approved_rows_and_keeps_existing_bytes(self):
        before = self.bytes(self.training)
        code, _, _ = self.pro("--apply")
        self.assertEqual(code, 0)
        self.assertTrue(self.bytes(self.training).startswith(before))
        rows = tc.load_rows(self.training)
        self.assertEqual([r["text"] for r in rows[1:]], ["Approved and fine.", "Approved with a changed label."])
        self.assertEqual([r["label"] for r in rows[1:]], ["PLAUSIBLE USAGE", "NEEDS CONTEXT"])  # the human's label, not the suggestion
        self.assertTrue(all(r["review_status"] == tc.REVIEWED for r in rows[1:]))
        tc.validate(rows, set(TERMS))

    def test_unreviewed_and_rejected_never_promoted(self):
        self.pro("--apply")
        texts = {r["text"] for r in tc.load_rows(self.training)}
        self.assertNotIn("Still waiting for a human.", texts)
        self.assertNotIn("Rejected by a human.", texts)

    def test_problems_stop_everything(self):
        cases = {
            "approved but label": cand("Toxic", "No label given.", "", tc.REVIEWED),
            "is not one of needs_review": cand("Toxic", "Typo status.", "NEEDS CONTEXT", "approved"),
            "unknown term": cand("Nonsense", "Wrong term.", "NEEDS CONTEXT", tc.REVIEWED),
        }
        for msg, bad in cases.items():
            with self.subTest(msg):
                write_csv(self.out, self.cands + [bad], fields=gen.COLUMNS)
                before = self.bytes(self.training)
                code, _, err = self.pro("--apply")
                self.assertEqual(code, 1)
                self.assertIn(msg, err)
                self.assertEqual(self.bytes(self.training), before)

    def test_malformed_candidate_rows_stop_promotion(self):
        self.out.write_text(",".join(gen.COLUMNS) + "\nToxic,only three fields\n", encoding="utf-8")  # row is short
        before = self.bytes(self.training)
        code, _, err = self.pro("--apply")
        self.assertEqual(code, 1)  # a missing review_status is a problem, not a silent skip
        self.assertIn("review_status '' is not one of", err)
        self.assertEqual(self.bytes(self.training), before)
        self.out.write_text(",".join(gen.COLUMNS) + "\nToxic,half a row,PLAUSIBLE USAGE,reviewed\n", encoding="utf-8")
        code, _, _ = self.pro("--apply")  # a row that is complete in the four core columns still promotes
        self.assertEqual(code, 0)
        self.assertEqual(tc.load_rows(self.training)[-1]["text"], "half a row")

    def test_duplicate_of_training_is_skipped(self):
        write_csv(self.out, [cand("Toxic", "A TRAINING sentence about nothing in particular.", "NEEDS CONTEXT", tc.REVIEWED)], fields=gen.COLUMNS)
        _, out, _ = self.pro("--apply")
        self.assertIn("already in training", out)
        self.assertEqual(len(tc.load_rows(self.training)), 1)

    def test_heldout_overlap_is_blocked(self):
        held = "A held-out sentence that stays separate."
        for text in (held, held + " honestly"):  # identical, and nearly identical
            with self.subTest(text):
                write_csv(self.out, [cand("Trauma", text, "NEEDS CONTEXT", tc.REVIEWED)], fields=gen.COLUMNS)
                _, out, _ = self.pro("--apply")
                self.assertIn("BLOCKED", out)
                self.assertEqual(len(tc.load_rows(self.training)), 1)

    def test_applying_twice_adds_nothing_the_second_time(self):
        self.pro("--apply")
        once = self.bytes(self.training)
        code, out, _ = self.pro("--apply")
        self.assertEqual(code, 0)
        self.assertIn("0 approved candidate(s) ready", out)
        self.assertEqual(self.bytes(self.training), once)

    def test_candidate_file_is_not_modified(self):
        before = self.bytes(self.out)
        self.pro("--apply")
        self.assertEqual(self.bytes(self.out), before)

    def test_missing_files_are_errors(self):
        self.out.unlink()
        self.assertEqual(self.pro()[0], 1)
        write_csv(self.out, self.cands, fields=gen.COLUMNS)
        self.training.unlink()
        code, _, err = self.pro()
        self.assertEqual(code, 1)
        self.assertIn("not found", err)
        self.assertFalse(self.training.exists())


if __name__ == "__main__":
    unittest.main()
