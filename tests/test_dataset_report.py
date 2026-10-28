"""Tests for scripts/dataset_report.py and for design properties of the candidate templates.

Temp directories only. The report never writes, and these tests check that.
"""
import contextlib
import io
import json
import re
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

from app.dictionary import load_entries
from scripts import dataset_report as rep
from scripts import generate_candidates as gen
from scripts import train_classifier as tc
from tests.test_candidates import Base, TERMS, cand, row
from tests.test_classifier import write_csv

E = namedtuple("E", "term patterns misuse_clues plausible_clues")
ALL_TEXTS = [(term, ctx, label, text) for term, items in gen.TEMPLATES.items() for ctx, label, text in items]


class TemplateDesign(unittest.TestCase):
    """Properties the redesigned template library is meant to have (not sentence counts)."""

    def test_no_duplicate_sentences(self):
        keys = [gen._key(t) for *_, t in ALL_TEXTS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_sentence_is_the_same_frame_with_another_term_slotted_in(self):
        entries = load_entries()
        rows = [{"term": term, "text": text} for term, _c, _l, text in ALL_TEXTS]
        pairs = rep.frame_pairs(rows, {e.term: tuple(e.patterns) for e in entries})
        self.assertEqual(pairs, [])

    def test_every_label_appears_with_and_without_negation(self):
        neg = re.compile(rep.CUES["negation"], re.IGNORECASE)
        for label in tc.LABELS:
            with_neg = [t for _t, _c, lb, t in ALL_TEXTS if lb == label and neg.search(t)]
            without = [t for _t, _c, lb, t in ALL_TEXTS if lb == label and not neg.search(t)]
            self.assertTrue(with_neg and without, label)

    def test_negation_is_not_tied_to_one_suggested_label(self):
        labels = {lb for _t, c, lb, _x in ALL_TEXTS if c == "negation"}
        self.assertGreater(len(labels), 1)

    def test_similar_contexts_carry_different_labels(self):
        by_context = {}
        for _t, c, lb, _x in ALL_TEXTS:
            by_context.setdefault(c, set()).add(lb)
        for context in ("casual", "negation", "question", "reported", "disagreement"):
            self.assertGreater(len(by_context[context]), 1, context)

    def test_informal_terms_never_get_a_misuse_guess(self):
        for term in ("Triggered", "Red flag", "Emotional damage"):
            self.assertNotIn(tc.LABELS[2], {lb for _c, lb, _t in gen.TEMPLATES[term]}, term)

    def test_generation_is_deterministic(self):
        outputs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                terms = Path(tmp) / "t.json"
                terms.write_text(json.dumps([_entry(t, p) for t, p in TERMS.items()]), encoding="utf-8")
                out = Path(tmp) / "c.csv"
                with contextlib.redirect_stdout(io.StringIO()):
                    gen.main(["--output", str(out), "--training", str(Path(tmp) / "none.csv"),
                              "--heldout", str(Path(tmp) / "none2.csv"), "--terminology", str(terms)])
                outputs.append(out.read_bytes())
        self.assertEqual(outputs[0], outputs[1])


def _entry(term, patterns):
    return {"id": term.lower(), "term": term, "term_category": "T", "definition": "d", "guidance": "g",
            "misuse_examples": ["a"], "legitimate_examples": ["b"], "patterns": patterns,
            "misuse_clues": ["rude"], "plausible_clues": ["for years"], "humor": None, "source_title": None, "source_url": None}


class Building(unittest.TestCase):
    def test_masked_replaces_term_patterns_only(self):
        self.assertEqual(rep.masked("He said it was Toxic Positivity, not toxic.", ("toxic positivity", "toxic")),
                         "He said it was <T>, not <T>.")
        self.assertEqual(rep.masked("A toxicology report.", ("toxic",)), "A toxicology report.")

    def test_exact_duplicates_across_files_ignore_case_and_spaces(self):
        groups = rep.exact_duplicates({"training": [row("Toxic", "Same text here.", "NEEDS CONTEXT")],
                                       "candidates": [cand("Toxic", "x"), cand("Toxic", "  SAME TEXT HERE.  ")]})
        self.assertEqual(groups, [[("training", 2), ("candidates", 3)]])

    def test_near_pairs_use_word_overlap_and_ignore_unrelated(self):
        items = [("a", "he disagreed with me about the film ending"), ("b", "he disagreed with me about the film"),
                 ("c", "the solvent needs a fume hood")]
        pairs = rep.near_pairs(items)
        self.assertEqual([(a, b) for _s, a, b in pairs], [("a", "b")])
        self.assertEqual(rep.near_pairs([("a", "x y z")], other=[("b", "q r s")]), [])

    def test_frame_pairs_need_different_terms(self):
        patterns = {"Toxic": ("toxic",), "Trauma": ("trauma",)}
        rows = [{"term": "Toxic", "text": "My cousin said the club was toxic last night."},
                {"term": "Trauma", "text": "My cousin said the club was trauma last night."},
                {"term": "Toxic", "text": "My cousin said the club was toxic last night!"}]
        pairs = rep.frame_pairs(rows, patterns)
        self.assertEqual(len(pairs), 2)  # Toxic~Trauma twice; the two Toxic rows are not paired with each other

    def test_repeated_openings_and_phrases(self):
        texts = ["She told me a story.", "She told me nothing.", "She told me twice.", "Another start entirely."]
        self.assertEqual(rep.repeated_openings(texts), [("she told me", 3)])
        self.assertEqual(rep.repeated_openings(texts, min_count=4), [])
        self.assertEqual(rep.repeated_phrases(["a b c d e", "a b c d f", "a b c d g"]), [("a b c d", 3)])

    def test_cue_table_counts_by_label(self):
        rows = [cand("Toxic", "It was not fair.", suggested="NEEDS CONTEXT"), cand("Toxic", "He never did.", suggested="PLAUSIBLE USAGE"),
                cand("Toxic", "Is this fine?", suggested="NEEDS CONTEXT")]
        table = {name: (n, dict(by)) for name, n, by in rep.cue_table(rows, "suggested_label")}
        self.assertEqual(table["negation"], (2, {"NEEDS CONTEXT": 1, "PLAUSIBLE USAGE": 1}))
        self.assertEqual(table["question"], (1, {"NEEDS CONTEXT": 1}))

    def test_clue_verdict_and_echo(self):
        entry = E("Toxic", ("toxic",), ("rude",), ("for years",))
        self.assertEqual(rep.clue_verdict("He is rude.", entry), "POSSIBLE MISUSE")
        self.assertEqual(rep.clue_verdict("It went on for years.", entry), "PLAUSIBLE USAGE")
        self.assertEqual(rep.clue_verdict("Rude, for years.", entry), "NEEDS CONTEXT")  # opposing clues cancel
        self.assertEqual(rep.clue_verdict("Nothing here.", entry), "NEEDS CONTEXT")
        rows = [{"term": "Toxic", "text": "He is rude.", "label": "POSSIBLE MISUSE"},
                {"term": "Toxic", "text": "He is rude.", "label": "PLAUSIBLE USAGE"},
                {"term": "Other", "text": "x", "label": "NEEDS CONTEXT"}]
        ok, n, _ = rep.clue_echo(rows, [entry], "label")
        self.assertEqual((ok, n), (1, 2))  # unknown term skipped

    def test_context_table(self):
        rows = [cand("T", "a", context="casual", suggested="POSSIBLE MISUSE"), cand("T", "b", context="casual", suggested="NEEDS CONTEXT")]
        self.assertEqual(rep.context_table(rows), {"casual": {"POSSIBLE MISUSE": 1, "NEEDS CONTEXT": 1}})


class ReportCommand(Base):
    def run_report(self, *extra):
        return self.call(rep.main, "--training", str(self.training), "--heldout", str(self.heldout),
                         "--candidates", str(self.out), "--terminology", str(self.terms), *extra)

    def test_covers_requested_sections(self):
        self.gen()
        code, out, err = self.run_report()
        self.assertEqual((code, err), (0, ""))
        for needle in ("Coverage by term", "Candidate review status: needs_review", "Exact duplicates", "Near-duplicate",
                       "Same-sentence-different-term", "Candidates overlapping training", "Candidates overlapping held-out",
                       "Repeated openings", "context vs suggested label", "Cues in candidates", "clue-list echo", "Warnings"):
            self.assertIn(needle, out)
        self.assertIn("do not show it is good", out)

    def test_status_counts_reviewed_pending_rejected(self):
        write_csv(self.out, [cand("Toxic", "a1"), cand("Toxic", "a2", "PLAUSIBLE USAGE", tc.REVIEWED),
                             cand("Toxic", "a3", status=gen.REJECTED), cand("Toxic", "a4")], fields=gen.COLUMNS)
        _, out, _ = self.run_report()
        self.assertIn("Candidate review status: needs_review 2, rejected 1, reviewed 1", out)

    def test_reports_overlap_duplicates_and_missing_labels_as_warnings(self):
        text = "A very specific sentence about a rainy afternoon and the bus."
        write_csv(self.training, [row("Toxic", text, "NEEDS CONTEXT")])
        write_csv(self.heldout, [row("Toxic", "Held out sentence about the green door and the old key.", "NEEDS CONTEXT", tc.NEEDS_REVIEW)])
        write_csv(self.out, [cand("Toxic", text.upper()), cand("Toxic", "Held out sentence about the green door and the old key today.")],
                  fields=gen.COLUMNS)
        _, out, _ = self.run_report()
        self.assertIn("Exact duplicates (ignoring case and spaces): 1 group(s)", out)
        self.assertIn("Candidates overlapping training: 1 by word overlap, 1 exact", out)
        self.assertIn("Candidates overlapping held-out: 1 by word overlap, 0 exact", out)
        self.assertIn("lack reviewed training rows for at least one label", out)
        self.assertIn("not a quota", out)

    def test_flags_a_context_and_a_cue_tied_to_one_label(self):
        write_csv(self.out, [cand("Toxic", "Is it? one", context="question", suggested="NEEDS CONTEXT"),
                             cand("Toxic", "Is it? two", context="question", suggested="NEEDS CONTEXT")], fields=gen.COLUMNS)
        _, out, _ = self.run_report()
        self.assertIn("context 'question' maps to one suggested label (NEEDS CONTEXT)", out)
        self.assertIn("cue 'question' appears only with NEEDS CONTEXT among candidates (n=2)", out)

    def test_tiny_samples_do_not_trigger_the_clue_echo_warning(self):
        _, out, _ = self.run_report()  # the fixture has 1-2 rows, all agreeing with the clue lists
        self.assertNotIn("relearn the scanner", out)

    def test_read_only_and_deterministic(self):
        self.gen()
        before = {p: self.bytes(p) for p in (self.training, self.heldout, self.out)}
        first = self.run_report()
        second = self.run_report()
        self.assertEqual(first, second)
        self.assertEqual({p: self.bytes(p) for p in before}, before)

    def test_missing_and_empty_inputs(self):
        self.training.unlink()
        self.heldout.write_bytes(b"")
        code, out, _ = self.run_report()  # candidates file missing too
        self.assertEqual(code, 0)
        self.assertEqual(out.count("WARNING"), 2)
        self.assertIn("candidates 0", out)
        self.assertFalse(self.out.exists())

    def test_malformed_inputs_are_errors_not_crashes(self):
        self.out.write_text("a,b\n1,2\n", encoding="utf-8")
        code, _, err = self.run_report()
        self.assertEqual((code, err.startswith("ERROR:")), (1, True))
        self.out.unlink()
        self.terms.unlink()
        code, _, err = self.run_report()
        self.assertEqual((code, "not found" in err), (1, True))

    def test_rows_with_invalid_labels_or_missing_fields_do_not_crash_the_report(self):
        self.training.write_text("term,text,label,review_status\nToxic,Some text,MAYBE,reviewed\nToxic\n", encoding="utf-8")
        self.out.write_text(",".join(gen.COLUMNS) + "\nToxic,only a few fields\n", encoding="utf-8")
        code, out, _ = self.run_report()
        self.assertEqual(code, 0)
        self.assertIn("Warnings", out)


if __name__ == "__main__":
    unittest.main()
