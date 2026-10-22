import json
import tempfile
import unittest
from pathlib import Path

from app.dictionary import DictionaryError, load_entries

EXPECTED_TERMS = {
    "Gaslighting", "Narcissist", "Toxic", "Trauma", "Triggered", "Red flag",
    "Love bombing", "Manipulative", "Boundaries", "Toxic positivity",
    "Emotional damage", "Attachment issues", "Gaslight", "Narcissistic abuse",
    "Trauma dumping",
}


def _good_entry(**overrides):
    entry = {
        "id": "x", "term": "X", "term_category": "Test", "definition": "d", "guidance": "g",
        "misuse_examples": ["a"], "legitimate_examples": ["b"], "patterns": ["x"],
        "misuse_clues": [], "plausible_clues": [], "humor": None,
        "source_title": None, "source_url": None,
    }
    entry.update(overrides)
    return entry


class BundledDictionary(unittest.TestCase):
    def test_exactly_15_valid_entries(self):
        entries = load_entries()
        self.assertEqual(len(entries), 15)
        self.assertEqual({e.term for e in entries}, EXPECTED_TERMS)
        self.assertEqual(len({e.id for e in entries}), 15)

    def test_every_entry_has_examples_and_patterns(self):
        for e in load_entries():
            self.assertTrue(e.misuse_examples and e.legitimate_examples and e.patterns, e.id)


class MalformedData(unittest.TestCase):
    def load(self, content: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            path.write_text(content, encoding="utf-8")
            return load_entries(path)

    def test_valid_minimal_entry_loads(self):
        self.assertEqual(len(self.load(json.dumps([_good_entry()]))), 1)

    def test_missing_file(self):
        with self.assertRaisesRegex(DictionaryError, "not found"):
            load_entries(Path("does-not-exist.json"))

    def test_invalid_json(self):
        with self.assertRaisesRegex(DictionaryError, "not valid JSON"):
            self.load("{nope")

    def test_root_not_a_list(self):
        with self.assertRaisesRegex(DictionaryError, "list of entries"):
            self.load("{}")

    def test_missing_field_names_the_field(self):
        bad = _good_entry()
        del bad["definition"]
        with self.assertRaisesRegex(DictionaryError, "'definition'"):
            self.load(json.dumps([bad]))

    def test_empty_patterns_rejected(self):
        with self.assertRaisesRegex(DictionaryError, "'patterns'"):
            self.load(json.dumps([_good_entry(patterns=[])]))

    def test_bad_url_rejected(self):
        bad = _good_entry(source_title="T", source_url="ftp://example.com")
        with self.assertRaisesRegex(DictionaryError, "source_url"):
            self.load(json.dumps([bad]))

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(DictionaryError, "Duplicate"):
            self.load(json.dumps([_good_entry(), _good_entry()]))


if __name__ == "__main__":
    unittest.main()
