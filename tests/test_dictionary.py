import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.dictionary import DEFAULT_PATH, DictionaryError, load_entries, resource_path

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


class ResourcePath(unittest.TestCase):
    def test_source_mode_points_at_repo_data_file(self):
        self.assertTrue(DEFAULT_PATH.is_file(), DEFAULT_PATH)
        self.assertTrue(DEFAULT_PATH.is_absolute())

    def test_source_mode_ignores_working_directory(self):
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                self.assertEqual(len(load_entries(resource_path("data", "terminology.json"))), 15)
            finally:
                os.chdir(old)

    def test_frozen_mode_uses_meipass_not_module_location(self):
        # Simulates PyInstaller --onedir: data under <dist>/_internal/data/.
        with tempfile.TemporaryDirectory() as tmp:
            internal = Path(tmp) / "dist" / "app" / "_internal"
            (internal / "data").mkdir(parents=True)
            shutil.copy(DEFAULT_PATH, internal / "data" / "terminology.json")
            with mock.patch.object(sys, "frozen", True, create=True), \
                 mock.patch.object(sys, "_MEIPASS", str(internal), create=True):
                path = resource_path("data", "terminology.json")
                self.assertEqual(path, internal / "data" / "terminology.json")
                self.assertEqual(len(load_entries(path)), 15)

    def test_frozen_mode_missing_file_gives_useful_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(sys, "_MEIPASS", tmp, create=True):
            with self.assertRaisesRegex(DictionaryError, "not found"):
                load_entries(resource_path("data", "terminology.json"))


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
