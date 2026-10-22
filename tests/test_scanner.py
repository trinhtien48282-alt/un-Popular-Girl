import unittest

from app.dictionary import load_entries
from app.models import Category
from app.scanner import Scanner


class ScannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = load_entries()
        cls.scanner = Scanner(cls.entries)

    def ids(self, text):
        return [m.entry.id for m in self.scanner.scan(text)]

    def test_empty_and_whitespace(self):
        self.assertEqual(self.scanner.scan(""), [])
        self.assertEqual(self.scanner.scan("  \n\t "), [])

    def test_unrelated_text(self):
        self.assertEqual(self.scanner.scan("The weather is lovely and the soup is hot."), [])

    def test_case_insensitive(self):
        m = self.scanner.scan("GaSlIgHtInG")
        self.assertEqual([x.entry.id for x in m], ["gaslighting"])
        self.assertEqual(m[0].text, "GaSlIgHtInG")  # original casing preserved

    def test_multiword_variants(self):
        for text in ("love bombing", "Love-Bombing", "love   bombing", "red flag", "RED-FLAG"):
            self.assertEqual(len(self.scanner.scan(text)), 1, text)

    def test_word_boundaries(self):
        for text in ("untoxic", "toxicology", "traumatized", "redflagged", "xtrauma", "boundariesless"):
            self.assertEqual(self.scanner.scan(text), [], text)

    def test_punctuation_is_a_boundary(self):
        self.assertEqual(self.ids("(toxic) 'trauma', red flag!"), ["toxic", "trauma", "red-flag"])

    def test_multiple_occurrences_and_positions(self):
        text = "Toxic this, toxic that."
        m = self.scanner.scan(text)
        self.assertEqual([(x.start, x.end) for x in m], [(0, 5), (12, 17)])
        for x in m:
            self.assertEqual(text[x.start:x.end], x.text)

    def test_context_surrounds_match(self):
        text = "x" * 100 + " toxic " + "y" * 100
        m = self.scanner.scan(text)[0]
        self.assertEqual(text[m.start:m.end], "toxic")
        self.assertEqual(len(m.before), 40)
        self.assertEqual(len(m.after), 40)
        self.assertEqual(m.context, text[m.start - 40:m.end + 40])

    def test_context_at_text_edges(self):
        m = self.scanner.scan("toxic")[0]
        self.assertEqual((m.before, m.after), ("", ""))

    def test_overlap_longest_wins(self):
        self.assertEqual(self.ids("toxic positivity"), ["toxic-positivity"])
        self.assertEqual(self.ids("trauma dumping"), ["trauma-dumping"])
        self.assertEqual(self.ids("narcissistic abuse"), ["narcissistic-abuse"])
        self.assertEqual(self.ids("gaslighting"), ["gaslighting"])
        self.assertEqual(self.ids("to gaslight someone"), ["gaslight"])

    def test_overlap_does_not_hide_neighbours(self):
        self.assertEqual(self.ids("toxic positivity and trauma"), ["toxic-positivity", "trauma"])

    def test_bare_match_needs_context(self):
        m = self.scanner.scan("Trauma.")[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)

    def test_misuse_clue(self):
        m = self.scanner.scan("He disagreed with me, that's gaslighting.")[0]
        self.assertEqual(m.category, Category.POSSIBLE_MISUSE)
        self.assertIn("disagree", m.reason)

    def test_plausible_not_flagged_as_misuse(self):
        cases = [
            "My therapist helped me process trauma after the accident.",
            "The factory dumped toxic waste.",
            "He denied it happened and made me doubt my memory, which is gaslighting.",
        ]
        for text in cases:
            self.assertEqual(self.scanner.scan(text)[0].category, Category.PLAUSIBLE_USAGE, text)

    def test_mixed_clues_need_context(self):
        text = "He disagreed with me and kept saying it never happened, which is gaslighting."
        self.assertEqual(self.scanner.scan(text)[0].category, Category.NEEDS_CONTEXT)

    def test_clues_only_count_within_same_sentence(self):
        text = "He disagreed with me. Later I felt gaslighting was too strong a word."
        self.assertEqual(self.scanner.scan(text)[0].category, Category.NEEDS_CONTEXT)

    def test_informal_terms_never_flagged_as_misuse(self):
        for text in ("I'm so triggered lol", "emotional damage, lol", "that is a red flag"):
            for m in self.scanner.scan(text):
                self.assertNotEqual(m.category, Category.POSSIBLE_MISUSE, text)

    def test_curly_apostrophe_clue(self):
        m = self.scanner.scan("He didn\u2019t text back, that's gaslighting.")[0]
        self.assertEqual(m.category, Category.POSSIBLE_MISUSE)

    def test_deterministic(self):
        text = "Toxic positivity, trauma dumping, a red flag, and gaslighting. Toxic!"
        first = self.scanner.scan(text)
        for _ in range(5):
            self.assertEqual(self.scanner.scan(text), first)
        self.assertEqual(first, Scanner(self.entries).scan(text))

    def test_every_pattern_finds_its_own_entry(self):
        for entry in self.entries:
            for pattern in entry.patterns:
                found = [m.entry.id for m in self.scanner.scan(f"so {pattern} here")]
                self.assertEqual(found, [entry.id], pattern)

    def test_results_sorted_by_position(self):
        starts = [m.start for m in self.scanner.scan("trauma, toxic, red flag, boundaries")]
        self.assertEqual(starts, sorted(starts))


if __name__ == "__main__":
    unittest.main()
