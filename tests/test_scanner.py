import unittest

from app.dictionary import load_entries
from app.models import Category
from app.scanner import Scanner, line_col


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

    def test_trivial_disagreement_is_flagged(self):
        text = "My brother said I was wrong about the movie's ending, so he's gaslighting me."
        m = self.scanner.scan(text)[0]
        self.assertEqual(m.category, Category.POSSIBLE_MISUSE)
        text = "She argued about dinner, so she tried to gaslight me."
        self.assertEqual(self.scanner.scan(text)[0].category, Category.POSSIBLE_MISUSE)

    def test_repetition_clue_is_plausible(self):
        text = "He repeatedly denied it happened and made me doubt myself, which is gaslighting."
        self.assertEqual(self.scanner.scan(text)[0].category, Category.PLAUSIBLE_USAGE)

    def test_no_clue_reason_is_honest_about_its_limits(self):
        self.assertIn("fixed list", self.scanner.scan("Trauma.")[0].reason)

    def test_line_col(self):
        self.assertEqual(line_col("abc", 0), (1, 1))
        self.assertEqual(line_col("abc\ndef", 5), (2, 2))
        text = "one\ntwo toxic"
        self.assertEqual(line_col(text, text.index("toxic")), (2, 5))

    def test_negated_term_is_not_flagged_by_clue_words(self):
        text = "My brother wasn't gaslighting me; we simply remembered the movie differently."
        m = self.scanner.scan(text)[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertIn("negation", m.reason)
        self.assertIn("wasn't", m.reason)

    def test_negation_with_curly_apostrophe(self):
        m = self.scanner.scan("He wasn\u2019t gaslighting me, just disagreeing about the movie.")[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertIn("negation", m.reason)

    def test_dont_think_is_negation(self):
        m = self.scanner.scan("I don't think that was gaslighting.")[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertIn("negation", m.reason)

    def test_plain_accusation_is_not_treated_as_negated(self):
        m = self.scanner.scan("He was gaslighting me.")[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertNotIn("negation", m.reason)  # bare match: the ordinary no-clue reason
        flagged = self.scanner.scan("He was gaslighting me about the movie.")[0]
        self.assertEqual(flagged.category, Category.POSSIBLE_MISUSE)

    def test_negation_with_contrast_is_not_a_denial(self):
        text = ("He wasn't gaslighting me at first, but later repeatedly denied conversations "
                "we'd both had and insisted I was imagining things.")
        m = self.scanner.scan(text)[0]
        self.assertEqual(m.category, Category.PLAUSIBLE_USAGE)
        self.assertIn("set aside", m.reason)
        self.assertIn("repeatedly", m.reason)

    def test_contrast_is_conservative_one_clue_is_not_enough(self):
        m = self.scanner.scan("He wasn't rude but he was gaslighting me about the movie.")[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertIn("Only one clue", m.reason)
        two = self.scanner.scan("He wasn't rude but he was gaslighting me about the movie ending.")[0]
        self.assertEqual(two.category, Category.POSSIBLE_MISUSE)  # "movie" + "ending"

    def test_only_text_after_contrast_counts_as_evidence(self):
        # "movie"/"ending" sit before the contrast and must not count.
        text = "He wasn't gaslighting me about the movie ending at first, but then he apologised."
        m = self.scanner.scan(text)[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertNotIn("movie", m.reason)

    def test_far_or_clause_separated_negation_is_ignored(self):
        for text in (
            "I don't know why he keeps gaslighting me about the movie.",
            "He didn't text back, that's gaslighting.",
            "It was not a good week and honestly the way he talks is just gaslighting about the movie.",
        ):
            self.assertNotIn("negation", self.scanner.scan(text)[0].reason, text)

    # --- context window: borrowing from the previous sentence -------------
    def test_examples_from_the_audit(self):
        a = self.scanner.scan("I don't think that was gaslighting.")[0]
        self.assertEqual(a.category, Category.NEEDS_CONTEXT)
        self.assertIn("negation", a.reason)
        b = self.scanner.scan(
            "He wasn't gaslighting me at first, but later repeatedly denied conversations "
            "we'd both had and insisted I was imagining things.")[0]
        self.assertEqual(b.category, Category.PLAUSIBLE_USAGE)
        self.assertIn("imagining things", b.reason)
        c = self.scanner.scan(
            "For months, he repeatedly denied conversations we'd both had and insisted "
            "I was imagining things. That's gaslighting.")[0]
        self.assertEqual(c.category, Category.PLAUSIBLE_USAGE)
        self.assertIn("sentence before it", c.reason)

    def test_ordinary_disagreement(self):
        same = self.scanner.scan("He disagreed with me about the movie ending, so he's gaslighting me.")[0]
        self.assertEqual(same.category, Category.POSSIBLE_MISUSE)
        split = self.scanner.scan("He disagreed with me about the movie. That's gaslighting.")[0]
        self.assertEqual(split.category, Category.POSSIBLE_MISUSE)
        self.assertIn("sentence before it", split.reason)

    def test_educational_mention(self):
        m = self.scanner.scan("In psychology class we learned what gaslighting means.")[0]
        self.assertEqual(m.category, Category.PLAUSIBLE_USAGE)
        mixed = self.scanner.scan("In psychology class we compared gaslighting with ordinary disagreement.")[0]
        self.assertNotEqual(mixed.category, Category.POSSIBLE_MISUSE)

    def test_simple_negation(self):
        m = self.scanner.scan("He wasn't gaslighting me.")[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertIn("negation", m.reason)

    def test_repeated_pattern_accusation_across_sentences(self):
        text = "He kept denying it, again and again, and made me doubt my memory. He is gaslighting me."
        self.assertEqual(self.scanner.scan(text)[0].category, Category.PLAUSIBLE_USAGE)

    def test_borrowed_context_needs_two_clues(self):
        m = self.scanner.scan("We watched the movie. That's gaslighting.")[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertIn("Only one clue", m.reason)

    def test_no_borrowing_across_line_break_or_for_long_sentences(self):
        evidence = "He repeatedly denied it happened and made me doubt myself"
        for text in (
            evidence + ".\nThat's gaslighting.",
            evidence + ". I really do think that what he did there was gaslighting.",
            "That's gaslighting. " + evidence + ".",  # never borrows from the sentence after
        ):
            m = self.scanner.scan(text)[0]
            self.assertEqual(m.category, Category.NEEDS_CONTEXT, text)
            self.assertNotIn("sentence before it", m.reason, text)

    def test_negated_short_sentence_does_not_borrow(self):
        text = "He repeatedly denied it happened and made me doubt myself. That wasn't gaslighting."
        m = self.scanner.scan(text)[0]
        self.assertEqual(m.category, Category.NEEDS_CONTEXT)
        self.assertIn("negation", m.reason)

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
