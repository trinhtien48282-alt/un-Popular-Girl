"""Deterministic, rule-based terminology scanner. No ML, no network.

HOW IT WORKS
1. Every entry lists detection patterns (plain phrases). Each is compiled to a
   case-insensitive regex with word boundaries; spaces and hyphens inside a
   phrase match any run of spaces/hyphens ("love bombing" == "love-bombing").
2. Overlaps: longest match wins ("toxic positivity" beats "toxic").
3. Category, per match, from clue phrases found in the SAME SENTENCE:
     misuse clue only    -> POSSIBLE MISUSE
     plausible clue only -> PLAUSIBLE USAGE
     both, or neither    -> NEEDS CONTEXT   (the default for a bare match)
   Clue phrases match word prefixes ("isolat" matches "isolating").
There is no score and no confidence number: the category is the whole result.

LIMITATIONS (also shown in the UI)
This is keyword matching, not language understanding. It cannot see intent,
sarcasm, negation ("that is NOT gaslighting"), quotation, or who is speaking,
and sentence splitting is naive (abbreviations like "e.g." end a "sentence").
Expect false positives and false negatives. A result is a prompt to think,
never a verdict about a person's words, experience, or mind.
"""
import re

from .models import Category, Entry, Match

CONTEXT_CHARS = 40
_SENTENCE_END = ".!?\n"


def _phrase(phrase: str, *, whole_word: bool) -> re.Pattern[str]:
    parts = re.split(r"[\s-]+", phrase.strip())
    body = r"[\s-]+".join(re.escape(p) for p in parts)
    return re.compile(r"(?<!\w)" + body + (r"(?!\w)" if whole_word else ""), re.IGNORECASE)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(text.rfind(c, 0, start) for c in _SENTENCE_END) + 1
    rights = [i for i in (text.find(c, end) for c in _SENTENCE_END) if i != -1]
    return left, (min(rights) if rights else len(text))


class Scanner:
    def __init__(self, entries: list[Entry]):
        self._rules = [
            (
                entry,
                [_phrase(p, whole_word=True) for p in entry.patterns],
                [_phrase(c, whole_word=False) for c in entry.misuse_clues],
                [_phrase(c, whole_word=False) for c in entry.plausible_clues],
            )
            for entry in entries
        ]

    def scan(self, text: str) -> list[Match]:
        if not text.strip():
            return []
        # Same-length swap keeps offsets valid while letting "didn't" match "didn’t".
        norm = text.replace("\u2019", "'")
        found = [
            (m.start(), m.end(), rule)
            for rule in self._rules
            for rx in rule[1]
            for m in rx.finditer(norm)
        ]
        found.sort(key=lambda f: (f[0] - f[1], f[0]))  # longest first, then leftmost
        kept: list[tuple[int, int, tuple]] = []
        for start, end, rule in found:
            if all(end <= ks or start >= ke for ks, ke, _ in kept):
                kept.append((start, end, rule))
        kept.sort(key=lambda k: k[0])
        return [self._build(text, norm, s, e, rule) for s, e, rule in kept]

    @staticmethod
    def _build(text: str, norm: str, start: int, end: int, rule: tuple) -> Match:
        entry, _, misuse_rx, plausible_rx = rule
        left, right = _sentence_bounds(norm, start, end)
        sentence = norm[left:right]
        misuse = next((m.group(0) for rx in misuse_rx if (m := rx.search(sentence))), None)
        plausible = next((m.group(0) for rx in plausible_rx if (m := rx.search(sentence))), None)
        if misuse and not plausible:
            category = Category.POSSIBLE_MISUSE
            reason = f'Nearby wording ("{misuse}") often goes with loose or casual use of this term.'
        elif plausible and not misuse:
            category = Category.PLAUSIBLE_USAGE
            reason = f'Nearby wording ("{plausible}") often goes with a specific or literal use of this term.'
        elif misuse and plausible:
            category = Category.NEEDS_CONTEXT
            reason = f'Mixed signals: "{misuse}" and "{plausible}" point different ways.'
        else:
            category = Category.NEEDS_CONTEXT
            reason = "The term appears, and no nearby wording points either way."
        return Match(
            entry=entry,
            text=text[start:end],
            start=start,
            end=end,
            category=category,
            reason=reason,
            before=text[max(0, start - CONTEXT_CHARS):start],
            after=text[end:end + CONTEXT_CHARS],
        )
