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
4. Narrow negation rule, checked first: if a negation cue ("wasn't", "not",
   "never", "no", "without", "don't think", ...) sits within the 5 words before
   the term, in the same clause, and no contrast word ("but", "however",
   "later", "at first", ...) appears in that window or after the term in the
   sentence, the term is treated as denied, not applied. Result: NEEDS CONTEXT,
   clues ignored (they describe an accusation nobody is making).
There is no score and no confidence number: the category is the whole result.

LIMITATIONS (also shown in the UI)
This is keyword matching, not language understanding. It cannot see intent,
sarcasm, quotation, or who is speaking, and only handles negation through the
narrow rule above (it misses "I doubt that was gaslighting", and can wrongly
treat "I am not sure he's gaslighting me" as a denial). Sentence splitting is naive (abbreviations like "e.g." end a "sentence").
Expect false positives and false negatives. A result is a prompt to think,
never a verdict about a person's words, experience, or mind.
"""
import re

from .models import Category, Entry, Match

CONTEXT_CHARS = 40
_SENTENCE_END = ".!?\n"
_CLAUSE_BREAK = ",;:()"
_NEGATION_WINDOW = 5  # words before the term
# Deliberately narrow: bare "don't"/"can't" are excluded ("I don't know why he keeps gaslighting me").
_NEGATION = re.compile(
    r"\b(?:not|never|no|without|(?:is|was|are|were|ai)n't"
    r"|(?:do|does|did)(?:n't|\s+not)\s+(?:really\s+)?(?:think|believe|say|call|consider|see|feel))\b",
    re.IGNORECASE,
)
_CONTRAST = re.compile(
    r"\b(?:but|however|although|though|yet|until|eventually|later|then|initially|anymore)\b|\bat first\b",
    re.IGNORECASE,
)


def line_col(text: str, index: int) -> tuple[int, int]:
    """1-based (line, column) of a character offset."""
    return text.count("\n", 0, index) + 1, index - (text.rfind("\n", 0, index) + 1) + 1


def _phrase(phrase: str, *, whole_word: bool) -> re.Pattern[str]:
    parts = re.split(r"[\s-]+", phrase.strip())
    body = r"[\s-]+".join(re.escape(p) for p in parts)
    return re.compile(r"(?<!\w)" + body + (r"(?!\w)" if whole_word else ""), re.IGNORECASE)


def _negation_cue(norm: str, left: int, right: int, start: int, end: int) -> str | None:
    """The negation cue denying the term at norm[start:end], else None."""
    clause = max([left] + [norm.rfind(c, left, start) + 1 for c in _CLAUSE_BREAK])
    window = " ".join(re.findall(r"[\w']+", norm[clause:start])[-_NEGATION_WINDOW:])
    cue = _NEGATION.search(window)
    if cue and not _CONTRAST.search(window) and not _CONTRAST.search(norm[end:right]):
        return cue.group(0)
    return None


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
        negation = _negation_cue(norm, left, right, start, end)
        if negation:
            category = Category.NEEDS_CONTEXT
            reason = f'The term follows a negation ("{negation}"), so it looks denied rather than applied. Clue words are ignored.'
        elif misuse and not plausible:
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
            reason = "The term appears. Only a fixed list of clue phrases is checked nearby, and none was found, so this scanner cannot say more."
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
