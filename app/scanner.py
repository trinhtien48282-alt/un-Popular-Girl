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
4. Negation, checked first: a cue ("wasn't", "not", "never", "no", "without",
   "don't think", ...) within the 5 words before the term, in the same clause.
     - no contrast word ("but", "however", "later", "at first", ...) anywhere
       around it -> the term looks DENIED: NEEDS CONTEXT, clues ignored.
     - contrast present -> the denial is set aside, but conservatively: only
       text after a contrast word that follows the term is examined (or the
       whole sentence if the contrast comes first), and TWO distinct clues
       are required instead of one.
5. Borrowed context: if the term's own sentence has no clues and is short
   (<= 6 words, e.g. "That's gaslighting."), the sentence right before it
   (same line, ended by . ! ?) is checked too. Borrowed evidence needs TWO
   distinct clues, so one stray word in a neighbouring sentence decides nothing.
   Never borrowed: across a line break, or from the sentence after.
Opposing clues always cancel to NEEDS CONTEXT. A single clue still decides
inside the term's own, un-negated sentence (that is the weakest remaining spot).
There is no score and no confidence number: the category is the whole result.

LIMITATIONS (also shown in the UI)
This is keyword matching, not language understanding. It cannot see intent,
sarcasm, quotation, or who is speaking, and only handles negation through
rule 4 (it misses "I doubt that was gaslighting", and can wrongly
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
_SHORT_SENTENCE_WORDS = 6  # at most this many words may borrow context
_WORD = re.compile(r"[\w']+")
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


def _negation(norm: str, left: int, right: int, start: int, end: int) -> tuple[str, bool, int] | None:
    """Negation cue before the term, as (cue, contrasted, evidence_start), else None.

    contrasted=False: the term is denied. contrasted=True: a contrast word is
    near, so the denial is set aside; evidence_start is where clue-checking
    begins (just after a contrast word that follows the term, else sentence start).
    """
    clause = max([left] + [norm.rfind(c, left, start) + 1 for c in _CLAUSE_BREAK])
    window = " ".join(_WORD.findall(norm[clause:start])[-_NEGATION_WINDOW:])
    cue = _NEGATION.search(window)
    if not cue:
        return None
    after = _CONTRAST.search(norm, end, right)
    if after:
        return cue.group(0), True, after.end()
    return cue.group(0), bool(_CONTRAST.search(window)), left


def _clues(rxs: list[re.Pattern[str]], text: str) -> list[str]:
    """First match of each distinct clue pattern, in list order."""
    return [m.group(0) for rx in rxs if (m := rx.search(text))]


def _previous_sentence(norm: str, left: int) -> str:
    """The sentence just before `left`, only if the same line and ended by . ! ?"""
    if left == 0 or norm[left - 1] not in ".!?":
        return ""
    pleft = max(norm.rfind(c, 0, left - 1) for c in _SENTENCE_END) + 1
    return norm[pleft:left - 1]


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
        negation = _negation(norm, left, right, start, end)
        if negation and not negation[1]:
            category = Category.NEEDS_CONTEXT
            reason = f'The term follows a negation ("{negation[0]}"), so it looks denied rather than applied. Clue words are ignored.'
            return Scanner._match(text, start, end, entry, category, reason)

        need, lead = 1, ""
        zone = norm[left:right]
        if negation:  # contrast: be conservative
            zone = norm[negation[2]:right]
            need, lead = 2, f'A negation ("{negation[0]}") near a contrast word was set aside; two clues are required. '
        misuse, plausible = _clues(misuse_rx, zone), _clues(plausible_rx, zone)
        if not negation and not misuse and not plausible and len(_WORD.findall(zone)) <= _SHORT_SENTENCE_WORDS:
            prev = _previous_sentence(norm, left)
            misuse, plausible = _clues(misuse_rx, prev), _clues(plausible_rx, prev)
            if misuse or plausible:
                need, lead = 2, "This sentence is short, so the sentence before it was checked too; two clues are required there. "

        def quoted(items: list[str]) -> str:
            return ", ".join(f'"{i}"' for i in items[:3])

        category = Category.NEEDS_CONTEXT
        if misuse and plausible:
            reason = f"{lead}Mixed signals: {quoted(misuse)} and {quoted(plausible)} point different ways."
        elif (misuse or plausible) and len(misuse or plausible) < need:
            reason = f"{lead}Only one clue ({quoted(misuse or plausible)}) was found, which is not enough to decide."
        elif misuse:
            category = Category.POSSIBLE_MISUSE
            reason = f"{lead}Nearby wording ({quoted(misuse)}) often goes with loose or casual use of this term."
        elif plausible:
            category = Category.PLAUSIBLE_USAGE
            reason = f"{lead}Nearby wording ({quoted(plausible)}) often goes with a specific or literal use of this term."
        else:
            reason = f"{lead}The term appears. Only a fixed list of clue phrases is checked nearby, and none was found, so this scanner cannot say more."
        return Scanner._match(text, start, end, entry, category, reason)

    @staticmethod
    def _match(text: str, start: int, end: int, entry: Entry, category: Category, reason: str) -> Match:
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
