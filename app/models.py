from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    POSSIBLE_MISUSE = "POSSIBLE MISUSE"
    NEEDS_CONTEXT = "NEEDS CONTEXT"
    PLAUSIBLE_USAGE = "PLAUSIBLE USAGE"


@dataclass(frozen=True)
class Entry:
    id: str
    term: str
    term_category: str  # severity-neutral label, e.g. "Clinical-adjacent term"
    definition: str
    misuse_examples: tuple[str, ...]
    legitimate_examples: tuple[str, ...]
    guidance: str
    patterns: tuple[str, ...]
    misuse_clues: tuple[str, ...]
    plausible_clues: tuple[str, ...]
    humor: str | None
    source_title: str | None
    source_url: str | None


@dataclass(frozen=True)
class Match:
    entry: Entry
    text: str  # matched text exactly as typed
    start: int  # character offsets into the scanned text
    end: int
    category: Category
    reason: str  # plain-language explanation of why this category was chosen
    before: str  # surrounding text
    after: str

    @property
    def context(self) -> str:
        return f"{self.before}{self.text}{self.after}"
