"""Load and validate the bundled terminology file."""
import json
from pathlib import Path

from .models import Entry

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "terminology.json"

_STR_FIELDS = ("id", "term", "term_category", "definition", "guidance")
_LIST_FIELDS = ("misuse_examples", "legitimate_examples", "patterns", "misuse_clues", "plausible_clues")
# Clue lists may be empty on purpose (e.g. "triggered" has no misuse clues).
_MAY_BE_EMPTY = ("misuse_clues", "plausible_clues")


class DictionaryError(ValueError):
    """Bundled terminology data is missing or malformed."""


def _entry(raw: object, index: int) -> Entry:
    where = f"entry #{index + 1}"
    if not isinstance(raw, dict):
        raise DictionaryError(f"{where}: expected an object, got {type(raw).__name__}")
    where = f"entry #{index + 1} ({raw.get('id', '?')})"
    for key in _STR_FIELDS:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise DictionaryError(f"{where}: '{key}' must be a non-empty string")
    for key in _LIST_FIELDS:
        value = raw.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise DictionaryError(f"{where}: '{key}' must be a list of non-empty strings")
        if not value and key not in _MAY_BE_EMPTY:
            raise DictionaryError(f"{where}: '{key}' must not be empty")
    for key in ("humor", "source_title", "source_url"):
        value = raw.get(key)
        if value is not None and not isinstance(value, str):
            raise DictionaryError(f"{where}: '{key}' must be a string or null")
    url = raw.get("source_url")
    if url is not None and not url.startswith(("https://", "http://")):
        raise DictionaryError(f"{where}: 'source_url' must start with http:// or https://")
    if url is not None and not raw.get("source_title"):
        raise DictionaryError(f"{where}: 'source_url' needs a 'source_title'")
    return Entry(
        id=raw["id"],
        term=raw["term"],
        term_category=raw["term_category"],
        definition=raw["definition"],
        misuse_examples=tuple(raw["misuse_examples"]),
        legitimate_examples=tuple(raw["legitimate_examples"]),
        guidance=raw["guidance"],
        patterns=tuple(raw["patterns"]),
        misuse_clues=tuple(raw["misuse_clues"]),
        plausible_clues=tuple(raw["plausible_clues"]),
        humor=raw.get("humor"),
        source_title=raw.get("source_title"),
        source_url=url,
    )


def load_entries(path: Path = DEFAULT_PATH) -> list[Entry]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DictionaryError(f"Terminology file not found: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DictionaryError(f"Terminology file is not valid JSON ({path}): {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise DictionaryError("Terminology file must contain a non-empty JSON list of entries")
    entries = [_entry(item, i) for i, item in enumerate(raw)]
    ids = [e.id for e in entries]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise DictionaryError(f"Duplicate entry ids: {', '.join(dupes)}")
    return entries
