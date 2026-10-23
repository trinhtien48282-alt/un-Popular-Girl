# un-Popular Girl

A small, fully offline desktop app (Python + PySide6) that flags possibly loose, exaggerated, or oversimplified uses of therapy-related and pop-psychology words, and explains each term accurately and without drama. The explanations are balanced and respectful; the UI copy is allowed a dry sense of humor. The jokes are aimed at word misuse, never at people with mental health difficulties.

**Version 0.1.0** - 15 bundled terms.

## Features

- Paste text, press **Inspect the vocabulary** (or Ctrl+Enter), get every occurrence of a bundled term.
- Three result categories: `POSSIBLE MISUSE`, `NEEDS CONTEXT`, `PLAUSIBLE USAGE`. Each result carries a marker symbol and a text label, so color is never the only signal.
- Detail pane: definition, when the term may be appropriate, commonly oversimplified usage, legitimate usage, and a reference (shown as text, never opened automatically).
- Dictionary tab to browse all 15 entries without scanning anything.
- Matched-term count, empty states, resizable window, dark theme.

The 15 terms: Gaslighting, Narcissist, Toxic, Trauma, Triggered, Red flag, Love bombing, Manipulative, Boundaries, Toxic positivity, Emotional damage, Attachment issues, Gaslight, Narcissistic abuse, Trauma dumping.

## Requirements

- Python 3.12 or newer
- Windows 10/11 (PySide6 also runs on macOS and Linux, but Windows is the target)
- One dependency: `PySide6` (the GUI toolkit). The scanner and dictionary loader use only the standard library.

## Install (Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Launch

```
python main.py
```

## Tests

Tests use the standard library `unittest`, need no network, and do not import Qt.

```
python -m unittest discover -s tests -v
```

## Packaging (Windows, PyInstaller `--onedir`, uv)

```
uv run --with-requirements requirements.txt --with pyinstaller pyinstaller --noconfirm --clean --onedir --windowed --name un-Popular-Girl --add-data "data\terminology.json;data" main.py
```

The JSON is bundled once, to `dist\un-Popular-Girl\_internal\data\terminology.json`. The app finds it through `resource_path()` in `app/dictionary.py`, which uses `sys._MEIPASS` when frozen and the project root otherwise. Verify: run the exe by full path from a different folder (for example `cd %TEMP%`), then check the Dictionary tab lists 15 entries. Temporarily renaming the JSON should produce an error dialog naming the missing path.

## Architecture

```
main.py                 entry point; loads data, applies theme, shows window
app/
    models.py           Entry, Match, Category (small dataclasses)
    dictionary.py       loads and validates data/terminology.json
    scanner.py          deterministic scanning + clue rules (no Qt)
    window.py           PySide6 UI only
    styles.py           the one Qt stylesheet
data/terminology.json   all 15 entries
tests/                  offline unit tests
```

The scanner never imports Qt, so it can be tested and reused on its own.

## How detection works

1. **Find terms.** Each entry lists detection phrases. They are matched case-insensitively with word boundaries, so `toxic` does not match `toxicology`. Spaces and hyphens inside a phrase are interchangeable (`love bombing` = `love-bombing`).
2. **Resolve overlaps.** The longest match wins, so `toxic positivity` is reported once, as Toxic positivity, not also as Toxic.
3. **Check nearby clues.** In the term's own sentence, each entry's clue phrases are checked (a short sentence like "That's gaslighting." may also borrow the sentence right before it, but then needs two clues; a negation such as "wasn't" before the term marks it as denied unless a contrast word like "but" is nearby, in which case two clues are required):
   - only a *misuse clue* found: `POSSIBLE MISUSE`
   - only a *plausible clue* found: `PLAUSIBLE USAGE`
   - both, or neither: `NEEDS CONTEXT` (the default for a bare match)

   Clue phrases match word prefixes (`isolat` matches `isolating`).
4. Each result includes a plain-language reason naming the clue that was used.

There are no scores, no confidence percentages, no machine learning, and no randomness. The same input always produces the same output. "Triggered", "Red flag", and "Emotional damage" deliberately have no misuse clues, since informal use of them is not inherently wrong.

## Limitations and false positives

This is keyword matching, not language understanding. It cannot see intent, sarcasm, quotation, or who is speaking. Negation is handled by one narrow rule (a cue such as "wasn't" or "don't think" within five words before the term, with no "but"/"later"-style contrast), which misses many phrasings ("I doubt that was gaslighting") and can misfire ("I am not sure he is gaslighting me"). Sentence splitting is naive. It will sometimes flag fine usage and miss real misuse. A clue word is a hint, never proof, and a result never says anything about a person's experience, intent, or mental state.

## Offline guarantees

- No network code anywhere. No telemetry, accounts, cloud calls, or update checks.
- All content lives in `data/terminology.json`.
- Source URLs are shown as plain text; the app never opens them.

## Disclaimer

Educational use only. This is not a diagnostic instrument and gives no medical, psychological, or legal advice. Several terms here (for example narcissistic personality disorder, trauma, attachment disorders) are clinical concepts that only qualified professionals can apply to a real person. If you are struggling, a licensed professional or local crisis service is the right place to turn.

## Adding or editing terms

Edit `data/terminology.json`. Each entry is an object with:

| Field | Notes |
|---|---|
| `id` | unique, stable slug |
| `term`, `term_category`, `definition`, `guidance` | non-empty strings |
| `misuse_examples`, `legitimate_examples` | non-empty lists of strings |
| `patterns` | non-empty list of phrases to detect (list each inflection you want, e.g. `gaslight`, `gaslighted`) |
| `misuse_clues`, `plausible_clues` | lists of clue phrases; may be empty |
| `humor` | optional string or `null` |
| `source_title`, `source_url` | both `null`, or a title plus an `http(s)` URL |

The file is validated at startup; a problem produces an error naming the entry and field. If you add or remove entries, update the count assertion in `tests/test_dictionary.py`. Only cite sources you have actually checked.

## Source notes

The reference URLs were written without internet access in the build environment and have not been opened to verify them. Please click through each one once before publishing. Entries with no single authoritative source (for example Love bombing, Boundaries, Trauma dumping) say so rather than invent one.
