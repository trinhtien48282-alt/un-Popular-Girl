"""Try the saved terminology classifier from the command line. Offline; not used by the GUI.

    uv run python -m scripts.predict_classifier --term Gaslighting --text "He disagreed with me. That's gaslighting."
    uv run python -m scripts.predict_classifier --term Toxic --text "..." --term Trauma --text "..."   # several, paired in order
    uv run python -m scripts.predict_classifier --file examples.csv                                    # columns: term,text
    uv run python -m scripts.predict_classifier --evaluate                                             # held-out, human-reviewed rows only

Output numbers are model scores from a small TF-IDF + logistic-regression model,
not the chance that a label is correct. The model sees only wording; it cannot
tell what happened between people.

--evaluate reads data/heldout_examples.csv, keeps only `reviewed` rows, and drops
any row that is identical or near-identical (Jaccard >= SIMILARITY) to a reviewed
training row. It assumes the model was trained from data/training_examples.csv.
It never scores training rows as if they were unseen.
"""
import argparse
import csv
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from app.classifier import DEFAULT_MODEL_PATH, Classifier, ClassifierError
from app.dictionary import DictionaryError, load_entries
from scripts import train_classifier as tc

HELDOUT_PATH = tc.ROOT / "data" / "heldout_examples.csv"
BANNER = ("Model scores, not guarantees: they are not the chance a label is right. "
          "The model matches wording learned from a small example set.")
CAVEAT = "NOTE: small, project-authored data; these numbers do not show the model is reliable."


def _training_texts(path: Path) -> list[tuple[int, str]] | None:
    """Reviewed training rows as (row number in the original CSV, text).

    Row 1 is the header, so the first data row is 2. This equals the file's line number
    unless a field contains a line break. needs_review rows are skipped but still counted.
    """
    try:
        rows = tc.load_rows(path)
    except tc.DatasetError:
        return None
    return [(n, r["text"]) for n, r in enumerate(rows, start=2) if r["review_status"] == tc.REVIEWED]


def _closest(text: str, refs: list[tuple[int, str]]) -> tuple[float, int]:
    """(similarity, original CSV row number) of the closest reviewed training row."""
    score, i = tc.closest(text, [t for _, t in refs])
    return score, (refs[i][0] if i >= 0 else -1)


def _familiarity(text: str, refs: list[tuple[int, str]] | None) -> str:
    if refs is None:
        return "training data unavailable, so familiarity was not checked"
    score, row = _closest(text, refs)
    if score >= tc.SIMILARITY:
        return f"SEEN IN TRAINING: similarity {score:.2f} to reviewed training row {row}"
    return f"not in training (closest reviewed training row: similarity {score:.2f})"


def _read_file(path: Path) -> list[tuple[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"term", "text"} <= set(reader.fieldnames or []):
                raise tc.DatasetError(f"{path}: needs columns 'term' and 'text'")
            rows = [((r.get("term") or "").strip(), (r.get("text") or "").strip()) for r in reader]
    except FileNotFoundError:
        raise tc.DatasetError(f"Input file not found: {path}") from None
    if any(not t or not x for t, x in rows):
        raise tc.DatasetError(f"{path}: every row needs a non-empty term and text")
    return rows


def _read_examples(args) -> list[tuple[str, str]]:
    if len(args.term) != len(args.text):
        raise tc.DatasetError("each --term needs a matching --text (counts differ)")
    examples = list(zip(args.term, args.text))
    if args.file:
        examples += _read_file(args.file)
    return examples


def predict_mode(clf: Classifier, examples: list[tuple[str, str]], train_texts, known: set[str]) -> None:
    print(BANNER)
    for n, (term, text) in enumerate(examples, 1):
        print(f"\n[{n}] term: {term}" + ("" if term in known else "   (not a dictionary term; score is meaningless)"))
        print(f"    text: {text}")
        pred = clf.predict(term, text)
        print(f"    predicted: {pred.label.value}")
        for cat, p in sorted(pred.probabilities.items(), key=lambda kv: (-kv[1], kv[0].value)):
            print(f"      {cat.value:16} {p:.3f}")
        print(f"    {_familiarity(text, train_texts)}")


def evaluate_mode(clf: Classifier, heldout: Path, train_texts, known: set[str]) -> None:
    if train_texts is None:
        raise tc.DatasetError("Training data is needed to prove the held-out rows are unseen, and it could not be read")
    rows = tc.load_rows(heldout)
    tc.validate(rows, known)
    reviewed = tc.reviewed_rows(rows)
    print(f"Held-out file: {heldout}\n  rows: {len(rows)}  reviewed: {len(reviewed)}  needs_review (ignored): {len(rows) - len(reviewed)}")
    kept, leaked = [], []
    for r in reviewed:
        score, row = _closest(r["text"], train_texts)
        (leaked if score >= tc.SIMILARITY else kept).append((r, score, row))
    for r, score, row in leaked:
        print(f"  EXCLUDED (similarity {score:.2f} to training row {row}): {r['text'][:60]}")
    if not kept:
        raise tc.DatasetError("No usable held-out rows: none are both human-reviewed and unseen in training. "
                              "Review rows in the held-out CSV (set review_status to 'reviewed') first.")
    y = [r["label"] for r, _, _ in kept]
    pred = [clf.predict(r["term"], r["text"]).label.value for r, _, _ in kept]
    print(f"\nEvaluated on {len(kept)} reviewed, unseen rows:")
    print(classification_report(y, pred, labels=list(tc.LABELS), zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("  order:", " | ".join(tc.LABELS))
    print(confusion_matrix(y, pred, labels=list(tc.LABELS)))
    print(f"\nAccuracy {accuracy_score(y, pred):.3f}   macro-F1 {f1_score(y, pred, average='macro', zero_division=0):.3f}")
    missing = [lb for lb in tc.LABELS if lb not in y]
    if missing:
        print("WARNING: no held-out examples for: " + ", ".join(missing))
    print(CAVEAT)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--term", action="append", default=[], help="term; repeat, paired with --text in order")
    ap.add_argument("--text", action="append", default=[], help="sentence or passage; repeat")
    ap.add_argument("--file", type=Path, help="CSV with columns term,text")
    ap.add_argument("--evaluate", action="store_true", help="score reviewed, unseen held-out rows")
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--training-data", type=Path, default=tc.DATA_PATH)
    args = ap.parse_args(argv)
    if not (args.term or args.text or args.file or args.evaluate):
        ap.error("give --term/--text, --file, or --evaluate")
    try:
        clf = Classifier.load(args.model)
        known = {e.term for e in load_entries()}
        train_texts = _training_texts(args.training_data)
        if args.term or args.text or args.file:
            predict_mode(clf, _read_examples(args), train_texts, known)
        if args.evaluate:
            evaluate_mode(clf, args.heldout, train_texts, known)
    except (ClassifierError, tc.DatasetError, DictionaryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
