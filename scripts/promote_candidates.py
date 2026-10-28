"""Copy candidates YOU approved into data/training_examples.csv. Dry run unless --apply.

    uv run python -m scripts.promote_candidates            # dry run: shows what would be added
    uv run python -m scripts.promote_candidates --apply    # append to the training CSV

A candidate is copied only if you set review_status to exactly "reviewed" AND gave it a valid label.
needs_review and rejected rows are ignored. Problems (empty or unknown label, misspelled status,
unknown term) stop everything, so nothing is half-applied. Existing training rows are never changed:
rows are only appended. Rows that duplicate a training row are skipped. Rows that duplicate or closely
overlap a held-out row are BLOCKED, so held-out text cannot leak into training.
"""
import argparse
import csv
import sys
from pathlib import Path

from app.dictionary import DictionaryError, load_entries
from scripts import train_classifier as tc
from scripts.generate_candidates import CANDIDATES_PATH, HELDOUT_PATH, REJECTED, _key, _read_optional, read_candidates


def plan(candidates, training, heldout, known_terms):
    """Returns (to_add, skipped_messages, problems). Pure function, no file access."""
    problems, skipped, to_add = [], [], []
    train_keys = {_key(r["text"]) for r in training}
    held_keys = {_key(r["text"]) for r in heldout}
    held_texts = [r["text"] for r in heldout]
    train_texts = [r["text"] for r in training if r["review_status"] == tc.REVIEWED]
    taken: set[str] = set()
    for n, c in enumerate(candidates, start=2):  # row 1 is the header
        status = c["review_status"]
        if status in (tc.NEEDS_REVIEW, REJECTED):
            continue
        where = f"candidate row {n}"
        if status != tc.REVIEWED:
            problems.append(f"{where}: review_status {status!r} is not one of needs_review, reviewed, rejected")
            continue
        if c["label"] not in tc.LABELS:
            problems.append(f"{where}: approved but label {c['label']!r} is not one of {', '.join(tc.LABELS)}")
        if c["term"] not in known_terms:
            problems.append(f"{where}: unknown term {c['term']!r}")
        if not c["text"].strip():
            problems.append(f"{where}: empty text")
        k = _key(c["text"])
        if k in held_keys or tc.closest(c["text"], held_texts)[0] >= tc.SIMILARITY:
            skipped.append(f"{where}: BLOCKED, identical or too similar to a held-out row: {c['text'][:60]}")
        elif k in train_keys or k in taken:
            skipped.append(f"{where}: already in training, skipped: {c['text'][:60]}")
        else:
            taken.add(k)
            sim = tc.closest(c["text"], train_texts)[0]
            note = f" (word overlap {sim:.2f} with a training row)" if sim >= tc.SIMILARITY else ""
            to_add.append({"term": c["term"], "text": c["text"].strip(), "label": c["label"], "review_status": tc.REVIEWED})
            if note:
                skipped.append(f"{where}: added, but note{note}")
    return to_add, skipped, problems


def _append(path: Path, rows) -> None:
    with open(path, "rb") as f:
        f.seek(-1, 2)
        needs_newline = f.read(1) not in (b"\n", b"\r")
    with open(path, "a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\r\n")
        csv.DictWriter(f, fieldnames=tc.COLUMNS).writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--training", type=Path, default=tc.DATA_PATH)
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH, help="read only")
    ap.add_argument("--terminology", type=Path, default=None, help="default: data/terminology.json")
    ap.add_argument("--apply", action="store_true", help="really append (default: dry run)")
    args = ap.parse_args(argv)
    try:
        if not args.training.is_file():
            raise tc.DatasetError(f"Training file not found: {args.training}")
        if not args.candidates.is_file():
            raise tc.DatasetError(f"Candidates file not found: {args.candidates}")
        known = {e.term for e in (load_entries(args.terminology) if args.terminology else load_entries())}
        candidates = read_candidates(args.candidates)
        training = tc.load_rows(args.training)
        warnings: list[str] = []
        heldout = _read_optional(args.heldout, "Held-out", warnings)
        for w in warnings:
            print(f"WARNING: {w}")
        to_add, notes, problems = plan(candidates, training, heldout, known)
        if problems:
            raise tc.DatasetError("Fix these in the candidates file first; nothing was changed:\n  " + "\n  ".join(problems))
        tc.validate(training + to_add, known)  # the combined file must still be valid
        for line in notes:
            print(line)
        waiting = sum(c["review_status"] == tc.NEEDS_REVIEW for c in candidates)
        print(f"{len(to_add)} approved candidate(s) ready; {waiting} still needs_review (ignored).")
        if not args.apply:
            print("Dry run: nothing written. Re-run with --apply to append them to the training file.")
            return 0
        if to_add:
            _append(args.training, to_add)
            after = tc.load_rows(args.training)
            if after[:len(training)] != training or after[len(training):] != to_add:
                raise tc.DatasetError("Post-write check failed: the training file is not what was expected. Restore it from git.")
        print(f"Appended {len(to_add)} row(s) to {args.training}. Existing rows were not touched.")
        print("Next: validate, then retrain (see docs/CANDIDATE_WORKFLOW.md).")
    except (tc.DatasetError, DictionaryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
