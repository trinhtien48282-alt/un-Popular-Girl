"""Train and evaluate the TF-IDF + logistic-regression terminology classifier.

Run from the repo root:
    uv run python -m scripts.train_classifier                    # validate, evaluate, train, save
    uv run python -m scripts.train_classifier --validate-only    # dataset checks only
    uv run python -m scripts.train_classifier --evaluate-only    # checks + evaluation, no model saved

Only rows with review_status == "reviewed" are used. "needs_review" rows are
validated but never trained on or evaluated on.

Evaluation: near-duplicate rows (word-set Jaccard >= SIMILARITY, any term) are
merged into groups; StratifiedGroupKFold keeps each group in one fold, so a
sentence and its paraphrase never sit on both sides of a split. Reported
metrics are out-of-fold predictions. The saved model is then fit on all
reviewed rows. This is a prototype check on a small set, not proof of reliability.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from app.classifier import build_input
from app.dictionary import DictionaryError, load_entries
from app.models import Category

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "training_examples.csv"
MODEL_PATH = ROOT / "models" / "terminology_classifier.joblib"

COLUMNS = ("term", "text", "label", "review_status")
LABELS = tuple(sorted(c.value for c in Category))
REVIEWED, NEEDS_REVIEW = "reviewed", "needs_review"
SEED = 42
FOLDS = 3
MIN_PER_CLASS = 8  # reviewed examples per label, a floor for "can evaluate at all", not for "good"
SIMILARITY = 0.6


class DatasetError(ValueError):
    """Training data is missing, malformed, or too small."""


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.lower().replace("\u2019", "'")))


def load_rows(path: Path) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = set(COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise DatasetError(f"{path}: missing column(s): {', '.join(sorted(missing))}")
            return [{k: (r.get(k) or "").strip() for k in COLUMNS} for r in reader]
    except FileNotFoundError:
        raise DatasetError(f"Training file not found: {path}") from None


def validate(rows: list[dict[str, str]], known_terms: set[str] | None = None) -> None:
    """Raise DatasetError listing every problem (row numbers match the CSV file's lines)."""
    problems = [] if rows else ["no rows"]
    seen: dict[tuple[str, str], tuple[int, str]] = {}
    for i, r in enumerate(rows, start=2):
        for k in COLUMNS:
            if not r[k]:
                problems.append(f"row {i}: '{k}' is empty")
        if r["label"] and r["label"] not in LABELS:
            problems.append(f"row {i}: invalid label {r['label']!r}")
        if r["review_status"] and r["review_status"] not in (REVIEWED, NEEDS_REVIEW):
            problems.append(f"row {i}: review_status must be '{REVIEWED}' or '{NEEDS_REVIEW}', got {r['review_status']!r}")
        if known_terms is not None and r["term"] and r["term"] not in known_terms:
            problems.append(f"row {i}: unknown term {r['term']!r}")
        key = (r["term"].lower(), _norm(r["text"]))
        if r["text"] and key in seen:
            first, label = seen[key]
            kind = "duplicate of" if label == r["label"] else "conflicting label for the same text as"
            problems.append(f"row {i}: {kind} row {first}")
        else:
            seen[key] = (i, r["label"])
    if problems:
        raise DatasetError("Dataset problems:\n  " + "\n  ".join(problems))


def reviewed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["review_status"] == REVIEWED]


def group_ids(texts: list[str], threshold: float = SIMILARITY) -> list[int]:
    """Union rows whose word sets overlap (Jaccard >= threshold); return a group id per row."""
    sets = [set(_norm(t).split()) for t in texts]
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union and len(sets[i] & sets[j]) / len(union) >= threshold:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def closest(text: str, references: list[str]) -> tuple[float, int]:
    """(best Jaccard similarity, index) of `text` against `references`; (0.0, -1) if none."""
    a = set(_norm(text).split())
    best = (0.0, -1)
    for i, ref in enumerate(references):
        b = set(_norm(ref).split())
        if a | b and len(a & b) / len(a | b) > best[0]:
            best = (len(a & b) / len(a | b), i)
    return best


def check_coverage(labels: list[str], groups: list[int], folds: int) -> None:
    problems = []
    for label in LABELS:
        n = labels.count(label)
        g = len({gr for gr, lb in zip(groups, labels) if lb == label})
        if n < MIN_PER_CLASS or g < folds:
            problems.append(f"{label}: {n} reviewed examples in {g} groups (need at least {MIN_PER_CLASS} examples and {folds} groups)")
    if problems:
        raise DatasetError("Not enough reviewed data to train and evaluate:\n  " + "\n  ".join(problems))


def make_pipeline(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[\w']+")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])


def grouped_splits(labels: list[str], groups: list[int], folds: int = FOLDS, seed: int = SEED):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(labels)), labels, groups))


def evaluate(X: list[str], y: list[str], groups: list[int], folds: int, seed: int) -> list[str]:
    pred = [""] * len(y)
    for train_idx, test_idx in grouped_splits(y, groups, folds, seed):
        pipe = make_pipeline(seed).fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        for i, p in zip(test_idx, pipe.predict([X[i] for i in test_idx])):
            pred[i] = p
    return pred


def coverage_lines(rows: list[dict[str, str]], known_terms: set[str] | None = None) -> list[str]:
    """Reviewed-row counts per term and label, with gaps called out.

    Counts show where data is absent; they say nothing about whether the data is good.
    No minimum is applied here: a gap is only a zero.
    """
    reviewed = Counter((r["term"], r["label"]) for r in rows if r["review_status"] == REVIEWED)
    unreviewed = Counter(r["term"] for r in rows if r["review_status"] != REVIEWED)
    terms = sorted({r["term"] for r in rows} | set(known_terms or ()))
    width = max([len("term")] + [len(t) for t in terms])
    out = ["Coverage by term (reviewed rows only; counts show gaps, not data quality):",
           f"  {'term':{width}}  " + "  ".join(f"{label:>15}" for label in LABELS) + "  unreviewed  gaps"]
    for t in terms:
        counts = [reviewed[(t, label)] for label in LABELS]
        missing = [label for label, c in zip(LABELS, counts) if c == 0]
        if len(missing) == len(LABELS):
            gap = "NO REVIEWED ROWS"
        else:
            gap = ("no reviewed: " + ", ".join(missing)) if missing else ""
        out.append(f"  {t:{width}}  " + "  ".join(f"{c:>15}" for c in counts) + f"  {unreviewed[t]:>10}  {gap}")
    for label in LABELS:
        by_term = Counter({t: reviewed[(t, label)] for t in terms if reviewed[(t, label)]})
        total = sum(by_term.values())
        if total:
            top, n = by_term.most_common(1)[0]
            out.append(f"  {label}: {total} reviewed; most rows from one term: {top} ({n} of {total})")
        else:
            out.append(f"  {label}: NO reviewed rows")
    return out


def _print_counts(rows: list[dict[str, str]]) -> None:
    counts = Counter((r["label"], r["review_status"]) for r in rows)
    n_rev = sum(r["review_status"] == REVIEWED for r in rows)
    print(f"Rows: {len(rows)}  reviewed: {n_rev}  needs_review: {len(rows) - n_rev}")
    for label in LABELS:
        print(f"  {label:16} reviewed {counts[(label, REVIEWED)]:3}   needs_review {counts[(label, NEEDS_REVIEW)]:3}")


def _print_report(y: list[str], pred: list[str], n_groups: int) -> None:
    print(f"\nGrouped {FOLDS}-fold cross-validation on {len(y)} reviewed rows ({n_groups} near-duplicate groups):")
    print(classification_report(y, pred, labels=list(LABELS), zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("  order:", " | ".join(LABELS))
    print(confusion_matrix(y, pred, labels=list(LABELS)))
    print(f"\nAccuracy {accuracy_score(y, pred):.3f}   macro-F1 {f1_score(y, pred, average='macro', zero_division=0):.3f}")
    print("NOTE: small, project-authored dataset; these numbers do not show the model is reliable.")


def run(data: Path = DATA_PATH, model: Path = MODEL_PATH, *, seed: int = SEED, folds: int = FOLDS,
        known_terms: set[str] | None = None, validate_only: bool = False, evaluate_only: bool = False):
    """Returns the trained Pipeline, or None for --validate-only / --evaluate-only."""
    rows = load_rows(data)
    validate(rows, known_terms)
    _print_counts(rows)
    print()
    print("\n".join(coverage_lines(rows, known_terms)))
    rev = reviewed_rows(rows)
    X = [build_input(r["term"], r["text"]) for r in rev]
    y = [r["label"] for r in rev]
    groups = group_ids([r["text"] for r in rev])
    check_coverage(y, groups, folds)
    print("\nDataset valid; enough reviewed data to evaluate.")
    if validate_only:
        return None
    _print_report(y, evaluate(X, y, groups, folds, seed), len(set(groups)))
    if evaluate_only:
        return None
    pipe = make_pipeline(seed).fit(X, y)
    model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model)
    print(f"\nSaved model (fit on all {len(y)} reviewed rows, seed {seed}): {model}")
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    try:
        terms = {e.term for e in load_entries()}
        run(a.data, a.model, seed=a.seed, known_terms=terms,
            validate_only=a.validate_only, evaluate_only=a.evaluate_only)
    except (DatasetError, DictionaryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
