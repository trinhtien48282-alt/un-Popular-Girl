"""Read-only diagnostics for the training, candidate and held-out CSVs. Writes nothing.

    uv run python -m scripts.dataset_report
    uv run python -m scripts.dataset_report --candidates data\\candidate_examples.csv --pairs 10

What it can and cannot tell you:
  * Duplicates are exact (case/space-insensitive). "Near-duplicates" use WORD OVERLAP (Jaccard),
    not meaning, so paraphrases with different words are not caught.
  * "Frame" duplicates mask the term, then compare word overlap: they find sentences that differ
    mainly by which term is slotted in.
  * Cue tables are descriptive. A cue that appears in only one label is a warning that a model
    could learn the cue instead of the meaning. Nothing here shows the data is good or the model improved.
  * "Clue echo" re-checks each sentence against the scanner's clue phrase lists (prefix matching, no
    negation or sentence logic). It approximates the scanner; it does not run it.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from app.dictionary import DictionaryError, load_entries
from scripts import generate_candidates as gen
from scripts import train_classifier as tc

CUES = {
    "negation": r"\b(?:not|never|no|nobody|without)\b|n't\b",
    "hedge or unknown": r"\b(?:not sure|can't tell|couldn't tell|hard to say|unsure|maybe|perhaps|i think|apparently)\b",
    "question": r"\?",
    "quoted words": r"(?<!\w)'[^']+'(?!\w)|\"[^\"]+\"",
    "reporting verb": r"\b(?:says?|said|told|called|claims?|accused|calling)\b",
    "mentions detail words": r"\b(?:for (?:months|years|weeks)|every|constantly|repeatedly|again and again)\b",
}
MIN_ROWS_FOR_RATE_WARNING = 10  # below this, a percentage is too noisy to warn about (display choice, not a quality threshold)
MIN_REPEAT = 3  # display cutoff for repeated openings/phrases, only to keep output short


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower().replace("\u2019", "'"))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def masked(text: str, patterns) -> str:
    """Replace the term's detection patterns with <T> so frames can be compared across terms."""
    for p in sorted(patterns, key=len, reverse=True):
        body = r"[\s-]+".join(re.escape(x) for x in re.split(r"[\s-]+", p.strip()))
        text = re.sub(r"(?<!\w)" + body + r"(?!\w)", "<T>", text, flags=re.IGNORECASE)
    return text


def exact_duplicates(sources: dict[str, list[dict[str, str]]]) -> list[list[tuple[str, int]]]:
    """Groups of (source, row number) whose text is identical ignoring case and surrounding spaces."""
    seen: dict[str, list[tuple[str, int]]] = {}
    for name, rows in sources.items():
        for n, r in enumerate(rows, start=2):
            seen.setdefault(gen._key(r["text"]), []).append((name, n))
    return [v for v in seen.values() if len(v) > 1]


def near_pairs(items: list[tuple[str, str]], threshold: float = tc.SIMILARITY, other: list[tuple[str, str]] | None = None):
    """Word-overlap pairs >= threshold. items/other are (label, text). Within `items`, or items vs other."""
    sets = [set(_words(t)) for _, t in items]
    out = []
    if other is None:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s = _jaccard(sets[i], sets[j])
                if s >= threshold:
                    out.append((s, items[i][0], items[j][0]))
    else:
        osets = [set(_words(t)) for _, t in other]
        for i in range(len(items)):
            for j in range(len(other)):
                s = _jaccard(sets[i], osets[j])
                if s >= threshold:
                    out.append((s, items[i][0], other[j][0]))
    return sorted(out, reverse=True)


def frame_pairs(rows: list[dict[str, str]], patterns_by_term: dict[str, tuple[str, ...]], threshold: float = tc.SIMILARITY):
    """Sentences of DIFFERENT terms that match once the term is masked."""
    frames = [(f"{r['term']}: {r['text'][:50]}", set(_words(masked(r["text"], patterns_by_term.get(r["term"], ()))))) for r in rows]
    terms = [r["term"] for r in rows]
    return sorted(((_jaccard(frames[i][1], frames[j][1]), frames[i][0], frames[j][0])
                   for i in range(len(rows)) for j in range(i + 1, len(rows))
                   if terms[i] != terms[j] and _jaccard(frames[i][1], frames[j][1]) >= threshold), reverse=True)


def repeated_openings(texts: list[str], n: int = 3, min_count: int = MIN_REPEAT) -> list[tuple[str, int]]:
    c = Counter(" ".join(_words(t)[:n]) for t in texts if len(_words(t)) >= n)
    return [(k, v) for k, v in c.most_common() if v >= min_count]


def repeated_phrases(texts: list[str], n: int = 4, min_count: int = MIN_REPEAT) -> list[tuple[str, int]]:
    """Word n-grams that appear in at least min_count different sentences."""
    c: Counter = Counter()
    for t in texts:
        w = _words(t)
        c.update({" ".join(w[i:i + n]) for i in range(len(w) - n + 1)})
    return [(k, v) for k, v in c.most_common() if v >= min_count]


def cue_table(rows: list[dict[str, str]], label_key: str) -> list[tuple[str, int, Counter]]:
    out = []
    for name, rx in CUES.items():
        hits = [r[label_key] for r in rows if r[label_key] and re.search(rx, r["text"], re.IGNORECASE)]
        out.append((name, len(hits), Counter(hits)))
    return out


def clue_verdict(text: str, entry) -> str:
    """What the scanner's clue lists alone would suggest (no negation or sentence logic)."""
    def hit(clues):
        return any(re.search(r"(?<!\w)" + re.escape(c), text, re.IGNORECASE) for c in clues)
    m, p = hit(getattr(entry, "misuse_clues", ())), hit(getattr(entry, "plausible_clues", ()))
    return "POSSIBLE MISUSE" if m and not p else "PLAUSIBLE USAGE" if p and not m else "NEEDS CONTEXT"


def clue_echo(rows: list[dict[str, str]], entries, label_key: str) -> tuple[int, int, Counter]:
    """(rows whose clue-list verdict equals their label, rows checked, (verdict, label) counts)."""
    by_term = {e.term: e for e in entries}
    pairs = Counter()
    for r in rows:
        if r[label_key] in tc.LABELS and r["term"] in by_term:
            pairs[(clue_verdict(r["text"], by_term[r["term"]]), r[label_key])] += 1
    return sum(n for (v, label), n in pairs.items() if v == label), sum(pairs.values()), pairs


def context_table(candidates: list[dict[str, str]]) -> dict[str, Counter]:
    table: dict[str, Counter] = {}
    for r in candidates:
        table.setdefault(r["context"], Counter())[r["suggested_label"]] += 1
    return table


def _cue_lines(title: str, scope: str, rows, label_key: str) -> tuple[list[str], list[str]]:
    lines, warns = [f"{title}:"], []
    for name, n, by in cue_table(rows, label_key):
        spread = " | ".join(f"{lb} {c}" for lb, c in sorted(by.items())) or "-"
        lines.append(f"  {name:22} rows {n:3}   {spread}")
        if n >= 2 and len(by) == 1:
            warns.append(f"cue '{name}' appears only with {next(iter(by))} among {scope} (n={n})")
    return lines, warns


def report(training, heldout, candidates, entries, pairs: int = 5) -> str:
    """Whole report as text (so tests can check it). Reads nothing, writes nothing."""
    out: list[str] = []
    warns: list[str] = []
    terms = [e.term for e in entries]
    patterns = {e.term: tuple(e.patterns) for e in entries}

    out += [f"Rows: training {len(training)} (reviewed {len(tc.reviewed_rows(training))}), held-out {len(heldout)}, candidates {len(candidates)}", ""]
    out += tc.coverage_lines(training, set(terms)) + [""]
    gaps = [t for t in terms if not all(any(r["term"] == t and r["label"] == label and r["review_status"] == tc.REVIEWED
                                            for r in training) for label in tc.LABELS)]
    if gaps:
        warns.append(f"{len(gaps)} term(s) lack reviewed training rows for at least one label: {', '.join(gaps)} "
                     "(a gap to review, not a quota to fill with weak examples)")

    status = Counter(r["review_status"] for r in candidates)
    out += ["Candidate review status: " + (", ".join(f"{s} {n}" for s, n in sorted(status.items())) or "none"), ""]
    out += gen.candidate_lines(training, candidates, terms) + [""]

    sources = {"training": training, "held-out": heldout, "candidates": candidates}
    dups = exact_duplicates(sources)
    out.append(f"Exact duplicates (ignoring case and spaces): {len(dups)} group(s)")
    for g in dups[:pairs]:
        out.append("  " + ", ".join(f"{s} row {n}" for s, n in g))
    if dups:
        warns.append(f"{len(dups)} exact duplicate group(s) across the files")

    cands = [(f"candidate row {n}: {r['text'][:50]}", r["text"]) for n, r in enumerate(candidates, start=2)]
    near = near_pairs(cands)
    out.append(f"Near-duplicate candidate pairs (word overlap >= {tc.SIMILARITY:.0%}, not meaning): {len(near)}")
    for s, a, b in near[:pairs]:
        out.append(f"  {s:.2f}  {a}  ~  {b}")
    fr = frame_pairs(candidates, patterns)
    out.append(f"Same-sentence-different-term frames (term masked, different terms): {len(fr)}")
    for s, a, b in fr[:pairs]:
        out.append(f"  {s:.2f}  {a}  ~  {b}")
    if near or fr:
        warns.append(f"{len(near)} near-duplicate and {len(fr)} frame pair(s) among candidates")
    for name, rows in (("training", training), ("held-out", heldout)):
        refs = [(f"{name} row {n}", r["text"]) for n, r in enumerate(rows, start=2)]
        hits = near_pairs(cands, other=refs)
        exact = sum(gen._key(c["text"]) in {gen._key(r["text"]) for r in rows} for c in candidates)
        out.append(f"Candidates overlapping {name}: {len({a for _, a, _ in hits})} by word overlap, {exact} exact")
        if hits:
            warns.append(f"{len({a for _, a, _ in hits})} candidate(s) overlap {name} rows; promotion blocks held-out overlap")
    out.append("")

    texts = [r["text"] for r in candidates]
    op, ph = repeated_openings(texts), repeated_phrases(texts)
    out.append(f"Repeated openings (first 3 words, shown when >= {MIN_REPEAT}; cutoff only limits output): " + (", ".join(f"'{k}' x{v}" for k, v in op) or "none"))
    out.append("Repeated 4-word phrases: " + (", ".join(f"'{k}' x{v}" for k, v in ph[:8]) or "none"))
    if op or ph:
        warns.append("repeated openings or phrases among candidates (see above)")
    out.append("")

    ctx = context_table(candidates)
    out.append("Candidate context vs suggested label (a context tied to one label lets a model learn the context):")
    for c, by in sorted(ctx.items()):
        out.append(f"  {c:14} " + " | ".join(f"{lb} {n}" for lb, n in sorted(by.items())))
        if len(by) == 1 and sum(by.values()) >= 2:
            warns.append(f"context '{c}' maps to one suggested label ({next(iter(by))})")
    out.append("")

    for title, scope, rows, key in (("Cues in reviewed training rows (by label)", "reviewed training rows", tc.reviewed_rows(training), "label"),
                                    ("Cues in candidates (by SUGGESTED label, unverified)", "candidates", candidates, "suggested_label")):
        lines, w = _cue_lines(title, scope, rows, key)
        out += lines + [""]
        warns += w

    for title, rows, key in (("reviewed training rows", tc.reviewed_rows(training), "label"), ("candidates (suggested label)", candidates, "suggested_label")):
        ok, n, _ = clue_echo(rows, entries, key)
        out.append(f"Scanner clue-list echo, {title}: label equals what the clue lists alone suggest in {ok} of {n} rows"
                   + (f" ({ok / n:.0%})" if n else ""))
        if n >= MIN_ROWS_FOR_RATE_WARNING and ok / n >= 0.75:
            warns.append(f"{title}: clue lists alone agree with the label for {ok / n:.0%} of rows; a model may mostly relearn the scanner")
    out.append("")

    out.append(f"Warnings ({len(warns)}):")
    out += [f"  - {w}" for w in warns] or ["  none"]
    out.append("Diagnostics describe the data; they do not show it is good or that the model improved.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--training", type=Path, default=tc.DATA_PATH)
    ap.add_argument("--heldout", type=Path, default=gen.HELDOUT_PATH)
    ap.add_argument("--candidates", type=Path, default=gen.CANDIDATES_PATH)
    ap.add_argument("--terminology", type=Path, default=None)
    ap.add_argument("--pairs", type=int, default=5, help="how many example pairs to list")
    a = ap.parse_args(argv)
    try:
        entries = load_entries(a.terminology) if a.terminology else load_entries()
        warnings: list[str] = []
        training = gen._read_optional(a.training, "Training", warnings)
        heldout = gen._read_optional(a.heldout, "Held-out", warnings)
        candidates = gen.read_candidates(a.candidates)
        for w in warnings:
            print(f"WARNING: {w}")
        print(report(training, heldout, candidates, entries, a.pairs))
    except (tc.DatasetError, DictionaryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
