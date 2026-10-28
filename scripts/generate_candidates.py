"""Generate candidate example sentences for HUMAN REVIEW. Offline: plain Python templates only.

    uv run python -m scripts.generate_candidates                 # append new candidates to data\\candidate_examples.csv
    uv run python -m scripts.generate_candidates --dry-run       # show what would happen, write nothing
    uv run python -m scripts.generate_candidates --coverage-only # just print coverage

Nothing generated here is trusted:
  * every row is written with review_status = needs_review and an EMPTY label;
  * `suggested_label` is only an unverified guess by whoever wrote the sentence; `context` says how it is built, not what it is;
  * data/training_examples.csv and data/heldout_examples.csv are only read, never written.

The sentences are hand-written (one distinct sentence per term and context, no word-swapping).
A term/context pair with no natural sentence is simply left out.
Generation is deterministic: the same dictionary and files always give the same rows in the same order.
Run `python -m scripts.dataset_report` for diagnostics (repetition, near-duplicates, label cues).
Existing rows in the candidates file are never rewritten; new rows are appended.
Duplicates are exact matches only (ignoring case and surrounding spaces). A separate
word-overlap warning is printed, but it is NOT a semantic duplicate check.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from app.dictionary import DictionaryError, load_entries
from scripts import train_classifier as tc

CANDIDATES_PATH = tc.ROOT / "data" / "candidate_examples.csv"
HELDOUT_PATH = tc.ROOT / "data" / "heldout_examples.csv"
REJECTED = "rejected"
COLUMNS = tc.COLUMNS + ("suggested_label", "context")
# What each context tag means. A context says how the sentence is built, NOT what its label is:
# every context is used with more than one suggested label where a natural sentence exists.
CONTEXTS = {
    "casual": "everyday remark or joke",
    "disagreement": "a quarrel or difference of opinion",
    "academic": "explanation, definition or study of the term",
    "in-context": "a concrete situation with enough detail to judge",
    "negation": "the sentence contains a negation (what it means depends on the sentence)",
    "ambiguous": "key information is missing",
    "reported": "someone else's claim or quoted words",
    "question": "asks about the term instead of applying it",
    "counterexample": "contrasts a non-example with an example",
}
P, M, N = "PLAUSIBLE USAGE", "POSSIBLE MISUSE", "NEEDS CONTEXT"

# term -> [(context, provisional label, sentence)]. Hand-written, one distinct sentence each.
# The label is a PROVISIONAL GUESS for the reviewer to confirm or overrule, never a final label.
# Triggered, Red flag and Emotional damage have no POSSIBLE MISUSE guess: this project treats casual
# use of those as ordinary slang. Terms and contexts without a natural sentence are simply absent.
TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "Gaslighting": [
        ("casual", M, "He said the concert started at eight, I said seven, and now I'm telling everyone it's gaslighting."),
        ("disagreement", M, "Group chat verdict on my coworker pointing out a typo in my slide: gaslighting."),
        ("in-context", P, "She told me I had never asked for the day off, then showed me a calendar I had no memory of editing. It happened with the rent, the car and the holiday plans, and by spring I was keeping a diary just to trust myself. My counsellor called it gaslighting."),
        ("academic", P, "Researchers describing gaslighting usually stress that it unfolds across many interactions and aims to erode someone's confidence in their own judgement."),
        ("question", N, "Is it gaslighting if my manager says he never promised me the promotion, or did we just hear different things in that meeting?"),
        ("reported", N, "My neighbour keeps telling people her landlord is gaslighting her, though all she has told me is that he won't return her calls."),
        ("counterexample", P, "A single wrong date is just a mistake. What I mean by gaslighting is the year she spent denying events I had watched happen."),
        ("negation", N, "I'm not saying it was gaslighting; I only know that our accounts of that evening don't match."),
        ("negation", P, "She never admitted anything, and over the following months every conversation I remembered was met with 'that didn't happen'. I now understand that this was gaslighting."),
        ("question", P, "Was my roommate gaslighting me? Every week for a year she insisted she had never borrowed my things, even when I watched her carrying them out."),
        ("disagreement", P, "We argued about money again last night; she insists the hidden account never existed, as she has every time I raise it for two years, and I'm starting to think it's gaslighting."),
        ("ambiguous", N, "A coworker mentioned gaslighting after the meeting, and I couldn't tell who she meant or why."),
    ],
    "Gaslight": [
        ("casual", M, "My brother tried to gaslight me into believing the leftovers were his, which is dramatic for half a pizza."),
        ("disagreement", M, "My favourite band is overrated, she said, and I accused her of trying to gaslight me."),
        ("academic", P, "In the glossary, to gaslight is defined as manipulating a person over time so that they distrust their own memory and perception."),
        ("in-context", P, "At first he would only gaslight her about small things like where the keys were; by the end he was telling her the arguments she remembered had never happened."),
        ("question", N, "Can a coworker gaslight you by accident, or does it have to be on purpose?"),
        ("reported", N, "The review said the studio tried to gaslight its players about the bug, but it didn't list any examples."),
    ],
    "Narcissist": [
        ("casual", M, "She talked about her promotion for a whole minute, what a narcissist."),
        ("negation", M, "She didn't ask about my day once, so she must be a narcissist."),
        ("disagreement", M, "After our argument about the holiday route my sister called me a narcissist for wanting the coastal road."),
        ("academic", P, "Clinicians reserve the diagnosis of narcissistic personality disorder for a long-standing pattern across many situations, which is different from occasionally acting self-centred."),
        ("in-context", P, "Her father was assessed by a psychiatrist over several sessions and diagnosed with narcissistic personality disorder, which gave the family a name for decades of behaviour they had struggled to explain."),
        ("question", M, "Is my sister a narcissist just because she picked the other restaurant?"),
        ("question", N, "Does anyone else think our new manager is a narcissist, or is she just confident?"),
        ("reported", P, "My aunt says his ex was a narcissist: according to her, the woman forged his signature, lied about their money and cut him off from his friends."),
        ("negation", N, "I'd hesitate to call my dad a narcissist, though he can be hard to talk to."),
    ],
    "Toxic": [
        ("casual", M, "When I said her playlist was boring, my friend announced that I'm toxic."),
        ("negation", M, "He didn't laugh at my joke, so I told everyone he's toxic."),
        ("academic", P, "The safety sheet classifies the solvent as toxic if inhaled and requires gloves and a fume hood."),
        ("in-context", P, "Our old team lead mocked people in meetings, took credit for their work and punished anyone who objected. People who left called the culture toxic."),
        ("question", N, "Would you call a workplace toxic if everyone is stressed but nobody is being mistreated?"),
        ("reported", M, "My cousin says her book club is toxic because two members voted for a different novel."),
        ("counterexample", P, "Disagreeing about politics at the table is not the same as an environment where people are belittled daily, and only the second one I'd call toxic."),
        ("ambiguous", N, "Her last job was toxic, she says, but I never found out what happened there."),
    ],
    "Trauma": [
        ("casual", M, "The group presentation got moved to 8 a.m. and I swear I'm going to need trauma counselling."),
        ("casual", M, "Spilling coffee on my new shirt was a trauma I will never recover from."),
        ("academic", P, "Clinical definitions of trauma focus on exposure to serious injury, violence or threat to life, and acknowledge that people respond very differently."),
        ("in-context", P, "After the flood he couldn't sleep and startled at heavy rain for months, and eventually he started seeing a specialist for trauma."),
        ("question", N, "Where does stress end and trauma begin, in your experience?"),
        ("reported", N, "She told the interviewer her childhood was full of trauma, and the interviewer did not ask her to elaborate."),
        ("negation", P, "He hadn't talked about the accident in years, and when he finally did, his doctor explained that avoidance is common after trauma."),
        ("counterexample", P, "The surgeon examined the blunt trauma to his ribs."),
        ("ambiguous", N, "Something happened at that job that she calls trauma, and she changes the subject whenever I ask."),
    ],
    "Triggered": [
        ("casual", N, "My friend said she got triggered when I changed the subject, then laughed and ordered another round."),
        ("casual", N, "Don't get triggered, it was just a joke."),
        ("academic", P, "In trauma-informed practice, being triggered means that a sound, smell or other reminder brings distressing memories of a past event back."),
        ("in-context", P, "When a car backfired near the market, he froze and covered his head; later he explained that the noise had triggered memories of the shelling."),
        ("question", N, "When people say 'triggered' to mean annoyed, does the word lose its serious meaning?"),
        ("reported", N, "Several commenters said the video left them triggered, though none said what it reminded them of."),
    ],
    "Red flag": [
        ("casual", N, "He left a review of a restaurant he'd never been to. Red flag, in my book."),
        ("casual", N, "Pineapple on pizza is a red flag, fight me."),
        ("academic", P, "Fraud investigators train staff to notice red flags like invoices that sit just under the approval limit."),
        ("in-context", P, "He insisted on reading her messages, got angry whenever she saw her sister, and threatened to leave if she went on the trip; her friends told her those were red flags."),
        ("question", N, "Should someone with no friends from their past count as a red flag, or are some people just private?"),
        ("reported", P, "She told me the guy had 'a huge red flag': he'd lied about being single."),
        ("negation", N, "Being quiet on the first date isn't automatically a red flag."),
    ],
    "Love bombing": [
        ("casual", M, "My neighbour waved and brought me cookies on the day I moved in, and my sister immediately called it love bombing."),
        ("negation", M, "She hadn't even met me twice before she said she missed me, which I decided was love bombing."),
        ("academic", P, "The term love bombing is used in popular writing for an early flood of affection that is later withdrawn, though it is not an official diagnosis."),
        ("in-context", P, "For three weeks he sent flowers to her office and called her his soulmate; as soon as she mentioned a weekend away with friends, the warmth switched off and he began to criticise her. She now sees it as love bombing."),
        ("question", N, "If someone you've been seeing for two weeks says they're falling for you, is that love bombing?"),
        ("reported", N, "The article quotes several women who say they were love bombed, without describing what happened afterwards."),
        ("negation", P, "He never gave her a moment alone: gifts every day, messages at all hours, and when she pushed back, silence for a week. The therapist's word for it was love bombing."),
    ],
    "Manipulative": [
        ("casual", M, "My roommate asked nicely if I could keep the music down and I told her she was being manipulative."),
        ("academic", P, "Persuasion becomes manipulative when it relies on deception or on exploiting someone's fears rather than on honest reasons."),
        ("in-context", P, "He told the elderly couple the offer expired that evening, which was false, and kept calling to scare them into signing; the regulator described the sales tactics as manipulative."),
        ("question", N, "My friend always makes me feel guilty for saying no: manipulative, or just needy?"),
        ("reported", M, "Several reviewers called the ad manipulative because it asked viewers to donate."),
        ("negation", N, "She pushes hard, but she's very direct rather than manipulative."),
        ("counterexample", P, "Asking a favour isn't manipulative. Pretending a deadline exists so that someone feels forced to agree is."),
    ],
    "Boundaries": [
        ("casual", M, "I set a boundary that my partner can't have friends over, because those are my boundaries."),
        ("disagreement", M, "My mum asked when I would be home for dinner, and I said she was violating my boundaries."),
        ("academic", P, "Therapists often describe a boundary as a statement about what you will do, such as leaving a conversation, rather than an instruction for someone else."),
        ("in-context", P, "Once her brother started shouting, she said calmly that she would hang up, and she did. She has kept to that boundary ever since."),
        ("counterexample", P, "The county surveyor updated the boundaries of the parcel after the river shifted."),
        ("question", M, "Is it a boundary if I tell my partner he can't see his sister?"),
        ("question", N, "When you tell someone to choose between you and their friends, is that a boundary or an ultimatum?"),
        ("reported", N, "My manager keeps saying he 'respects boundaries', and I haven't seen what that looks like."),
    ],
    "Toxic positivity": [
        ("casual", M, "She congratulated me on the promotion with three exclamation marks and a party emoji, and my friend called it toxic positivity."),
        ("academic", P, "Writers on emotional well-being use toxic positivity for responses that brush aside distress, such as telling a grieving person to be grateful."),
        ("in-context", P, "When he told his team that his father had died, the manager said everything happens for a reason and asked everyone to stay upbeat for the client call. It felt like toxic positivity."),
        ("question", N, "Telling a friend their interview will probably go fine: kind encouragement or toxic positivity?"),
        ("reported", N, "Reddit threads accuse the app of toxic positivity, though the posts I read were short."),
        ("in-context", P, "She answered every mention of my layoff with 'good vibes only' and changed the subject, which is toxic positivity."),
    ],
    "Emotional damage": [
        ("casual", N, "He roasted my haircut in front of everyone and I just whispered 'emotional damage' to myself."),
        ("casual", N, "Lost to a ten-year-old at chess. Emotional damage."),
        ("academic", P, "Support-service guides describe emotional damage as lasting harm that can follow prolonged bullying, coercion or neglect."),
        ("in-context", P, "Years of being told she was useless left her anxious and withdrawn, and her therapist spoke openly about the emotional damage."),
        ("question", N, "Does calling a bad review 'emotional damage' water down the phrase, or is it harmless fun?"),
        ("reported", N, "He claims the divorce caused him emotional damage but hasn't said how."),
        ("in-context", P, "The insult was cruel, and the weeks of silence afterwards did real emotional damage."),
    ],
    "Attachment issues": [
        ("casual", M, "My boyfriend likes to say goodnight every evening, so my friend says he has attachment issues."),
        ("disagreement", M, "She said I text too slowly and told me I clearly have attachment issues."),
        ("academic", P, "Developmental researchers study attachment styles rather than attachment issues, a phrase that is not itself a diagnosis."),
        ("in-context", P, "After being moved between six foster homes before the age of four, the boy found it hard to trust any carer, and the assessment noted attachment issues."),
        ("question", N, "Do attachment issues in adults always trace back to childhood, or can they develop later?"),
        ("reported", M, "People in the comments keep diagnosing him with attachment issues because he answered an interview question slowly."),
        ("negation", N, "He's clingy sometimes, but I'm not claiming it's attachment issues."),
    ],
    "Narcissistic abuse": [
        ("casual", M, "My flatmate rolled her eyes at my cooking and I told the group chat I was a victim of narcissistic abuse."),
        ("academic", P, "Authors disagree about narcissistic abuse: some use it for sustained emotional abuse, while critics note it is not a clinical category."),
        ("in-context", P, "For eleven years her husband mocked her in front of friends, controlled the bank card and told her she was too stupid to leave. In group therapy she learned to describe this as narcissistic abuse."),
        ("question", N, "Does narcissistic abuse describe something specific, or does it blur emotional abuse in general?"),
        ("reported", N, "A podcast host says her ex committed narcissistic abuse but only gave a general summary."),
        ("negation", N, "Our fight was ugly, but calling it narcissistic abuse feels like a stretch."),
    ],
    "Trauma dumping": [
        ("casual", M, "My friend told me about her awful commute and my other friend whispered 'trauma dumping'."),
        ("disagreement", M, "My sister wanted to talk through her exam stress for twenty minutes, so I accused her of trauma dumping."),
        ("academic", P, "Etiquette writers define trauma dumping as sharing distressing material without checking that the listener is ready to hear it."),
        ("in-context", P, "A man I had just met at the wedding began describing his worst year in intense detail and didn't pause when I looked away. I felt trapped, and later realised it was trauma dumping."),
        ("question", N, "If a friend asks how you are and you answer honestly, is that trauma dumping?"),
        ("reported", N, "Someone replied 'that's trauma dumping' to her post, though the post only said she'd had a rough week."),
        ("negation", P, "She never asked if I had time; she simply started in on her whole history while I was trying to leave. That is trauma dumping."),
    ],
}


def _key(text: str) -> str:
    """Exact-duplicate key: ignores case and surrounding whitespace, nothing else."""
    return text.strip().lower()


def _has_pattern(text: str, patterns) -> bool:
    for p in patterns:
        body = r"[\s-]+".join(re.escape(part) for part in re.split(r"[\s-]+", p.strip()))
        if re.search(r"(?<!\w)" + body + r"(?!\w)", text, re.IGNORECASE):
            return True
    return False


def build_pool(entries) -> tuple[list[tuple[str, str, str, str]], dict[str, list[str]]]:
    """(term, context, provisional label, text) from the templates, plus a report of anything left out and why."""
    by_term = {e.term: e for e in entries}
    report = {"no_templates": [], "unknown_template_terms": [], "dropped": []}
    pool = []
    for term, entry in by_term.items():
        if term not in TEMPLATES:
            report["no_templates"].append(term)
            continue
        for context, guess, text in TEMPLATES[term]:
            if context in CONTEXTS and guess in tc.LABELS and _has_pattern(text, entry.patterns):
                pool.append((term, context, guess, text))
            else:
                report["dropped"].append(f"{term}: {text[:50]}")  # the term as the scanner sees it is missing
    report["unknown_template_terms"] = sorted(set(TEMPLATES) - set(by_term))
    return pool, report


def _read_optional(path: Path, what: str, warnings: list[str]) -> list[dict[str, str]]:
    """Read-only helper. A missing or empty file is a warning, not an error; a malformed one is an error."""
    if not path.is_file() or path.stat().st_size == 0:
        warnings.append(f"{what} file missing or empty ({path}); duplicates against it were not checked")
        return []
    return tc.load_rows(path)


def read_candidates(path: Path) -> list[dict[str, str]]:
    """Existing candidates. Missing or empty file = none. Any other header is refused, so we never append to the wrong file."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise tc.DatasetError(f"{path}: unexpected header {reader.fieldnames}; expected {list(COLUMNS)}. Not touching it.")
        return [{k: (r.get(k) or "") for k in COLUMNS} for r in reader]


def _append(path: Path, rows: list[dict[str, str]]) -> None:
    """Append rows without rewriting what is already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    needs_newline = False
    if not fresh:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) not in (b"\n", b"\r")
    with open(path, "a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\r\n")
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def candidate_lines(training: list[dict[str, str]], candidates: list[dict[str, str]], terms: list[str]) -> list[str]:
    """Candidate status per term, plus the label gaps they might (provisionally) help with."""
    reviewed = Counter((r["term"], r["label"]) for r in training if r["review_status"] == tc.REVIEWED)
    pending = Counter((r["term"], r["suggested_label"]) for r in candidates if r["review_status"] == tc.NEEDS_REVIEW)
    status = Counter((r["term"], r["review_status"]) for r in candidates)
    totals = {t: sum(reviewed[(t, label)] for label in tc.LABELS) for t in terms}
    out = ["Candidates by term (provisional suggestions only; none are trusted):"]
    for t in terms:
        line = (f"  {t}: {status[(t, tc.NEEDS_REVIEW)]} pending, {status[(t, tc.REVIEWED)]} approved "
                f"(not yet in training), {status[(t, REJECTED)]} rejected")
        gaps = [f"no reviewed {label} ({pending[(t, label)]} pending suggest it)" for label in tc.LABELS if reviewed[(t, label)] == 0]
        out.append(line + (" | gaps: " + "; ".join(gaps) if gaps else ""))
    thin = sorted(terms, key=lambda t: (totals[t], t))[:3]
    out.append("  Fewest reviewed rows (relative ranking, no minimum implied): "
               + ", ".join(f"{t} ({totals[t]})" for t in thin))
    out.append("  Counts show gaps, not quality. Read every candidate before approving it.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--training", type=Path, default=tc.DATA_PATH, help="read only")
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH, help="read only")
    ap.add_argument("--terminology", type=Path, default=None, help="default: data/terminology.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--coverage-only", action="store_true")
    args = ap.parse_args(argv)
    try:
        protected = {args.training.resolve(), args.heldout.resolve()}
        if args.output.resolve() in protected:
            raise tc.DatasetError(f"Refusing to write to {args.output}: that is a training or held-out file.")
        entries = load_entries(args.terminology) if args.terminology else load_entries()
        warnings: list[str] = []
        training = _read_optional(args.training, "Training", warnings)
        heldout = _read_optional(args.heldout, "Held-out", warnings)
        existing = read_candidates(args.output)
        for w in warnings:
            print(f"WARNING: {w}")

        new_rows: list[dict[str, str]] = []
        if not args.coverage_only:
            seen: dict[str, str] = {}
            for source, rows in (("training", training), ("held-out", heldout), ("earlier candidates", existing)):
                for r in rows:
                    seen.setdefault(_key(r["text"]), source)
            pool, report = build_pool(entries)
            skipped: Counter = Counter()
            for term, context, guess, text in pool:
                k = _key(text)
                if k in seen:
                    skipped[seen[k]] += 1
                    continue
                seen[k] = "repeated in this run"
                new_rows.append({"term": term, "text": text, "label": "", "review_status": tc.NEEDS_REVIEW,
                                 "suggested_label": guess, "context": context})
            refs = [r["text"] for r in training + heldout]
            overlap = [r for r in new_rows if tc.closest(r["text"], refs)[0] >= tc.SIMILARITY]
            if not args.dry_run and new_rows:
                _append(args.output, new_rows)
            verb = "Would write" if args.dry_run else "Wrote"
            print(f"{verb} {len(new_rows)} new candidate(s) to {args.output} (all review_status=needs_review, label empty).")
            print(f"Skipped {sum(skipped.values())} exact duplicate(s): " + (", ".join(f"{n} in {s}" for s, n in sorted(skipped.items())) or "none")
                  + ". Only exact matches (ignoring case and surrounding spaces) are detected, not paraphrases.")
            print(f"Word-overlap warnings: {len(overlap)} new row(s) share >= {tc.SIMILARITY:.0%} of their words with a training or held-out row"
                  " (a rough overlap measure, not a semantic check; they were still written).")
            for r in overlap:
                print(f"  overlap: {r['text'][:70]}")
            for key, label in (("no_templates", "Terms with no templates (skipped)"),
                               ("unknown_template_terms", "Templates for terms missing from the dictionary (ignored)"),
                               ("dropped", "Templates dropped (term not in the sentence, or unknown context or label)")):
                if report[key]:
                    print(f"{label}: " + "; ".join(report[key]))
            print()

        terms = [e.term for e in entries]
        print("\n".join(tc.coverage_lines(training, set(terms))))
        print()
        print("\n".join(candidate_lines(training, existing + new_rows, terms)))
        print("\nDo not evaluate the model on rows it was trained or tuned on, and never copy candidates into the held-out file.")
    except (tc.DatasetError, DictionaryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
