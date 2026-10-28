# Annotation guide: what the three labels mean

A label describes **how a term is used in the text, judged from the text alone**. It never says anything about
what really happened between people, and never about anyone's mind or health. Use the dictionary (`data/terminology.json`:
`definition` and `guidance`) as the reference for what each term means.

| Label | Use it when the text... | Not when... |
|---|---|---|
| `POSSIBLE MISUSE` | gives facts that, by the dictionary's own guidance, are *not* what the term covers: a trivial disagreement, a single everyday annoyance, a plain request or preference, hyperbole about a minor event | the facts are missing, or could fit the term |
| `PLAUSIBLE USAGE` | gives facts consistent with the dictionary definition (a sustained pattern, a clinical or literal sense, a specific concrete situation) or discusses the term the way the definition does (explanation, teaching) | it only asserts the term with no facts |
| `NEEDS CONTEXT` | does not give enough to judge, points both ways, or does not apply the term (denies it, asks about it) | it merely contains a negation, quote or hedge but still gives enough facts to decide |

"Plausible" means consistent with the definition. It is not a verdict that the term is right.

## Borderline cases (current convention, please confirm)

1. **Negation: read the meaning.** Denying the term applies ("it wasn't gaslighting") is not a use of it, so `NEEDS CONTEXT`. This matches the scanner. A negation inside the evidence ("she never admitted it, and for months...") does not change the label: judge by the evidence.
2. **Reported speech and accusations.** Label what the text itself establishes. A bare quote or claim with no facts: `NEEDS CONTEXT`. A report that includes facts: judge those facts.
3. **Questions.** `NEEDS CONTEXT` unless the question states facts that decide it.
4. **Academic or explanatory sentences** that describe the term as the dictionary does: `PLAUSIBLE USAGE`. One that describes it loosely or wrongly: reject it or fix the sentence.
5. **Literal or non-psychological senses** (toxic waste, blunt trauma, property boundaries): `PLAUSIBLE USAGE`.
6. **Jokes and hyperbole.** Label the facts described. A joke about a trivial event is `POSSIBLE MISUSE` (loose use), not an insult to anyone. A joke with no facts: `NEEDS CONTEXT`.
7. **Informal terms** (Triggered, Red flag, Emotional damage): casual use is ordinary slang, so `NEEDS CONTEXT`. This project's scanner never flags misuse for these terms, so few or no `POSSIBLE MISUSE` rows are expected.
8. **Two reasonable readers would pick different labels and editing the sentence can't fix it:** reject the sentence. Do not use `NEEDS CONTEXT` as a dumping ground.

## Examples

| Text | Label | Why |
|---|---|---|
| "He said the concert started at eight, I said seven, and now I'm telling everyone it's gaslighting." | `POSSIBLE MISUSE` | a disagreement about a time |
| "She never admitted anything, and for months every conversation I remembered was met with 'that didn't happen'. I now understand that this was gaslighting." | `PLAUSIBLE USAGE` | negation, but sustained pattern described |
| "I'm not saying it was gaslighting; I only know our accounts don't match." | `NEEDS CONTEXT` | the term is denied; no facts |
| "Is it gaslighting if my manager says he never promised me the promotion?" | `NEEDS CONTEXT` | a question with too little detail |
| "Was my roommate gaslighting me? Every week for a year she insisted she had never borrowed my things, even when I watched her carry them out." | `PLAUSIBLE USAGE` | a question, but it states a sustained pattern |
| "The county surveyor updated the boundaries of the parcel." | `PLAUSIBLE USAGE` | literal sense |

## Open questions in the current project definitions

These are real inconsistencies. I have not changed the labelling policy; each needs your decision.

1. **Scanner labels vs dataset labels measure different things.** The scanner's category says what nearby clue words suggest; the dataset label says how the term is used. They can legitimately differ (the dictionary example "In psychology class we compared gaslighting with ordinary disagreement." is `PLAUSIBLE USAGE` by definition, but the scanner's clue lists mix "psychology" and "disagree"). Proposal: keep dataset labels about meaning, and never treat scanner agreement as correctness.
2. **`NEEDS CONTEXT` mixes four situations:** no information, conflicting information, denied, asked-about. One class, four reasons. Proposal: if the model keeps struggling, record a `reason` in the review notes first, before splitting the label.
3. **Denial vs. the instruction not to label negation as `NEEDS CONTEXT` automatically.** Rule 1 reconciles them (the denial itself is `NEEDS CONTEXT`, negation inside evidence is not), but this is a judgement call on "denied" sentences that also give facts, e.g. "It wasn't gaslighting; we both had the date wrong." Proposal: keep `NEEDS CONTEXT`, since the term is not applied.
4. **Existing training labels were largely derived from the scanner's own tests and examples.** Many reviewed training rows therefore echo the scanner's clue lists (see the clue-echo line in `dataset_report`). A model trained on them may mostly relearn the scanner.
5. **Dictionary "misuse" examples for Triggered, Red flag and Emotional damage** conflict with the scanner's design (those terms never get `POSSIBLE MISUSE`). They were set to `NEEDS CONTEXT` and `needs_review` earlier; they still need your decision.
