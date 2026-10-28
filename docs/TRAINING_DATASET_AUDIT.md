# Training dataset quality audit

Audit date: 2026-10-24. Refreshed after user-approved label changes in training rows 67, 68, and 90, and promotion of candidate rows 16, 25, and 73. The dataset report was run with `python -m scripts.dataset_report`; all warnings below are heuristic review prompts, not proof that an example is wrong or redundant.

## Highest-priority issues

1. **NEEDS CONTEXT coverage is still concentrated in Gaslighting.** Training has 86 reviewed rows: 17 `NEEDS CONTEXT` (13 Gaslighting, and 1 each for Trauma, Toxic, Triggered, and Trauma dumping), 39 `PLAUSIBLE USAGE`, and 30 `POSSIBLE MISUSE`. Ten terms still have no reviewed `NEEDS CONTEXT` row: Attachment issues, Boundaries, Emotional damage, Gaslight, Love bombing, Manipulative, Narcissist, Narcissistic abuse, Red flag, and Toxic positivity. This is a gap to consider, not a quota.
2. **Gaslighting dominates the near-duplicate alerts and training set.** All 18 training near-pairs are for Gaslighting. Four are very close versions of the same “movie disagreement = gaslighting” misuse example; multiple other pairs repeat the same alleged-denial scenario. The model may learn repeated phrasing more strongly than the intended distinctions.
3. **The reviewed-label conflicts in the identified pairs have been resolved by approval.** Rows 67, 68, and 86 now agree at `NEEDS CONTEXT`; rows 73 and 90 now agree at `PLAUSIBLE USAGE`. Row 73/90 remains a near-duplicate warning, but its difference is clause order rather than a meaningful training distinction.
4. **Two training sentences contain U+FFFD replacement characters** (rows 63 and 82: `didn�t`). This is visible text corruption, likely from encoding conversion. Do not silently repair it; confirm the intended apostrophe and re-save as UTF-8 during human review.
5. **Many non-reviewed training rows still contain labels.** The report counts 38 training rows as `needs_review`; their labels are not verified ground truth. The candidate set has 72 rows: 3 reviewed and promoted, 69 still `needs_review` with blank labels; held-out has 21 `needs_review` rows with guessed labels. The held-out labels remain unverified.

## Likely near-duplicate pairs in training

The report threshold is word-set Jaccard >= 0.60, ignoring order and repeated words. Every pair below has term `Gaslighting`; every row status is `reviewed`. Row numbers are CSV line numbers (header is line 1). Keep the complete text and status visible so the warnings can be judged semantically.

| Rows; score | Labels | Complete sentences | Assessment |
|---|---|---|---|
| 2, 62; 0.60 | `POSSIBLE MISUSE` / `POSSIBLE MISUSE` | “He disagreed with me about the movie ending. That's gaslighting.” / “He disagreed with me, that's gaslighting.” | Redundant: same unsupported leap from disagreement to gaslighting. |
| 2, 65; 0.75 | `POSSIBLE MISUSE` / `POSSIBLE MISUSE` | “He disagreed with me about the movie ending. That's gaslighting.” / “He disagreed with me about the movie ending, so he's gaslighting me.” | Redundant: only claim framing changes. |
| 2, 66; 0.90 | `POSSIBLE MISUSE` / `POSSIBLE MISUSE` | “He disagreed with me about the movie ending. That's gaslighting.” / “He disagreed with me about the movie. That's gaslighting.” | Highly redundant paraphrase. |
| 62, 66; 0.67 | `POSSIBLE MISUSE` / `POSSIBLE MISUSE` | “He disagreed with me, that's gaslighting.” / “He disagreed with me about the movie. That's gaslighting.” | Redundant: same distinction as the three rows above. |
| 65, 66; 0.67 | `POSSIBLE MISUSE` / `POSSIBLE MISUSE` | “He disagreed with me about the movie ending, so he's gaslighting me.” / “He disagreed with me about the movie. That's gaslighting.” | Redundant. |
| 66, 67; 0.60 | `POSSIBLE MISUSE` / `NEEDS CONTEXT` | “He disagreed with me about the movie. That's gaslighting.” / “He was gaslighting me about the movie.” | Different evidence levels: the first explicitly overstates a disagreement; the second is a bare allegation and does not establish misuse. |
| 67, 68; 0.64 | `NEEDS CONTEXT` / `NEEDS CONTEXT` | “He was gaslighting me about the movie.” / “He wasn't rude but he was gaslighting me about the movie ending.” | Near paraphrases; neither provides behavior evidence to judge the allegation. “Wasn't rude” does not establish gaslighting. |
| 67, 82; 0.60 | `NEEDS CONTEXT` / `NEEDS CONTEXT` | “He was gaslighting me about the movie.” / “He wasn�t gaslighting me, just disagreeing about the movie.” | Meaningful polarity contrast; row 82 has corrupted punctuation and needs text confirmation. |
| 67, 86; 0.70 | `NEEDS CONTEXT` / `NEEDS CONTEXT` | “He was gaslighting me about the movie.” / “He wasn't rude but he was gaslighting me about the movie.” | Similar unsupported allegations; labels now align under the approved policy. |
| 68, 86; 0.91 | `NEEDS CONTEXT` / `NEEDS CONTEXT` | “He wasn't rude but he was gaslighting me about the movie ending.” / “He wasn't rude but he was gaslighting me about the movie.” | Near-identical examples; labels now align under the approved policy. |
| 68, 89; 0.60 | `NEEDS CONTEXT` / `NEEDS CONTEXT` | “He wasn't rude but he was gaslighting me about the movie ending.” / “He wasn't gaslighting me about the movie ending at first, but then he apologised.” | Different polarity and timeline; potentially useful contrast. Row 89 adds an apology but not a clear denial pattern. |
| 72, 73; 0.73 | `PLAUSIBLE USAGE` / `PLAUSIBLE USAGE` | “He denied it happened and made me doubt my memory, which is gaslighting.” / “He repeatedly denied it happened and made me doubt myself, which is gaslighting.” | Redundant: both describe the same pattern; row 73 adds repetition but the pattern is already in the concept. |
| 72, 77; 0.62 | `PLAUSIBLE USAGE` / `PLAUSIBLE USAGE` | “He denied it happened and made me doubt my memory, which is gaslighting.” / “He kept denying it, again and again, and made me doubt my memory. He is gaslighting me.” | Mostly redundant; row 77 adds emphasis, not a distinct boundary. |
| 73, 88; 0.73 | `PLAUSIBLE USAGE` / `NEEDS CONTEXT` | “He repeatedly denied it happened and made me doubt myself, which is gaslighting.” / “He repeatedly denied it happened and made me doubt myself. That wasn't gaslighting.” | Meaningful assertion/negation contrast, but the same stated evidence supports opposite conclusions. Human review should establish whether this is intentionally an uncertainty example. |
| 73, 90; 0.79 | `PLAUSIBLE USAGE` / `PLAUSIBLE USAGE` | “He repeatedly denied it happened and made me doubt myself, which is gaslighting.” / “That's gaslighting. He repeatedly denied it happened and made me doubt myself.” | Redundant paraphrases: same evidence and assertion, with clause order changed. Keep one unless the order difference is intentionally useful. |
| 74, 75; 0.61 | `PLAUSIBLE USAGE` / `PLAUSIBLE USAGE` | “He wasn't gaslighting me at first, but later repeatedly denied conversations we'd both had and insisted I was imagining things.” / “For months, he repeatedly denied conversations we'd both had and insisted I was imagining things. That's gaslighting.” | Similar evidence; row 74 adds a before/after contrast, which is a useful temporal distinction. Consider keeping only if that contrast is a deliberate target. |
| 84, 85; 0.60 | `NEEDS CONTEXT` / `NEEDS CONTEXT` | “He was gaslighting me.” / “He wasn't gaslighting me.” | Useful minimal polarity pair, but both have no supporting context; reasonable `NEEDS CONTEXT` examples if that is the intended unit of annotation. |
| 88, 90; 0.79 | `NEEDS CONTEXT` / `PLAUSIBLE USAGE` | “He repeatedly denied it happened and made me doubt myself. That wasn't gaslighting.” / “That's gaslighting. He repeatedly denied it happened and made me doubt myself.” | Useful assertion/negation contrast: row 88's conclusion conflicts with the stated evidence; row 90's conclusion fits it under the approved policy. |

Overall: the most obvious redundancy is the cluster rows 2/62/65/66, all making the same ordinary-disagreement misuse point. Rows 67/68/86 now form a consistent `NEEDS CONTEXT` cluster of bare allegations. Rows 73/90 are same-label near-duplicates; rows 88/90 form a useful contradiction/affirmation contrast under the approved policy. Rows 84/85 also show polarity, but both lack supporting context.

## Coverage and behavior distinctions

- **Ordinary disagreement:** well represented for Gaslighting, but repeated with nearly identical movie examples. Gaslight has two reviewed misuse examples (one additional dinner argument in row 69), but no reviewed `NEEDS CONTEXT` example.
- **Ambiguous allegation:** Gaslighting has several `NEEDS CONTEXT` rows, though short bare assertions such as “He was gaslighting me.” cannot teach what context makes the claim uncertain. Toxic, Triggered, and Trauma dumping now each have one reviewed ambiguous-context example; ten other terms still have no reviewed `NEEDS CONTEXT` row.
- **Negation:** there are 11 reviewed training sentences with negation cues; 8 are `NEEDS CONTEXT`, 2 `PLAUSIBLE USAGE`, and 1 `POSSIBLE MISUSE` (dataset report counts). Most are Gaslighting; the minimal pair at rows 84/85 is helpful, but negation plus later correction/contrast needs careful labeling.
- **Technical or literal meanings:** several useful examples already exist: Toxic waste (rows 12, 71), trauma in a medical context (row 17), attachment theory/diagnostic discussion (row 49), and Narcissistic Personality Disorder discussion (rows 8–9). These are positive distinctions, though some are two similar textbook/technical examples and several terms have only two reviewed rows total.
- **Interpret missing label coverage in light of term guidance:** Emotional damage, Red flag, and Triggered have no reviewed `POSSIBLE MISUSE` rows. The scanner deliberately has no misuse clue rules for these terms, but `terminology.json` still includes misuse examples (for example, a minor inconvenience called “emotional damage” or a quirk called a “red flag”). Scanner cues are not the training annotation policy; these coverage gaps should be filled only if a human-reviewed sentence genuinely demonstrates the approved definition's overreach, not by quota.
- **Well-supported contextual usage:** Gaslighting (rows 4, 72–77), Toxic, Love bombing, Manipulative, Narcissistic abuse, and Trauma dumping have examples naming patterns/behaviors. Gaslighting's adequate pattern evidence is undermined by repeated variants. More varied, concise contexts would be more valuable than adding further paraphrases.
- **Annotation policy limit:** repository guidance says ask whether the term is natural and fits the label; when a reasonable reader could choose differently, use `NEEDS CONTEXT` or reject. It does not fully specify whether annotation judges the sentence's *claim* against the described evidence, the *word's use*, or the speaker's assertion. That ambiguity affects negated and contradictory gaslighting examples. Resolve it before relabeling any row.

## Suggested new examples for human review

All labels below are **proposals, not ground truth**. These target meaningful missing `NEEDS CONTEXT` coverage without trying to equalize counts. Review naturalness and policy fit; edit or reject rather than accepting a suggestion automatically.

| Term | Proposed sentence | Proposed label | Why review it |
|---|---|---|---|
| Attachment issues | “My partner asks for reassurance and checks in often; I can't tell whether that reflects attachment issues or simply a stressful week.” | `NEEDS CONTEXT` | Distinguishes observable behavior from a psychological interpretation. |
| Emotional damage | “After the joke, he said it caused emotional damage, but I couldn't tell whether he meant lasting hurt or was joking with the group.” | `NEEDS CONTEXT` | Separates serious impact from a possibly playful phrase. |
| Red flag | “She canceled our plans twice and I felt uneasy, but I don't know whether that is a red flag or a temporary problem.” | `NEEDS CONTEXT` | Tests uncertainty about whether limited evidence indicates a warning sign. |
| Narcissistic abuse | “A support group member called the relationship narcissistic abuse; I heard only part of the story and can't judge the label from that alone.” | `NEEDS CONTEXT` | Addresses a contested popular category and limited second-hand evidence. |
| Gaslight | “My sister says I changed my story; I remember it differently, and I can't tell whether either of us is misremembering or gaslighting the other.” | `NEEDS CONTEXT` | Adds an ambiguous allegation to the spelling variant without duplicating the simple movie disagreement examples. |

The previously proposed Triggered, Toxic, and Trauma dumping examples are no longer needed for label coverage; an approved existing candidate batch now supplies reviewed `NEEDS CONTEXT` rows for those terms.

## Suggested review order

1. Confirm intended apostrophes in rows 63 and 82; repair only after human confirmation.
2. Review the ordinary-disagreement cluster rows 2, 62, 65, and 66 together. Decide which example best teaches the boundary and whether the others add any distinct signal. Rows 67 and 68 are now context-needed bare allegations, not explicit disagreement examples.
3. Review polarity/context pairs 73/88/90 and 84/85, plus timeline pair 74/75. Rows 73/90 are redundant; keep both only if clause order is an intentional contrast. Keep other contrasts only if annotators apply the labels consistently.
4. Consider the remaining proposed `NEEDS CONTEXT` examples for terms that still have no reviewed row. Do not add examples to equalize counts; require human approval of each sentence and label.

## Approved candidate review batch

The user approved these existing suggestions as `NEEDS CONTEXT`; their candidate status was changed to `reviewed` and the examples were appended to training as reviewed rows 123–125. This adds one reviewed `NEEDS CONTEXT` example each for Toxic, Triggered, and Trauma dumping.

| Candidate row | Term | Complete sentence | Why it may fill a gap |
|---|---|---|---|
| 16 → 123 | Toxic | “Her last job was toxic, she says, but she did not explain what happened.” | Hearsay and missing behavior details make the claim hard to assess. |
| 25 → 124 | Triggered | “He said he got triggered at the meeting, which could mean anything from irritation to something much heavier.” | Explicitly leaves informal and trauma-related meanings unresolved. |
| 73 → 125 | Trauma dumping | “Someone on the forum accused her of trauma dumping, but I can't tell whether anyone had agreed to listen.” | Consent context is central to the term and is deliberately missing. |

Before promotion, their maximum word-set Jaccard similarities were: row 16, 0.15 to training / 0.14 to held-out; row 25, 0.12 / 0.12; row 73, 0.24 / 0.14. The human approved these labels and text; the other candidate suggestions remain unverified.

## Validation

- `python -m scripts.dataset_report`: exited 0; 124 training rows, 72 candidates, 21 held-out rows; 0 exact cross/within-dataset pairs; 18 thresholded near-pairs.
- Direct split check using the same word-set Jaccard function: 0 exact and 0 >=0.60 near-duplicate pairs between training and held-out; all 21 held-out rows remain `needs_review`.
- `python -m scripts.train_classifier --validate-only`: exited 0; dataset valid with 86 reviewed rows and enough data to evaluate.
- Full tests after candidate promotion: `python -m pytest` with the repository `.venv` Python: **155 passed in 10.83s**.
- Rows 67 and 68 changed from `POSSIBLE MISUSE` to `NEEDS CONTEXT`; row 90 changed from `NEEDS CONTEXT` to `PLAUSIBLE USAGE`; candidate rows 16, 25, and 73 were approved as `NEEDS CONTEXT` and appended as reviewed training rows 123–125. No held-out rows, model code, or dependencies were changed.
