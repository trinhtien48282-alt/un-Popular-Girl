# Annotation policy

**Status: approved by the user on 2026-10-24.** This policy consolidates the existing workflow guidance and terminology entries. It does not itself change any dataset labels or annotation records.

## What the label describes

Label how the term is used in the supplied sentence and the context that sentence provides. Do not decide whether an allegation is factually true, diagnose a person, infer intent, or judge anyone's lived experience. Use only evidence stated in the text. When a term is clinical-adjacent or contested, apply that term's `definition` and `guidance` in `data/terminology.json`.

## Label definitions

### `PLAUSIBLE USAGE`

The sentence gives enough context for the term to fit its intended meaning or a recognized ordinary, literal, or technical meaning. For interpersonal or clinical-adjacent claims, the sentence should state behavior that supports the term's meaning; the label does not certify that the described events happened.

Examples of the distinction already represented in the data: repeated denial that undermines someone's confidence may support *gaslighting*; toxic waste is a literal use of *toxic*; medical blunt-force injury is a literal use of *trauma*; a textbook discussion can use a term accurately without applying it to a person.

### `POSSIBLE MISUSE`

The sentence itself gives a clear reason the term is being applied too broadly, used in the wrong sense, or treated as a substitute for evidence. Examples include calling one disagreement *gaslighting*, an ordinary inconvenience *trauma*, or a single selfish act proof of a clinical diagnosis. Label the wording and stated evidence, not the speaker's character or intent.

### `NEEDS CONTEXT`

The supplied text does not support a stable choice between plausible use and misuse. Use it for bare allegations, missing or disputed context, ambiguous senses, internally conflicting statements, or a reasonable alternative reading that the sentence does not resolve. Do not use it merely because a term has several meanings when the sentence clearly selects one.

## Cross-cutting rules

- **Negation is not a label by itself.** Read what the complete sentence says. A well-supported denial may be `PLAUSIBLE USAGE`; an unsupported bare denial may be `NEEDS CONTEXT`; a clearly overbroad use remains `POSSIBLE MISUSE`. A sentence whose evidence conflicts with its conclusion is usually `NEEDS CONTEXT` unless the wording clearly signals a deliberate misuse example.
- **Quoted and reported claims:** attribute the claim to the speaker. Quotation marks or phrases like “she said” do not prove or disprove it. Use `NEEDS CONTEXT` if the sentence provides too little evidence to assess how the term is being applied.
- **Ordinary and technical senses:** do not call common informal usage misuse merely because it is not clinical. Use the sentence and the term's entry to determine whether the intended sense is clear and fits.
- **Impact and consent:** for terms such as *trauma dumping*, use stated consent, capacity, setting, and impact. Sharing distress is not itself misuse; do not shame someone for seeking support.
- **Uncertainty:** if a reasonable annotator could choose a different label from the sentence as written, choose `NEEDS CONTEXT` or reject the example as unhelpful. Do not resolve uncertainty by guessing what happened off-page.
- **Text quality:** correct encoding or wording only after confirming intended text. A corrupted or unnatural sentence should be fixed or rejected before its label is reviewed.

## Review record

For each reviewed example, the annotator should be able to answer:

1. What sense of the term is used?
2. What words in the sentence support that sense or show overreach?
3. What context is missing, if `NEEDS CONTEXT` is selected?
4. Is the text natural and intact, including punctuation and encoding?

Similarity warnings identify sentences to compare; they do not justify deletion by themselves. Keep a near-duplicate only when it teaches a meaningful contrast, such as polarity, timeframe, or a different sense, and the labels apply consistently.

## Approval and review state

The user approved this policy on 2026-10-24, including the rule that labels assess term use against evidence stated in the sentence and do not judge whether an alleged event is objectively true. Negation alone does not determine a label; contradictory or unsupported claims need context when the sentence does not resolve them.

The approved corrections changed training CSV rows 67 and 68 to `NEEDS CONTEXT`, retained row 86 as `NEEDS CONTEXT`, and changed row 90 to `PLAUSIBLE USAGE`, matching row 73's label for the same described behavior. The user also approved candidate rows 16 (Toxic), 25 (Triggered), and 73 (Trauma dumping) as `NEEDS CONTEXT`; they were marked reviewed and appended to training.
