# Candidate example workflow

Generated sentences are **suggestions to read**, never training data. Nothing is promoted automatically.
All commands are PowerShell, from the repo root.

## Safety rules built in

- Every generated row is `review_status=needs_review` with an **empty `label`**. `suggested_label` is only a guess from the sentence's context type.
- The generator only *reads* `data\training_examples.csv` and `data\heldout_examples.csv`. It appends to `data\candidate_examples.csv` and never rewrites existing candidate rows.
- The training script ignores `needs_review` rows, and an unlabelled row fails dataset validation, so a candidate file cannot be trained on by accident.
- Duplicate checks are **exact** (ignoring case and surrounding spaces) against training, held-out and earlier candidates. A separate word-overlap warning is printed; it is not a semantic check, so read for paraphrases yourself.

Read `docs/ANNOTATION_GUIDE.md` before reviewing: it defines the three labels and the borderline cases.

## 1. Generate

```powershell
git status                                              # start from a clean tree so changes are easy to see
uv run python -m scripts.generate_candidates --dry-run  # preview, writes nothing
uv run python -m scripts.generate_candidates            # append new candidates to data\candidate_examples.csv
uv run python -m scripts.generate_candidates --coverage-only   # coverage report only
```

Running it again adds nothing new, and the output is the same every time. The sentences live in `TEMPLATES` in `scripts\generate_candidates.py`; each has a `context` (how it is built) and a provisional label guess. A term/context pair with no natural sentence is left out on purpose.

If your candidate file still holds rows from an older generator version, they stay as they are (rows are never rewritten) and the new sentences are added next to them. Mark old rows you don't want as `rejected`; rejected rows are remembered and never regenerated.

### Look at the data before reviewing it

```powershell
uv run python -m scripts.dataset_report
```

Read-only. It shows counts per term, label and review status, exact duplicates, word-overlap near-duplicates, same-sentence-different-term frames, repeated openings, which surface cues (negation, questions, quotes) go with only one label, how often the label matches what the scanner's clue lists alone would say, and overlap with training and held-out rows. It prints warnings, not pass/fail: a warning is a prompt to look, and a clean report does not show the data is good.

## 2. Review the CSV

Open `data\candidate_examples.csv` in VS Code (or a spreadsheet; if you use Excel, save as "CSV UTF-8", not tested here). Keep the header. Columns you edit: `label` and `review_status`.

| You want to | Set `label` | Set `review_status` |
|---|---|---|
| Approve | exactly one of `POSSIBLE MISUSE`, `PLAUSIBLE USAGE`, `NEEDS CONTEXT` (your decision, not the suggestion) | `reviewed` |
| Fix the sentence | edit `text` first, then as above | `reviewed` |
| Reject | leave empty | `rejected` |
| Decide later | leave empty | leave `needs_review` |

Ask of each sentence: is it natural, is the term used the way the label says, and could a reasonable reader pick a different label? If yes, `NEEDS CONTEXT` or reject.

## 3. Transfer approved rows

```powershell
uv run python -m scripts.promote_candidates           # dry run: lists what would be added
uv run python -m scripts.promote_candidates --apply   # appends to data\training_examples.csv
git diff data\training_examples.csv                   # should show added lines only
```

Only rows with `review_status` exactly `reviewed` and a valid label are copied. A typo or missing label stops the whole run with a message and changes nothing. Rows already in training are skipped. Rows identical or close to a held-out row are **blocked**. Existing training rows are never changed.

## 4. Validate, retrain, test

```powershell
uv run python -m scripts.train_classifier --validate-only   # also prints the coverage table
uv run python -m scripts.train_classifier                    # evaluate (grouped cross-validation) + train + save model
uv run pytest
```

## Held-out data

- Never copy candidates into `data\heldout_examples.csv`, and never evaluate on rows used to train or tune the model.
- Better scores from `train_classifier` after adding data are cross-validation numbers on your own data. They do not show the model improved on unseen text. Real evidence needs a separate, human-reviewed, never-trained-on set (`uv run python -m scripts.predict_classifier --evaluate`).

## Limits

- Hand-written templates are few and repetitive in style; many reviewed rows will still be needed.
- Suggested labels are guesses. Context type does not decide the correct label.
- Coverage counts show gaps, not data quality.
