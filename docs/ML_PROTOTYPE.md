# ML prototype (not in the GUI)

TF-IDF + logistic regression that predicts `POSSIBLE MISUSE` / `NEEDS CONTEXT` / `PLAUSIBLE USAGE` from `(term, text)`.
It learns from labelled example sentences. It is not an LLM and cannot tell what happened between people.
Offline, no GUI use, scanner untouched.

## Commands (PowerShell, repo root)

```powershell
uv sync                                                  # 1. install/sync (adds scikit-learn, updates uv.lock)
uv run python -m scripts.train_classifier --validate-only   # 2. validate dataset
uv run python -m scripts.train_classifier                    # 3. evaluate + train + save models\terminology_classifier.joblib
uv run python -m scripts.train_classifier --evaluate-only    # 4. evaluation only, saves nothing
uv run pytest                                            # 5. all tests (or: uv run python -m unittest discover -s tests -v)
```

## Dataset: `data/training_examples.csv`

Columns: `term, text, label, review_status` (`reviewed` | `needs_review`).
Only `reviewed` rows are trained on or evaluated. `needs_review` rows are validated and ignored.
Rows are `reviewed` only because you wrote them (dictionary examples, scanner test sentences). Every row I wrote or relabeled is `needs_review`. Read the CSV and flip rows to `reviewed` as you approve them.

## Evaluation

Near-duplicate rows (word-set Jaccard >= 0.6, any term) form groups. `StratifiedGroupKFold` (3 folds, seed 42) keeps each group in one fold. Reported metrics are out-of-fold. The saved model is then fit on all reviewed rows.
Training refuses to run with fewer than 8 reviewed rows or 3 groups per label.

## Inference

```python
from app.classifier import Classifier
clf = Classifier.load()            # models/terminology_classifier.joblib
clf.predict("Gaslighting", "He disagreed with me. That's gaslighting.")   # label + probabilities
clf.predict(term, text, min_confidence=0.5)  # optional; below it -> NEEDS CONTEXT. Off by default: no threshold evaluated yet.
```
Model files are pickles (joblib): only load ones you trained.

## Before hybrid integration

- Review and grow the dataset; NEEDS CONTEXT is thin and mostly about "gaslighting".
- Evaluate a `min_confidence` value before using one.
- Decide how ML and scanner disagreements are shown.
- Add scikit-learn to `requirements.txt` and the PyInstaller command (not done: the GUI does not use it).

## Trying it: `scripts/predict_classifier.py`

```powershell
uv run python -m scripts.predict_classifier --term Gaslighting --text "He disagreed with me. That's gaslighting."
uv run python -m scripts.predict_classifier --term Toxic --text "..." --term Trauma --text "..."   # several, paired in order
uv run python -m scripts.predict_classifier --file examples.csv                                    # columns: term,text
uv run python -m scripts.predict_classifier --file data\heldout_examples.csv                       # predictions only, no scoring
uv run python -m scripts.predict_classifier --evaluate                                             # held-out evaluation
```

Output is model scores, not a chance of being right. Each example also says whether it is near-identical (similarity >= 0.6) to a reviewed training row: `SEEN IN TRAINING` or `not in training`. That check reads `data/training_examples.csv`, not the model file, so it assumes the model was trained from that CSV.

### Held-out evaluation

`data/heldout_examples.csv` holds 21 candidate rows (negation, academic discussion, ordinary disagreement, sustained pattern, ambiguous or quoted wording, one longer passage). All are `needs_review`: I wrote them and the labels are my guess. None is near-identical to a training row (max similarity 0.38 at the time of writing).
`--evaluate` scores only `reviewed` rows, and drops any that are near-identical to a reviewed training row. It exits with an error until you review rows (set `review_status` to `reviewed` after reading and fixing the label). Do not copy held-out rows into the training CSV and then still call them held out.
Word overlap is a crude leakage check: it catches rephrasings and repeats, not sentences on the same theme.

## Candidate examples

To reduce hand-writing, see `docs/CANDIDATE_WORKFLOW.md` (`scripts/generate_candidates.py` and `scripts/promote_candidates.py`). Generated rows are always `needs_review` and are never trained on until you approve them.
