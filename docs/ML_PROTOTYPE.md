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
