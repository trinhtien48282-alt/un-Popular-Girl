"""Standalone ML classifier for terminology usage (prototype). Offline, no Qt.

Not wired into the GUI or the rule-based scanner yet.

Predicts one of the scanner's three categories from (term, text) with a saved
scikit-learn Pipeline (TF-IDF + logistic regression) trained by
scripts/train_classifier.py. It learns from labelled example sentences. It is
not an LLM and cannot tell what actually happened between people.
Same saved model + same input = same output. No training here, no network.

Only load model files you trained yourself: joblib files are pickles.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from .dictionary import resource_path
from .models import Category

DEFAULT_MODEL_PATH = resource_path("models", "terminology_classifier.joblib")


class ClassifierError(RuntimeError):
    """Model file is missing, unreadable, or not a usable classifier."""


def build_input(term: str, text: str) -> str:
    """Model input: a term token plus the text. Shared by training and inference."""
    slug = re.sub(r"\W+", "_", term.strip().lower()).strip("_")
    return f"term_{slug} " + text.replace("\u2019", "'")


@dataclass(frozen=True)
class Prediction:
    label: Category
    probabilities: dict[Category, float]


class Classifier:
    def __init__(self, pipeline: Pipeline):
        valid = {c.value for c in Category}
        if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
            raise ClassifierError("Model file does not contain a scikit-learn Pipeline with predict_proba")
        if not set(pipeline.classes_) <= valid:
            raise ClassifierError(f"Model has unknown labels: {sorted(set(pipeline.classes_) - valid)}")
        self._pipe = pipeline

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "Classifier":
        path = Path(path)
        if not path.is_file():
            raise ClassifierError(
                f"Model file not found: {path}. Train one with: uv run python -m scripts.train_classifier"
            )
        try:
            pipeline = joblib.load(path)
        except Exception as exc:  # pickle/joblib raise many different types
            raise ClassifierError(f"Model file could not be read ({path}): {exc}") from exc
        return cls(pipeline)

    def predict(self, term: str, text: str, min_confidence: float | None = None) -> Prediction:
        """Argmax label plus class probabilities.

        min_confidence is off by default: no threshold has been evaluated yet.
        If you set one, a top probability below it returns NEEDS CONTEXT.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        probs = self._pipe.predict_proba([build_input(term, text)])[0]
        scores = {Category(c): float(p) for c, p in zip(self._pipe.classes_, probs)}
        label = max(scores, key=scores.get)  # ties: first (alphabetical) class, so deterministic
        if min_confidence is not None and scores[label] < min_confidence:
            label = Category.NEEDS_CONTEXT
        return Prediction(label, scores)
