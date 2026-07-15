from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_MODEL_PATH = Path("models/expense_classifier.joblib")
MIN_CONFIDENCE = 0.55


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working = working.dropna(subset=["category"])
    working = working[working["amount"] < 0]
    working["text"] = (
        working["merchant"].fillna("").astype(str)
        + " "
        + working["description"].fillna("").astype(str)
    ).str.strip()
    working = working[working["text"] != ""]
    return working


def build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words="english",
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_classifier(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    training_df = prepare_training_frame(df)
    if training_df["category"].nunique() < 2:
        raise ValueError("Need at least two labeled categories to train the classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        training_df["text"],
        training_df["category"],
        test_size=0.2,
        random_state=42,
        stratify=training_df["category"],
    )

    model = build_classifier()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return model, metrics


def save_model(model: Pipeline, model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier artifact not found at {model_path}. Run training first."
        )
    return joblib.load(model_path)


def predict_categories(
    df: pd.DataFrame,
    model: Optional[Pipeline] = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> pd.DataFrame:
    working = df.copy()
    if model is None:
        model = load_model()

    working["combined_text"] = (
        working["merchant"].fillna("").astype(str)
        + " "
        + working["description"].fillna("").astype(str)
    ).str.strip()

    probabilities = model.predict_proba(working["combined_text"])
    labels = model.classes_
    best_index = probabilities.argmax(axis=1)

    working["predicted_category"] = [labels[index] for index in best_index]
    working["prediction_confidence"] = probabilities.max(axis=1)
    working["needs_llm_review"] = working["prediction_confidence"] < min_confidence
    return working


if __name__ == "__main__":
    from data_loader import load_transactions

    dataset = load_transactions("data/sample_transactions.csv")
    classifier, report = train_classifier(dataset)
    path = save_model(classifier)
    print(f"Saved model to {path}")
    print(pd.DataFrame(report).transpose()[["precision", "recall", "f1-score"]].fillna(""))
