"""Predict invoice late-payment risk using a small educational ML model."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["amount_inr", "payment_terms_days", "previous_late_payments"]
TARGET = "paid_late"


def load_invoices(path, training=False):
    """Reject missing, negative, nonnumeric, and incomplete feature values."""
    frame = pd.read_csv(path)
    required = ["invoice_id", *FEATURES, *([TARGET] if training else [])]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("The CSV must contain rows with no missing required values.")
    if frame["invoice_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Invoice IDs must not be blank.")
    if frame["invoice_id"].duplicated().any():
        raise ValueError("Invoice IDs must be unique within each file.")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if ((frame[column] < 0) | (frame[column] == float("inf"))).any():
            raise ValueError(f"{column} must contain finite, nonnegative numbers.")
    for column in ["payment_terms_days", "previous_late_payments"]:
        if (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain whole numbers.")
    if training:
        if not frame[TARGET].isin([0, 1]).all():
            raise ValueError("paid_late must be 0 (on time) or 1 (late).")
        counts = frame[TARGET].value_counts()
        if len(frame) < 20 or len(counts) != 2 or counts.min() < 5:
            raise ValueError("Training needs at least 20 invoices and 5 of each class.")
    return frame


def train_model(frame):
    """Evaluate on held-out rows, then refit on all history for predictions."""
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.25,
        stratify=frame[TARGET], random_state=42,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    print(f"Held-out test invoices: {len(y_test)}")
    print(f"Model accuracy: {accuracy_score(y_test, predicted):.1%}")
    print(f"Majority baseline accuracy: {baseline.score(x_test, y_test):.1%}")
    print(classification_report(y_test, predicted, labels=[0, 1],
                                target_names=["On time", "Late"], zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted; order: on time, late):")
    print(confusion_matrix(y_test, predicted, labels=[0, 1]))
    model.fit(frame[FEATURES], frame[TARGET])
    return model


def predict_invoices(model, frame):
    result = frame[["invoice_id", *FEATURES]].copy()
    late_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(frame[FEATURES])[:, late_index]
    result["late_probability"] = probabilities.round(4)
    result["prediction"] = ["Likely late" if p >= 0.5 else "Likely on time"
                            for p in probabilities]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "data/invoices.csv")
    parser.add_argument("--input", type=Path, default=ROOT / "data/new_invoices.csv")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    try:
        if args.output.resolve() in {args.history.resolve(), args.input.resolve()}:
            raise ValueError("Output must differ from the history and input paths.")
        history = load_invoices(args.history, training=True)
        incoming = load_invoices(args.input)
        model = train_model(history)
        predictions = predict_invoices(model, incoming)
        predictions.to_csv(args.output, index=False)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print("\nPredictions:")
    print(predictions.to_string(index=False))
    print(f"\nSaved to: {args.output.resolve()}")
    print("Educational demo: fictional data; these scores are not validated business probabilities.")


if __name__ == "__main__":
    main()
