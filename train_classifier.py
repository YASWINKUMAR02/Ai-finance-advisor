from __future__ import annotations

import json
from pathlib import Path

from data_loader import load_transactions
from expense_classifier import save_model, train_classifier


def main() -> None:
    dataset = load_transactions("data/sample_transactions.csv")
    model, metrics = train_classifier(dataset)
    model_path = save_model(model)
    metrics_path = Path("models/classifier_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved classifier artifact to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
