import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .config import MODELS_DIR


def load_model_and_threshold():
    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    threshold_path = MODELS_DIR / "threshold.json"

    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError(
            "Model or threshold not found. Make sure to run train_model.py and evaluate.py first."
        )

    model = joblib.load(model_path)
    with threshold_path.open() as f:
        threshold_info = json.load(f)

    threshold = float(threshold_info["threshold"])
    return model, threshold


def score_file(
    input_csv: str | Path,
    output_csv: Optional[str | Path] = None,
) -> Path:
    model, threshold = load_model_and_threshold()

    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_scored = df.copy()
    df_scored["fraud_probability"] = probs
    df_scored["fraud_flag"] = preds

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_scored.csv")

    output_csv = Path(output_csv)
    df_scored.to_csv(output_csv, index=False)
    print(f"Saved scored transactions to: {output_csv}")
    return output_csv


def main():
    # Example usage:
    # python -m src.score_new_transactions data/raw/synthetic_fraud_dataset.csv
    import argparse

    parser = argparse.ArgumentParser(description="Score new transactions for fraud risk.")
    parser.add_argument("input_csv", type=str, help="Path to CSV with transaction data.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Where to save scored CSV. Defaults to '<input>_scored.csv'",
    )
    args = parser.parse_args()

    score_file(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
