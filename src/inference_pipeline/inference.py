"""
Inference pipeline for Housing Regression MLE.

- Loads the trained/tuned model and saved encoders.
- Applies preprocessing + feature engineering (using fitted encoders).
- Returns predictions for new data.
- Can be run programmatically (predict(df)) or via CLI.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, drop_unused_columns

# ----------------------------
# Default paths
# ----------------------------
# Resolve project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("📂 Inference using project root:", PROJECT_ROOT)


# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    # Step 1: Preprocess
    df = clean_and_merge(input_df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    # Step 2: Feature engineering
    df = add_date_features(df)

    # Load encoders
    if Path(freq_encoder_path).exists():
        freq_map = load(freq_encoder_path)
        if "zipcode" in df.columns:
            df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
            df = df.drop(columns=["zipcode"], errors="ignore")

    if Path(target_encoder_path).exists():
        target_encoder = load(target_encoder_path)
        if "city_full" in df.columns:
            df["city_full_encoded"] = target_encoder.transform(df["city_full"])
            df = df.drop(columns=["city_full"], errors="ignore")

    # Drop unused/leakage columns
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Drop target column if present (prevents feature_names mismatch)
    if "price" in df.columns:
        df = df.drop(columns=["price"])

    # Step 3: Load model
    model = load(model_path)

    # Step 4: Predict
    preds = model.predict(df)
    df["predicted_price"] = preds

    return df


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new housing data.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    parser.add_argument("--freq_encoder", type=str, default=str(DEFAULT_FREQ_ENCODER), help="Path to frequency encoder pickle")
    parser.add_argument("--target_encoder", type=str, default=str(DEFAULT_TARGET_ENCODER), help="Path to target encoder pickle")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
