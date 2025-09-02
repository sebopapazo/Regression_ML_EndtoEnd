"""
Inference pipeline for Housing Regression MLE.

- Loads the trained/tuned model from disk.
- Applies the same preprocessing + feature engineering steps as training.
- Returns predictions for new data.
- Can be used programmatically (predict(df)) or via CLI.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering
from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, frequency_encode, target_encode, drop_unused_columns


# ----------------------------
# Default paths
# ----------------------------
DEFAULT_MODEL = Path("models/xgb_best_model.pkl")
DEFAULT_OUTPUT = Path("predictions.csv")


# ----------------------------
# Core inference function
# ----------------------------
def predict(input_df: pd.DataFrame, model_path: Path | str = DEFAULT_MODEL) -> pd.DataFrame:
    # Step 1: Preprocess (same as training)
    df = clean_and_merge(input_df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    # Step 2: Feature engineering
    df = add_date_features(df)

    # For inference, we need consistency:
    # Use training data encoders to fit frequency/target encodings.
    # Here we reload train/eval to build encoders.
    train_df = pd.read_csv("/Users/riadanas/Desktop/housing regression MLE/data/processed/feature_engineered_train.csv")
    eval_df = pd.read_csv("/Users/riadanas/Desktop/housing regression MLE/data/processed/feature_engineered_eval.csv")

    # Frequency encode zipcode
    train_df, eval_df = frequency_encode(train_df, eval_df, "zipcode")
    _, df = frequency_encode(train_df, df, "zipcode")

    # Target encode city_full
    train_df, eval_df = target_encode(train_df, eval_df, "city_full", "price")
    _, df = target_encode(train_df, df, "city_full", "price")

    # Drop unused/leakage columns
    df, _ = drop_unused_columns(df.copy(), eval_df.copy())

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

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    raw_df = pd.read_csv(input_path)
    preds_df = predict(raw_df, model_path=model_path)

    preds_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

