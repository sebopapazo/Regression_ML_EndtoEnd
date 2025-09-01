"""
⚡ Feature Engineering Script for Housing Regression MLE

- Loads cleaned train/eval datasets from data/processed/.
- Adds time features (year, quarter, month).
- Frequency encodes zipcode.
- Target encodes city_full using median price.
- Drops leakage/unneeded columns.
- Saves feature-engineered datasets to data/processed/.

"""

import pandas as pd
from pathlib import Path
from category_encoders import TargetEncoder

# ----------------------------
# Paths
# ----------------------------
PROCESSED_DIR = Path("data/processed")

TRAIN_PATH = PROCESSED_DIR / "cleaning_train.csv"
EVAL_PATH = PROCESSED_DIR / "cleaning_eval.csv"

OUT_TRAIN_PATH = PROCESSED_DIR / "feature_engineered_train.csv"
OUT_EVAL_PATH = PROCESSED_DIR / "feature_engineered_eval.csv"

# ----------------------------
# Feature Engineering Functions
# ----------------------------
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, quarter, month features from date column."""
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month

    # Reorder: year, quarter, month directly after date
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))

    return df


def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frequency encode a column (fit on train, apply to eval)."""
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    eval[f"{col}_freq"] = eval[col].map(freq_map).fillna(0)
    return train, eval


def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target encode a categorical feature (fit on train, apply to eval)."""
    te = TargetEncoder(cols=[col])
    train[f"{col}_encoded"] = te.fit_transform(train[col], train[target])
    eval[f"{col}_encoded"] = te.transform(eval[col])
    return train, eval


def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns not used in modeling (leakage or raw categorical)."""
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
    eval = eval.drop(columns=[c for c in drop_cols if c in eval.columns], errors="ignore")
    return train, eval

# ----------------------------
# Main pipeline
# ----------------------------
def run_feature_engineering():
    # Load cleaned datasets
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())

    # Date features
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)

    # Frequency encode zipcode
    train_df, eval_df = frequency_encode(train_df, eval_df, "zipcode")

    # Target encode city_full
    train_df, eval_df = target_encode(train_df, eval_df, "city_full", "price")

    # Drop unused/leakage columns
    train_df, eval_df = drop_unused_columns(train_df, eval_df)

    # Save processed datasets
    train_df.to_csv(OUT_TRAIN_PATH, index=False)
    eval_df.to_csv(OUT_EVAL_PATH, index=False)

    print("✅ Feature engineering complete.")
    print("Train shape:", train_df.shape)
    print("Eval shape:", eval_df.shape)


if __name__ == "__main__":
    run_feature_engineering()
