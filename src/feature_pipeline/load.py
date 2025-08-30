import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_and_split_data(raw_path: str = "data/raw/untouched_raw_original.csv"):
    """Load raw dataset, split into train/eval/holdout by date."""
    df = pd.read_csv(raw_path)

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date (important for time-series style splits)
    df = df.sort_values("date")

    # Define cutoffs
    cutoff_date_eval = "2020-01-01"     # eval starts
    cutoff_date_holdout = "2022-01-01"  # holdout starts

    # Train: before 2020
    train_df = df[df["date"] < cutoff_date_eval]

    # Eval: 2020–2021
    eval_df = df[(df["date"] >= cutoff_date_eval) & (df["date"] < cutoff_date_holdout)]

    # Holdout: 2022–2023
    holdout_df = df[df["date"] >= cutoff_date_holdout]

    # Save splits
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    eval_df.to_csv(DATA_DIR / "eval.csv", index=False)
    holdout_df.to_csv(DATA_DIR / "holdout.csv", index=False)

    print("✅ Data split completed.")
    print(f"Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()
