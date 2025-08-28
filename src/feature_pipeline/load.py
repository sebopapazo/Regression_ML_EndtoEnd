import pandas as pd
from pathlib import Path

# Base path for data
DATA_DIR = Path("data")

def load_train() -> pd.DataFrame:
    """Load raw training data (excludes 2022–2023)."""
    return pd.read_csv(DATA_DIR / "raw" / "train.csv")

def load_holdout() -> pd.DataFrame:
    """Load holdout data (2022–2023)."""
    return pd.read_csv(DATA_DIR / "raw" / "holdout.csv")

def load_usmetros() -> pd.DataFrame:
    """Load auxiliary dataset for cleaning (optional)."""
    return pd.read_csv(DATA_DIR / "raw" / "usmetros.csv")

def load_train_leakage_safe() -> pd.DataFrame:
    """Load final processed dataset used for ML training (leakage-free)."""
    return pd.read_csv(DATA_DIR / "processed" / "train_leakage_safe.csv")

def save_data(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to CSV."""
    df.to_csv(path, index=False)
