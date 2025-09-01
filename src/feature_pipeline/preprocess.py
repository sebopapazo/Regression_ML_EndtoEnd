"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

import pandas as pd
import re
from pathlib import Path

# ----------------------------
# Path setup
# ----------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# City name mapping (manual fixes)
# ----------------------------
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}

# ----------------------------
# Normalization utilities
# ----------------------------
def normalize_city(s: str) -> str:
    """Normalize city names: lowercase, strip, unify dashes."""
    if pd.isna(s):
        return s
    s = s.strip().lower()
    # replace en-dash, em-dash, and hyphen with plain "-"
    s = re.sub(r"[–—-]", "-", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s

# ----------------------------
# Core cleaning functions
# ----------------------------
def clean_and_merge(df: pd.DataFrame, metros_path: str = "data/raw/usmetros.csv") -> pd.DataFrame:
    """Normalize city names, apply mapping, merge lat/lng from metros dataset."""
    metros = pd.read_csv(metros_path)

    # Normalize city_full + metro_full
    df["city_full"] = df["city_full"].apply(normalize_city)
    metros["metro_full"] = metros["metro_full"].apply(normalize_city)

    # Apply city mapping (normalized)
    norm_mapping = {normalize_city(k): normalize_city(v) for k, v in CITY_MAPPING.items()}
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # Merge with lat/lng
    df = df.merge(
        metros[["metro_full", "lat", "lng"]],
        how="left",
        left_on="city_full",
        right_on="metro_full"
    )
    df.drop(columns=["metro_full"], inplace=True)

    # Log missing matches
    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("⚠️ Still missing lat/lng for:", missing)
    else:
        print("✅ All cities matched with metros dataset")

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    after = df.shape[0]
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(split: str):
    """Run preprocessing pipeline for a given split (train/eval/holdout)."""
    path = RAW_DIR / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = PROCESSED_DIR / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")

    return df

# ----------------------------
# Main runner
# ----------------------------
if __name__ == "__main__":
    for split in ["train", "eval", "holdout"]:
        preprocess_split(split)
