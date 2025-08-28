import pandas as pd
from feature_pipeline.load import load_train, load_holdout

def test_train_quality():
    df = load_train()

    # --- Schema ---
    expected_cols = [
        "date", "median_sale_price", "median_list_price", "median_ppsf", 
        "homes_sold", "inventory", "year", "price", "city", "zipcode"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    # --- Sanity checks ---
    assert len(df) > 0, "Training dataset is empty"
    assert df["price"].notnull().all(), "Price column has null values"
    assert (df["price"] > 0).all(), "Non-positive prices found in training set"
    assert df["year"].between(2011, 2021).all(), "Train should not include 2022+ years"

def test_holdout_quality():
    df = load_holdout()

    # --- Schema consistency with train ---
    train_df = load_train()
    assert set(df.columns) == set(train_df.columns), "Train/Holdout schema mismatch"

    # --- Holdout year check ---
    assert df["year"].min() >= 2022, "Holdout must only contain 2022+ data"

    # --- Target checks ---
    assert df["price"].notnull().all(), "Price column has null values in holdout"
    assert (df["price"] > 0).all(), "Non-positive prices found in holdout"

def test_basic_ranges():
    train_df = load_train()
    holdout_df = load_holdout()

    combined = pd.concat([train_df, holdout_df])

    # Year range sanity
    assert combined["year"].between(2010, 2025).all(), "Year out of range"

    # Price range sanity (very broad just to catch broken values)
    assert combined["price"].between(1000, 10_000_000).all(), "Suspicious price values found"
