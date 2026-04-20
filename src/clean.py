"""Cleaning pipeline for InsideAirbnb Madrid listings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import ROOT

KEEP_COLS = [
    "id",
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    "room_type",
    "accommodates",
    "bedrooms",
    "bathrooms_text",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "host_is_superhost",
    "host_response_rate",
    "host_listings_count",
    "amenities",
    "price",
]


def clean_listings(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load raw listings, clean, impute, encode, save parquet."""
    raw_path = Path(raw_path) if raw_path is not None else ROOT / "data/raw/listings.csv"
    out_path = ROOT / "data/processed/listings_clean.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Loaded raw: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Column names: {list(df.columns)}")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw file missing required columns: {missing}. "
            f"InsideAirbnb may have renamed fields."
        )

    df = df[KEEP_COLS].copy()

    # Price
    df["price"] = (
        df["price"].astype(str).str.replace(r"[\$,]", "", regex=True)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    n_bad_price = df["price"].isna().sum() + (df["price"] == 0).sum()
    df = df[df["price"].notna() & (df["price"] > 0)]
    print(
        f"Dropped {n_bad_price:,} rows with NaN or zero price; {len(df):,} rows remain"
    )

    # Inactive listings
    mask_inactive = (df["availability_365"] == 0) & (df["number_of_reviews"] == 0)
    n_inactive = mask_inactive.sum()
    df = df[~mask_inactive].copy()
    print(
        f"After removing inactive listings: {len(df):,} rows remain ({n_inactive:,} dropped)"
    )

    # Extreme prices: winsorize top 1%, drop < 5 EUR
    p01, p99 = df["price"].quantile([0.01, 0.99])
    print(f"Price percentiles before trimming: 1st={p01:.2f} €, 99th={p99:.2f} €")
    n_below5 = (df["price"] < 5).sum()
    df = df[df["price"] >= 5].copy()
    n_above99 = (df["price"] > p99).sum()
    df.loc[df["price"] > p99, "price"] = p99
    print(
        f"Winsorized top 1% to 99th percentile ({n_above99:,} capped); "
        f"dropped {n_below5:,} rows below €5/night; {len(df):,} rows remain"
    )

    # Bathrooms
    df["bathrooms"] = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)").astype(float)
    df = df.drop(columns=["bathrooms_text"])

    # Host response rate
    df["host_response_rate"] = (
        df["host_response_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .replace("nan", np.nan)
    )
    df["host_response_rate"] = pd.to_numeric(df["host_response_rate"], errors="coerce")
    df["host_response_rate"] = df["host_response_rate"] / 100.0

    # Superhost
    def _superhost_to_int(x):
        if pd.isna(x):
            return 0
        if x in (True, "t", "True"):
            return 1
        if x in (False, "f", "False"):
            return 0
        return 0

    df["host_is_superhost"] = df["host_is_superhost"].map(_superhost_to_int).astype(int)

    # Missingness flags + imputation
    impute_cols = [
        "bedrooms",
        "bathrooms",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "host_response_rate",
        "host_listings_count",
    ]

    miss_rows = []
    for col in impute_cols:
        flag = f"{col}_missing"
        df[flag] = df[col].isna().astype(int)
        miss_rows.append(
            {
                "column": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_fraction": float(df[col].isna().mean()),
            }
        )

    miss_df = pd.DataFrame(miss_rows)
    print("Missingness before imputation:")
    print(miss_df.to_string(index=False))

    med_by_room = df.groupby("room_type")[["bedrooms", "bathrooms"]].transform("median")
    df["bedrooms"] = df["bedrooms"].fillna(med_by_room["bedrooms"])
    df["bathrooms"] = df["bathrooms"].fillna(med_by_room["bathrooms"])
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    for col in [
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "host_response_rate",
    ]:
        df[col] = df[col].fillna(df[col].median())

    df["host_listings_count"] = df["host_listings_count"].fillna(1)

    df["log_price"] = np.log1p(df["price"])

    df = pd.get_dummies(df, columns=["room_type"], drop_first=False)
    room_dummy_cols = [c for c in df.columns if c.startswith("room_type_")]
    for c in room_dummy_cols:
        df[c] = df[c].astype(int)

    df["neighbourhood_id"] = pd.Categorical(df["neighbourhood_cleansed"]).codes

    df.to_parquet(out_path, index=False)
    print(f"Saved cleaned data: {len(df):,} rows → {out_path}")
    return df
