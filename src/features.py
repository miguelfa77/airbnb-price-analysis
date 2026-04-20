"""Feature engineering for Madrid Airbnb listings."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.paths import ROOT
import pandas as pd

LAT_SOL = 40.4168
LON_SOL = -3.7038


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(
        dlambda / 2
    ) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _amenity_slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _parse_amenities_cell(cell) -> list[str]:
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    try:
        val = ast.literal_eval(str(cell))
        if isinstance(val, list):
            return [str(x) for x in val]
    except (ValueError, SyntaxError):
        pass
    return []


def engineer_features(
    df: pd.DataFrame, metro_path: str | Path | None = None
) -> pd.DataFrame:
    """Add geospatial, amenity, and derived features; apply outlier rules; save parquet."""
    df = df.copy()
    metro_path = (
        Path(metro_path) if metro_path is not None else ROOT / "data/raw/metro_stations.json"
    )

    df["dist_sol_km"] = haversine_km(
        df["latitude"], df["longitude"], LAT_SOL, LON_SOL
    )
    df["dist_sol_km_sq"] = df["dist_sol_km"] ** 2

    if metro_path.exists() and metro_path.stat().st_size > 0:
        import json

        with open(metro_path, encoding="utf-8") as f:
            gj = json.load(f)
        coords_list = []
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") == "Point":
                lon_m, lat_m = geom["coordinates"][:2]
                coords_list.append([lat_m, lon_m])
            elif geom.get("type") == "MultiPoint":
                for pt in geom.get("coordinates", []):
                    lon_m, lat_m = pt[:2]
                    coords_list.append([lat_m, lon_m])
        if coords_list:
            metro_coords = np.asarray(coords_list, dtype=float)
            lat_arr = df["latitude"].to_numpy(dtype=float)[:, np.newaxis]
            lon_arr = df["longitude"].to_numpy(dtype=float)[:, np.newaxis]
            mlat = metro_coords[:, 0].reshape(1, -1)
            mlon = metro_coords[:, 1].reshape(1, -1)
            dists = haversine_km(lat_arr, lon_arr, mlat, mlon)
            df["dist_metro_km"] = dists.min(axis=1)
            print(
                f"Computed distance to nearest metro for {len(df):,} listings "
                f"({metro_coords.shape[0]} stations)."
            )
        else:
            df["dist_metro_km"] = np.nan
            print("Warning: metro GeoJSON had no point coordinates; dist_metro_km set to NaN.")
    else:
        df["dist_metro_km"] = np.nan
        print(
            f"Warning: metro file missing or empty at {metro_path}; "
            "dist_metro_km set to NaN."
        )

    # Amenities
    parsed = df["amenities"].apply(_parse_amenities_cell)
    flat: list[str] = []
    for lst in parsed:
        flat.extend(lst)
    freq = Counter(flat)
    TOP_AMENITIES = [a for a, _ in freq.most_common(30)]
    print("Top 30 amenities (name: count):")
    for a in TOP_AMENITIES:
        print(f"  {a}: {freq[a]}")

    for amenity in TOP_AMENITIES:
        col = f"amenity_{_amenity_slug(amenity)}"
        df[col] = parsed.apply(lambda lst, am=amenity: int(am in lst)).astype(int)

    df = df.drop(columns=["amenities"])

    n0 = len(df)
    df = df[df["accommodates"] <= 20].copy()
    print(f"Dropped {n0 - len(df):,} listings with accommodates > 20")
    n1 = len(df)
    df = df[df["bedrooms"] <= 10].copy()
    print(f"Dropped {n1 - len(df):,} listings with bedrooms > 10")

    # Sanitize column names (no spaces / odd chars)
    bad = [c for c in df.columns if not re.match(r"^[A-Za-z0-9_]+$", str(c))]
    if bad:
        rename = {c: re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_") for c in bad}
        df = df.rename(columns=rename)
    assert all(re.match(r"^[A-Za-z0-9_]+$", str(c)) for c in df.columns), (
        "Column names must be alphanumeric/underscore only."
    )

    feature_like = [
        c
        for c in df.columns
        if c
        not in ("id", "price", "log_price", "latitude", "longitude", "neighbourhood_cleansed")
    ]
    print(f"Final feature count (excluding id/target/geo/raw neighbourhood): {len(feature_like)}")

    out_path = ROOT / "data/processed/listings_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Overwriting existing feature file {out_path}")
    df.to_parquet(out_path, index=False)
    print(f"Saved feature-engineered data: {len(df):,} rows → {out_path}")
    return df
