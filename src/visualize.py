"""Report figures (300 DPI PNG)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from src.paths import ROOT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)
PALETTE = "Blues_r"

FIG_DIR = ROOT / "outputs" / "figures"
MADRID_DISTRICTS_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/madrid.geojson"
)


def _ensure_fig_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def plot_price_distribution(df: pd.DataFrame) -> Path:
    _ensure_fig_dir()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    med = df["price"].median()
    sns.histplot(df["price"], kde=True, ax=axes[0], color="steelblue")
    axes[0].axvline(med, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Raw price")
    axes[0].set_xlabel("Nightly price (€)")
    axes[0].annotate(
        f"Median: €{med:.0f}",
        xy=(med, axes[0].get_ylim()[1] * 0.9),
        fontsize=10,
        color="red",
    )

    med_log = df["log_price"].median()
    sns.histplot(df["log_price"], kde=True, ax=axes[1], color="steelblue")
    axes[1].axvline(med_log, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Log-transformed price")
    axes[1].set_xlabel("log(1 + price)")

    plt.tight_layout()
    out = FIG_DIR / "fig1_price_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_model_comparison(results_df: pd.DataFrame) -> Path:
    _ensure_fig_dir()
    d = results_df.sort_values("test_mae_eur", ascending=True).copy()
    best = d.iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(d))
    colors = []
    for m in d["model"]:
        colors.append("#2c5282" if m == best else "#d0d0d0")
    bars = ax.barh(y_pos, d["test_mae_eur"], color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["model"])
    ax.set_xlabel("Test MAE (€)")
    ax.set_title("Test MAE by model (€, on price scale)")
    xmax = d["test_mae_eur"].max() * 1.35
    ax.set_xlim(0, xmax)
    for i, (_, row) in enumerate(d.iterrows()):
        ax.text(
            row["test_mae_eur"] + 0.3,
            i,
            f"€{row['test_mae_eur']:.1f}",
            va="center",
            fontsize=9,
        )
        ax.text(
            row["test_mae_eur"] + xmax * 0.12,
            i,
            f"R²={row['test_r2']:.3f}",
            va="center",
            fontsize=8,
            color="dimgray",
        )
    plt.tight_layout()
    out = FIG_DIR / "fig2_model_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def _group_color(feat: str) -> str:
    f = feat.lower()
    if f.startswith("amenity_"):
        return "#e07a5f"
    if f.startswith("room_type_"):
        return "#6c757d"
    if any(
        x in f
        for x in (
            "review_scores",
            "number_of_reviews",
            "host_response",
            "superhost",
        )
    ):
        return "#f4a261"
    if any(x in f for x in ("dist_", "metro", "neighbourhood")):
        return "#2a9d8f"
    return "#1d3557"


def plot_feature_importance(imp_df: pd.DataFrame) -> Path:
    _ensure_fig_dir()
    d = imp_df.sort_values("importance_mean", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(d))
    bar_cols = [_group_color(str(x)) for x in d["feature"]]
    xerr = d["importance_std"].values
    ax.barh(
        y_pos,
        d["importance_mean"],
        xerr=xerr,
        color=bar_cols,
        height=0.65,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    labels = [str(x).replace("_", " ").title() for x in d["feature"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean decrease in R² (permutation importance)")
    plt.tight_layout()
    out = FIG_DIR / "fig3_feature_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def _feat_label(feat: str) -> str:
    mapping = {
        "dist_sol_km": "Distance to centre (km)",
        "dist_sol_km_sq": "Squared distance to centre (km²)",
        "dist_metro_km": "Distance to nearest metro (km)",
        "accommodates": "Accommodates (guests)",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "minimum_nights": "Minimum nights",
        "availability_365": "Availability (days/year)",
        "host_listings_count": "Host listing count",
        "number_of_reviews": "Number of reviews",
        "review_scores_rating": "Overall rating",
        "review_scores_accuracy": "Accuracy score",
        "review_scores_cleanliness": "Cleanliness score",
        "review_scores_checkin": "Check-in score",
        "review_scores_communication": "Communication score",
        "review_scores_location": "Location score",
        "review_scores_value": "Value score",
        "neighbourhood_id": "Neighbourhood id",
    }
    return mapping.get(feat, feat.replace("_", " ").title())


def plot_pdps(pdp_results: dict) -> Path:
    _ensure_fig_dir()
    feats = list(pdp_results.keys())
    n = len(feats)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No PDP data", ha="center")
        out = FIG_DIR / "fig4_pdp_plots.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, feats):
        grid = pdp_results[feat]["grid_values"]
        avg = np.asarray(pdp_results[feat]["average"]).ravel()
        centered = avg - np.mean(avg)
        ax.plot(grid, centered, color="#2c5282", linewidth=2)
        ax.axhspan(-0.05, 0.05, color="0.85", zorder=0)
        ax.set_title(_feat_label(feat))
        ax.set_xlabel(_feat_label(feat))
        ax.set_ylabel("Δ predicted log-price (from curve mean)")
        # rug at bottom
        ymin, ymax = ax.get_ylim()
        rug_y = ymin + 0.02 * (ymax - ymin)
        ax.scatter(
            grid,
            np.full_like(grid, rug_y, dtype=float),
            s=4,
            alpha=0.15,
            color="gray",
            marker="|",
            linewidths=0.5,
        )

    for j in range(len(feats), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig4_pdp_plots.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_choropleth(df: pd.DataFrame, district_predictions_df: pd.DataFrame) -> Path:
    _ensure_fig_dir()
    raw_path = ROOT / "data/raw/madrid_districts.geojson"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        import requests

        try:
            r = requests.get(
                MADRID_DISTRICTS_URL,
                timeout=120,
                headers={"User-Agent": "airbnb-madrid-analysis/1.0"},
            )
            r.raise_for_status()
            raw_path.write_bytes(r.content)
        except Exception as exc:
            print(f"Could not download Madrid districts GeoJSON: {exc}")
            return FIG_DIR / "fig_choropleth.png"

    gdf = gpd.read_file(raw_path)
    merge_df = district_predictions_df.rename(
        columns={"neighbourhood_cleansed": "name"}
    )
    gdf = gdf.merge(
        merge_df,
        on="name",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(
        column="median_actual_price",
        cmap="YlOrRd",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey"},
        edgecolor="0.7",
        linewidth=0.3,
    )
    ax.set_title("Median nightly Airbnb price by district (€)")
    ax.axis("off")

    # Label every district that matched the data merge (inside polygon when possible)
    valid = gdf.dropna(subset=["median_actual_price"])
    n_lbl = len(valid)
    if n_lbl > 0:
        fs = max(4.0, min(7.0, 11.0 - 0.18 * n_lbl))
        for _, row in valid.iterrows():
            pt = row.geometry.representative_point()
            ax.annotate(
                text=row["name"],
                xy=(pt.x, pt.y),
                ha="center",
                va="center",
                fontsize=fs,
                color="black",
            )

    plt.tight_layout()
    out = FIG_DIR / "fig_choropleth.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out
