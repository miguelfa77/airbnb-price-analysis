"""Permutation importance, PDPs, and district-level predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.paths import ROOT
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance


def _prep_feature_name(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def run_interpretation(pipeline_results: dict, df: pd.DataFrame) -> dict:
    """Permutation importance, partial dependence, full-dataset predictions."""
    pipe = pipeline_results["best_pipe"]
    X_test = pipeline_results["X_test"]
    y_test = pipeline_results["y_test"]
    X_trainval = pipeline_results["X_trainval"]
    numerical_cols = set(pipeline_results["numerical_cols"])

    print("Computing permutation feature importance...")
    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )
    # Permutation importance operates in original feature space (before prep)
    feat_names = list(pipeline_results["feature_cols"])

    imp_df = (
        pd.DataFrame(
            {
                "feature": feat_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    imp_path = out_dir / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved {imp_path}")

    # Top 5 continuous (original column names) by importance
    top_cont = []
    for fname in imp_df["feature"]:
        orig = _prep_feature_name(fname)
        if orig in numerical_cols and orig not in top_cont:
            top_cont.append(orig)
        if len(top_cont) >= 5:
            break

    pdp_results: dict = {}
    print(f"Partial dependence for: {top_cont}")
    X_pdp = X_trainval.copy()
    for c in pipeline_results["numerical_cols"]:
        if c in X_pdp.columns:
            X_pdp[c] = X_pdp[c].astype(float)
    for feat in top_cont:
        pd_res = partial_dependence(
            pipe,
            X_pdp,
            features=[feat],
            kind="average",
            grid_resolution=50,
        )
        pdp_results[feat] = {
            "average": np.asarray(pd_res["average"][0]),
            "grid_values": np.asarray(pd_res["grid_values"][0]),
        }

    X_full = df[pipeline_results["feature_cols"]]
    df_out = df.copy()
    df_out["predicted_log_price"] = pipe.predict(X_full)
    df_out["predicted_price"] = np.expm1(df_out["predicted_log_price"])

    dist_pred = (
        df_out.groupby("neighbourhood_cleansed", observed=True)
        .agg(
            median_actual_price=("price", "median"),
            median_predicted_price=("predicted_price", "median"),
            listing_count=("id", "count"),
        )
        .reset_index()
    )
    pred_path = out_dir / "district_predictions.csv"
    dist_pred.to_csv(pred_path, index=False)
    print(f"Saved {pred_path}")

    return {
        "imp_df": imp_df,
        "pdp_results": pdp_results,
        "df_with_preds": df_out,
    }
