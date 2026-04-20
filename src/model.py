"""Model training, cross-validation, and evaluation."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb

from src.paths import ROOT
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "log_price"
DROP_COLS = [
    "id",
    "price",
    "log_price",
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    "amenities",
]


def rmse_scorer(estimator, X, y):
    preds = estimator.predict(X)
    return -np.sqrt(np.mean((preds - y) ** 2))


def mae_scorer(estimator, X, y):
    preds = estimator.predict(X)
    return -np.mean(np.abs(preds - y))


def r2_scorer(estimator, X, y):
    return r2_score(y, estimator.predict(X))


SCORING = {
    "rmse": rmse_scorer,
    "mae": mae_scorer,
    "r2": r2_scorer,
}


def _cv_metrics(pipe, X_trainval, y_trainval):
    cv_res = cross_validate(
        pipe,
        X_trainval,
        y_trainval,
        cv=5,
        scoring=SCORING,
        n_jobs=1,
    )
    return {
        "cv_mae_mean": -np.mean(cv_res["test_mae"]),
        "cv_mae_std": np.std(-cv_res["test_mae"]),
        "cv_r2_mean": np.mean(cv_res["test_r2"]),
        "cv_r2_std": np.std(cv_res["test_r2"]),
    }


def _test_metrics(pipe, X_test, y_test, y_raw_test):
    y_pred_log = pipe.predict(X_test)
    y_pred_price = np.expm1(y_pred_log)
    return {
        "test_mae_log": mean_absolute_error(y_test, y_pred_log),
        "test_mae_eur": mean_absolute_error(y_raw_test, y_pred_price),
        "test_mape": float(
            np.mean(np.abs((y_raw_test - y_pred_price) / y_raw_test)) * 100
        ),
        "test_r2": r2_score(y_test, y_pred_log),
    }


def run_full_pipeline(df: pd.DataFrame) -> dict:
    """Train all models with CV, evaluate on test set, save metrics and district MAE."""
    y_raw = df["price"].copy()
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    nan_cols = [c for c in feature_cols if df[c].isna().all()]
    if nan_cols:
        print(f"Dropping all-NaN feature columns: {nan_cols}")
        feature_cols = [c for c in feature_cols if c not in nan_cols]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    X_trainval, X_test, y_trainval, y_test, y_raw_trainval, y_raw_test = (
        train_test_split(
            X,
            y,
            y_raw,
            test_size=0.30,
            random_state=42,
        )
    )

    binary_cols = [
        c for c in feature_cols if X_trainval[c].dropna().isin([0, 1]).all()
    ]
    numerical_cols = [c for c in feature_cols if c not in binary_cols]

    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    bin_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_transformer, numerical_cols),
            ("bin", bin_transformer, binary_cols),
        ],
        remainder="drop",
    )

    rows = []
    fitted: dict[str, Pipeline] = {}

    # Linear models
    linear_specs = [
        (
            "OLS (Ridge α=0.01)",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    ("model", Ridge(alpha=0.01)),
                ]
            ),
        ),
        (
            "RidgeCV",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
                ]
            ),
        ),
        (
            "LassoCV",
            Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    (
                        "model",
                        LassoCV(cv=5, random_state=42, max_iter=5000),
                    ),
                ]
            ),
        ),
    ]

    for name, pipe in linear_specs:
        print(f"Cross-validating {name}...")
        m_cv = _cv_metrics(pipe, X_trainval, y_trainval)
        pipe.fit(X_trainval, y_trainval)
        fitted[name] = pipe
        m_te = _test_metrics(pipe, X_test, y_test, y_raw_test)
        rows.append({"model": name, **m_cv, **m_te})

    # XGBoost: CV with n_estimators=200, then final with early stopping
    print("Cross-validating XGBoost (n_estimators=200)...")
    pipe_xgb_cv = Pipeline(
        [
            ("prep", clone(preprocessor)),
            (
                "model",
                xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    m_cv_xgb = _cv_metrics(pipe_xgb_cv, X_trainval, y_trainval)

    print("Fitting final XGBoost (500 trees, early stopping)...")
    prep_xgb = clone(preprocessor)
    prep_xgb.fit(X_trainval)
    X_tr_sub, X_es_sub, y_tr_sub, y_es_sub = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.15,
        random_state=42,
    )
    X_tr_t = prep_xgb.transform(X_tr_sub)
    X_es_t = prep_xgb.transform(X_es_sub)
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr_t,
        y_tr_sub,
        eval_set=[(X_es_t, y_es_sub)],
        verbose=False,
    )
    best_n = getattr(xgb_model, "best_iteration", None)
    if best_n is None:
        best_n = 499
    n_trees = int(best_n) + 1
    xgb_final = xgb.XGBRegressor(
        n_estimators=n_trees,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_final.fit(prep_xgb.transform(X_trainval), y_trainval)
    pipe_xgb_final = Pipeline([("prep", prep_xgb), ("model", xgb_final)])
    fitted["XGBoost (final, early stopping)"] = pipe_xgb_final
    m_te_xgb = _test_metrics(pipe_xgb_final, X_test, y_test, y_raw_test)
    rows.append(
        {
            "model": "XGBoost (final, early stopping)",
            **m_cv_xgb,
            **m_te_xgb,
        }
    )

    # LightGBM: CV n=200, final n=500
    print("Cross-validating LightGBM (n_estimators=200)...")
    pipe_lgb_cv = Pipeline(
        [
            ("prep", clone(preprocessor)),
            (
                "model",
                lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    m_cv_lgb = _cv_metrics(pipe_lgb_cv, X_trainval, y_trainval)

    print("Fitting final LightGBM (500 trees)...")
    pipe_lgb_final = Pipeline(
        [
            ("prep", clone(preprocessor)),
            (
                "model",
                lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    pipe_lgb_final.fit(X_trainval, y_trainval)
    fitted["LightGBM (final, n=500)"] = pipe_lgb_final
    m_te_lgb = _test_metrics(pipe_lgb_final, X_test, y_test, y_raw_test)
    rows.append(
        {
            "model": "LightGBM (final, n=500)",
            **m_cv_lgb,
            **m_te_lgb,
        }
    )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("test_r2", ascending=False).reset_index(
        drop=True
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    print("Model comparison (test R² descending):")
    print(results_df.to_string(index=False))
    print(f"Saved {results_path}")

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"Best model by test R²: {best_name}")

    y_pred_log = best_pipe.predict(X_test)
    y_pred_price = np.expm1(y_pred_log)
    test_df = X_test.copy()
    test_df["y_true_price"] = y_raw_test.values
    test_df["y_pred_price"] = y_pred_price
    test_df["neighbourhood_cleansed"] = df.loc[
        X_test.index, "neighbourhood_cleansed"
    ].values

    test_df["abs_err"] = np.abs(test_df["y_true_price"] - test_df["y_pred_price"])
    district_mae = (
        test_df.groupby("neighbourhood_cleansed", observed=True)["abs_err"]
        .mean()
        .sort_values(ascending=False)
        .rename("MAE_EUR")
    )
    district_path = out_dir / "district_mae.csv"
    district_mae.to_csv(district_path)
    print(f"Saved {district_path}")

    return {
        "results_df": results_df,
        "best_pipe": best_pipe,
        "X_trainval": X_trainval,
        "X_test": X_test,
        "y_test": y_test,
        "y_raw_test": y_raw_test,
        "feature_cols": feature_cols,
        "numerical_cols": numerical_cols,
        "binary_cols": binary_cols,
        "preprocessor": preprocessor,
        "district_mae": district_mae,
    }
